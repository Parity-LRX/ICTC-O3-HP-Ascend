# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Equivalent CSR-neighbor splitting for the live five-input first VJP.

Experimental schedules share the current algebra and edge writes. Only the
source-feature reduction has S partial owners; its final sum is deterministic.
"""
import torch

from mace_ictc.models.generated_cartesian_message import _sum


_EXTENSIONS = {}


def _source(p, projection, splits, readonly=False):
    from .coherent import _csr_adjoint_source, _outputs
    original = _csr_adjoint_source(p, projection)
    code = original[:original.index("void coherent_adjoint(")]
    code = code.replace("const int64_t task=", "const int64_t task0=", 1)
    code = code.replace("if(task>=", f"const int split=task0%{splits};const int64_t task=task0/{splits};\nif(task>=", 1)
    code = code.replace("index=PTR[src];index<PTR[src+1];++index",
                        f"index=PTR[src]+split;index<PTR[src+1];index+={splits}")
    code = code.replace(f"DX[src*{p['xdim']}+", f"DX[(split*N+src)*{p['xdim']}+")
    if readonly:
        code = code.replace('float* ', 'float* __restrict__ ').replace('int64_t* ', 'int64_t* __restrict__ ')
    if p["c"] % 32 == 0:
        code = code.replace(f"const bool valid=channel<{p['c']};", "const bool valid=true;")
    flags = ("needx", "needh", "needg", "needr", "needi")
    for flag in flags:
        code = code.replace("bool "+flag, "bool unused_"+flag)
    code = code.replace("const int lane=", "constexpr bool needx=true,needh=true,needg=true,needr=true,needi=true;\n const int lane=", 1)
    code += f"""
__global__ void coherent_csr_dx_reduce(const float* P,float* Y,int64_t count) {{
 const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count)Y[i]={_sum(f'P[{s}*count+i]' for s in range(splits))};
}}
"""
    outputs = _outputs(p)
    wrapper_args = ",".join(f"torch::Tensor v{l}" for l, _, _ in outputs)
    pointer_args = ",".join(f"v{l}.data_ptr<float>()" for l, _, _ in outputs)
    groups = len({(row[2], row[1]) for row in p["path_specs"]})
    tiles = (p["c"]+31)//32
    code += f"""
void coherent_csr_split(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor dst,torch::Tensor mask,{wrapper_args},
 torch::Tensor ptr,torch::Tensor order,torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,
 torch::Tensor dr,torch::Tensor di,torch::Tensor px) {{
 const c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
 if(x.size(0)){{
 coherent_adjoint_kernel<<<(x.size(0)*{tiles*groups*splits}+3)/4,128,0,stream>>>(
 x.data_ptr<float>(),h.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
 r.data_ptr<float>(),i.data_ptr<float>(),dst.data_ptr<int64_t>(),mask.data_ptr<float>(),
 {pointer_args},ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),px.data_ptr<float>(),
 dh.data_ptr<float>(),dg.data_ptr<float>(),dr.data_ptr<float>(),di.data_ptr<float>(),
 x.size(0),dst.numel(),mask.numel()!=0,true,true,true,true,true);
 coherent_csr_dx_reduce<<<(dx.numel()+255)/256,256,0,stream>>>(px.data_ptr<float>(),dx.data_ptr<float>(),dx.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 }}
}}
"""
    return code


def adjoint(x, sh, w, g, real, imag, src, dst, mask, p, projection, needed, grads, *, splits=4, readonly=False, factored=False):
    if not all(needed):
        raise ValueError("Specialized CSR first adjoint requires all five input gradients")
    from .coherent import _HEADERS, _outputs
    from mace_ictc.models.message_topology import topology
    outputs = _outputs(p)
    p = {**p, 'factor_edge_scalars': factored}
    key = (p["path_specs"], p["xdim"], p["odim"], p["c"], p["phase_heads"], p["phase_angular"], splits, readonly, factored,
           tuple(tuple(row) for row in projection["static_rows"]))
    if key not in _EXTENSIONS:
        from torch.utils.cpp_extension import load_inline
        name = f"ictc_coherent_csr_split_v1_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}_a{int(p['phase_angular'])}_s{splits}"
        if readonly:name += '_readonly'
        if factored:name += '_factor'
        declarations = "#include <torch/extension.h>\nvoid coherent_csr_split("+",".join(
            ["torch::Tensor"]*(16+len(outputs)))+");\n"
        _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declarations,
            cuda_sources=_HEADERS+_source(p, projection, splits, readonly), functions=["coherent_csr_split"],
            extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    dx, dh, dg, dr, di = torch.zeros_like(x), torch.zeros_like(sh), torch.empty_like(g), torch.zeros_like(real), torch.zeros_like(imag)
    # Restricted TP path sets need exact zeros for any unused input degree.
    px = x.new_zeros(splits, *x.shape)
    cotangents = [grad.transpose(-1, -2).contiguous() if grad is not None else x.new_zeros(
        x.shape[0], 3, 2*l+1, width//(2*l+1)) for (l, _, width), grad in zip(outputs, grads)]
    if src.numel():
        _EXTENSIONS[key].coherent_csr_split(x, sh, w, g, real, imag, dst, mask,
            *cotangents, *topology(src, x.shape[0]), dx, dh, dg, dr, di, px)
    else:
        dg.zero_()
    return dx, dh, dg, dr, di
