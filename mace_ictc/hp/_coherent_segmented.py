# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Experimental edge/input-degree ownership for the coherent first adjoint.

Small harmonic/phase partials replace many source-CSR atomic additions. DX is
either source-owned, or emitted per edge then deterministically CSR-reduced.
Both schedules evaluate every original TP path; no model branch is omitted.
"""
import torch

from mace_ictc.models.generated_cartesian_message import _literal, _sum


_EXTENSIONS = {}


def _source(p, projection, *, edge_dx):
    from .coherent import _columns, _outputs
    columns, outputs = _columns(p, projection), _outputs(p)
    c, hdim, phases = p["c"], projection["hdim"], p["phase_count"]
    threads = 32 if c <= 32 else 128
    warps = threads // 32
    groups = {}
    for row in p["path_specs"]:
        groups.setdefault((row[2], row[1]), []).append(row)
    slots = hdim + 2 * phases
    cotangents = ",".join(f"const float* V{l}" for l, _, _ in outputs)
    lines = ["""
__global__ void coherent_segmented_edge_kernel(const float* X,const float* H,const float* W,
 const float* G,const float* R,const float* I,const int64_t* SRC,const int64_t* DST,
 const float* MASK,""" + cotangents + """,float* DH,float* DG,float* DR,float* DI,float* DX,
 int64_t E,bool masked,bool needx,bool needh,bool needg,bool needr,bool needi) {
 const int64_t e=blockIdx.x,src=SRC[e],dst=DST[e];
 const int group=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
 const float mask=masked?MASK[e]:1.f;
"""]
    for h in range(hdim):
        lines += [f"const float h{h}=H[e*{hdim}+{h}];", f"float dh{h}=0.f;"]
    for phase in range(phases):
        lines += [f"float dr{phase}=0.f,di{phase}=0.f;"]
    for number, ((xb, ni), paths) in enumerate(groups.items()):
        lines += [f"if(group=={number}){{", f"for(int channel=threadIdx.x;channel<{c};channel+={threads}){{",
                  f"const int head=channel/{c//p['phase_heads']};"]
        for j in range(ni):
            lines.append(f"const float x{j}=X[src*{p['xdim']}+{xb+j*c}+channel];")
            if edge_dx:
                lines.append(f"float dx{j}=0.f;")
        for path, _, _, bb, bs, ob, m in paths:
            degree = (m-1)//2
            _, base, width = next(row for row in outputs if row[0] == degree)
            phase = degree*p["phase_heads"] if p["phase_angular"] else 0
            lines += ["{", f"const int phase={phase}+head;",
                      f"const float cr=R[e*{phases}+phase],ci=I[e*{phases}+phase];",
                      f"const float gate=G[e*{p['p']*c}+{path*c}+channel];",
                      f"const float scale=W[{path*c}+channel]*mask;"]
            for mag in range(m):
                offset = mag*(width//m)+(ob-base)//m
                for stream, prefix in enumerate(("v", "vr", "vi")):
                    lines.append(f"const float {prefix}{mag}=V{degree}[(dst*3+{stream})*{width}+{offset}+channel];")
                lines.append(f"const float b{mag}=v{mag}+cr*vr{mag}+ci*vi{mag};")
                for j in range(ni):
                    if columns[bb+j*bs+mag]:
                        lines.append(f"const float q{j}_{mag}="+_sum(
                            f"{_literal(value)}*h{h}" for h, value in columns[bb+j*bs+mag])+";")
                lines.append(f"const float t{mag}="+_sum(
                    f"x{j}*q{j}_{mag}" for j in range(ni) if columns[bb+j*bs+mag])+";")
            value = _sum(f"b{mag}*t{mag}" for mag in range(m))
            lines.append(f"if(needg)DG[e*{p['p']*c}+{path*c}+channel]=({value})*scale;")
            if edge_dx:
                for j in range(ni):
                    value = _sum(f"b{mag}*q{j}_{mag}" for mag in range(m) if columns[bb+j*bs+mag])
                    lines.append(f"if(needx)dx{j}+=gate*scale*({value});")
            for head in range(p["phase_heads"]):
                dr = _sum(f"vr{mag}*t{mag}" for mag in range(m))
                di = _sum(f"vi{mag}*t{mag}" for mag in range(m))
                lines += [f"if(head=={head}){{dr{phase+head}+=({dr})*gate*scale;",
                          f"di{phase+head}+=({di})*gate*scale;}}"]
            lines.append("if(needh){")
            for j in range(ni):
                for mag in range(m):
                    if columns[bb+j*bs+mag]:
                        lines.append(f"const float db{j}_{mag}=x{j}*b{mag}*gate*scale;")
                        for h, value in columns[bb+j*bs+mag]:
                            lines.append(f"dh{h}+=({_literal(value)})*db{j}_{mag};")
            lines += ["}", "}"]
        if edge_dx:
            for j in range(ni):
                lines.append(f"if(needx)DX[e*{p['xdim']}+{xb+j*c}+channel]=dx{j};")
        lines += ["}", "}"]
    lines.append(f"__shared__ float partial[{slots*warps}];")
    reductions = [(f"dh{h}", "DH", h, hdim, "needh") for h in range(hdim)]
    reductions += [(f"dr{i}", "DR", i, phases, "needr") for i in range(phases)]
    reductions += [(f"di{i}", "DI", i, phases, "needi") for i in range(phases)]
    for slot, (value, _, _, _, need) in enumerate(reductions):
        lines += [f"if({need}){{",
                  f"for(int d=16;d>0;d/=2){value}+=__shfl_down_sync(0xffffffff,{value},d);",
                  f"if(lane==0)partial[{slot*warps}+warp]={value};", "}"]
    lines.append("__syncthreads();")
    for slot, (_, target, offset, stride, need) in enumerate(reductions):
        value = _sum(f"partial[{slot*warps+warp}]" for warp in range(warps))
        lines.append(f"if(threadIdx.x=={slot%threads}&&{need}){target}[(group*E+e)*{stride}+{offset}]={value};")
    lines += ["}", f"""
__global__ void coherent_partial_reduce_kernel(const float* P,float* Y,int64_t count) {{
 const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count)Y[i]={_sum(f'P[{g}*count+i]' for g in range(len(groups)))};
}}
__global__ void coherent_edge_dx_reduce_kernel(const float* P,const int64_t* PTR,
 const int64_t* ORDER,float* Y,int64_t N) {{
 const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=N*{p['xdim']})return;
 const int64_t node=i/{p['xdim']},channel=i%{p['xdim']};float value=0.f;
 for(int64_t j=PTR[node];j<PTR[node+1];++j)value+=P[ORDER[j]*{p['xdim']}+channel];
 Y[i]=value;
}}
"""]
    # Keep the established deterministic source DX expression for the first
    # candidate, so only the expensive harmonic/phase ownership changes.
    from ._coherent_adjoint import source
    old = source(p, projection, columns, outputs)
    lines.append(old[old.index("__global__ void coherent_source_adjoint_kernel"):
                     old.index("void coherent_adjoint(")])
    wrapper_args = ",".join(f"torch::Tensor v{l}" for l, _, _ in outputs)
    pointer_args = ",".join(f"v{l}.data_ptr<float>()" for l, _, _ in outputs)
    lines.append(f"""
void coherent_segmented(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor dst,torch::Tensor mask,{wrapper_args},
 torch::Tensor ptr,torch::Tensor order,torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,
 torch::Tensor dr,torch::Tensor di,torch::Tensor ph,torch::Tensor pr,torch::Tensor pi,torch::Tensor px,
 bool needx,bool needh,bool needg,bool needr,bool needi) {{
 const c10::cuda::CUDAGuard guard(x.device());
 auto stream=at::cuda::getCurrentCUDAStream();const int64_t E=src.numel();
 if(E){{
 coherent_segmented_edge_kernel<<<dim3(E,{len(groups)}),{threads},0,stream>>>(
 x.data_ptr<float>(),h.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
 r.data_ptr<float>(),i.data_ptr<float>(),src.data_ptr<int64_t>(),dst.data_ptr<int64_t>(),
 mask.data_ptr<float>(),{pointer_args},ph.data_ptr<float>(),dg.data_ptr<float>(),
 pr.data_ptr<float>(),pi.data_ptr<float>(),px.data_ptr<float>(),E,mask.numel()!=0,
 needx,needh,needg,needr,needi);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
""")
    for partial, target, need in (("ph", "dh", "needh"), ("pr", "dr", "needr"), ("pi", "di", "needi")):
        lines.append(f"if({need})coherent_partial_reduce_kernel<<<({target}.numel()+255)/256,256,0,stream>>>("
                     f"{partial}.data_ptr<float>(),{target}.data_ptr<float>(),{target}.numel());")
    lines.append("}")
    if edge_dx:
        lines.append(f"if(needx&&x.numel())coherent_edge_dx_reduce_kernel<<<(x.numel()+255)/256,256,0,stream>>>("
                     "px.data_ptr<float>(),ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),dx.data_ptr<float>(),x.size(0));")
    else:
        tiles = (c+31)//32
        lines.append(f"""
 if(needx&&x.size(0)){{
 coherent_source_adjoint_kernel<<<(x.size(0)*{tiles*len(groups)}+3)/4,128,0,stream>>>(
 h.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),r.data_ptr<float>(),i.data_ptr<float>(),
 dst.data_ptr<int64_t>(),mask.data_ptr<float>(),{pointer_args},ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),
 dx.data_ptr<float>(),x.size(0),mask.numel()!=0);
 }}
""")
    lines += ["C10_CUDA_KERNEL_LAUNCH_CHECK();", "}"]
    return "\n".join(lines)


def adjoint(x, sh, w, g, real, imag, src, dst, mask, p, projection, needed, grads, *, edge_dx=False):
    from .coherent import _HEADERS, _outputs
    from mace_ictc.models.message_topology import topology
    outputs = _outputs(p)
    key = (p["path_specs"], p["xdim"], p["odim"], p["c"], p["phase_heads"], p["phase_angular"], edge_dx,
           tuple(tuple(row) for row in projection["static_rows"]))
    if key not in _EXTENSIONS:
        from torch.utils.cpp_extension import load_inline
        name = f"ictc_coherent_segmented_v1_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}_a{int(p['phase_angular'])}_dx{int(edge_dx)}"
        declarations = "#include <torch/extension.h>\nvoid coherent_segmented(" + ",".join(
            ["torch::Tensor"]*(20+len(outputs)) + ["bool"]*5) + ");\n"
        _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declarations,
            cuda_sources=_HEADERS+_source(p, projection, edge_dx=edge_dx),
            functions=["coherent_segmented"], extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    empty = x.new_empty(0)
    values = [torch.empty_like(v) if need else empty for v, need in zip((x, sh, g, real, imag), needed)]
    groups = len({(row[2], row[1]) for row in p["path_specs"]})
    partials = [v.new_empty(groups, *v.shape) if need else empty
                for v, need in zip((sh, real, imag), (needed[1], needed[3], needed[4]))]
    px = x.new_empty(src.numel(), x.shape[1]) if edge_dx and needed[0] else empty
    cotangents = [grad.transpose(-1, -2).contiguous() if grad is not None else x.new_zeros(
        x.shape[0], 3, 2*l+1, width//(2*l+1)) for (l, _, width), grad in zip(outputs, grads)]
    if src.numel():
        _EXTENSIONS[key].coherent_segmented(x, sh, w, g, real, imag, src, dst, mask,
            *cotangents, *topology(src, x.shape[0]), *values, *partials, px, *needed)
    else:
        for value in values:
            value.zero_()
    return tuple(v if need else None for v, need in zip(values, needed))
