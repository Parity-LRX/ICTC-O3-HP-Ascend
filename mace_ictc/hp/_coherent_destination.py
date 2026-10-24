# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Destination-cached first VJP with deterministic source-feature gather.

The wide three-stream node cotangent is read once per destination/tile, not
once per incident edge. Source gradients use a CSR gather or an explicit FP32
atomic variant without the E*X scratch. Harmonic/phase partials have unique
writers. No gradient is omitted, compressed, or reduced in precision.
"""
import torch
import re
from mace_ictc.models.generated_cartesian_message import _sum

_EXTENSIONS = {}


def _source(p, projection, splits, atomic_x=False, needx=True, block_warps=4,
            needh=True, needg=True, specialized=False):
    from .coherent import _csr_adjoint_source, _outputs, _adjoint_groups
    original = _csr_adjoint_source(p, projection, owned_edges=True, destination_owned=True)
    code = original[:original.index('void coherent_adjoint(')]
    code = code.replace('const int64_t task=', 'const int64_t task0=', 1)
    code = code.replace('if(task>=', f'const int split=task0%{splits};const int64_t task=task0/{splits};\nif(task>=', 1)
    code = code.replace('index=PTR[dst];index<PTR[dst+1];++index',
                        f'index=PTR[dst]+split;index<PTR[dst+1];index+={splits}')
    if atomic_x:
        code = re.sub(r'if\(needx&&valid\)DX\[e\*(\d+)\+([^\]]+)\]=dx(\d+);',
            r'if(needx&&valid)atomicAdd(DX+src*\1+\2,dx\3);', code)
    if p['c'] % 32 == 0:
        code = code.replace(f'const bool valid=channel<{p["c"]};', 'const bool valid=true;')
    for flag in ('needx', 'needh', 'needg', 'needr', 'needi'):
        code = code.replace('bool '+flag, 'bool unused_'+flag)
    code = code.replace('const int lane=', f'constexpr bool needx={str(needx).lower()},needh={str(needh).lower()},needg={str(needg).lower()},needr=true,needi=true;\n const int lane=', 1)
    # All written arrays have distinct storage from the read-only operands.
    code = code.replace('float* ', 'float* __restrict__ ').replace('int64_t* ', 'int64_t* __restrict__ ')
    groups = len(_adjoint_groups(p))
    tiles = (p['c']+31)//32
    slots = groups*tiles
    def partial(array, width, slot):
        return (f'{array}[(i/{width}*{slots}+{slot})*{width}+i%{width}]'
            if p.get('coalesced_edge_partials', False) else f'{array}[i*{slots}+{slot}]')
    code += f'''
__global__ void coherent_destination_dx_reduce(const float* __restrict__ P,
 const int64_t* __restrict__ PTR,const int64_t* __restrict__ ORDER,
 float* __restrict__ Y,int64_t nodes) {{
 const int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(index>=nodes*{p['xdim']})return;
 const int64_t src=index/{p['xdim']},component=index%{p['xdim']};
 float value=0.f;
 for(int64_t j=PTR[src];j<PTR[src+1];++j)value+=P[ORDER[j]*{p['xdim']}+component];
 Y[index]=value;
}}
__global__ void coherent_destination_edge_reduce(const float* H,const float* R,const float* I,
 float* DH,float* DR,float* DI,int64_t edges) {{
 const int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if({str(needh).lower()}&&i<edges*{projection['hdim']})DH[i]={_sum(partial('H',projection['hdim'],s) for s in range(slots))};
 if(i<edges*{p['phase_count']}){{
  DR[i]={_sum(partial('R',p['phase_count'],s) for s in range(slots))};
  DI[i]={_sum(partial('I',p['phase_count'],s) for s in range(slots))};
 }}
}}
'''
    outputs = _outputs(p)
    wrapper_args = ','.join(f'torch::Tensor v{l}' for l, _, _ in outputs)
    pointer_args = ','.join(f'v{l}.data_ptr<float>()' for l, _, _ in outputs)
    code += f'''
void coherent_destination(torch::Tensor x,torch::Tensor h,torch::Tensor w,torch::Tensor g,
 torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor mask,{wrapper_args},
 torch::Tensor ptr,torch::Tensor order,torch::Tensor xptr,torch::Tensor xorder,
 torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,torch::Tensor dr,torch::Tensor di,
 torch::Tensor px,torch::Tensor ph,torch::Tensor pr,torch::Tensor pi) {{
 const c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
 if(x.size(0)){{
 coherent_adjoint_kernel<<<(x.size(0)*{tiles*groups*splits}+{block_warps-1})/{block_warps},{32*block_warps},0,stream>>>(
 x.data_ptr<float>(),h.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),
 r.data_ptr<float>(),i.data_ptr<float>(),src.data_ptr<int64_t>(),mask.data_ptr<float>(),
 {pointer_args},ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),{'dx' if atomic_x else 'px'}.data_ptr<float>(),
 ph.data_ptr<float>(),dg.data_ptr<float>(),pr.data_ptr<float>(),pi.data_ptr<float>(),
 x.size(0),src.numel(),mask.numel()!=0,true,true,true,true,true);
 {'' if atomic_x or not needx else 'coherent_destination_dx_reduce<<<(dx.numel()+255)/256,256,0,stream>>>(px.data_ptr<float>(),xptr.data_ptr<int64_t>(),xorder.data_ptr<int64_t>(),dx.data_ptr<float>(),x.size(0));'}
 coherent_destination_edge_reduce<<<(src.numel()*{max(projection['hdim'],p['phase_count'])}+255)/256,256,0,stream>>>(
 ph.data_ptr<float>(),pr.data_ptr<float>(),pi.data_ptr<float>(),dh.data_ptr<float>(),dr.data_ptr<float>(),di.data_ptr<float>(),src.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 }}
}}
'''
    if specialized:
        markers = (
            ('__global__ void coherent_adjoint_kernel', 'template<int GROUP>\n__global__ void coherent_adjoint_kernel'),
            (f'if(task>=N*{slots})return;', f'if(task>=N*{tiles})return;'),
            (f'const int group=(task/{tiles})%{groups};', 'constexpr int group=GROUP;'),
            (f'const int64_t dst=task/{slots};', f'const int64_t dst=task/{tiles};'),
        )
        for old, new in markers:
            if code.count(old) != 1:
                raise RuntimeError(f'First-adjoint specialization marker changed: {old!r}')
            code = code.replace(old, new, 1)
        before, launch = code.split('\n coherent_adjoint_kernel<<<', 1)
        launch, after = launch.split(';\n ', 1)
        launch = launch.replace(f'x.size(0)*{tiles*groups*splits}', f'x.size(0)*{tiles*splits}', 1)
        code = before+'\n'+'\n'.join(f' coherent_adjoint_kernel<{group}><<<{launch};'
            for group in range(groups))+'\n '+after
    return code


def adjoint(x, sh, w, g, real, imag, src, dst, mask, p, projection, needed, grads, *, splits=1, atomic_x=False, factored=False, coalesced=False, group_paths=0,
            block_warps=4, register_limit=0, disable_fma=False, schedule='legacy'):
    if schedule == 'legacy' and not all(needed[1:]):
        raise ValueError('Destination-cached first adjoint requires harmonic/gate/phase gradients')
    needx = bool(needed[0])
    needh, needg = (True, True) if schedule == 'legacy' else (bool(needed[1]), bool(needed[2]))
    from .coherent import _HEADERS, _outputs, _adjoint_groups
    from mace_ictc.models.message_topology import topology
    outputs = _outputs(p)
    if group_paths not in range(7) or (group_paths and not atomic_x):
        raise ValueError('Path groups require atomic source accumulation and a 1..6 path limit')
    if block_warps not in (1, 2, 4, 8) or register_limit not in (0, 64, 96, 128, 160, 192):
        raise ValueError('Invalid explicit destination launch/register schedule')
    p = {**p, 'factor_edge_scalars': factored, 'coalesced_edge_partials': coalesced,
         'destination_group_paths': group_paths,
         'componentwise_first': schedule in ('compact', 'compact_grouped')}
    key = (p['path_specs'], p['xdim'], p['odim'], p['c'], p['phase_heads'], p['phase_angular'], splits, atomic_x, factored, coalesced, needx, group_paths,
           block_warps, register_limit, disable_fma, schedule, needh, needg,
           tuple(tuple(row) for row in projection['static_rows']))
    if key not in _EXTENSIONS:
        from torch.utils.cpp_extension import load_inline
        name = f'ictc_coherent_destination_v1_{p["c"]}_{p["xdim"]}_{p["odim"]}_h{p["phase_heads"]}_a{int(p["phase_angular"])}_s{splits}_x{int(atomic_x)}'
        if factored:name += '_factor'
        if coalesced:name += '_coalesce'
        if not needx:name += '_nodx'
        if group_paths:name += f'_gp{group_paths}'
        if block_warps != 4:name += f'_bw{block_warps}'
        if register_limit:name += f'_reg{register_limit}'
        if disable_fma:name += '_nofma'
        if schedule != 'legacy':name += f'_{schedule}_h{int(needh)}_g{int(needg)}'
        declarations = '#include <torch/extension.h>\nvoid coherent_destination('+','.join(
            ['torch::Tensor']*(21+len(outputs)))+');\n'
        _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declarations,
            cuda_sources=_HEADERS+_source(p, projection, splits, atomic_x, needx, block_warps,
                needh, needg, schedule in ('grouped', 'compact_grouped')), functions=['coherent_destination'],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--ptxas-options=-v',
                *([f'--maxrregcount={register_limit}'] if register_limit else []),
                *(['--fmad=false'] if disable_fma else [])], verbose=False)
    dx, dh, dg, dr, di = ((torch.zeros_like if atomic_x else torch.empty_like)(x) if needx else x.new_empty(0), torch.empty_like(sh) if needh else sh.new_empty(0), torch.empty_like(g) if needg else g.new_empty(0),
                          torch.empty_like(real), torch.empty_like(imag))
    coverage = {j for _, ni, xb, *_ in p['path_specs'] for j in range(xb, xb+ni*p['c'])}
    allocate_x = x.new_empty if len(coverage) == p['xdim'] else x.new_zeros
    px = x.new_empty(0) if atomic_x or not needx else allocate_x(src.numel(), p['xdim'])
    slots = ((p['c']+31)//32)*len(_adjoint_groups(p))
    ph = sh.new_empty(*sh.shape, slots) if needh else sh.new_empty(0)
    pr, pi = real.new_empty(*real.shape, slots), imag.new_empty(*imag.shape, slots)
    cotangents = [grad.transpose(-1, -2).contiguous() if grad is not None else x.new_zeros(
        x.shape[0], 3, 2*l+1, width//(2*l+1)) for (l, _, width), grad in zip(outputs, grads)]
    if src.numel():
        source_csr = (src.new_empty(0), src.new_empty(0)) if atomic_x or not needx else topology(src, x.shape[0])
        _EXTENSIONS[key].coherent_destination(x, sh, w, g, real, imag, src, mask,
            *cotangents, *topology(dst, x.shape[0]), *source_csr,
            dx, dh, dg, dr, di, px, ph, pr, pi)
    else:
        for value in (dx, dh, dg, dr, di):value.zero_()
    return tuple(value if need else None for value, need in zip((dx, dh, dg, dr, di), needed))
