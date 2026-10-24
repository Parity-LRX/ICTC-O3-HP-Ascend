# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded radial scalar VJP with destination-cached node cotangents.

Reuses the ICTC original generated CG algebra and FP32 source accumulation.
Only one tile of harmonic/phase partials exists; no E*path*C gate gradient.
"""
import re
import torch

_EXTENSIONS = {}


def extension(p, projection, needx, inline_width=0, cached_gates=False):
    from .coherent import _HEADERS, _outputs, _adjoint_groups, _csr_adjoint_source
    from mace_ictc.models.generated_cartesian_message import _sum
    p = {**p, 'factor_edge_scalars': True, 'coalesced_edge_partials': True}
    key = (p['path_specs'], p['c'], p['xdim'], p['odim'], p['phase_heads'],
           p['phase_angular'], needx, inline_width, cached_gates,
           tuple(tuple(row) for row in projection['static_rows']))
    if key in _EXTENSIONS:return _EXTENSIONS[key]
    outputs = _outputs(p)
    slots = ((p['c']+31)//32)*len(_adjoint_groups(p))
    code = _csr_adjoint_source(p, projection, owned_edges=True, destination_owned=True)
    code = code[:code.index('void coherent_adjoint(')]
    code = code.replace('coherent_adjoint_kernel', 'coherent_distance_destination_kernel')
    code = code.replace('const float* G,', 'const float* G,const float* GP,')
    if inline_width:code = code.replace('const float* GP,', 'const float* GP,const float* PW,')
    code = code.replace('bool needi)', 'bool needi,int64_t BEGIN,int64_t END)')
    marker = f'const int64_t dst=task/{slots};'
    if marker not in code:raise RuntimeError('Destination radial scheduling marker changed')
    code = code.replace(marker, marker+'\nconst int64_t lo=PTR[dst]>BEGIN?PTR[dst]:BEGIN;'
        '\nconst int64_t hi=PTR[dst+1]<END?PTR[dst+1]:END;\nif(lo>=hi)return;')
    code = code.replace('index=PTR[dst];index<PTR[dst+1];++index', 'index=lo;index<hi;++index')
    code = code.replace('const float mask=masked?MASK[e]:1.f;',
                        'const float mask=masked?MASK[e]:1.f;float radial=0.f;')
    width = p['p']*p['c']
    code = re.sub(r'if\(needg&&valid\)DG\[e\*\d+\+(\d+)\+channel\]=([^;\n]+);',
        lambda m: f'if(needg&&valid)radial+=({m[2]})*'+('radial_tangent;' if inline_width else
            f'GP[(index-BEGIN)*{width}+{m[1]}+channel];'), code)
    if 'DG[e*' in code:raise RuntimeError('Unconverted full radial cotangent write')
    if not cached_gates:code = code.replace('G[e*', 'G[(index-BEGIN)*')
    if cached_gates and inline_width:raise ValueError('Cached gates and inline projection are distinct schedules')
    if inline_width:
        pattern = rf'const float gate=valid\?G\[\(index-BEGIN\)\*{width}\+(\d+)\+channel\]:0.f;'
        code, count = re.subn(pattern, lambda match:
            f'const float2 radial_pair=valid?ictc_radial_pair(G,GP,PW,e,{match[1]}+channel):make_float2(0.f,0.f);'
            'const float gate=radial_pair.x,radial_tangent=radial_pair.y;', code)
        if count != p['p']:raise RuntimeError('Inline adjoint projection did not replace every radial gate')
    code = code.replace('float pr_value=0.f,pi_value=0.f;',
        'for(int d=16;d>0;d/=2)radial+=__shfl_down_sync(0xffffffff,radial,d);'
        'if(lane==0)atomicAdd(DG+e,radial);\nfloat pr_value=0.f,pi_value=0.f;')
    code = re.sub(r'if\(needx&&valid\)DX\[e\*(\d+)\+([^\]]+)\]=dx(\d+);',
                  r'if(needx&&valid)atomicAdd(DX+src*\1+\2,dx\3);', code)
    for field in ('DH', 'DR', 'DI'):
        code = code.replace(field+'[(e*', field+'[((index-BEGIN)*')
    if p['c'] % 32 == 0:
        code = code.replace(f'const bool valid=channel<{p["c"]};', 'const bool valid=true;')
    for flag in ('needx', 'needh', 'needg', 'needr', 'needi'):
        code = code.replace('bool '+flag, 'bool unused_'+flag)
    code = code.replace('const int lane=',
        f'constexpr bool needx={str(needx).lower()},needh=true,needg=true,needr=true,needi=true;\nconst int lane=', 1)
    code = code.replace('float* ', 'float* __restrict__ ').replace('int64_t* ', 'int64_t* __restrict__ ')
    hdim, phases = projection['hdim'], p['phase_count']
    def partial(name, width):
        return _sum(f'{name}[(j*{slots}+{s})*{width}+c]' for s in range(slots))
    code += f'''
__global__ void coherent_distance_destination_reduce(const float* PH,const float* PR,const float* PI,
 const int64_t* ORDER,float* DH,float* DR,float* DI,int64_t BEGIN,int64_t count) {{
 const int64_t index=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 if(index<count*{hdim}){{
  const int64_t j=index/{hdim},c=index%{hdim},e=ORDER[BEGIN+j];
  DH[e*{hdim}+c]={partial('PH',hdim)};
 }}
 if(index<count*{phases}){{
  const int64_t j=index/{phases},c=index%{phases},e=ORDER[BEGIN+j];
  DR[e*{phases}+c]={partial('PR',phases)};
  DI[e*{phases}+c]={partial('PI',phases)};
 }}
}}
'''
    args = ','.join(f'torch::Tensor v{l}' for l, _, _ in outputs)
    pointers = ','.join(f'v{l}.data_ptr<float>()' for l, _, _ in outputs)
    code += f'''
void coherent_distance_destination(torch::Tensor x,torch::Tensor h,torch::Tensor w,
 torch::Tensor g,torch::Tensor gp,{"torch::Tensor pw," if inline_width else ""}torch::Tensor r,torch::Tensor i,torch::Tensor src,torch::Tensor mask,
 {args},torch::Tensor ptr,torch::Tensor order,torch::Tensor dx,torch::Tensor dh,torch::Tensor dg,
 torch::Tensor dr,torch::Tensor di,torch::Tensor ph,torch::Tensor pr,torch::Tensor pi,
 int64_t BEGIN,int64_t END) {{
 const c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
 if(END<=BEGIN)return;
 coherent_distance_destination_kernel<<<(x.size(0)*{slots}+3)/4,128,0,stream>>>(
 x.data_ptr<float>(),h.data_ptr<float>(),w.data_ptr<float>(),g.data_ptr<float>(),gp.data_ptr<float>(),
 {"pw.data_ptr<float>()," if inline_width else ""}
 r.data_ptr<float>(),i.data_ptr<float>(),src.data_ptr<int64_t>(),mask.data_ptr<float>(),
 {pointers},ptr.data_ptr<int64_t>(),order.data_ptr<int64_t>(),dx.data_ptr<float>(),ph.data_ptr<float>(),
 dg.data_ptr<float>(),pr.data_ptr<float>(),pi.data_ptr<float>(),x.size(0),src.numel(),
 mask.numel()!=0,true,true,true,true,true,BEGIN,END);
 coherent_distance_destination_reduce<<<((END-BEGIN)*{max(hdim,phases)}+255)/256,256,0,stream>>>(
 ph.data_ptr<float>(),pr.data_ptr<float>(),pi.data_ptr<float>(),order.data_ptr<int64_t>(),
 dh.data_ptr<float>(),dr.data_ptr<float>(),di.data_ptr<float>(),BEGIN,END-BEGIN);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
'''
    from torch.utils.cpp_extension import load_inline
    name = f'ictc_radial_destination_v1_{p["c"]}_{p["xdim"]}_{p["odim"]}_h{p["phase_heads"]}_a{int(p["phase_angular"])}_x{int(needx)}'
    if inline_width:name = name.replace('_v1_', '_v2_')+f'_k{inline_width}'
    if cached_gates:name = name.replace('_v1_', '_v3_')+'_cachedg'
    declaration = '#include <torch/extension.h>\nvoid coherent_distance_destination('+','.join(
        ['torch::Tensor']*(19+len(outputs)+int(bool(inline_width)))+['int64_t']*2)+');\n'
    helper = ''
    if inline_width:
        from ._radial_projection import device_source
        helper = device_source(inline_width, width)
    _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration,
        cuda_sources=_HEADERS+helper+code, functions=['coherent_distance_destination'],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--ptxas-options=-v'], verbose=False)
    return _EXTENSIONS[key]


def workspace(x, p, projection, tile):
    from .coherent import _adjoint_groups
    slots = ((p['c']+31)//32)*len(_adjoint_groups(p))
    return (x.new_empty(tile, slots, projection['hdim']),
            x.new_empty(tile, slots, p['phase_count']), x.new_empty(tile, slots, p['phase_count']))
