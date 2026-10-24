# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Experimental IEEE-FP32 radial output projection inside native ICTC TP."""
import re

_EXTENSIONS = {}


def device_source(hidden, width):
    if not 1 <= hidden <= 128:
        raise ValueError('Inline radial output projection supports 1..128 hidden channels')
    return f'''
__device__ __forceinline__ float ictc_radial_value(const float* H,const float* W,
 int64_t e,int column) {{
 float value=0.f;
 #pragma unroll 4
 for(int k=0;k<{hidden};++k)value=fmaf(H[e*{hidden}+k],W[k*{width}+column],value);
 return value;
}}
__device__ __forceinline__ float2 ictc_radial_pair(const float* H,const float* T,const float* W,
 int64_t e,int column) {{
 float value=0.f,tangent=0.f;
 #pragma unroll 4
 for(int k=0;k<{hidden};++k){{
  const float w=W[k*{width}+column];
  value=fmaf(H[e*{hidden}+k],w,value);
  tangent=fmaf(T[e*{hidden}+k],w,tangent);
 }}
 return make_float2(value,tangent);
}}
'''


def forward_extension(p, projection, hidden):
    from .coherent import _outputs, _forward_source
    from torch.utils.cpp_extension import load_inline
    key = (p['path_specs'], p['c'], p['xdim'], p['phase_heads'], p['phase_angular'], hidden,
           tuple(tuple(row) for row in projection['static_rows']))
    if key in _EXTENSIONS:return _EXTENSIONS[key]
    width = p['p']*p['c']
    code = _forward_source(p, projection).replace('coherent_forward', 'coherent_inline_radial_forward')
    code = code.replace('const float* G,', 'const float* G,const float* PW,')
    code = code.replace('torch::Tensor g,', 'torch::Tensor g,torch::Tensor pw,')
    code = code.replace('g.data_ptr<float>(),', 'g.data_ptr<float>(),pw.data_ptr<float>(),')
    pattern = rf'G\[e\*{width}\+(\d+)\+channel\]'
    code, count = re.subn(pattern,
        lambda match: f'ictc_radial_value(G,PW,e,{match[1]}+channel)', code)
    if count != p['p']:raise RuntimeError('Inline forward projection did not replace every live radial gate')
    args = ['torch::Tensor']*(11+len(_outputs(p)))
    declaration = '#include <torch/extension.h>\nvoid coherent_inline_radial_forward('+','.join(args)+');\n'
    name = f'ictc_radial_projection_v1_{p["c"]}_{p["xdim"]}_{p["odim"]}_h{p["phase_heads"]}_a{int(p["phase_angular"])}_k{hidden}'
    from .coherent import _HEADERS
    _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration,
        cuda_sources=_HEADERS+device_source(hidden, width)+code,
        functions=['coherent_inline_radial_forward'], extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--ptxas-options=-v'], verbose=False)
    return _EXTENSIONS[key]
