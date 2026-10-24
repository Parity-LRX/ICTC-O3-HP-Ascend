# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Equivalent coherent forward schedules with bounded register accumulators.

The independent-stream schedule repeats the TP arithmetic but needs one third
as many live accumulators. Both preserve every path and edge reduction order.
This module is screened independently before a policy is selected by default.
"""
import re

from .coherent import _forward_source, _outputs


_EXTENSIONS = {}


def _source(p, projection, limit, separate_streams, readonly=False, neighbor_warps=1):
    source = _forward_source(p, projection, accumulator_limit=limit, neighbor_warps=neighbor_warps)
    if readonly:
        source = source.replace('float* ', 'float* __restrict__ ').replace('int64_t* ', 'int64_t* __restrict__ ')
    if not separate_streams:
        return source
    source = source.replace("const int64_t task=", "const int64_t task0=", 1)
    source = source.replace("if(task>=", "const int stream_id=task0%3;const int64_t task=task0/3;\nif(task>=", 1)
    source = re.sub(r"float a(\d+)_(\d+)=0\.f,r\1_\2=0\.f,i\1_\2=0\.f;",
                    r"float a\1_\2=0.f;", source)
    source = re.sub(r"a(\d+)_(\d+)\+=v(\d+);r\1_\2\+=v\3\*cr;i\1_\2\+=v\3\*ci;",
                    r"a\1_\2+=v\3*(stream_id==0?1.f:(stream_id==1?cr:ci));", source)
    lines = []
    for line in source.splitlines():
        if re.search(r"=r\d+_\d+;|=i\d+_\d+;", line):
            continue
        if re.search(r"=a\d+_\d+;", line):
            line = line.replace("(dst*3+0)", "(dst*3+stream_id)")
        lines.append(line)
    source = "\n".join(lines)
    source = re.sub(r"const int64_t blocks=\(x.size\(0\)\*(\d+)\+3\)/4;",
                    lambda m: f"const int64_t blocks=(x.size(0)*{int(m.group(1))*3}+3)/4;", source)
    return source


def forward(x, sh, w, g, real, imag, src, dst, mask, p, projection, out, *,
            accumulator_limit=8, separate_streams=False, readonly=False, neighbor_warps=1):
    from torch.utils.cpp_extension import load_inline
    from mace_ictc.models.message_topology import topology
    key = (p["path_specs"], p["xdim"], p["odim"], p["c"], p["phase_heads"], p["phase_angular"],
           tuple(tuple(row) for row in projection["static_rows"]), accumulator_limit, separate_streams, readonly, neighbor_warps)
    if neighbor_warps not in (1, 2, 4) or (neighbor_warps > 1 and p['c'] % 32):
        raise ValueError('Neighbor-shared candidate requires full channel warps and 1/2/4 owners')
    if neighbor_warps > 1 and separate_streams:
        raise ValueError('Independent-stream and neighbor-shared schedules are screened separately')
    if key not in _EXTENSIONS:
        name = f"ictc_coherent_forward_v1_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}_a{int(p['phase_angular'])}_w{accumulator_limit}_s{int(separate_streams)}"
        if readonly:name += '_readonly'
        if neighbor_warps > 1:name += f'_nw{neighbor_warps}'
        declaration = "#include <torch/extension.h>\nvoid coherent_forward("+",".join(
            ["torch::Tensor"]*(10+len(_outputs(p))))+");\n"
        _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declaration,
            cuda_sources=_source(p, projection, accumulator_limit, separate_streams, readonly, neighbor_warps),
            functions=["coherent_forward"], extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    if src.numel():
        _EXTENSIONS[key].coherent_forward(x, sh, w, g, real, imag, src, mask,
                                         *topology(dst, x.shape[0]), *out)
