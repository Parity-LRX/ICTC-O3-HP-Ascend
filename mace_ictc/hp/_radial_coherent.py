# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bounded-memory radial gate/TP VJP pipeline using native ICTC and cuBLAS.

The original CSR joint adjoint feeds a reusable bounded gate-cotangent tile,
immediately consumed by IEEE FP32 GEMM. No full E*path*C gate gradient exists.
Mixed/higher derivatives use the exact original differentiable polynomial.
"""
import re

import torch
from .coherent import _outputs, _extension, _reference

_EXTENSIONS = {}


def _chunk_extension(p, projection, gate_pipeline=False):
    key = (p["path_specs"], p["xdim"], p["c"], p["phase_heads"], p["phase_angular"],
           tuple(tuple(row) for row in projection["static_rows"]), gate_pipeline)
    if key in _EXTENSIONS:
        return _EXTENSIONS[key]
    from torch.utils.cpp_extension import load_inline
    from .coherent import _HEADERS, _csr_adjoint_source, _forward_source
    source = _csr_adjoint_source(p, projection)
    source = source.replace("coherent_adjoint_kernel", "coherent_radial_chunk_kernel")
    source = source.replace("void coherent_adjoint(", "void coherent_radial_chunk(")
    source = source.replace("bool needi)", "bool needi,int64_t BEGIN,int64_t END)")
    source = source.replace("needx,needh,needg,needr,needi);",
                            "needx,needh,needg,needr,needi,BEGIN,END);")
    tasks = ((p["c"]+31)//32)*len({(row[2], row[1]) for row in p["path_specs"]})
    source = source.replace(f"const int64_t src=task/{tasks};",
        f"const int64_t src=task/{tasks};\n"
        "const int64_t lo=PTR[src]>BEGIN?PTR[src]:BEGIN;\n"
        "const int64_t hi=PTR[src+1]<END?PTR[src+1]:END;\n"
        "if(lo>=hi)return;")
    source = source.replace("for(int64_t index=PTR[src];index<PTR[src+1];++index)",
                            "for(int64_t index=lo;index<hi;++index)")
    source = source.replace("DG[e*", "DG[(index-BEGIN)*")
    if gate_pipeline:
        source = source.replace("G[e*", "G[(index-BEGIN)*")
    source = re.sub(r"(DX\[src\*\d+\+\d+\+channel\])=dx(\d+);", r"\1+=dx\2;", source)
    outputs = len(_outputs(p))
    declarations = "#include <torch/extension.h>\nvoid coherent_radial_chunk("+",".join(
        ["torch::Tensor"]*(15+outputs)+["bool"]*5+["int64_t"]*2)+");\n"
    forward = _forward_source(p, projection).replace("coherent_forward", "coherent_radial_chunk_forward")
    forward = forward.replace("int64_t E,bool masked)", "int64_t E,bool masked,int64_t BEGIN,int64_t END)")
    forward = re.sub(r"(torch::Tensor out\d+)\)", r"\1,int64_t BEGIN,int64_t END)", forward)
    forward = forward.replace("mask.numel()!=0);", "mask.numel()!=0,BEGIN,END);")
    forward = re.sub(r"const int64_t dst=task/(\d+);",
        r"const int64_t dst=task/\1;\nconst int64_t lo=PTR[dst]>BEGIN?PTR[dst]:BEGIN;\n"
        r"const int64_t hi=PTR[dst+1]<END?PTR[dst+1]:END;\nif(lo>=hi)return;", forward)
    forward = forward.replace("for(int64_t index=PTR[dst];index<PTR[dst+1];++index)",
                               "for(int64_t index=lo;index<hi;++index)")
    forward = forward.replace("G[e*", "G[(index-BEGIN)*")
    forward = re.sub(r"(O\d+\[[^\n]+?\])=([ari]\d+_\d+);", r"\1+=\2;", forward)
    declarations += "void coherent_radial_chunk_forward("+",".join(
        ["torch::Tensor"]*(10+outputs)+["int64_t"]*2)+");\n"
    name = f"ictc_radial_chunk_v2_{p['c']}_{p['xdim']}_{p['odim']}_h{p['phase_heads']}_a{int(p['phase_angular'])}_g{int(gate_pipeline)}"
    _EXTENSIONS[key] = load_inline(name=name, cpp_sources=declarations,
        cuda_sources=_HEADERS+source+forward,
        functions=["coherent_radial_chunk", "coherent_radial_chunk_forward"],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--ptxas-options=-v"], verbose=False)
    return _EXTENSIONS[key]


class _RadialCoherent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sh, w, hidden, radial_weight, real, imag, src, dst, mask, p, projection,
                tile_edges, gate_pipeline):
        ctx.set_materialize_grads(False)
        gates = x.new_empty(0) if gate_pipeline else torch.nn.functional.linear(hidden, radial_weight)
        ctx.save_for_backward(x, sh, w, hidden, radial_weight, gates, real, imag, src, dst, mask)
        ctx.p, ctx.projection, ctx.tile_edges = p, projection, tile_edges
        ctx.gate_pipeline = gate_pipeline
        output = tuple((x.new_empty if src.numel() and not gate_pipeline else x.new_zeros)(
            x.shape[0], 3, 2*l+1, width//(2*l+1)).transpose(-1, -2)
            for l, _, width in _outputs(p))
        if src.numel():
            from mace_ictc.models.message_topology import topology
            if gate_pipeline:
                ptr, order = topology(dst, x.shape[0])
                tile = min(tile_edges, src.numel())
                tile_hidden = hidden.new_empty(tile, hidden.shape[1])
                tile_gates = hidden.new_empty(tile, radial_weight.shape[0])
                extension = _chunk_extension(p, projection, True)
                for begin in range(0, src.numel(), tile):
                    end = min(begin+tile, src.numel())
                    torch.index_select(hidden, 0, order[begin:end], out=tile_hidden[:end-begin])
                    torch.mm(tile_hidden[:end-begin], radial_weight.T, out=tile_gates[:end-begin])
                    extension.coherent_radial_chunk_forward(x, sh, w, tile_gates, real, imag, src, mask,
                                                           ptr, order, *output, begin, end)
            else:
                _extension(p, projection).coherent_forward(x, sh, w, gates, real, imag, src, mask,
                                                          *topology(dst, x.shape[0]), *output)
        return output

    @staticmethod
    def backward(ctx, *grads):
        x, sh, w, hidden, radial_weight, gates, real, imag, src, dst, mask = ctx.saved_tensors
        p, projection = ctx.p, ctx.projection
        needed = tuple(ctx.needs_input_grad[index] for index in (0, 1, 3, 5, 6))
        if torch.is_grad_enabled():
            local = [value.clone() for value in (x, sh, hidden, real, imag)]
            gate = torch.nn.functional.linear(local[2], radial_weight)
            output = _reference(local[0], local[1], w, gate, *local[3:], src, dst, mask, p, projection)
            parts, cotangents = [], []
            for (l, offset, width), grad in zip(_outputs(p), grads):
                if grad is not None:
                    value = output.reshape(x.shape[0], 3, p["odim"])[..., offset:offset+width]
                    value = value.reshape(x.shape[0], 3, -1, 2*l+1, p["c"]).transpose(-1, -2)
                    parts.append(value.reshape(x.shape[0], 3, width//(2*l+1), 2*l+1))
                    cotangents.append(grad)
            active = [value for value, need in zip(local, needed) if need]
            values = iter(torch.autograd.grad(parts, active, cotangents, create_graph=True, allow_unused=True))
            dx, dh, dz, dr, di = [next(values) if need else None for need in needed]
        else:
            empty = x.new_empty(0)
            dx, dh, dz, dr, di = [torch.zeros_like(value) if need else empty
                for value, need in zip((x, sh, hidden, real, imag), needed)]
            if src.numel():
                from mace_ictc.models.message_topology import topology
                ptr, order = topology(src, x.shape[0])
                cotangents = [grad.transpose(-1, -2).contiguous() if grad is not None else x.new_zeros(
                    x.shape[0], 3, 2*l+1, width//(2*l+1))
                    for (l, _, width), grad in zip(_outputs(p), grads)]
                tile = min(ctx.tile_edges, src.numel())
                dg = x.new_empty(tile, radial_weight.shape[0]) if needed[2] else empty
                partial = x.new_empty(tile, hidden.shape[1]) if needed[2] else empty
                extension = _chunk_extension(p, projection, ctx.gate_pipeline)
                if ctx.gate_pipeline:
                    tile_hidden = hidden.new_empty(tile, hidden.shape[1])
                    gates = hidden.new_empty(tile, radial_weight.shape[0])
                for begin in range(0, src.numel(), tile):
                    end = min(begin+tile, src.numel())
                    if ctx.gate_pipeline:
                        torch.index_select(hidden, 0, order[begin:end], out=tile_hidden[:end-begin])
                        torch.mm(tile_hidden[:end-begin], radial_weight.T, out=gates[:end-begin])
                    extension.coherent_radial_chunk(x, sh, w, gates, real, imag, dst, mask,
                        *cotangents, ptr, order, dx, dh, dg, dr, di, *needed, begin, end)
                    if needed[2]:
                        # The cotangent tile is consumed before the next native
                        # launch overwrites it; its footprint is controlled by
                        # tile_edges rather than the full graph edge count.
                        torch.mm(dg[:end-begin], radial_weight, out=partial[:end-begin])
                        dz.index_copy_(0, order[begin:end], partial[:end-begin])
            dx, dh, dz, dr, di = [value if need else None
                for value, need in zip((dx, dh, dz, dr, di), needed)]
        return dx, dh, None, dz, None, dr, di, None, None, None, None, None, None, None


def radial_coherent(owner, blocks, harmonics, hidden, weight, src, dst, mask, real, imag, *,
                    tile_edges=4096, gate_pipeline=False):
    """Frozen bias-free radial output linear, original first/mixed derivatives."""
    from .operators import _eager, _operand
    _eager(blocks[0])
    if weight.requires_grad or weight.ndim != 2 or not 1 <= weight.shape[1] <= 128:
        raise ValueError("Radial TP fusion requires a frozen FP32 bank with 1..128 inputs")
    if not isinstance(tile_edges, int) or isinstance(tile_edges, bool) or tile_edges < 1:
        raise ValueError("Radial TP tile_edges must be a positive integer")
    e = src.numel()
    p0, projection = owner.edge._prepare()
    phases = owner.heads*(owner.output_lmax+1 if owner.angular else 1)
    sample = owner.edge.tp.weight
    _operand(hidden, (e, weight.shape[1]), sample, "radial hidden")
    _operand(weight, (owner.edge.gate_width, weight.shape[1]), sample, "radial weight")
    for name, value in (("src", src), ("dst", dst)):
        if value.shape != (e,) or value.dtype != torch.int64 or value.device != sample.device:
            raise ValueError(f"{name} must be int64 [E] on the execution device")
    for value in (real, imag):
        _operand(value, (e, phases, 1), sample, "phase coefficient")
    if mask is not None:
        if mask.requires_grad or mask.shape not in ((e,), (e, 1)):
            raise ValueError("Coherent mask must be fixed [E] or [E,1]")
        _operand(mask, mask.shape, sample, "mask")
    p = {**p0, "native_packed": True, "phase_heads": owner.heads,
         "phase_angular": owner.angular, "phase_count": phases}
    x = owner.edge.pack_input(blocks)
    sh = torch.cat([harmonics[l].squeeze(-2) for l in projection["keys"]], -1).contiguous()
    _operand(sh, (e, projection["hdim"]), sample, "harmonics")
    m = x.new_empty(0) if mask is None else mask.reshape(e).contiguous()
    output = _RadialCoherent.apply(x, sh, sample, hidden.contiguous(), weight.contiguous(),
        real.squeeze(-1).contiguous(), imag.squeeze(-1).contiguous(),
        src.contiguous(), dst.contiguous(), m, p, projection, tile_edges, gate_pipeline)
    return {l: value for (l, _, _), value in zip(_outputs(p), output)}
