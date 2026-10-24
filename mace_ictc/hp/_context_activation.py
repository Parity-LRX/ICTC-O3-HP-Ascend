# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Joint first/mixed adjoints of a live node-to-edge normalized SiLU.

The operation is scale*silu(D[dst] + S[src] + R). All projections and the
incoming output cotangent remain live. No fixed graph or address is retained.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from mace_ictc.models.adjoint_requirements import required_gradients


@triton.jit
def _first(D, S, R, SRC, DST, G, DD, DS, DR,
           TOTAL: tl.constexpr, H: tl.constexpr, SCALE: tl.constexpr,
           ND: tl.constexpr, NS: tl.constexpr, NR: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = j < TOTAL
    edge, channel = j // H, j % H
    src = tl.load(SRC + edge, valid, 0)
    dst = tl.load(DST + edge, valid, 0)
    a = tl.load(D + dst * H + channel, valid, 0.)
    a += tl.load(S + src * H + channel, valid, 0.)
    a += tl.load(R + j, valid, 0.)
    sigmoid = tl.div_rn(1., 1.+libdevice.exp(-a))
    q = tl.load(G + j, valid, 0.) * SCALE * sigmoid * (1. + a * (1. - sigmoid))
    if ND:
        tl.atomic_add(DD + dst * H + channel, q, valid, sem="relaxed")
    if NS:
        tl.atomic_add(DS + src * H + channel, q, valid, sem="relaxed")
    if NR:
        tl.store(DR + j, q, valid)


@triton.jit
def _mixed(D, S, R, G, SRC, DST, TD, TS, TR, DD, DS, DR, DG,
           TOTAL: tl.constexpr, H: tl.constexpr, SCALE: tl.constexpr,
           HD: tl.constexpr, HS: tl.constexpr, HR: tl.constexpr,
           ND: tl.constexpr, NS: tl.constexpr, NR: tl.constexpr, NG: tl.constexpr,
           BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = j < TOTAL
    edge, channel = j // H, j % H
    src = tl.load(SRC + edge, valid, 0)
    dst = tl.load(DST + edge, valid, 0)
    a = tl.load(D + dst * H + channel, valid, 0.)
    a += tl.load(S + src * H + channel, valid, 0.)
    a += tl.load(R + j, valid, 0.)
    tangent = tl.full((BLOCK,), 0., tl.float32)
    if HD:
        tangent += tl.load(TD + dst * H + channel, valid, 0.)
    if HS:
        tangent += tl.load(TS + src * H + channel, valid, 0.)
    if HR:
        tangent += tl.load(TR + j, valid, 0.)
    sigmoid = tl.div_rn(1., 1.+libdevice.exp(-a))
    complement = 1.-sigmoid
    left = tangent*(1.+a*complement)
    if (ND or NS) or NR:
        scaled_g = tl.load(G+j, valid, 0.)*SCALE
        right = tangent*(scaled_g*sigmoid)
        sigmoid_grad = left*scaled_g-right*a
        q = right*complement+(sigmoid_grad*complement)*sigmoid
        if ND:
            tl.atomic_add(DD + dst * H + channel, q, valid, sem="relaxed")
        if NS:
            tl.atomic_add(DS + src * H + channel, q, valid, sem="relaxed")
        if NR:
            tl.store(DR + j, q, valid)
    if NG:
        tl.store(DG+j, (left*sigmoid)*SCALE, valid)


class _GatherSiLUAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d, s, r, g, src, dst, scale, needed):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(d, s, r, g, src, dst)
        ctx.scale, ctx.needed = scale, needed
        dd = torch.zeros_like(d) if needed[0] else None
        ds = torch.zeros_like(s) if needed[1] else None
        dr = torch.empty_like(r) if needed[2] else None
        if r.numel() and any(needed):
            _first[(triton.cdiv(r.numel(), 256),)](
                d, s, r, src, dst, g, dd if needed[0] else d,
                ds if needed[1] else s, dr if needed[2] else r,
                r.numel(), r.shape[1], scale, *needed, 256, enable_fp_fusion=False)
        return dd, ds, dr

    @staticmethod
    def backward(ctx, td, ts, tr):
        d, s, r, g, src, dst = ctx.saved_tensors
        needs = required_gradients((d, s, r, g), ctx.needs_input_grad[:4])
        if not any(needs) or all(t is None for t in (td, ts, tr)):
            return (None,) * 8
        if torch.is_grad_enabled():
            # Third/higher derivatives retain the literal live input graph.
            a = d.index_select(0, dst) + s.index_select(0, src) + r
            sigmoid = a.sigmoid()
            tangent = torch.zeros_like(r)
            if td is not None:
                tangent = tangent + td.index_select(0, dst)
            if ts is not None:
                tangent = tangent + ts.index_select(0, src)
            if tr is not None:
                tangent = tangent + tr
            complement = 1.-sigmoid
            left = tangent*(1.+a*complement)
            scaled_g = g*ctx.scale
            right = tangent*(scaled_g*sigmoid)
            sigmoid_grad = left*scaled_g-right*a
            q = right*complement+(sigmoid_grad*complement)*sigmoid
            dd = torch.zeros_like(d).index_add(0, dst, q) if needs[0] else None
            ds = torch.zeros_like(s).index_add(0, src, q) if needs[1] else None
            dr = q if needs[2] else None
            dg = (left*sigmoid)*ctx.scale if needs[3] else None
        else:
            dd = torch.zeros_like(d) if needs[0] else None
            ds = torch.zeros_like(s) if needs[1] else None
            dr = torch.empty_like(r) if needs[2] else None
            dg = torch.empty_like(g) if needs[3] else None
            if r.numel():
                tangents = tuple(r if t is None else t.contiguous() for t in (td, ts, tr))
                _mixed[(triton.cdiv(r.numel(), 256),)](
                    d, s, r, g, src, dst, *tangents,
                    dd if needs[0] else d, ds if needs[1] else s,
                    dr if needs[2] else r, dg if needs[3] else g,
                    r.numel(), r.shape[1], ctx.scale,
                    *(t is not None for t in (td, ts, tr)), *needs, 256, enable_fp_fusion=False)
        return dd, ds, dr, dg, None, None, None, None


def gather_silu_adjoint(d, s, r, grad, src, dst, scale, needed=(True, True, True)):
    """First VJP with fused mixed-second derivatives, for CUDA FP32 inputs.

    d/s are [N,H], r/grad are [E,H], src/dst fixed int64 [E]. `needed`
    declares which first-VJP outputs the current autograd request consumes.
    """
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
            or len(needed) != 3 or not all(isinstance(v, bool) for v in needed)):
        raise TypeError("scale must be numeric and needed must contain three booleans")
    if d.ndim != 2 or s.shape != d.shape or r.ndim != 2 or r.shape[1] != d.shape[1] or grad.shape != r.shape:
        raise ValueError("Expected d/s [N,H] and r/grad [E,H]")
    if (not d.is_cuda or any(v.device != d.device or v.dtype != torch.float32 for v in (d, s, r, grad))
            or d.shape[1] < 1):
        raise ValueError("Context adjoints require CUDA FP32 projections on one device")
    if any(v.dtype != torch.int64 or v.device != d.device or v.shape != (r.shape[0],) for v in (src, dst)):
        raise ValueError("src/dst must be int64 [E] on the projection device")
    return _GatherSiLUAdjoint.apply(*(v.contiguous() for v in (d, s, r, grad, src, dst)),
                                     float(scale), tuple(needed))
