# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Live path-weight/gate product with bounded first and mixed adjoints.

No E-by-path-by-channel product is materialized just to reduce a coefficient
gradient. The first and mixed VJPs produce bounded edge-chunk partials instead.
Features, coefficients and incoming cotangents remain live across every call.
"""
import torch
import triton
import triton.language as tl

from mace_ictc.models.adjoint_requirements import required_gradients


@triton.jit
def _value(X, W, Y, TOTAL: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    j = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    value = tl.load(X+j, j<TOTAL, 0.)*tl.load(W+j%WIDTH, j<TOTAL, 0.)
    tl.store(Y+j, value, j<TOTAL)


@triton.jit
def _first(X, W, G, DX, PART, EDGES: tl.constexpr, WIDTH: tl.constexpr,
           NEED_X: tl.constexpr, NEED_W: tl.constexpr, BE: tl.constexpr, BC: tl.constexpr):
    chunk = tl.program_id(0)
    edge = chunk*BE+tl.arange(0, BE)
    col = tl.program_id(1)*BC+tl.arange(0, BC)
    j = edge[:, None]*WIDTH+col[None, :]
    valid = (edge[:, None]<EDGES)&(col[None, :]<WIDTH)
    g = tl.load(G+j, valid, 0.)
    if NEED_X:
        w = tl.load(W+col, col<WIDTH, 0.)
        tl.store(DX+j, g*w[None, :], valid)
    if NEED_W:
        x = tl.load(X+j, valid, 0.)
        tl.store(PART+chunk*WIDTH+col, tl.sum(g*x, 0), col<WIDTH)


@triton.jit
def _mixed(X, W, G, HX, HW, DX, PART, DG, EDGES: tl.constexpr, WIDTH: tl.constexpr,
           HAS_X: tl.constexpr, HAS_W: tl.constexpr,
           NEED_X: tl.constexpr, NEED_W: tl.constexpr, NEED_G: tl.constexpr,
           BE: tl.constexpr, BC: tl.constexpr):
    chunk = tl.program_id(0)
    edge = chunk*BE+tl.arange(0, BE)
    col = tl.program_id(1)*BC+tl.arange(0, BC)
    j = edge[:, None]*WIDTH+col[None, :]
    valid = (edge[:, None]<EDGES)&(col[None, :]<WIDTH)
    if NEED_X or NEED_W:
        g = tl.load(G+j, valid, 0.)
    if HAS_X:
        hx = tl.load(HX+j, valid, 0.)
    if HAS_W:
        hw = tl.load(HW+col, col<WIDTH, 0.)
    if NEED_X:
        tl.store(DX+j, g*hw[None, :], valid)
    if NEED_W:
        tl.store(PART+chunk*WIDTH+col, tl.sum(g*hx, 0), col<WIDTH)
    if NEED_G:
        result = tl.full((BE, BC), 0., tl.float32)
        if HAS_X:
            result = hx*tl.load(W+col, col<WIDTH, 0.)[None, :]
        if HAS_W:
            term = tl.load(X+j, valid, 0.)*hw[None, :]
            result = result+term if HAS_X else term
        tl.store(DG+j, result, valid)


def _empty_partial(x, needed):
    return x.new_empty(triton.cdiv(x.shape[0], 128), x.shape[1]) if needed else x.new_empty(0)


class _GateAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, g, needed):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, w, g)
        ctx.needed = needed
        dx = torch.empty_like(x) if needed[0] else None
        partial = _empty_partial(x, needed[1])
        if x.numel() and any(needed):
            _first[(triton.cdiv(x.shape[0], 128), triton.cdiv(x.shape[1], 32))](
                x, w, g, x if dx is None else dx, partial,
                *x.shape, *needed, 128, 32, num_warps=4, enable_fp_fusion=False)
        dw = partial.sum(0) if needed[1] else None
        return dx, dw

    @staticmethod
    def backward(ctx, hx, hw):
        x, w, g = ctx.saved_tensors
        needs = required_gradients((x, w, g), ctx.needs_input_grad[:3])
        ndx, ndw, ndg = needs[0] and hw is not None, needs[1] and hx is not None, needs[2]
        if hx is None and hw is None:
            return None, None, None, None
        if torch.is_grad_enabled():
            dx = g*hw if ndx else None
            dw = (g*hx).sum(0) if ndw else None
            terms = [value for value in (None if hx is None else hx*w,
                                         None if hw is None else x*hw) if value is not None]
            dg = sum(terms[1:], terms[0]) if ndg else None
        else:
            dx = torch.empty_like(x) if ndx else None
            partial = _empty_partial(x, ndw)
            dg = torch.empty_like(g) if ndg else None
            if x.numel() and (ndx or ndw or ndg):
                _mixed[(triton.cdiv(x.shape[0], 128), triton.cdiv(x.shape[1], 32))](
                    x, w, g, x if hx is None else hx.contiguous(), w if hw is None else hw.contiguous(),
                    x if dx is None else dx, partial, g if dg is None else dg,
                    *x.shape, hx is not None, hw is not None, ndx, ndw, ndg,
                    128, 32, num_warps=4, enable_fp_fusion=False)
            dw = partial.sum(0) if ndw else None
        return dx, dw, dg, None


class _GateProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, w)
        out = torch.empty_like(x)
        if x.numel():
            _value[(triton.cdiv(x.numel(), 256),)](
                x, w, out, x.numel(), x.shape[1], 256, num_warps=4, enable_fp_fusion=False)
        return out

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return None, None
        x, w = ctx.saved_tensors
        needed = required_gradients((x, w), ctx.needs_input_grad[:2])
        return _GateAdjoint.apply(x, w, grad.contiguous(), needed)


def path_weighted_gates(gates, weight):
    """Multiply [E,D] gates by live [D] or [P,C] path coefficients."""
    from torch.fx.experimental.proxy_tensor import get_proxy_mode
    from torch._subclasses.fake_tensor import is_fake
    flat = weight.reshape(-1)
    if gates.ndim != 2 or flat.numel() != gates.shape[1]:
        raise ValueError("Expected gates [E,D] and D path coefficients")
    if (not gates.is_cuda or gates.dtype != torch.float32 or weight.dtype != torch.float32
            or weight.device != gates.device or not gates.is_contiguous()
            or torch.compiler.is_compiling() or get_proxy_mode() is not None or is_fake(gates)
            or torch._C._are_functorch_transforms_active() or torch.is_autocast_enabled(gates.device.type)):
        return gates*flat
    return _GateProduct.apply(gates, flat.contiguous())
