# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Live indexed species GEMMs and recursive native bilinear adjoints.

Each node evaluates only its actual species bank. The two bilinear primitives
(indexed GEMM and grouped weight reduction) are each other's exact adjoints,
including force-loss double backward and higher derivatives. No CUDA Graph,
host species selection, N-sized matrix bank or external TP operator is used.
"""
import torch
import triton
import triton.language as tl

from mace_ictc.models.adjoint_requirements import required_gradients
from ._species_gemm import _grouped


@triton.jit
def _weight_gradient(X, G, PTR, ORDER, PART, N: tl.constexpr, M: tl.constexpr,
                     CI: tl.constexpr, CO: tl.constexpr, S: tl.constexpr,
                     SX0: tl.constexpr, SX1: tl.constexpr, SX2: tl.constexpr,
                     SG0: tl.constexpr, SG1: tl.constexpr, SG2: tl.constexpr,
                     CHUNKS: tl.constexpr, BI: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
    tile, chunk, species = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ni = tl.cdiv(CI, BI)
    ii = (tile%ni)*BI+tl.arange(0, BI)
    oo = (tile//ni)*BO+tl.arange(0, BO)
    reduction = tl.arange(0, BK)
    begin, end = tl.load(PTR+species), tl.load(PTR+species+1)
    rows = (end-begin)*M
    accumulator = tl.full((BO, BI), 0., tl.float32)
    for start in range(chunk*BK, rows, CHUNKS*BK):
        row = start+reduction
        node = tl.load(ORDER+begin+row//M, row<rows, 0)
        angular = row%M
        g = tl.load(G+node[None, :]*SG0+oo[:, None]*SG1+angular[None, :]*SG2,
                    (oo[:, None]<CO)&(row[None, :]<rows), 0.)
        x = tl.load(X+node[:, None]*SX0+ii[None, :]*SX1+angular[:, None]*SX2,
                    (row[:, None]<rows)&(ii[None, :]<CI), 0.)
        accumulator = tl.dot(g, x, accumulator, input_precision="ieee")
    address = ((chunk*S+species)*CO+oo[:, None])*CI+ii[None, :]
    tl.store(PART+address, accumulator, (oo[:, None]<CO)&(ii[None, :]<CI))


def _linear_raw(features, weight, transpose, ptr, order):
    n, ci, m = features.shape
    co = weight.shape[2] if transpose else weight.shape[1]
    out = features.new_empty(n, m, co)
    if n:
        _grouped[(triton.cdiv(n*m, 32), triton.cdiv(co, 64), weight.shape[0])](
            features, weight, features, ptr, order, out, n, m, ci, co,
            *features.stride(), transpose, False, 32, 64, 32, num_warps=4)
    return out.transpose(-1, -2)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, transpose, ptr, order):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, w, ptr, order)
        ctx.transpose = transpose
        return _linear_raw(x, w, transpose, ptr, order)

    @staticmethod
    def backward(ctx, grad):
        if grad is None:
            return (None,)*5
        x, w, ptr, order = ctx.saved_tensors
        needed = required_gradients((x, w), ctx.needs_input_grad[:2])
        dx = _Linear.apply(grad, w, not ctx.transpose, ptr, order) if needed[0] else None
        dw = (_Weight.apply(grad, x, ptr, order, w.shape[0]) if ctx.transpose else
              _Weight.apply(x, grad, ptr, order, w.shape[0])) if needed[1] else None
        return dx, dw, None, None, None


class _Weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad, ptr, order, species):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, grad, ptr, order)
        n, ci, m = x.shape
        co = grad.shape[1]
        chunks = min(16, max(1, triton.cdiv(n*m, 256)))
        partial = x.new_empty(chunks, species, co, ci)
        _weight_gradient[(triton.cdiv(ci, 32)*triton.cdiv(co, 32), chunks, species)](
            x, grad, ptr, order, partial, n, m, ci, co, species,
            *x.stride(), *grad.stride(), chunks, 32, 32, 32, num_warps=4)
        return partial.sum(0)

    @staticmethod
    def backward(ctx, incoming):
        if incoming is None:
            return (None,)*5
        x, grad, ptr, order = ctx.saved_tensors
        needed = required_gradients((x, grad), ctx.needs_input_grad[:2])
        bank = incoming.contiguous()
        dx = _Linear.apply(grad, bank, True, ptr, order) if needed[0] else None
        dg = _Linear.apply(x, bank, False, ptr, order) if needed[1] else None
        return dx, dg, None, None, None


def training_species_linear(features, weight, species, bias=None):
    from mace_ictc.models.message_topology import topology
    torch._assert_async(((species>=0)&(species<weight.shape[0])).all(),
                        "Species index outside the weight bank")
    ptr, order = topology(species, weight.shape[0])
    value = _Linear.apply(features, weight.contiguous(), False, ptr, order)
    if bias is not None:
        value = value+bias.index_select(0, species)[:, :, None]
    return value
