# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Native indexed FP32 GEMM for a frozen species bank, with exact adjoints."""
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped(X, W, B, PTR, ORDER, Y, N: tl.constexpr, M: tl.constexpr,
             CI: tl.constexpr, CO: tl.constexpr, SX0: tl.constexpr,
             SX1: tl.constexpr, SX2: tl.constexpr, TRANSPOSE: tl.constexpr,
             BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    species = tl.program_id(2)
    begin, end = tl.load(PTR+species), tl.load(PTR+species+1)
    row = tl.program_id(0)*BM+tl.arange(0, BM)
    if tl.program_id(0)*BM >= (end-begin)*M:
        return
    column = tl.program_id(1)*BN+tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    nodes = tl.load(ORDER+begin+row//M, row < (end-begin)*M, 0)
    angular = row % M
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for start in range(tl.cdiv(CI, BK)):
        k = start*BK+reduction
        a = tl.load(X+nodes[:, None]*SX0+k[None, :]*SX1+angular[:, None]*SX2,
                    (row[:, None] < (end-begin)*M) & (k[None, :] < CI), 0)
        if TRANSPOSE:
            address = species*CI*CO+k[:, None]*CO+column[None, :]
        else:
            address = species*CI*CO+column[None, :]*CI+k[:, None]
        b = tl.load(W+address, (k[:, None] < CI) & (column[None, :] < CO), 0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
    if BIAS:
        accumulator += tl.load(B+species*CO+column, column < CO, 0)[None, :]
    tl.store(Y+(nodes[:, None]*M+angular[:, None])*CO+column[None, :], accumulator,
             (row[:, None] < (end-begin)*M) & (column[None, :] < CO))


class _SpeciesGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weight, bias, transpose, ptr, order):
        n, ci, angular = features.shape
        co = weight.shape[2] if transpose else weight.shape[1]
        ctx.save_for_backward(weight, ptr, order)
        ctx.transpose = transpose
        # This view is already the following channel GEMM's preferred layout.
        output = features.new_empty(n, angular, co)
        if n:
            grid = (triton.cdiv(n*angular, 32), triton.cdiv(co, 64), weight.shape[0])
            _grouped[grid](features, weight, bias if bias is not None else features,
                           ptr, order, output, n, angular, ci, co, *features.stride(),
                           transpose, bias is not None, 32, 64, 32, num_warps=4)
        return output.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad):
        weight, ptr, order = ctx.saved_tensors
        dx = (_SpeciesGemm.apply(grad, weight, None, not ctx.transpose, ptr, order)
              if ctx.needs_input_grad[0] else None)
        return dx, None, None, None, None, None


def grouped_species_linear(features, weight, species, bias=None):
    """Frozen banks only; original Torch path owns all learned-bank gradients."""
    if weight.requires_grad or (bias is not None and bias.requires_grad):
        raise ValueError("Indexed inference GEMM requires frozen bank weights")
    from mace_ictc.models.message_topology import topology
    torch._assert_async(((species >= 0) & (species < weight.shape[0])).all(),
                        "Species index outside the weight bank")
    ptr, order = topology(species, weight.shape[0])
    return _SpeciesGemm.apply(features, weight.contiguous(),
                              None if bias is None else bias.contiguous(), False, ptr, order)
