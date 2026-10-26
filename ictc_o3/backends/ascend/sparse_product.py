"""Ascend AI Core sparse multilinear maps and differentiable adjoints.

Coefficients and reduction tables describe the mathematical map. Kernel tiling
is independent of channels, angular degrees, species count and graph size.
"""
from __future__ import annotations

import torch

from .runtime import _vector_grid

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _map(L, R, W, G, SPECIES, IDS, COEF, Y,
             N, C: tl.constexpr, RC: tl.constexpr,
             A: tl.constexpr, B: tl.constexpr, P: tl.constexpr,
             M: tl.constexpr, O: tl.constexpr, K: tl.constexpr,
             MODE: tl.constexpr, BANK: tl.constexpr, INDEXED: tl.constexpr,
             BO: tl.constexpr, BK: tl.constexpr):
        for tile in range(tl.program_id(0).to(tl.int64),
                          tl.cdiv(tl.cast(N, tl.int64)*C*O, BO),
                          tl.num_programs(0).to(tl.int64)):
            row = tile * BO + tl.arange(0, BO)
            f, c, n = row % O, (row // O) % C, row // (C * O)
            valid = row < tl.cast(N, tl.int64) * C * O
            k = tl.arange(0, BK)
            active = valid[:, None] & (k[None, :] < K)
            a = tl.load(IDS + f[:, None] * K + k[None, :], active, 0)
            b = tl.load(IDS + O*K + f[:, None] * K + k[None, :], active, 0)
            p = tl.load(IDS + 2*O*K + f[:, None] * K + k[None, :], active, 0)
            q = tl.load(IDS + 3*O*K + f[:, None] * K + k[None, :], active, 0)
            coefficient = tl.load(COEF + f[:, None] * K + k[None, :], active, 0.)
            if MODE < 3:
                if not BANK:
                    species = n
                elif INDEXED:
                    species = tl.load(SPECIES + n, valid, 0)
                else:
                    species = tl.full((BO,), 0, tl.int64)
                w = tl.load(W + (species[:, None]*C+c[:, None])*P+p, active, 0.)
            rc = tl.full((BO,), 0, tl.int64) if RC == 1 else c
            if MODE == 0:
                left = tl.load(L + (n[:, None]*C+c[:, None])*A+a, active, 0.)
                right = tl.load(R + (n[:, None]*RC+rc[:, None])*B+b, active, 0.)
                value = (left*right)*(coefficient*w)
            elif MODE == 1:
                right = tl.load(R + (n[:, None]*RC+rc[:, None])*B+b, active, 0.)
                grad = tl.load(G + (n[:, None]*C+c[:, None])*M+q, active, 0.)
                value = (right*grad)*(coefficient*w)
            elif MODE == 2:
                left = tl.load(L + (n[:, None]*C+c[:, None])*A+a, active, 0.)
                grad = tl.load(G + (n[:, None]*C+c[:, None])*M+q, active, 0.)
                value = (left*grad)*(coefficient*w)
            else:
                left = tl.load(L + (n[:, None]*C+c[:, None])*A+a, active, 0.)
                right = tl.load(R + (n[:, None]*RC+rc[:, None])*B+b, active, 0.)
                grad = tl.load(G + (n[:, None]*C+c[:, None])*M+q, active, 0.)
                value = (left*right)*(coefficient*grad)
            tl.store(Y + row, tl.sum(value, 1), valid)

    @triton.jit
    def _shared_right_adjoint(L, W, G, SPECIES, IDS, COEF, Y,
                              N, C: tl.constexpr, A: tl.constexpr,
                              B: tl.constexpr, P: tl.constexpr, M: tl.constexpr,
                              K: tl.constexpr, CB: tl.constexpr,
                              BANK: tl.constexpr, INDEXED: tl.constexpr,
                              BO: tl.constexpr, BC: tl.constexpr, BKC: tl.constexpr):
        for task in range(tl.program_id(0).to(tl.int64),
                          tl.cdiv(tl.cast(N, tl.int64)*B, BO)*CB,
                          tl.num_programs(0).to(tl.int64)):
            channel_block = task % CB
            row = (task // CB)*BO+tl.arange(0, BO)
            b, n = row % B, row // B
            kc = tl.arange(0, BKC)
            k, c = kc // BC, channel_block*BC+kc % BC
            active = (row[:, None] < tl.cast(N, tl.int64)*B) & (k[None, :] < K) & (c[None, :] < C)
            a = tl.load(IDS+b[:, None]*K+k[None, :], active, 0)
            p = tl.load(IDS+2*B*K+b[:, None]*K+k[None, :], active, 0)
            q = tl.load(IDS+3*B*K+b[:, None]*K+k[None, :], active, 0)
            coefficient = tl.load(COEF+b[:, None]*K+k[None, :], active, 0.)
            if not BANK:
                species = n
            elif INDEXED:
                species = tl.load(SPECIES+n, row < tl.cast(N, tl.int64)*B, 0)
            else:
                species = tl.full((BO,), 0, tl.int64)
            left = tl.load(L+(n[:, None]*C+c[None, :])*A+a, active, 0.)
            weight = tl.load(W+(species[:, None]*C+c[None, :])*P+p, active, 0.)
            grad = tl.load(G+(n[:, None]*C+c[None, :])*M+q, active, 0.)
            value = tl.sum((left*grad)*(coefficient*weight), 1)
            tl.store(Y+row*CB+channel_block, value, row < tl.cast(N, tl.int64)*B)

    @triton.jit
    def _bank_adjoint(L, R, G, SPECIES, IDS, COEF, Y,
                      N, C: tl.constexpr, RC: tl.constexpr,
                      A: tl.constexpr, B: tl.constexpr, P: tl.constexpr,
                      M: tl.constexpr, K: tl.constexpr, NB, TASKS,
                      INDEXED: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        for task in range(tl.program_id(0).to(tl.int64), tl.cast(TASKS, tl.int64),
                          tl.num_programs(0).to(tl.int64)):
            row, atom_block = task // NB, task % NB
            p, c, species = row % P, (row // P) % C, row // (C*P)
            n = atom_block * BN + tl.arange(0, BN)
            k = tl.arange(0, BK)
            valid = n < N
            if INDEXED:
                valid = valid & (tl.load(SPECIES+n, n < N, 0) == species)
            active = valid[:, None] & (k[None, :] < K)
            a = tl.load(IDS+p*K+k, k < K, 0)
            b = tl.load(IDS+P*K+p*K+k, k < K, 0)
            q = tl.load(IDS+3*P*K+p*K+k, k < K, 0)
            coefficient = tl.load(COEF+p*K+k, k < K, 0.)
            rc = 0 if RC == 1 else c
            left = tl.load(L+(n[:, None]*C+c)*A+a[None, :], active, 0.)
            right = tl.load(R+(n[:, None]*RC+rc)*B+b[None, :], active, 0.)
            grad = tl.load(G+(n[:, None]*C+c)*M+q[None, :], active, 0.)
            value = tl.sum(tl.sum((left*right)*(grad*coefficient[None, :]), 1), 0)
            tl.store(Y+row*NB+atom_block, value)


def available():
    from ictc_o3.ascendc.runtime import is_enabled
    if is_enabled():
        return True
    if triton is None:
        return False
    try:
        from triton.backends import backends
        return "ascend" in backends
    except ImportError:
        return False


def _raw(left, right, weight, index, plan, bank, mode=0, grad=None):
    from ictc_o3.ascendc.runtime import is_enabled
    if is_enabled() or '_ascendc_graph' in plan:
        from ictc_o3.ascendc.product import raw
        return raw(left,right,weight,index,plan,bank,mode,grad)
    if plan.get("_ascend_gemm",False):
        from .dense_product import raw
        return raw(left,right,weight,index,plan,bank,mode,grad)
    n, channels, a = left.shape
    _, rc, b = right.shape
    _, _, paths = weight.shape
    m = plan["sizes"][3]
    ids, coefficient = plan["reductions"][mode]
    k = ids.shape[-1]
    bk = triton.next_power_of_2(k)
    if mode == 2 and rc == 1:
        bc = triton.next_power_of_2(max(1, min(channels, 64, 1024 // bk)))
        cb = triton.cdiv(channels, bc)
        bo = triton.next_power_of_2(max(1, min(32, 2048 // (bk*bc))))
        partial = left.new_empty(n, b, cb)
        if partial.numel():
            _shared_right_adjoint[(_vector_grid(triton.cdiv(n*b, bo)*cb, left.device),)](
                left, weight, grad, left if index is None else index, ids,
                coefficient, partial, n, channels, a, b, paths, m, k, cb,
                bank, index is not None, bo, bc, bk*bc)
        return (partial.squeeze(-1) if cb == 1 else partial.sum(-1)).unsqueeze(1)
    if mode == 3 and bank:
        bn = max(1, min(32, 2048 // bk))
        bn = triton.next_power_of_2(bn)
        nb = triton.cdiv(n, bn)
        partial = left.new_empty(*weight.shape, nb)
        if partial.numel():
            _bank_adjoint[(_vector_grid(weight.numel()*nb, left.device),)](
                left, right, grad, left if index is None else index, ids,
                coefficient, partial, n, channels, rc, a, b, paths, m, k, nb, weight.numel()*nb,
                index is not None, bn, bk)
        return partial.sum(-1)
    out_width = (m, a, b, paths)[mode]
    result = left.new_empty(n, channels, out_width)
    bo = triton.next_power_of_2(max(1, min(32, 2048 // bk)))
    if result.numel():
        _map[(_vector_grid(triton.cdiv(result.numel(), bo), left.device),)](
            left, right, weight, left if grad is None else grad,
            left if index is None else index, ids, coefficient, result,
            n, channels, rc, a, b, paths, m, out_width, k, mode, bank,
            index is not None, bo, bk)
    return result


def _reference(left, right, weight, index, plan, bank):
    graph=plan.get('_ascendc_graph')
    nodes=left.shape[0] if graph is None else graph.get('destination_nodes',left.shape[0])
    if graph is not None:
        left=left.index_select(0,graph['src'])
    selected = weight if not bank else (weight[index] if index is not None else weight)
    result = left.new_zeros(left.shape[0], left.shape[1], plan["sizes"][3])
    for a, b, p, out, coefficient in plan["entries"]:
        term = left[:, :, a]*right[:, :, b]*(selected[:, :, p]*coefficient)
        result[:, :, out] = result[:, :, out]+term
    if graph is not None:
        if graph['mask'] is not None:
            result=result*graph['mask'][:,None,None]
        result=result.new_zeros(nodes,result.shape[1],result.shape[2]).index_add(0,graph['dst'],result)
    return result


def _graph_tensors(plan):
    graph=plan.get('_ascendc_graph')
    if graph is None:
        return ()
    return (graph['src'],graph['dst'],*(() if graph['mask'] is None else (graph['mask'],)))


class _Adjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, weight, grad, index, plan, bank, needed):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(left, right, weight, grad,
                              left.new_empty(0, dtype=torch.int64) if index is None else index,
                              *_graph_tensors(plan))
        ctx.spec = plan, bank, needed, index is not None
        return tuple(_raw(left, right, weight, index, plan, bank, mode+1, grad)
                     if need else None for mode, need in enumerate(needed))

    @staticmethod
    def backward(ctx, hleft, hright, hweight):
        left, right, weight, grad, saved_index, *_ = ctx.saved_tensors
        plan, bank, first_needed, indexed = ctx.spec
        index = saved_index if indexed else None
        from ictc_o3.autograd.adjoint_requirements import required_gradients
        needed = required_gradients((left, right, weight, grad), ctx.needs_input_grad[:4])
        if torch.is_grad_enabled():
            # Isolate partial derivatives when operands share an ancestor.
            with torch.enable_grad():
                local = [v.clone() for v in (left, right, weight, grad)]
                active = [i for i, v in enumerate(local) if v.requires_grad]
                output = _reference(*local[:3], index, plan, bank)
                first_ids = [i for i in range(3) if first_needed[i]]
                first = torch.autograd.grad(output, [local[i] for i in first_ids],
                                            local[3], create_graph=True, allow_unused=True)
                hs = (hleft, hright, hweight)
                terms = [(value*hs[i]).sum() for i, value in zip(first_ids, first)
                         if value is not None and hs[i] is not None and value.requires_grad]
                result = [None]*4
                if terms:
                    values = torch.autograd.grad(sum(terms), [local[i] for i in active],
                                                 create_graph=True, allow_unused=True)
                    for i, value in zip(active, values):
                        result[i] = value if needed[i] else None
            return *result, None, None, None, None
        result = [None]*4
        def add(slot, value):
            result[slot] = value if result[slot] is None else result[slot]+value
        for role, h in enumerate((hleft, hright, hweight)):
            if h is None:
                continue
            operands = [left, right, weight]
            operands[role] = h.contiguous()
            for target in range(3):
                if target != role and needed[target]:
                    add(target, _raw(*operands, index, plan, bank, target+1, grad))
            if needed[3]:
                add(3, _raw(*operands, index, plan, bank))
        return *result, None, None, None, None


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left, right, weight, index, plan, bank):
        ctx.save_for_backward(left, right, weight,
                              left.new_empty(0, dtype=torch.int64) if index is None else index,
                              *_graph_tensors(plan))
        ctx.spec = plan, bank, index is not None
        return _raw(left, right, weight, index, plan, bank)

    @staticmethod
    def backward(ctx, grad):
        left, right, weight, saved_index, *_ = ctx.saved_tensors
        plan, bank, indexed = ctx.spec
        index = saved_index if indexed else None
        from ictc_o3.autograd.adjoint_requirements import required_gradients
        needed = required_gradients((left, right, weight), ctx.needs_input_grad[:3])
        grad = grad.contiguous()
        if torch.is_grad_enabled():
            values = _Adjoint.apply(left, right, weight, grad, index, plan, bank, needed)
        else:
            values = tuple(_raw(left, right, weight, index, plan, bank, i+1, grad)
                           if need else None for i, need in enumerate(needed))
        return *values, None, None, None


class _QuadraticAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, grad, ones, plan, adjoint_plan):
        ctx.save_for_backward(x, grad)
        ctx.spec = ones, plan, adjoint_plan
        return _raw(x, x, ones, None, adjoint_plan, True, 1, grad)

    @staticmethod
    def backward(ctx, h):
        x, grad = ctx.saved_tensors
        ones, plan, adjoint_plan = ctx.spec
        from ictc_o3.autograd.adjoint_requirements import required_gradients
        needed = required_gradients((x, grad), ctx.needs_input_grad[:2])
        if torch.is_grad_enabled():
            # Partial derivatives use separate local graph nodes even when the
            # incoming cotangent depends on the original quadratic input.
            xx, gg = x.clone(), grad.clone()
            first = torch.autograd.grad(_reference(xx, xx, ones, None, plan, True),
                                        xx, gg, create_graph=True)[0]
            active = [v for v, need in zip((xx, gg), needed) if need]
            values = iter(torch.autograd.grad(first, active, h,
                                              create_graph=True, allow_unused=True))
            dx, dg = (next(values) if need else None for need in needed)
        else:
            h = h.contiguous()
            dx = _raw(h, h, ones, None, adjoint_plan, True, 1, grad) if needed[0] else None
            dg = _raw(h, x, ones, None, adjoint_plan, True) if needed[1] else None
        return dx, dg, None, None, None


class _Quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ones, plan, adjoint_plan):
        ctx.save_for_backward(x)
        ctx.spec = ones, plan, adjoint_plan
        return _raw(x, x, ones, None, plan, True)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        ones, plan, adjoint_plan = ctx.spec
        grad = grad.contiguous()
        value = (_QuadraticAdjoint.apply(x, grad, ones, plan, adjoint_plan)
                 if torch.is_grad_enabled()
                 else _raw(x, x, ones, None, adjoint_plan, True, 1, grad))
        return value, None, None, None


def sparse_quadratic(x, ones, plan, adjoint_plan):
    """Contract two identical inputs with a single combined input adjoint.

    The derivative plan contains both permutations of each original term.
    The coefficient bank must be fixed; learnable banks use sparse_product.
    """
    from .runtime import require_npu_fp32
    require_npu_fp32(x, ones)
    a, b, p, _ = plan['sizes']
    if (a != b or x.ndim != 3 or x.shape[-1] != a or
            ones.shape != (1, x.shape[1], p) or ones.requires_grad or
            tuple(adjoint_plan['sizes']) != tuple(plan['sizes'])):
        raise ValueError('Invalid fixed-bank quadratic sparse map')
    return _Quadratic.apply(x.contiguous(), ones.contiguous(), plan, adjoint_plan)


def sparse_product(left, right, weight, plan, *, index=None, bank=True):
    """Evaluate a sparse map with a shared, species or per-row weight bank.

    left [N,C,A], right [N,C|1,B], weight [S,C,P]. For bank=True, a missing
    index selects the single shared bank; otherwise index is an integer [N]
    species map. bank=False selects weight[n] without an index allocation.
    Plans come from sparse_product_plan and contain immutable coefficients.
    """
    from .runtime import require_npu_fp32
    require_npu_fp32(left, right, weight)
    n, channels, a = left.shape
    if (right.ndim != 3 or right.shape[0] != n or right.shape[1] not in (1, channels)
            or weight.ndim != 3 or weight.shape[1] != channels
            or tuple(plan["sizes"][:3]) != (a, right.shape[-1], weight.shape[-1])):
        raise ValueError("Sparse-map operand shapes do not match the plan")
    if (not bank and weight.shape[0] != n) or (bank and index is None and weight.shape[0] != 1):
        raise ValueError("Expected one shared bank or a per-row/species weight bank")
    if index is not None:
        if (not bank or index.shape != (n,) or index.device != left.device
                or index.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Species index must be an integer [N] tensor on the NPU")
        from .species import validate_species
        validate_species(index, weight.shape[0])
    if not available():
        raise RuntimeError("Ascend sparse maps require Triton-Ascend; install the NPU toolchain")
    return _Product.apply(left.contiguous(), right.contiguous(), weight.contiguous(),
                          None if index is None else index.contiguous(), plan, bank)
