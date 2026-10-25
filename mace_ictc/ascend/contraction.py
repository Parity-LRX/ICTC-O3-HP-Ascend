"""Shared CG prefixes with directly indexed species banks on Ascend."""
import os
import torch

from .sparse_product import sparse_product


def factorized_contractions(owner, x, index):
    from mace_ictc.models.factorized_contraction import _plan
    plan = _plan(owner, x)
    if plan is None:
        raise ValueError("Symmetric-contraction constants do not reconstruct within the FP32 budget")
    n, channels, _ = x.shape
    from mace_ictc.ascendc.runtime import is_enabled,mode
    native_enabled=is_enabled()
    from .structured_product import _FixedLinear
    fixed=lambda value,matrix: _FixedLinear.apply(value,matrix) if native_enabled else value@matrix
    native = fixed(x,plan["q"])
    banks = []
    for module, order, projection in plan["projections"]:
        weights = [module.weights_max, *module.weights]
        weight=weights[module.correlation-order]
        banks.append(fixed(weight.transpose(1,2),projection.T.contiguous()) if native_enabled else
                     torch.einsum("rp,zpc->zcr", projection, weight))
    bank = torch.cat(banks, -1)
    product=sparse_product
    if getattr(owner,"_ascend_sc_schedule","gemm")=="gemm":
        from .dense_product import product
    if plan["pair"] is None:
        pairs = native[:, :, :0]
    else:
        policy=os.environ.get('ICTC_ASCEND_SYMMETRIC_PAIR','1' if native_enabled and mode()=='performance' else '0')
        if policy not in ('0','1'):
            raise ValueError('The symmetric-pair policy must be zero or one')
        if native_enabled and policy=='1':
            from .sparse_product import sparse_quadratic
            pairs=sparse_quadratic(native,plan['ones'].unsqueeze(0),
                                   plan['pair'],plan['pair_adjoint'])
        else:
            pairs = product(native, native, plan["ones"].unsqueeze(0), plan["pair"])
    left = torch.cat((pairs, native), -1)
    right = torch.cat((native, native.new_ones(n, channels, 1)), -1)
    output = product(left, right, bank, plan["final"], index=index)
    return torch.cat([fixed(part,q.T.contiguous()).reshape(n, -1)
                      for part, q in zip(output.split(plan["sizes"], -1), plan["qout"])], -1)
