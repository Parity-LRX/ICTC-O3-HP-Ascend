"""Shared CG prefixes with directly indexed species banks on Ascend."""
import torch

from .sparse_product import sparse_product


def factorized_contractions(owner, x, index):
    from mace_ictc.models.factorized_contraction import _plan
    plan = _plan(owner, x)
    if plan is None:
        raise ValueError("Symmetric-contraction constants do not reconstruct within the FP32 budget")
    n, channels, _ = x.shape
    native = x @ plan["q"]
    banks = []
    for module, order, projection in plan["projections"]:
        weights = [module.weights_max, *module.weights]
        banks.append(torch.einsum("rp,zpc->zcr", projection, weights[module.correlation-order]))
    bank = torch.cat(banks, -1)
    product=sparse_product
    if getattr(owner,"_ascend_sc_schedule","gemm")=="gemm":
        from .dense_product import product
    if plan["pair"] is None:
        pairs = native[:, :, :0]
    else:
        pairs = product(native, native, plan["ones"].unsqueeze(0), plan["pair"])
    left = torch.cat((pairs, native), -1)
    right = torch.cat((native, native.new_ones(n, channels, 1)), -1)
    output = product(left, right, bank, plan["final"], index=index)
    return torch.cat([(part @ q.T).reshape(n, -1)
                      for part, q in zip(output.split(plan["sizes"], -1), plan["qout"])], -1)
