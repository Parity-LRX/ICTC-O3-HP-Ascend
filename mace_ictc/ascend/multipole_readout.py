# SPDX-License-Identifier: MIT
"""Exact fixed Cartesian recovery matrices for learned multipole sources."""
import os
import torch


def recover(recovery, block):
    policy=os.environ.get('ICTC_ASCEND_MULTIPOLE_RECOVERY','1')
    if policy not in ('0','1'):
        raise ValueError('ICTC_ASCEND_MULTIPOLE_RECOVERY must be zero or one')
    if policy=='0' or block.device.type!='npu' or block.dtype!=torch.float32:
        return None
    if not getattr(recovery,'_ascend_hp',False):return None
    from mace_ictc.models.pure_cartesian_ictd_layers import PhysicalTensorICTDRecovery
    from mace_ictc.models.pure_cartesian_ictd_layers import _sym_rank_linear_permutation_indices
    from torch.nn.modules import module as module_hooks
    if (module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
            or module_hooks._global_backward_hooks or module_hooks._global_backward_pre_hooks):
        return None
    if (type(recovery) is not PhysicalTensorICTDRecovery or recovery.rank not in (1,2)
            or recovery.include_trace_chain or recovery.rank2_mode!='symmetric'
            or recovery.internal_compute_dtype!=block.dtype
            or 'forward' in recovery.__dict__
            or recovery._forward_hooks or recovery._forward_pre_hooks
            or recovery._backward_hooks or recovery._backward_pre_hooks):
        return None
    from ictc_o3.ascendc.runtime import is_enabled
    if not is_enabled():return None
    basis=recovery._get_reconstructors(device=block.device,dtype=block.dtype)[recovery.rank]
    if basis.requires_grad:return None
    try:version=basis._version
    except RuntimeError:version=None
    key=id(basis),version,str(block.device),block.dtype,recovery.rank
    cached=getattr(recovery,'_ascend_cartesian_matrix',None)
    if version is None or cached is None or cached[0]!=key:
        indices,coefficients=_sym_rank_linear_permutation_indices(recovery.rank)
        mapping=[0]*(3**recovery.rank)
        divisor=[1.]*(3**recovery.rank)
        for index,entries in enumerate(indices):
            for entry in entries.tolist():
                mapping[entry]=index
                divisor[entry]=float(coefficients[index])
        selector=torch.tensor(mapping,device=block.device,dtype=torch.int64)
        scales=torch.tensor(divisor,device=block.device,dtype=block.dtype)
        matrix=(basis.index_select(0,selector)/scales[:,None]).contiguous()
        if version is not None:recovery._ascend_cartesian_matrix=key,matrix
    else:matrix=cached[1]
    from ictc_o3.ascendc.linear import linear
    return linear(block,matrix).reshape(*block.shape[:-1],*([3]*recovery.rank))
