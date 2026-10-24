# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Shared FP32 species GEMMs without an [N,C_out,C_in] weight expansion.

Small species banks trade a few dense GEMMs for eliminating the quadratic
per-node gather and its parameter-gradient scatter. All operations are ordinary
PyTorch, so learned banks and mixed/higher derivatives remain live.
"""
import torch
import torch.nn.functional as F


def shared_species_linear(block, weight, index, bias=None):
    """Apply weight[S,O,I] to block[N,I,M] selected by fixed index[N]."""
    torch._assert_async(((index>=0)&(index<weight.shape[0])).all(),"Species index outside the weight bank")
    angular=block.movedim(-2,-1).contiguous()
    flat=angular.reshape(-1,angular.shape[-1])
    result=None
    for species in range(weight.shape[0]):
        value=F.linear(flat,weight[species],None if bias is None else bias[species])
        value=value.reshape(*angular.shape[:-1],weight.shape[1])
        selected=torch.where(index[:,None,None]==species,value,0.)
        result=selected if result is None else result+selected
    return result.movedim(-1,-2)


def supported(block, weight):
    return (block.is_cuda and block.dtype==torch.float32 and weight.dtype==block.dtype
            and weight.device==block.device and 1<=weight.shape[0]<=4
            and not torch.is_autocast_enabled(block.device.type))
