"""Shared channel GEMMs with live first and higher derivatives."""
import torch
import torch.nn.functional as F

from .runtime import require_npu_fp32


def channel_linear(features, weight, bias=None):
    """features [...,I,M], weight [O,I], optional bias [O]."""
    require_npu_fp32(features, weight, *(() if bias is None else (bias,)))
    if features.ndim < 2 or weight.ndim != 2 or features.shape[-2] != weight.shape[1]:
        raise ValueError("Expected features [...,I,M] and weight [O,I]")
    if bias is not None and (features.shape[-1] != 1 or bias.shape != (weight.shape[0],)):
        raise ValueError("Bias is allowed only for scalar angular blocks with shape [O]")
    from ictc_o3.ascendc.runtime import is_enabled
    if is_enabled():
        from ictc_o3.ascendc.linear import channel_linear as native_linear
        return native_linear(features,weight,bias)
    angular = features.movedim(-2, -1).contiguous()
    result = F.linear(angular.reshape(-1, weight.shape[1]), weight, bias)
    return result.reshape(*angular.shape[:-1], weight.shape[0]).movedim(-1, -2)


def species_channel_linear(features, weight, species, bias=None):
    """Apply a small weight bank without an [N,O,I] temporary or host sorting.

    The shared GEMMs evaluate all species and select each node's result. This
    path trades species-count arithmetic for bounded memory and asynchronous
    execution, preserving every live weight and mixed derivative.
    """
    require_npu_fp32(features, weight, *(() if bias is None else (bias,)))
    if (features.ndim != 3 or weight.ndim != 3 or features.shape[1] != weight.shape[2]
            or species.shape != (features.shape[0],) or species.device != features.device
            or species.dtype not in (torch.int32, torch.int64) or weight.shape[0] < 1):
        raise ValueError("Expected features [N,I,M], weight [S,O,I] and integer species [N]")
    from .species import validate_species
    validate_species(species, weight.shape[0])
    if bias is not None and (features.shape[-1] != 1 or bias.shape != weight.shape[:2]):
        raise ValueError("Bias is allowed only for scalar angular blocks with shape [S,O]")
    from ictc_o3.ascendc.runtime import is_enabled
    if is_enabled():
        from ictc_o3.ascendc.linear import species_channel_linear as native_species_linear
        return native_species_linear(features,weight,species,bias)
    angular = features.transpose(1, 2).contiguous()
    flat = angular.reshape(-1, angular.shape[-1])
    result = None
    for s in range(weight.shape[0]):
        value = F.linear(flat, weight[s], None if bias is None else bias[s]).reshape(
            features.shape[0], features.shape[2], weight.shape[1])
        selected = torch.where(species[:, None, None] == s, value, 0.)
        result = selected if result is None else result+selected
    return result.transpose(1, 2)
