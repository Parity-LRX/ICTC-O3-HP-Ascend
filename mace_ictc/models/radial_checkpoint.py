"""Recompute radial MLP activations without changing modules or parameters."""
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch._subclasses.fake_tensor import is_fake


def radial_apply(fc,value,*,hidden=False,checkpoint=False):
    """Evaluate the original sequence, optionally excluding its last linear."""
    def evaluate(x):
        if not hidden:return fc(x)
        for index in range(len(fc)-1):x=fc[index](x)
        return x

    if (not checkpoint or not torch.is_grad_enabled() or not value.is_cuda
            or value.dtype!=torch.float32 or torch.compiler.is_compiling()
            or torch.is_autocast_enabled(value.device.type)
            or get_proxy_mode() is not None or is_fake(value)
            or torch._C._are_functorch_transforms_active()):
        return evaluate(value)
    from torch.utils.checkpoint import checkpoint as recompute
    return recompute(evaluate,value,use_reentrant=False)
