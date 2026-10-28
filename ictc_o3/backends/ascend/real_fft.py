# SPDX-License-Identifier: MIT
"""NPU real FFTs with explicit differentiable linear adjoints.

The installed NPU r2c backward does not register its higher-order derivative.
These paired adjoints use vendor forward transforms in both directions and
retain ordinary autograd for force-loss and cell/source mixed derivatives.
"""
import math
import torch

_WEIGHTS = {}


def _weights(value, shape):
    key = str(value.device), tuple(shape)
    if key not in _WEIGHTS:
        length, size = shape[-1], math.prod(shape)
        factors = [2. if i>0 and (length%2 or i<length//2) else 1. for i in range(length//2+1)]
        cpu = torch.tensor([factors, [1./v for v in factors], [v/size for v in factors]], dtype=torch.float32)
        if len(_WEIGHTS)>=32:_WEIGHTS.pop(next(iter(_WEIGHTS)))
        _WEIGHTS[key] = tuple(cpu.to(value.device).unbind(0))
    return _WEIGHTS[key]


def _scale(value, weight):
    pair = torch.view_as_real(value)
    return torch.view_as_complex((pair*weight[..., None]).contiguous())


class _RealForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.shape = tuple(value.shape[-3:])
        # Transform the real axis explicitly; the remaining axes use CANN's
        # complex transforms and the paired adjoints below.
        half = torch.fft.rfft(value.contiguous(), dim=-1)
        return torch.fft.fftn(half, dim=(-3, -2))

    @staticmethod
    def backward(ctx, gradient):
        _, inverse_weight, _ = _weights(gradient, ctx.shape)
        return _RealInverse.apply(_scale(gradient, inverse_weight), ctx.shape)*math.prod(ctx.shape)


class _RealInverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, shape):
        ctx.shape = tuple(shape)
        partial = torch.fft.ifftn(value.contiguous(), dim=(-3, -2))
        return torch.fft.irfft(partial, n=shape[-1], dim=-1)

    @staticmethod
    def backward(ctx, gradient):
        transformed = _RealForward.apply(gradient)
        _, _, weight = _weights(transformed, ctx.shape)
        return _scale(transformed, weight), None


def rfftn(value):
    return _RealForward.apply(value)


def irfftn(value, shape):
    return _RealInverse.apply(value, tuple(shape))
