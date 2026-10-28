# SPDX-License-Identifier: MIT
"""Bounded-memory Ascend C mesh assignment and analytic mixed adjoints."""
import os
import torch
from torch.autograd.function import once_differentiable
from ictc_o3.electrostatics.mesh import mesh_shape


def supported(value):
    if value.device.type != 'npu' or value.dtype != torch.float32:
        return False
    if torch.are_deterministic_algorithms_enabled() or torch.compiler.is_compiling() or torch.jit.is_tracing():
        return False
    from torch._subclasses.fake_tensor import FakeTensor
    from torch.fx.experimental.proxy_tensor import get_proxy_mode
    if isinstance(value, FakeTensor) or get_proxy_mode() is not None:
        return False
    from ictc_o3.ascendc import runtime
    return bool(runtime._LOADED or os.environ.get('ICTC_ASCENDC_LIBRARY'))


def prefer_native_assignment(frac, channels):
    """Select assignment independently of the real-FFT/spectral schedule."""
    policy = os.environ.get('ICTC_ASCEND_MESH_ASSIGNMENT', 'auto')
    if policy not in ('auto', 'native', 'vectorized'):
        raise ValueError('ICTC_ASCEND_MESH_ASSIGNMENT must be auto, native or vectorized')
    if policy != 'auto':return policy == 'native'
    from ictc_o3.ascendc.runtime import mode
    # Short channel vectors underfill per-particle DMA/vector work. Large
    # particle arrays can instead use the vectorized CANN index-add schedule.
    # Capacity and native modes keep bounded-UB assignment at every size.
    if mode() in ('capacity', 'native') or frac.shape[0] < 32768 or channels > 4:
        return True
    # The single-field particle tile also avoids the large stencil tensors
    # of the previous vectorized schedule. Explicit assignment policies above
    # retain their existing meaning.
    particle = os.environ.get('ICTC_ASCEND_MESH_PARTICLE', 'auto')
    if particle not in ('0', '1', 'auto'):
        raise ValueError('ICTC_ASCEND_MESH_PARTICLE must be auto, zero or one')
    if channels == 1 and particle != '0':
        # The bounded schedule wins for the qualified intermediate sizes.
        # Large force-training fields favor the existing CANN schedule.
        if particle == 'auto' and frac.shape[0] > 65536:
            return False
        from .runtime import has_triton_ascend
        return has_triton_ascend()
    return False


def _particle_assignment(source, mode, needed):
    policy = os.environ.get('ICTC_ASCEND_MESH_PARTICLE', 'auto')
    if policy not in ('0', '1', 'auto'):
        raise ValueError('ICTC_ASCEND_MESH_PARTICLE must be auto, zero or one')
    if policy == '0' or source.shape[1] != 1 or mode == 3 and needed[0]:
        return False
    # Small first adjoints are dominated by vector-kernel dispatch. Source and
    # mesh mixed adjoints retain their separate measured crossover.
    if policy == 'auto' and source.shape[0] < 8192 and mode != 3:
        return False
    from ictc_o3.ascendc.runtime import mode as execution_mode
    from .runtime import has_triton_ascend
    return execution_mode() != 'native' and has_triton_ascend()


def _run(frac, batch, source, mesh, config, mode, direction=None, needed=(True, True, True)):
    from ictc_o3.ascendc.runtime import load_library
    ops = load_library()
    if not hasattr(ops, 'electrostatic_assignment'):
        raise RuntimeError('Rebuild the Ascend C library to include electrostatic mesh assignment')
    shape, graphs, order, slab = config
    empty = frac.new_empty(0)
    particle = _particle_assignment(source, mode, needed)
    ds = torch.empty_like(source) if mode == 1 or mode == 3 and needed[1] else empty
    dm = source.new_zeros((graphs, *shape, source.shape[1])) if mode == 0 or mode == 3 and needed[2] else empty
    df = (torch.empty_like(frac) if particle else torch.zeros_like(frac)) if mode == 2 or mode == 3 and needed[0] else empty
    flags = sum(int(v) << i for i, v in enumerate(needed))
    if particle:
        from .scalar_assignment import run
        run(frac, batch, source, mesh, empty if direction is None else direction.contiguous(),
            ds, dm, df, config, mode, flags)
        return df, ds, dm
    ops.electrostatic_assignment(frac, batch, source, mesh,
        empty if direction is None else direction.contiguous(), ds, dm, df,
        [*shape, graphs, order, mode, int(slab), flags])
    return df, ds, dm


class _Spread(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frac, batch, source, config):
        ctx.save_for_backward(frac, batch, source)
        ctx.config = config
        ctx.set_materialize_grads(False)
        return _run(frac, batch, source, source.new_empty(0), config, 0)[2]

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:return None, None, None, None
        frac, batch, source = ctx.saved_tensors
        gradient = gradient.contiguous()
        from ictc_o3.autograd.adjoint_requirements import required_gradients
        needed = required_gradients((frac, source), (ctx.needs_input_grad[0], ctx.needs_input_grad[2]))
        df = _CoordinateVJP.apply(frac, batch, source, gradient, ctx.config) if needed[0] else None
        ds = _Gather.apply(frac, batch, gradient, ctx.config) if needed[1] else None
        return df, None, ds, None


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frac, batch, mesh, config):
        ctx.save_for_backward(frac, batch, mesh)
        ctx.config = config
        ctx.set_materialize_grads(False)
        source = frac.new_empty(frac.shape[0], mesh.shape[-1])
        return _run(frac, batch, source, mesh, config, 1)[1]

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:return None, None, None, None
        frac, batch, mesh = ctx.saved_tensors
        gradient = gradient.contiguous()
        from ictc_o3.autograd.adjoint_requirements import required_gradients
        needed = required_gradients((frac, mesh), (ctx.needs_input_grad[0], ctx.needs_input_grad[2]))
        df = _CoordinateVJP.apply(frac, batch, gradient, mesh, ctx.config) if needed[0] else None
        dm = _Spread.apply(frac, batch, gradient, ctx.config) if needed[1] else None
        return df, None, dm, None


class _CoordinateVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frac, batch, source, mesh, config):
        ctx.save_for_backward(frac, batch, source, mesh)
        ctx.config = config
        return _run(frac, batch, source, mesh, config, 2)[0]

    @staticmethod
    @once_differentiable
    def backward(ctx, direction):
        frac, batch, source, mesh = ctx.saved_tensors
        from ictc_o3.autograd.adjoint_requirements import required_gradients
        needed = required_gradients((frac, source, mesh),
            (ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]))
        df, ds, dm = _run(frac, batch, source, mesh, ctx.config, 3, direction, needed)
        return df if needed[0] else None, None, ds if needed[1] else None, dm if needed[2] else None, None


def _config(frac, batch, mesh_size, num_graphs, assignment, boundary):
    if not supported(frac):
        raise ValueError('Native electrostatics requires NPU FP32, a built Ascend C library and eager nondeterministic execution')
    if frac.ndim != 2 or frac.shape[1] != 3 or batch.shape != frac.shape[:1]:
        raise ValueError('Expected fractional positions [N,3] and batch [N]')
    if batch.dtype != torch.int64 or batch.device != frac.device:
        raise ValueError('batch must be int64 on the coordinate device')
    if assignment not in ('cic', 'tsc', 'pcs') or boundary not in ('periodic', 'slab'):
        raise ValueError('Expected cic/tsc/pcs assignment and periodic/slab boundary')
    if not isinstance(num_graphs, int) or isinstance(num_graphs, bool) or num_graphs < 1:
        raise ValueError('num_graphs must be a positive integer')
    # Invalid graph IDs must fail before the native kernel accesses the mesh.
    try:
        version = batch._version
    except RuntimeError:  # Tensors created under inference_mode have no counter.
        version = None
    bounds = version, num_graphs
    if version is None or getattr(batch, '_ictc_ascend_mesh_bounds', None) != bounds:
        if batch.numel() and bool(((batch < 0) | (batch >= num_graphs)).any()):
            raise ValueError('batch graph IDs must lie in [0,num_graphs)')
        batch._ictc_ascend_mesh_bounds = bounds
    return mesh_shape(mesh_size), num_graphs, {'cic': 2, 'tsc': 3, 'pcs': 4}[assignment], boundary == 'slab'


def spread(frac, batch, source, *, mesh_size, num_graphs, assignment='pcs', boundary='periodic', channels_first=False):
    config = _config(frac, batch, mesh_size, num_graphs, assignment, boundary)
    if source.ndim != 2 or source.shape[0] != frac.shape[0] or source.shape[1] < 1:
        raise ValueError('Expected source [N,C] with C > 0')
    if source.device != frac.device or source.dtype != frac.dtype:
        raise ValueError('Source and coordinates must share device and dtype')
    value = _Spread.apply(frac.contiguous(), batch.contiguous(), source.contiguous(), config)
    return value.permute(0, 4, 1, 2, 3).contiguous() if channels_first else value


def gather(frac, batch, mesh, *, mesh_size, num_graphs, assignment='pcs', boundary='periodic', channels_first=False):
    config = _config(frac, batch, mesh_size, num_graphs, assignment, boundary)
    if mesh.ndim != 5:
        raise ValueError('Expected a five-dimensional mesh')
    shape = config[0]
    channels = mesh.shape[1] if channels_first else mesh.shape[-1]
    expected = (num_graphs, channels, *shape) if channels_first else (num_graphs, *shape, channels)
    if mesh.shape != expected or channels < 1 or mesh.device != frac.device or mesh.dtype != frac.dtype:
        raise ValueError('Invalid mesh shape, device or dtype')
    if channels_first:mesh = mesh.permute(0, 2, 3, 4, 1)
    return _Gather.apply(frac.contiguous(), batch.contiguous(), mesh.contiguous(), config)
