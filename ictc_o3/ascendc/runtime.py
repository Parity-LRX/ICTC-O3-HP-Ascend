"""Load an explicitly built native library; imports never start a compiler."""
from pathlib import Path
import os
import torch

_LOADED = None
_ENABLED = False
_MODE = 'native'
_HARDWARE = {}


def hardware(device=None):
    """Cache immutable UB/core capacities for each NPU, never tensor values."""
    if device is None:
        index = torch.npu.current_device()
    elif isinstance(device, int) and not isinstance(device, bool):
        if device < 0:
            raise ValueError("The NPU device index must be nonnegative")
        index = device
    else:
        device = torch.device(device)
        if device.type != "npu":
            raise ValueError("Ascend hardware queries require an NPU device")
        index = torch.npu.current_device() if device.index is None else device.index
    if index not in _HARDWARE:
        with torch.npu.device(index):
            _HARDWARE[index]=tuple(load_library().hardware())
    return _HARDWARE[index]


def load_library(path=None):
    global _LOADED
    path = path or _LOADED or os.environ.get("ICTC_ASCENDC_LIBRARY")
    if not path:
        raise RuntimeError("Build Ascend C kernels and set ICTC_ASCENDC_LIBRARY to the resulting .so")
    path = str(Path(path).expanduser().resolve(strict=True))
    if _LOADED is not None and _LOADED != path:
        raise RuntimeError("A different ICTC Ascend C library is already loaded in this process")
    if _LOADED is None:
        torch.ops.load_library(path)
        _LOADED = path
    return torch.ops.ictc_ascendc


def enable(path=None, *, mode='performance'):
    """Select the native kernels for subsequent calls in this process."""
    global _ENABLED,_MODE
    if mode not in ('performance','capacity','native'):
        raise ValueError('Ascend C mode must be performance, capacity or native')
    ops = load_library(path)
    from ictc_o3.backends.ascend.runtime import strict_fp32
    strict_fp32()
    _MODE = mode
    _ENABLED = True
    return ops


def is_enabled():
    return _ENABLED


def mode():
    return _MODE


def native_graph_enabled():
    return _ENABLED and _MODE != 'performance'


def native_matrix_enabled():
    return _MODE == 'native'


def set_enabled(enabled):
    """Switch call dispatch; saved fused graph contexts retain their native adjoints."""
    global _ENABLED
    if enabled:
        load_library()
    previous = _ENABLED
    _ENABLED = bool(enabled)
    return previous
