"""Explicitly built Ascend C kernels with hardware-derived UB tiling."""

from .runtime import load_library, enable

__all__ = ["load_library", "enable"]
