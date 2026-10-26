"""Compatibility entry points for the standalone Ascend C backend."""
from ictc_o3.ascendc import enable, hardware, load_library

__all__ = ["enable", "load_library", "hardware"]
