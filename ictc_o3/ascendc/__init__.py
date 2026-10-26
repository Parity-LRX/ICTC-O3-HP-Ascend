"""Explicit Ascend C activation; importing never builds or loads the library."""
from .runtime import enable, hardware, load_library

__all__ = ["enable", "load_library", "hardware"]
