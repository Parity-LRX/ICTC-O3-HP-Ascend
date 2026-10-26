"""Explicit Ascend C activation for standalone ICTC-O3 operators.

These aliases share the model backend's process-wide library and dispatch state.
Importing this module does not load a native library or start compilation.
"""
from mace_ictc.ascendc.runtime import enable, hardware, load_library

__all__ = ["enable", "load_library", "hardware"]
