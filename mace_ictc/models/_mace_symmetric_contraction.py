"""Compatibility alias; implementation is owned by ictc_o3.ops._mace_symmetric_contraction."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("ictc_o3.ops._mace_symmetric_contraction")
