"""Compatibility alias; implementation is owned by ictc_o3.ops.sparse_weighted_product."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("ictc_o3.ops.sparse_weighted_product")
