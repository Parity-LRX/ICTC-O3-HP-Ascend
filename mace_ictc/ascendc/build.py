"""Compatibility alias; implementation is owned by ictc_o3.ascendc.build."""
from importlib import import_module as _import_module
import sys as _sys

if __name__ == "__main__":
    _import_module("ictc_o3.ascendc.build").main()
else:
    _sys.modules[__name__] = _import_module("ictc_o3.ascendc.build")
