"""Build the shared Ascend C library from the standalone operator namespace.

Run ``python -m ictc_o3.ascendc.build --output /path/to/build --jobs 4``
inside an Ascend compute allocation with the CANN environment sourced.
"""
from mace_ictc.ascendc.build import build, main

__all__ = ["build"]

if __name__ == "__main__":
    main()
