"""Standalone ICTC-O3 operators and native resources for Ascend NPUs.

Import and execution do not depend on mace_ictc. See docs/O3_API.md.
"""
import torch

# e3nn 0.4.4 stores slice objects in its immutable coefficient archive.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])

from .operators import *
from .operators import __all__
from .electrostatics import MeshElectrostatics, mesh_spread, mesh_gather, mesh_shape_for_cell
__all__ = [*__all__, "MeshElectrostatics", "mesh_spread", "mesh_gather", "mesh_shape_for_cell"]

__version__ = "0.1.0"
