"""
ICTC-O3-HP-Ascend - ICTC-O3 operators and MACE-compatible models for Ascend.

Standalone high-performance library. Stable HP operator and
whole-network entrypoints are exposed in ``mace_ictc.hp``. The package name
is retained for checkpoint compatibility; no O2U1 package is required.
"""

__version__ = "0.1.0"
# Contract checked by the optional CHORUS execution adapter, without renumbering the paper release.
__hp_api_version__ = 1

import torch
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([slice])

# Import the baseline ICTC-MACE model + small shared helpers (guarded so version metadata
# is still importable when heavy deps are absent).
try:
    from mace_ictc.models import (
        PureCartesianICTDFix,
        MainNet,
        MainNet2,
        RMSELoss,
        RobustScalarWeightedSum,
    )

    __all__ = [
        "PureCartesianICTDFix",
        "MainNet",
        "MainNet2",
        "RMSELoss",
        "RobustScalarWeightedSum",
    ]
except ImportError:
    # If dependencies are not installed, just define version
    __all__ = []
