"""Ascend AI Core backend for ICTC-O3 operators and MACE-ICTC models."""
from .runtime import check_model, configure_model, strict_fp32

__all__ = ["check_model", "configure_model", "strict_fp32"]
