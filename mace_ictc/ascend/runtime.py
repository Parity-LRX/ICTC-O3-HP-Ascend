"""MACE-specific configuration over the standalone Ascend operator backend."""
import torch
from ictc_o3.backends.ascend import runtime as _operators
from ictc_o3.backends.ascend.runtime import strict_fp32

def __getattr__(name):
    return getattr(_operators, name)


def check_model(model, *, training=False, capture=False):
    from mace_ictc.models.operator_plans import mace_backbone_core
    from mace_ictc.models.pure_cartesian_ictd_fix import (
        PureCartesianICTDFix, ICTDBridgeUSymmetricContractionSO3,
    )
    from .sparse_product import available
    core = mace_backbone_core(model)
    if not isinstance(core, PureCartesianICTDFix):
        return {"supported": False, "backend": "ascend", "reasons": ["Expected the MACE-ICTC backbone"]}
    reasons = []
    if not available():
        reasons.append("Triton-Ascend is unavailable")
    if any(p.device.type != "npu" or p.dtype != torch.float32 for p in core.parameters()):
        reasons.append("All model parameters must already be NPU FP32")
    if core.edge_compute_dtype != torch.float32:
        reasons.append("Geometry internal compute dtype must be FP32")
    if getattr(core, "angular_basis", "ictd") != "ictd":
        reasons.append("Expected the ICTC Cartesian angular basis")
    if capture:
        reasons.append("Graph capture is not qualified; use eager execution")
    products = [m for m in core.modules() if isinstance(m, ICTDBridgeUSymmetricContractionSO3)]
    if not products or len(products) != len(core.interactions):
        reasons.append("Every interaction must use a bridge-U symmetric contraction")
    if any(c.correlation not in (1, 2, 3) for m in products
           for c in m.symmetric_contractions.contractions):
        reasons.append("Native symmetric contraction currently supports correlation 1, 2 and 3")
    if any(getattr(m, "interaction_attn_heads", 0) or getattr(m, "tp_cueq", None) is not None
           for m in core.interactions):
        reasons.append("Interaction attention and cuEquivariance are outside current NPU coverage")
    if any(getattr(m.tp, "internal_compute_dtype", None) != torch.float32 for m in core.interactions):
        reasons.append("Tensor-product internal compute dtype must be FP32")
    return {"supported": not reasons, "backend": "ascend", "reasons": reasons}


def configure_model(model, *, training=False, policy="auto", capture=False,
                    edge_chunk_size=None):
    from mace_ictc.ascendc.runtime import is_enabled,mode,native_graph_enabled
    if policy not in ("auto", "generated", "reference"):
        raise ValueError("policy must be auto, generated or reference")
    if edge_chunk_size is not None and (not isinstance(edge_chunk_size, int)
            or isinstance(edge_chunk_size, bool) or edge_chunk_size < 1):
        raise ValueError("edge_chunk_size must be a positive integer or None for automatic sizing")
    from mace_ictc.models.operator_plans import mace_backbone_core, configure_mace_runtime_plan
    core = mace_backbone_core(model)
    coverage = check_model(model, training=training, capture=capture)
    if policy == "generated" and not coverage["supported"]:
        raise ValueError("Ascend HP unavailable: " + "; ".join(coverage["reasons"]))
    # Restore reference buffers before installing device-specific schedules.
    configure_mace_runtime_plan(model, policy="reference", training=training)
    selected = policy != "reference" and coverage["supported"]
    if not hasattr(core, "_ascend_original_edge_order"):
        core._ascend_original_edge_order = getattr(core, "preserve_edge_order", False)
    # index_add accepts arbitrary edge order; integer sorting would introduce
    # an unnecessary AiCPU operation and host synchronization on every call.
    core.preserve_edge_order = True if selected else core._ascend_original_edge_order
    if selected:
        if is_enabled():
            from mace_ictc.ascendc.linear import prepare_model
            prepare_model(core)
    for module in core.modules():
        module._ascend_hp = selected
    if selected:
        strict_fp32()
        core._fused_direction_harmonics = True
        for layer in core.interactions:
            layer._ascend_edge_chunk_size = edge_chunk_size
            # Standalone edge products retain their established schedules;
            # performance graph messages use channel-contiguous recursive GEMMs.
            layer.tp._ascend_tp_schedule = "gemm" if training else "structured"
    return {"requested": policy, "selected": "generated" if selected else "reference",
            "backend": "ascend", "reasons": coverage["reasons"] if policy != "reference" else [],
            "arithmetic": "strict FP32; HF32 and autocast disabled",
            "edge_chunk_size": ("auto" if edge_chunk_size is None else edge_chunk_size) if selected else None,
            "kernel_backend": ("ascendc" if is_enabled() else "triton_gemm") if selected else "reference",
            "ascendc_mode": mode() if selected and is_enabled() else None,
            "tensor_product_adjoint": ("chunk_recompute" if is_enabled() and mode()=="capacity" and not training else
                                       "automatic" if is_enabled() and mode()=="performance" else
                                       "analytic" if native_graph_enabled() or not training else "autograd") if selected else None,
            "counts": {"interactions": len(getattr(core, "interactions", ()))}}
