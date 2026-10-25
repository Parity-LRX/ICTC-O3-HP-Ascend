"""Eager Ascend runtime selection without parameter or dtype conversion."""
from __future__ import annotations

from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
import os

import torch


_graph_queue = ContextVar("ictc_ascend_graph_queue", default=None)


@lru_cache(maxsize=None)
def _device_memory(device):
    return torch.npu.get_device_properties(device).total_memory


@contextmanager
def graph_queue_scope():
    """Bound pending large calls without disabling the operator task queue.

    A message may request one stream fence at successful API completion. Small
    graphs retain asynchronous dispatch. No tensor data is cached.
    """
    policy = os.environ.get("ICTC_ASCEND_BOUND_GRAPH_QUEUE", "1")
    if policy not in ("0", "1"):
        raise ValueError("The graph queue policy must be zero or one")
    streams = {} if policy == "1" else None
    token = _graph_queue.set(streams)
    try:
        yield
        if streams:
            for stream in streams.values():
                stream.synchronize()
    finally:
        _graph_queue.reset(token)


def bound_graph_queue(device, working_bytes):
    streams = _graph_queue.get()
    if streams is not None and working_bytes >= _device_memory(device)//3:
        # Use the existing per-edge scratch estimate, not a channel/N threshold.
        # Finishing this stream before the next large API call avoids concurrent
        # graph allocations while preserving the default operator task queue.
        streams.setdefault(device, torch.npu.current_stream(device))


def drain_graph_queue(device):
    """Apply backpressure only when live allocations approach device capacity."""
    from mace_ictc.ascendc.runtime import is_enabled, mode
    if not is_enabled() or mode() != "performance":
        return False
    policy = os.environ.get("ICTC_ASCEND_BOUND_GRAPH_QUEUE", "1")
    if policy not in ("0", "1"):
        raise ValueError("The graph queue policy must be zero or one")
    if policy == "0":
        return False
    index = torch.npu.current_device() if device.index is None else device.index
    # Read one native allocator counter, without building a Python stats dict.
    allocated = torch.ops.ictc_ascendc.allocated_bytes(index)
    if allocated*4 < _device_memory(device)*3:
        return False
    # Queued native launches retain their operand tensors. Finishing this stream
    # releases eligible temporaries before another large native allocation.
    torch.npu.current_stream(device).synchronize()
    return True


@lru_cache(maxsize=None)
def _vector_cores(device_index):
    import triton
    properties = triton.runtime.driver.active.utils.get_device_properties(device_index)
    return max(1, min(65535, int(properties["num_vectorcore"])))


def _vector_grid(tasks, device):
    """Use physical vector cores; kernels loop over all remaining logical tiles."""
    index = torch.npu.current_device() if device.index is None else device.index
    return min(tasks, _vector_cores(index))


def require_npu_fp32(*values):
    if not values or any(v.device.type != "npu" or v.dtype != torch.float32 for v in values):
        raise ValueError("Ascend HP requires NPU FP32 tensors; precision is never changed automatically")
    if any(v.device != values[0].device for v in values):
        raise ValueError("All operands must be on the same NPU")
    if torch.is_autocast_enabled("npu"):
        raise ValueError("Ascend HP requires autocast to be disabled")
    if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
        raise ValueError("The current Ascend HP backend requires ordinary eager autograd")


def strict_fp32():
    import torch_npu  # Register the PrivateUse1 device before querying options.
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")


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
    # Clear inherited CUDA plans before installing device-specific schedules.
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
            "tensor_product_adjoint": ("automatic" if is_enabled() and mode()=="performance" else
                                       "analytic" if native_graph_enabled() or not training else "autograd") if selected else None,
            "counts": {"interactions": len(getattr(core, "interactions", ()))}}
