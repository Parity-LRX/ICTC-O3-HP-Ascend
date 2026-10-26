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
    from ictc_o3.ascendc.runtime import is_enabled, mode
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
