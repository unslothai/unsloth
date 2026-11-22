# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 gradient reduce-scatter that moves 16-bit gradients as 16-bit and adds them in float32.

With 16-bit compute (bf16, or fp16 on T4) and the default float32 `reduce_dtype`, FSDP2 upcasts each fresh
gradient into a float32 buffer and reduce-scatters 4 bytes per element. Every value in that buffer is a 16-bit
number, so casting it back is exact: exchanging the 16-bit copy with one all-to-all and adding the received
chunks in float32 gives the float32 reduce-scatter's sum (bitwise at 2 ranks, float32 summation order beyond)
from half the bytes. A buffer holding a gradient FSDP2 accumulated in float32 over micro-batches without a
reduce is not 16-bit exact, so that reduce (and any op other than SUM / AVG) runs the original collective.

UNSLOTH_FSDP2_REDUCE_TRANSPORT: "auto" (default) engages where it is exact, "0" / "off" keeps FSDP2's own.
"""

import os

import torch
import torch.distributed as dist

__all__ = ["install_half_transport_reduce", "half_transport_enabled"]

_HALF = (torch.float16, torch.bfloat16)


def half_transport_enabled():
    value = os.environ.get("UNSLOTH_FSDP2_REDUCE_TRANSPORT", "auto").strip().lower()
    return value not in ("0", "off", "false", "no", "none", "fp32")


def _op_name(op):
    op_type = getattr(op, "op", op)
    name = getattr(op_type, "name", None)
    return name if name is not None else str(op_type).rsplit(".", 1)[-1]


class _HalfTransportReduceScatter:
    """Wraps one param group's reduce-scatter comm; `fresh` is set by the group's post_backward wrapper."""

    def __init__(self, inner, group):
        self.inner = inner
        self.group = group
        self.dtype = None  # resolved on the first reduce: FSDP2 sets the dtypes at lazy init
        self.resolved = False
        self.fresh = False
        self.calls = 0
        self.fallbacks = 0

    def allocate(self, size, *, dtype, device):
        return self.inner.allocate(size, dtype = dtype, device = device)

    def __call__(
        self,
        output_tensor,
        input_tensor,
        group,
        op,
        async_op = False,
    ):
        name = _op_name(op)
        fresh, self.fresh = self.fresh, False
        if not self.resolved:
            self.dtype, self.resolved = _group_grad_dtype(self.group), True
        if (
            self.dtype is None
            or async_op
            or not fresh
            or name not in ("SUM", "AVG")
            or input_tensor.dtype != torch.float32
            or output_tensor.dtype != torch.float32
        ):
            self.fallbacks += 1
            return self.inner(
                output_tensor = output_tensor,
                input_tensor = input_tensor,
                group = group,
                op = op,
                async_op = async_op,
            )
        world = dist.get_world_size(group)
        send = input_tensor.to(self.dtype)  # exact: the buffer holds fresh 16-bit gradients
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group = group)
        del send
        recv = recv.view(world, -1)
        output_tensor.copy_(recv[0])
        for i in range(1, world):
            output_tensor.add_(recv[i])
        if name == "AVG":
            output_tensor.div_(world)
        self.calls += 1
        return None


def _group_grad_dtype(group):
    params = getattr(group, "fsdp_params", None) or []
    grad_dtypes, reduce_dtypes = set(), set()
    for p in params:
        grad_dtypes.add(getattr(p, "param_dtype", None) or getattr(p, "orig_dtype", None))
        reduce_dtypes.add(getattr(p, "reduce_dtype", None))
    if len(grad_dtypes) != 1 or reduce_dtypes != {torch.float32}:
        return None
    dtype = grad_dtypes.pop()
    return dtype if dtype in _HALF else None


def _wrap_post_backward(group, comm):
    original = group.post_backward

    def post_backward(*args, **kwargs):
        # A float32 accumulated gradient (a micro-batch that skipped its reduce) is not 16-bit exact.
        comm.fresh = all(
            getattr(p, "unsharded_accumulated_grad", None) is None for p in group.fsdp_params
        )
        return original(*args, **kwargs)

    group.post_backward = post_backward


def install_half_transport_reduce(modules):
    """Install on every FSDP2 param group of `modules` whose gradients are 16-bit and reduce_dtype float32.
    Returns the installed comms (empty when disabled or on torch without custom reduce-scatter comms); a
    group whose gradients turn out not to be 16-bit with a float32 reduce_dtype keeps the original collective."""
    if not half_transport_enabled() or not dist.is_available() or not dist.is_initialized():
        return []
    installed, seen = [], set()
    for module in modules:
        get_state = getattr(module, "_get_fsdp_state", None)
        if get_state is None:
            continue
        try:
            state = get_state()
        except Exception:
            continue
        groups = getattr(state, "_fsdp_param_groups", None)
        if groups is None:
            group = getattr(state, "_fsdp_param_group", None)
            groups = [group] if group is not None else []
        for group in groups:
            if id(group) in seen or not hasattr(group, "_reduce_scatter_comm"):
                continue
            seen.add(id(group))
            mesh_info = type(getattr(group, "mesh_info", None)).__name__
            if mesh_info != "FSDPMeshInfo":
                continue  # HSDP / replicate meshes keep the stock path
            comm = _HalfTransportReduceScatter(group._reduce_scatter_comm, group)
            group._reduce_scatter_comm = comm
            _wrap_post_backward(group, comm)
            installed.append(comm)
    return installed
