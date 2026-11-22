# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in native FSDP2 gradient reduce-scatter that moves 16-bit gradients as 16-bit and adds them in float32.

With 16-bit compute (bf16, or fp16 on T4) and the default float32 `reduce_dtype`, FSDP2 upcasts each fresh
gradient into a float32 buffer and reduce-scatters 4 bytes per element. When every gradient packed into that
buffer is a 16-bit tensor, casting the buffer back is exact: exchanging the 16-bit copy with one all-to-all and
adding the received chunks in float32 gives the float32 reduce-scatter's sum (bitwise at 2 ranks, float32
summation order beyond) from half the bytes. AVG scales each contribution by 1 / world before adding, as NCCL
does for float32, so a sum of finite values cannot overflow where the stock average does not.

Every choice that changes which collective runs is agreed across the group's ranks:
- at install: the switch and the piece size (one small all_reduce, opt-in path only);
- per reduce: whether this rank's packed gradients are all 16-bit (a gradient FSDP2 accumulated in float32 over
  micro-batches without a reduce, or a genuine float32 gradient, is not), voted over a CPU (gloo) group, so the
  vote never waits on the GPU. One rank that cannot carry its gradients sends every rank to the stock collective.

UNSLOTH_FSDP2_REDUCE_TRANSPORT: "on" engages where it is exact; anything else (default) keeps FSDP2's own.
UNSLOTH_FSDP2_REDUCE_TRANSPORT_MIB (default 32): 16-bit send buffer per all-to-all; a larger unit goes in pieces,
so the scratch stays at most twice this however large the unit (the embedding / LM head root unit).
"""

import math
import os

import torch
import torch.distributed as dist

__all__ = ["install_half_transport_reduce", "half_transport_enabled"]

_HALF = (torch.float16, torch.bfloat16)
_CODES = {None: 0, torch.float16: 1, torch.bfloat16: 2}
_DEFAULT_PIECE_BYTES = 32 * 2**20

try:
    from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
        DefaultReduceScatter as _DefaultReduceScatter,
    )
except ImportError:  # torch without custom reduce-scatter comms: nothing is installed
    _DefaultReduceScatter = None


def half_transport_enabled():
    value = os.environ.get("UNSLOTH_FSDP2_REDUCE_TRANSPORT", "off").strip().lower()
    return value in ("1", "on", "true", "yes")


def _piece_bytes():
    # Bytes of 16-bit send buffer per all-to-all (the receive buffer is the same size).
    try:
        mib = float(os.environ.get("UNSLOTH_FSDP2_REDUCE_TRANSPORT_MIB", "32"))
        size = int(mib * 2**20) if math.isfinite(mib) else 0
    except (ValueError, OverflowError):
        return _DEFAULT_PIECE_BYTES
    return size if 0 < size < 2**62 else _DEFAULT_PIECE_BYTES


def _op_name(op):
    op_type = getattr(op, "op", op)
    name = getattr(op_type, "name", None)
    return name if name is not None else str(op_type).rsplit(".", 1)[-1]


def _cpu_group(pg):
    """A group over `pg`'s ranks that reduces CPU tensors: `pg` itself when its backend has gloo, else a gloo
    group over the same ranks (created collectively by those ranks only)."""
    try:
        config = str(dist.get_backend_config(pg))
    except Exception:
        config = str(dist.get_backend(pg))
    if "gloo" in config.lower():
        return pg
    return dist.new_group(
        ranks = dist.get_process_group_ranks(pg),
        backend = "gloo",
        use_local_synchronization = True,
    )


def _min_over(values, group):
    t = torch.tensor(values, dtype = torch.int64)
    dist.all_reduce(t, op = dist.ReduceOp.MIN, group = group)
    return t.tolist()


_Base = _DefaultReduceScatter if _DefaultReduceScatter is not None else object


class _HalfTransportReduceScatter(_Base):
    """Wraps one param group's default reduce-scatter comm. A subclass of DefaultReduceScatter, so FSDP2's public
    setters (set_allocate_memory_from_process_group, set_symm_mem) accept it and replace it; the group's hooks
    then defer to whatever comm the group holds."""

    def __init__(self, inner, group, piece_bytes, vote_group):
        self.inner = inner
        self.group = group
        self.piece_bytes = piece_bytes  # agreed across the group's ranks at install
        self.vote_group = vote_group
        self.dtype = None  # resolved on first use: FSDP2 sets the dtypes at lazy init
        self.resolved = False
        self.inert = False  # every rank agreed the group has no 16-bit gradients: no further votes
        self.fresh = (
            False  # this rank's packed gradients are all `dtype` (set by the group's hooks)
        )
        self.calls = 0
        self.fallbacks = 0
        self.votes = 0

    def resolve(self):
        if not self.resolved:
            self.dtype, self.resolved = _group_grad_dtype(self.group), True
        return self.dtype

    def allocate(self, size, *, dtype, device):
        return self.inner.allocate(size, dtype = dtype, device = device)

    def _stock(self, output_tensor, input_tensor, group, op, async_op):
        self.fallbacks += 1
        return self.inner(
            output_tensor = output_tensor,
            input_tensor = input_tensor,
            group = group,
            op = op,
            async_op = async_op,
        )

    def __call__(
        self,
        output_tensor,
        input_tensor,
        group,
        op,
        async_op = False,
    ):
        fresh, self.fresh = self.fresh, False
        dtype = self.resolve()
        if self.inert or async_op:
            return self._stock(output_tensor, input_tensor, group, op, async_op)
        name = _op_name(op)
        world = dist.get_world_size(group)
        width = output_tensor.numel()
        local_ok = (
            fresh
            and dtype in _HALF
            and name in ("SUM", "AVG")
            and world > 1
            and not getattr(self.group, "reduce_scatter_unused_params", False)
            and input_tensor.dtype == torch.float32
            and output_tensor.dtype == torch.float32
            and input_tensor.is_contiguous()
            and output_tensor.is_contiguous()
            and input_tensor.device == output_tensor.device
            and input_tensor.numel() == world * width
        )
        code = _CODES.get(dtype, 0)
        # One CPU vote per reduce: [all ranks can carry, min dtype code, -max dtype code].
        ok, lo, neg_hi = _min_over([int(local_ok), code, -code], self.vote_group)
        self.votes += 1
        if lo == -neg_hi == 0:
            self.inert = True  # agreed: no 16-bit gradients in this group, ever
        if not ok or lo != -neg_hi:
            return self._stock(output_tensor, input_tensor, group, op, async_op)
        rows = input_tensor.view(world, width)  # row r: this rank's contribution to rank r's shard
        out_flat = output_tensor.view(-1)
        # Pieces bound the 16-bit send / receive scratch (a whole-unit copy grew peak memory by the
        # embedding's size); piece k of every row goes out in one all-to-all. Two flat buffers, reused.
        piece = max(1, min(width, self.piece_bytes // (2 * world)))
        send_buffer = torch.empty(world * piece, dtype = dtype, device = input_tensor.device)
        recv_buffer = torch.empty_like(send_buffer)
        scale = 1.0 / world if name == "AVG" else None
        for start in range(0, width, piece):
            count = min(width, start + piece) - start
            send = send_buffer[: world * count].view(world, count)
            recv = recv_buffer[: world * count].view(world, count)
            send.copy_(rows[:, start : start + count])  # exact: every packed gradient is `dtype`
            dist.all_to_all_single(recv, send, group = group)
            out = out_flat[start : start + count]
            out.copy_(recv[0])
            if scale is None:
                for i in range(1, world):
                    out.add_(recv[i])
            else:
                # Scale each contribution before adding (NCCL's float32 AVG): no overflow of finite sums.
                out.mul_(scale)
                for i in range(1, world):
                    out.add_(recv[i], alpha = scale)
        self.calls += 1
        return None


def _group_grad_dtype(group):
    """The 16-bit dtype every trainable parameter's gradient has, or None. Frozen parameters produce no gradient
    and are ignored (FSDP2 picks its dtypes from trainable parameters too). A parameter whose gradient may be
    reduced outside data parallel (TP, a DTensor / spmd parameter) or has an explicit gradient dtype keeps the
    stock collective for the whole group."""
    params = [
        p
        for p in (getattr(group, "fsdp_params", None) or [])
        if getattr(getattr(p, "sharded_param", None), "requires_grad", False)
    ]
    if not params:
        return None
    grad_dtypes, reduce_dtypes = set(), set()
    for p in params:
        if (
            getattr(p, "is_dtensor", False)
            or getattr(p, "is_spmd_types", False)
            or getattr(p, "may_reduce_grad_outside_dp", False)
            or getattr(p, "_has_sharded_grad_dtype_override", False)
        ):
            return None
        grad_dtypes.add(getattr(p, "param_dtype", None) or getattr(p, "orig_dtype", None))
        reduce_dtypes.add(getattr(p, "reduce_dtype", None))
    if len(grad_dtypes) != 1 or reduce_dtypes != {torch.float32}:
        return None
    dtype = grad_dtypes.pop()
    return dtype if dtype in _HALF else None


def _all_have_dtype(grads, dtype):
    return bool(grads) and dtype in _HALF and all(getattr(g, "dtype", None) == dtype for g in grads)


def _legacy_fresh(group, dtype):
    # torch <= 2.14: a gradient FSDP2 accumulated in float32 without a reduce sits in unsharded_accumulated_grad;
    # the rest are the unsharded parameters' own .grad.
    grads = []
    for p in group.fsdp_params:
        if getattr(p, "unsharded_accumulated_grad", None) is not None:
            return False
        param = getattr(p, "_unsharded_param", None)
        grad = getattr(param, "grad", None) if param is not None else None
        if grad is not None:
            grads.append(grad)
    return _all_have_dtype(grads, dtype)


def _wrap_group(group, comm):
    original = group.post_backward
    take = getattr(group, "_take_unsharded_grads_to_reduce", None)

    if take is not None:
        # torch 2.15+: classify exactly the gradients FSDP2 packs into the reduce-scatter input.
        def take_grads(*args, **kwargs):
            params, grads = take(*args, **kwargs)
            if group._reduce_scatter_comm is comm:
                comm.fresh = _all_have_dtype(grads, comm.resolve())
            return params, grads

        group._take_unsharded_grads_to_reduce = take_grads

    def post_backward(*args, **kwargs):
        if group._reduce_scatter_comm is not comm:  # a public setter replaced the comm
            return original(*args, **kwargs)
        comm.fresh = False
        try:
            if take is None and group.reduce_grads:
                comm.fresh = _legacy_fresh(group, comm.resolve())
            return original(*args, **kwargs)
        finally:
            comm.fresh = False

    group.post_backward = post_backward


def _param_groups(modules):
    seen = set()
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
            mesh_info = getattr(group, "mesh_info", None)
            if type(mesh_info).__name__ != "FSDPMeshInfo":
                continue  # HSDP / replicate meshes keep the stock path
            if getattr(mesh_info, "shard_mesh_size", 1) < 2:
                continue  # one rank: FSDP2 skips the collective
            yield group


def install_half_transport_reduce(modules, enabled = None):
    """Install on the FSDP2 param groups of `modules` (plain FSDP meshes, 2+ ranks). `enabled`: the switch
    already agreed across ranks (shard_model folds it into its layout agreement); None reads the environment
    and agrees it here. Every rank of a group must call this. Idempotent: a group already carrying the comm
    returns it. Returns the installed comms (empty when off, or on torch without custom reduce-scatter comms)."""
    if enabled is False or _DefaultReduceScatter is None:
        return []
    if not dist.is_available() or not dist.is_initialized():
        return []
    local = (1 if (enabled or half_transport_enabled()) else 0, _piece_bytes())
    installed, settings = [], {}
    for group in _param_groups(modules):
        existing = getattr(group, "_unsloth_half_transport", None)
        if existing is not None:
            if group._reduce_scatter_comm is existing:
                installed.append(existing)
            continue
        pg = group._reduce_scatter_process_group
        key = id(pg)
        if key not in settings:
            vote_group = _cpu_group(pg)
            on, piece = _min_over(list(local), vote_group)
            settings[key] = (on, piece, vote_group)
        on, piece, vote_group = settings[key]
        if not on or type(group._reduce_scatter_comm) is not _DefaultReduceScatter:
            continue  # off on some rank, or a comm someone customised: keep it
        comm = _HalfTransportReduceScatter(group._reduce_scatter_comm, group, piece, vote_group)
        group._reduce_scatter_comm = comm
        group._unsloth_half_transport = comm
        _wrap_group(group, comm)
        installed.append(comm)
    return installed
