# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 block attention over the valid key prefix, with no mask.

The text trim puts each row's pads first (``[pad | valid text | image]``; text rows all sit at the rotary origin), so
blocks run ``attention(q, k[:, start:], v[:, start:])`` mask-free: same math, and flash / the INT8 route can take it.
The start lives in a CPU tensor an opaque custom op reads (one compiled graph for every prompt). A non-zero start is
never CUDA-graph recorded (the graph key hook raises = run eager). Unplanned calls (CFG-stacked batches, deepcopy
replicas whose root hooks never run) read the plan off their own mask; grad calls run the stock processor.
``UNSLOTH_DIFFUSION_KREA2_PREFIX_KV=0`` turns it off.
"""

from __future__ import annotations

import copy
import os
import weakref
from typing import Any, Optional

PREFIX_ENV = "UNSLOTH_DIFFUSION_KREA2_PREFIX_KV"
# auto: only auto's cuDNN pick moves to native flash, on FLASH_FASTER_CAPABILITIES cards; explicit backends stay.
BACKEND_ENV = "UNSLOTH_DIFFUSION_KREA2_PREFIX_KV_BACKEND"
CLASS_NAME = "Krea2Transformer2DModel"
STARTS_ATTR = "_unsloth_krea2_kv_starts"
_INSTALLED_ATTR = "_unsloth_krea2_prefix_kv"
_OP_NAME = "unsloth_studio::krea2_prefix_attention"
MAX_ROWS = 64
_OFF = ("0", "off", "false", "no")
STATS = {"prefix": 0, "masked": 0, "from_mask": 0}
_MASK_MEMO: dict = {}
_MASK_MEMO_MAX = 8
_SLOTS: dict = {}
_FLASH_OK: dict = {}
SLOT_ATTR = "_unsloth_krea2_kv_slot"
_STATE: dict = {"op": None, "proc_cls": None}


def enabled() -> bool:
    return (os.environ.get(PREFIX_ENV) or "").strip().lower() not in _OFF


def new_starts() -> Any:
    """[rows, start_0, ..., start_{MAX_ROWS-1}]; rows 0 = no plan (masked call)."""
    import torch
    return torch.zeros(MAX_ROWS + 1, dtype = torch.int64)


def set_plan(transformer: Any, starts: Optional[list]) -> None:
    held = getattr(transformer, "__dict__", {}).get(STARTS_ATTR)
    if held is None:
        return
    if not starts or len(starts) > MAX_ROWS:
        held[0] = 0
        return
    held[0] = len(starts)
    for i, s in enumerate(starts):
        held[1 + i] = int(s)


def _dead() -> None:
    return None


class _Slot:
    """Per-transformer state. A deep copy (CFG replica) is never planned: its root hooks do not run."""

    __slots__ = ("starts", "proc", "owner", "planned", "__weakref__")

    def __init__(self, starts: Any, proc: Any, owner: Any, planned: bool):
        self.starts = starts
        self.proc = weakref.ref(proc) if proc is not None else _dead
        self.owner = weakref.ref(owner) if owner is not None else _dead
        self.planned = planned

    def __deepcopy__(self, memo: dict) -> "_Slot":
        new = _Slot.__new__(_Slot)
        memo[id(self)] = new
        new.starts = copy.deepcopy(self.starts, memo)
        new.starts[0] = 0
        proc = self.proc()
        new.proc = weakref.ref(copy.deepcopy(proc, memo)) if proc is not None else _dead
        owner = self.owner()
        copied = memo.get(id(owner)) if owner is not None else None
        new.owner = weakref.ref(copied) if copied is not None else _dead
        new.planned = False
        _register(new)
        return new


def _register(slot: _Slot) -> None:
    _SLOTS[slot.starts.data_ptr()] = weakref.ref(slot)
    # list(): atomic snapshot, a replica's worker thread may register concurrently
    for key in [k for k, ref in list(_SLOTS.items()) if ref() is None]:
        _SLOTS.pop(key, None)


def _slot_of(starts: Any) -> Optional[_Slot]:
    ref = _SLOTS.get(starts.data_ptr())
    slot = ref() if ref is not None else None
    return slot if slot is not None and slot.starts.data_ptr() == starts.data_ptr() else None


class PlanNotRecordable(RuntimeError):
    """Graph-key hook raise for a non-zero key start: the CUDA graph layers then run the call eager."""


def plan_key(transformer: Any) -> tuple:
    held = getattr(transformer, "__dict__", {}).get(STARTS_ATTR)
    if held is None:
        return ()
    rows = int(held[0])
    return tuple(int(x) for x in held[: 1 + rows].tolist())


def _flash_runs(device: Any, dtype: Any) -> bool:
    """Whether flash SDPA runs here (PyTorch Windows 2.10-2.14 ship without it); never probed under graph capture."""
    import torch

    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (f"cuda:{index}", str(dtype))
    hit = _FLASH_OK.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        return False
    from types import SimpleNamespace

    from .diffusion_attention import SDPA_FLASH, available_sdpa_kernels

    try:
        kernels = available_sdpa_kernels(SimpleNamespace(device = key[0], dtype = dtype))
    except Exception:  # noqa: BLE001
        kernels = ()
    if not kernels:
        return False  # not cached: retried next call
    _FLASH_OK[key] = hit = SDPA_FLASH in kernels
    return hit


def _prefix_backend(
    backend: Any,
    device: Any,
    dtype: Any = None,
    auto_pick: bool = False,
) -> Any:
    from diffusers.models.attention_dispatch import AttentionBackendName

    if type(backend).__name__ == "Int8Route":
        return backend
    raw = (os.environ.get(BACKEND_ENV) or "auto").strip().lower()
    if raw == "flash":
        return AttentionBackendName._NATIVE_FLASH
    if raw == "cudnn":
        return AttentionBackendName._NATIVE_CUDNN
    if raw == "incumbent":
        return backend
    name = str(getattr(backend, "value", backend) or "native")
    if name != "_native_cudnn" or not auto_pick:
        return backend
    try:
        import torch

        from .diffusion_attention import FLASH_FASTER_CAPABILITIES, NATIVE_FLASH_ATTN_ENV
        from .rocm_bf16 import is_rocm_torch

        if (os.environ.get(NATIVE_FLASH_ATTN_ENV) or "").strip().lower() in _OFF:
            return backend
        if (
            getattr(device, "type", None) == "cuda"
            and not is_rocm_torch(torch)
            and dtype in (torch.bfloat16, torch.float16)
            and tuple(torch.cuda.get_device_capability(device)) in FLASH_FASTER_CAPABILITIES
            and _flash_runs(device, dtype)
        ):
            return AttentionBackendName._NATIVE_FLASH
    except Exception:  # noqa: BLE001
        pass
    return backend


def _dispatch() -> Any:
    # read at call time so the INT8 route's module attribute swap is seen
    from diffusers.models.transformers import transformer_krea2
    return transformer_krea2.dispatch_attention_fn


def mask_plan(mask: Any, batch: int) -> Optional[list]:
    """Each row's key start when its masked keys are exactly a leading run, else None. One host read per mask."""
    import torch

    held = _MASK_MEMO.get(id(mask))
    if held is not None and held[0]() is mask:
        return held[1]
    if mask.dtype != torch.bool or mask.numel() % max(batch, 1):
        return None
    if mask.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    m = mask.reshape(batch, -1)
    first, pads = torch.stack((m.int().argmax(dim = 1), (~m).sum(dim = 1))).tolist()
    plan = first if first == pads and all(p < m.shape[1] for p in pads) else None
    try:
        items = list(_MASK_MEMO.items())
        if len(items) >= _MASK_MEMO_MAX:
            for key in [k for k, v in items if v[0]() is None] or [items[0][0]]:
                _MASK_MEMO.pop(key, None)
        _MASK_MEMO[id(mask)] = (weakref.ref(mask), plan)
    except TypeError:
        pass
    return plan


def _backend_of(starts: Any) -> Any:
    """Keyed by the starts storage, so no per-block constant reaches the compiled graph."""
    slot = _slot_of(starts)
    proc = slot.proc() if slot is not None else None
    return getattr(proc, "_attention_backend", None)


def _auto_pick(slot: Optional[_Slot]) -> bool:
    owner = slot.owner() if slot is not None else None
    if owner is None:
        return False
    from .diffusion_attention import ATTENTION_AUTO_PICK_ATTR

    return bool(getattr(owner, "__dict__", {}).get(ATTENTION_AUTO_PICK_ATTR))


def attention(q: Any, k: Any, v: Any, mask: Any, starts: Any) -> Any:
    """[B, S, H, D] x 3 -> [B, S, H, D]; the custom op's eager body."""
    import torch

    slot = _slot_of(starts)
    proc = slot.proc() if slot is not None else None
    backend = getattr(proc, "_attention_backend", None)
    dispatch = _dispatch()
    B = int(q.shape[0])
    planned = slot is not None and slot.planned
    rows = int(starts[0]) if planned else 0
    plan = starts[1 : 1 + rows].tolist() if planned and rows == B else None
    if plan is not None and mask is None and any(plan):
        plan = None
    if plan is None and mask is not None and (rows or not planned):
        plan = mask_plan(mask, B)
        if plan is not None:
            STATS["from_mask"] += 1
    if plan is None or any(s < 0 or s >= k.shape[1] for s in plan):
        STATS["masked"] += 1
        return dispatch(q, k, v, attn_mask = mask, backend = backend).contiguous()
    STATS["prefix"] += 1
    fast = _prefix_backend(backend, q.device, q.dtype, _auto_pick(slot))
    if len(set(plan)) == 1:
        s = plan[0]
        return dispatch(q, k[:, s:], v[:, s:], backend = fast).contiguous()
    out = torch.empty_like(q)
    for s in sorted(set(plan)):
        idx = [b for b, x in enumerate(plan) if x == s]
        if len(idx) == idx[-1] - idx[0] + 1:
            a, z = idx[0], idx[-1] + 1
            out[a:z] = dispatch(q[a:z], k[a:z, s:], v[a:z, s:], backend = fast)
        else:
            sel = torch.tensor(idx, device = q.device)
            out.index_copy_(
                0,
                sel,
                dispatch(
                    q.index_select(0, sel),
                    k.index_select(0, sel)[:, s:],
                    v.index_select(0, sel)[:, s:],
                    backend = fast,
                ).contiguous(),
            )
    return out


def _op() -> Any:
    if _STATE["op"] is not None:
        return _STATE["op"]
    import torch

    lib_ns, name = _OP_NAME.split("::")
    existing = getattr(getattr(torch.ops, lib_ns, None), name, None)
    if existing is None:
        # explicit schema: annotations are strings here (__future__)
        @torch.library.custom_op(
            _OP_NAME,
            mutates_args = (),
            schema = "(Tensor q, Tensor k, Tensor v, Tensor? mask, Tensor starts) -> Tensor",
        )
        def _krea2_prefix_attention(q, k, v, mask, starts):
            return attention(q, k, v, mask, starts)

        @_krea2_prefix_attention.register_fake
        def _(q, k, v, mask, starts):
            return q.new_empty(q.shape)

    _STATE["op"] = getattr(torch.ops.unsloth_studio, name)
    return _STATE["op"]


def processor_cls() -> Any:
    if _STATE["proc_cls"] is not None:
        return _STATE["proc_cls"]
    import torch
    from diffusers.models.transformers.transformer_krea2 import (
        Krea2AttnProcessor,
        apply_rotary_emb,
    )

    op = _op()

    class Krea2PrefixKVAttnProcessor(Krea2AttnProcessor):
        # read by the INT8 dispatch route
        _calls_dispatch_attention_fn = True

        def __call__(
            self,
            attn,
            hidden_states,
            attention_mask = None,
            image_rotary_emb = None,
        ):
            starts = getattr(self, "_unsloth_starts", None)
            if (
                starts is None
                or getattr(self, "_parallel_config", None) is not None
                or (torch.is_grad_enabled() and _wants_grad(attn, hidden_states))
            ):
                # the op has no autograd formula
                return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
            query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
            key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
            value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
            gate = attn.to_gate(hidden_states)
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim = 1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim = 1)
            groups = attn.num_heads // attn.num_kv_heads
            if groups > 1:
                key = key.repeat_interleave(groups, dim = 2)
                value = value.repeat_interleave(groups, dim = 2)
            hidden_states = op(query, key, value, attention_mask, starts)
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states * torch.sigmoid(gate)
            return attn.to_out[0](hidden_states)

    _STATE["proc_cls"] = Krea2PrefixKVAttnProcessor
    return Krea2PrefixKVAttnProcessor


def _wants_grad(attn: Any, hidden_states: Any) -> bool:
    return bool(hidden_states.requires_grad) or any(p.requires_grad for p in attn.parameters())


def installed(transformer: Any) -> bool:
    return bool(getattr(transformer, "__dict__", {}).get(_INSTALLED_ATTR))


def install(transformer: Any, logger: Any = None) -> int:
    """Swap the main-block processors (before the attention backend / INT8 route); returns how many."""
    if type(transformer).__name__ != CLASS_NAME or not enabled():
        return 0
    if installed(transformer):
        return int(transformer.__dict__[_INSTALLED_ATTR])
    try:
        cls = processor_cls()
        starts = new_starts()
        slot = _Slot(starts, None, transformer, True)
        n = 0
        for blk in getattr(transformer, "transformer_blocks", []):
            attn = getattr(blk, "attn", None)
            proc = getattr(attn, "processor", None)
            if proc is None or type(proc).__name__ != "Krea2AttnProcessor":
                continue
            new = cls()
            new._attention_backend = getattr(proc, "_attention_backend", None)
            new._parallel_config = getattr(proc, "_parallel_config", None)
            new._unsloth_starts = starts
            new._unsloth_slot = slot
            if not n:
                slot.proc = weakref.ref(new)
            attn.processor = new
            # the INT8 route needs ``heads``
            if not hasattr(attn, "heads"):
                attn.heads = attn.num_heads
            n += 1
        if not n:
            return 0
        transformer.__dict__[STARTS_ATTR] = starts
        transformer.__dict__[SLOT_ATTR] = slot
        _register(slot)
        prev = getattr(transformer, "_unsloth_graph_key_extra", None)
        ref = weakref.ref(transformer)

        def _key() -> tuple:
            t = ref()
            mine = plan_key(t) if t is not None else ()
            if any(mine[1:]):
                # one recording per start would fill the graph cap within a few prompts
                raise PlanNotRecordable("Krea 2 prefix K/V plan " + repr(mine))
            return (prev(), mine) if callable(prev) else mine

        transformer._unsloth_graph_key_extra = _key
        transformer.__dict__[_INSTALLED_ATTR] = n
        if logger is not None:
            logger.info(
                "diffusion.krea2: block attention reads the valid key prefix with no mask (%d blocks)",
                n,
            )
        return n
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warning("diffusion.krea2: prefix K/V attention unavailable: %s", exc)
        return 0
