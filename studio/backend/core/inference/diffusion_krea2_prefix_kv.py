# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 block attention over the valid key prefix, with no mask.

The text trim (``diffusion_krea2_text_trim``) keeps a 128-row bucket of text rows, so a partly filled bucket still
carries padded rows, and diffusers orders the joint sequence ``[text | image]``: the pads sit mid-sequence and every
block attends with a key-padding mask (cuDNN or mem-efficient attention, never flash or the INT8 kernel). With this
on, the trim puts each row's pads FIRST (``[pad | valid text | image]``; every text row sits at the rotary origin, so
the order of text rows moves no phase) and the 28 block attentions read keys from the first valid row on:
``attention(q, k[:, start:], v[:, start:])`` with ``attn_mask=None``. The padded query rows still run (their outputs
are sliced away by the model, as before) and see only valid keys. Same math as the masked call, fewer keys read, and
the call can take the mask-free kernels: native flash, or Studio's INT8 attention route (Krea 2 races it like the
other ``AUTO_DITS``).

The per-call key start lives in a small CPU tensor the pre-hook fills and an opaque custom op reads, so the compiled
blocks hold one graph for every prompt (no guard on the start). The start is fixed at CUDA graph capture time and
follows the prompt, so a call with a non-zero start is never recorded (its ``_unsloth_graph_key_extra`` raises, which
the graph layers read as "run eager"); calls with no plan or a zero start record and replay as before. On an RTX PRO
6000 (Krea-2 Turbo resident, INT8 route) the unrecorded prefix step still beat main's replayed masked step. A call the pre-hook did not plan (a CFG router stacking two planned branches into
one batch) reads its plan off the mask once per forward when the masked keys of every row are a leading run, else
runs the masked call as before.

The pre-hook's plan belongs to one call of one transformer: it is reset before and cleared after every forward, and
only the transformer whose pre-hook writes it reads it. A copy (a CFG replica made by ``copy.deepcopy``, whose root
hooks never run and whose starts would hold the source's value at copy time) reads every plan off its own mask. A call
that needs gradients runs the stock masked processor (the op has no autograd formula).
``UNSLOTH_DIFFUSION_KREA2_PREFIX_KV=0`` turns it off.
"""

from __future__ import annotations

import copy
import os
import weakref
from typing import Any, Optional

PREFIX_ENV = "UNSLOTH_DIFFUSION_KREA2_PREFIX_KV"
# auto: auto's cuDNN pick (the masked-DiT pin) runs the mask-free call on native flash where flash beats the cuDNN
# pin unmasked (diffusion_attention.FLASH_FASTER_CAPABILITIES: A100, L4) and this torch build has a flash SDPA kernel,
# unless UNSLOTH_DIFFUSION_NATIVE_FLASH_ATTN=0; an explicit backend (cudnn included), native (torch picks flash itself
# for a mask-free call), and every other card keep the processor's own. flash / cudnn / incumbent force one.
BACKEND_ENV = "UNSLOTH_DIFFUSION_KREA2_PREFIX_KV_BACKEND"
CLASS_NAME = "Krea2Transformer2DModel"
STARTS_ATTR = "_unsloth_krea2_kv_starts"
_INSTALLED_ATTR = "_unsloth_krea2_prefix_kv"
_OP_NAME = "unsloth_studio::krea2_prefix_attention"
MAX_ROWS = 64
_OFF = ("0", "off", "false", "no")
# calls by path: "prefix" ran mask-free on the valid keys (of them "from_mask": planned off the mask, not the
# pre-hook), "masked" kept the masked call
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
    """[rows, start_0, ..., start_{MAX_ROWS-1}]; rows 0 = no plan for this call (masked call)."""
    import torch
    return torch.zeros(MAX_ROWS + 1, dtype = torch.int64)


def set_plan(transformer: Any, starts: Optional[list]) -> None:
    """Called by the pre-hook for every forward: the key start of each batch row, or None (masked call)."""
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
    """One per transformer: its starts tensor, its first block processor (the attention backend), the transformer,
    and whether its own pre-hook plans every call. A deep copy (a CFG replica) is never planned: its root hooks do
    not run, so the starts it carries are the source's at copy time; it reads each plan off its own mask."""

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
        # the copied processor (memo: the same object the copied block holds)
        new.proc = weakref.ref(copy.deepcopy(proc, memo)) if proc is not None else _dead
        owner = self.owner()
        copied = memo.get(id(owner)) if owner is not None else None
        new.owner = weakref.ref(copied) if copied is not None else _dead
        new.planned = False
        _register(new)
        return new


def _register(slot: _Slot) -> None:
    _SLOTS[slot.starts.data_ptr()] = weakref.ref(slot)
    for key in [k for k, ref in _SLOTS.items() if ref() is None]:
        del _SLOTS[key]


def _slot_of(starts: Any) -> Optional[_Slot]:
    ref = _SLOTS.get(starts.data_ptr())
    slot = ref() if ref is not None else None
    return slot if slot is not None and slot.starts.data_ptr() == starts.data_ptr() else None


class PlanNotRecordable(RuntimeError):
    """Raised by the graph-key hook for a call with a non-zero key start: every CUDA graph layer treats a key hook
    that raises as a call it must not record (``diffusion_cuda_graph`` / ``diffusion_block_graph``: run eager)."""


def plan_key(transformer: Any) -> tuple:
    """The per-call state a CUDA graph must key on (the op reads it at capture time)."""
    held = getattr(transformer, "__dict__", {}).get(STARTS_ATTR)
    if held is None:
        return ()
    rows = int(held[0])
    return tuple(int(x) for x in held[: 1 + rows].tolist())


def _flash_runs(device: Any, dtype: Any) -> bool:
    """Whether this torch build runs flash SDPA on ``device`` at ``dtype``: the probe diffusion_attention runs before
    it pins exclusive flash (PyTorch Windows builds 2.10-2.14 ship without it). Never probed under CUDA graph capture
    (an unanswered card then keeps the incumbent)."""
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
    except Exception:  # noqa: BLE001 - an unreadable probe never forces flash
        kernels = ()
    if not kernels:
        return False  # the probe itself failed: asked again next call
    _FLASH_OK[key] = hit = SDPA_FLASH in kernels
    return hit


def _prefix_backend(
    backend: Any,
    device: Any,
    dtype: Any = None,
    auto_pick: bool = False,
) -> Any:
    """The backend of a mask-free prefix call: an INT8 route keeps itself (it routes unmasked calls), else per
    ``BACKEND_ENV``. ``auto_pick``: the incumbent is auto's cuDNN pick, not an explicit request."""
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

        if (os.environ.get(NATIVE_FLASH_ATTN_ENV) or "").strip().lower() in _OFF:
            return backend
        if (
            getattr(device, "type", None) == "cuda"
            and getattr(torch.version, "hip", None) is None
            and dtype in (torch.bfloat16, torch.float16)
            and tuple(torch.cuda.get_device_capability(device)) in FLASH_FASTER_CAPABILITIES
            and _flash_runs(device, dtype)
        ):
            return AttentionBackendName._NATIVE_FLASH
    except Exception:  # noqa: BLE001
        pass
    return backend


def _dispatch() -> Any:
    # read at call time, as the stock processor does, so the INT8 dispatch route (a module attribute swap) sees it
    from diffusers.models.transformers import transformer_krea2
    return transformer_krea2.dispatch_attention_fn


def mask_plan(mask: Any, batch: int) -> Optional[list]:
    """Each row's key start when its masked keys are exactly a leading run (the pads-first order), else None. One
    host read per mask tensor (the blocks of a forward share it); never under CUDA graph capture."""
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
        if len(_MASK_MEMO) >= _MASK_MEMO_MAX:
            for key in [k for k, v in _MASK_MEMO.items() if v[0]() is None] or list(_MASK_MEMO)[:1]:
                _MASK_MEMO.pop(key, None)
        # keyed per mask (a replica's worker thread and the source each keep theirs)
        _MASK_MEMO[id(mask)] = (weakref.ref(mask), plan)
    except TypeError:
        pass
    return plan


def _backend_of(starts: Any) -> Any:
    """The attention backend of the processors sharing this starts tensor (one per transformer; keyed by its
    storage, so no per-block constant reaches the compiled graph)."""
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
    """[B, S, H, D] x 3 -> [B, S, H, D] (contiguous). The eager body of the custom op."""
    import torch

    slot = _slot_of(starts)
    proc = slot.proc() if slot is not None else None
    backend = getattr(proc, "_attention_backend", None)
    dispatch = _dispatch()
    B = int(q.shape[0])
    # only the transformer whose pre-hook wrote the plan for this call trusts it; a copy reads its own mask
    planned = slot is not None and slot.planned
    rows = int(starts[0]) if planned else 0
    plan = starts[1 : 1 + rows].tolist() if planned and rows == B else None
    if plan is not None and mask is None and any(plan):
        plan = None
    if plan is None and mask is not None and (rows or not planned):
        # a batch the pre-hook did not plan (a CFG router stacking two planned branches, or a replica): read the plan
        # off the mask
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
        # explicit schema: this module's annotations are strings (``from __future__ import annotations``)
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
        """The stock processor with the attention call through the prefix op (masked call when not planned)."""

        # the INT8 dispatch route reads this to know the attention still goes through dispatch_attention_fn
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
                # the op has no autograd formula: a differentiated call runs the stock masked attention
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
    """Swap the main-block processors of a Krea 2 transformer (before the attention backend and the INT8 route are
    applied). Returns how many were swapped; 0 when off or not Krea 2."""
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
            # the INT8 dispatch route takes attention modules that name their head count ``heads``
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
                # the start is read at capture time and changes with the prompt: a recording per start would fill the
                # graph cap in a few prompts (G4: cap of 4 hit, later shapes eager), so planned calls run unrecorded
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
    except Exception as exc:  # noqa: BLE001 - optimisation only: the stock processors stay
        if logger is not None:
            logger.warning("diffusion.krea2: prefix K/V attention unavailable: %s", exc)
        return 0
