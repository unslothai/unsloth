# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager DiT LoRA forward without PEFT's per-call casts, plus a budgeted nf4 dequant cache.

Matches PEFT bit for bit per LoRA linear (same GEMMs, roundings and add); a folded ``addmm`` gained nothing.
Kill switches: ``UNSLOTH_DIFFUSION_FAST_LORA=0``, ``UNSLOTH_DIFFUSION_DEQUANT_CACHE=0``
(``UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB=N`` pins the budget).
"""

from __future__ import annotations

import os
import types
from typing import Optional

import torch

GIB = 1 << 30
_MARGIN_MIN_BYTES = 2 * GIB
_MARGIN_FRAC = 0.10
_BUDGET_FRAC = 0.85
# Hysteresis before regrowing a shrunk budget (avoids evict / refill churn).
_GROW_HYSTERESIS = 1 * GIB


def _env_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("0", "false", "off", "no")


def fast_lora_enabled() -> bool:
    return not _env_off("UNSLOTH_DIFFUSION_FAST_LORA")


def dequant_cache_enabled() -> bool:
    return not _env_off("UNSLOTH_DIFFUSION_DEQUANT_CACHE")


class _LoraProduct(torch.autograd.Function):
    """``x @ a.T @ b.T`` on the bf16 copies; the backward runs PEFT's GEMMs and returns fp32 grads to the masters."""

    @staticmethod
    def forward(ctx, x, a_master, b_master, a_lp, b_lp):
        x2 = x.reshape(-1, x.shape[-1])
        h = torch.mm(x2, a_lp.t())
        out = torch.mm(h, b_lp.t())
        ctx.save_for_backward(x, h, a_lp, b_lp)
        ctx.master_dtypes = (a_master.dtype, b_master.dtype)
        return out.view(*x.shape[:-1], b_lp.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x, h, a_lp, b_lp = ctx.saved_tensors
        need_x, need_a, need_b = ctx.needs_input_grad[:3]
        g2 = grad_out.reshape(-1, grad_out.shape[-1])
        if g2.dtype != b_lp.dtype:
            g2 = g2.to(b_lp.dtype)
        grad_x = grad_a = grad_b = None
        if need_x or need_a:
            grad_h = torch.mm(g2, b_lp)
            if need_x:
                grad_x = torch.mm(grad_h, a_lp).view(x.shape)
            if need_a:
                grad_a = torch.mm(grad_h.t(), x.reshape(-1, x.shape[-1])).to(ctx.master_dtypes[0])
        if need_b:
            grad_b = torch.mm(g2.t(), h).to(ctx.master_dtypes[1])
        return grad_x, grad_a, grad_b, None, None


_GEN = [0]


def mark_lora_updated() -> None:
    """Call after anything that writes the LoRA weights (optimizer step, resume load, EMA swap)."""
    _GEN[0] += 1


def _lp_weights(module, adapter: str, dtype: torch.dtype):
    a = module.lora_A[adapter].weight
    b = module.lora_B[adapter].weight
    scale = float(module.scaling[adapter])
    # Fused AdamW does not bump version counters, so the update generation is the primary key.
    key = (_GEN[0], a._version, b._version, a.data_ptr(), b.data_ptr(), scale, dtype)
    cached = module.__dict__.get("_unsloth_lp")
    if cached is not None and cached[0] == key:
        return a, b, cached[1], cached[2], scale
    with torch.no_grad():
        a_lp = a.detach().to(dtype)
        b_lp = b.detach().to(dtype)
    module.__dict__["_unsloth_lp"] = (key, a_lp, b_lp)
    STATS["casts"] += 1
    return a, b, a_lp, b_lp, scale


STATS = {"calls": 0, "fallback": 0, "casts": 0}


class _SplitCast(torch.autograd.Function):
    """One cast of ``x`` for both branches; backward adds the grads in ``x``'s dtype, as PEFT does."""

    @staticmethod
    def forward(ctx, x, dtype):
        ctx.in_dtype = x.dtype
        lp = x.to(dtype)
        return lp, lp.view_as(lp)

    @staticmethod
    def backward(ctx, g_base, g_lora):
        if g_base is None:
            return (None if g_lora is None else g_lora.to(ctx.in_dtype)), None
        g = g_base.to(ctx.in_dtype)
        return (g if g_lora is None else g + g_lora), None


def fast_lora_stats() -> dict:
    return {"fast_lora_" + k: v for k, v in STATS.items()}


def _is_bnb4(base) -> bool:
    w = getattr(base, "weight", None)
    return getattr(w, "quant_state", None) is not None and type(w).__name__ == "Params4bit"


def _fast_forward(self, x, *args, **kwargs):
    orig = self.__dict__["_unsloth_orig_forward"]
    adapters = self.active_adapters
    if (
        args
        or kwargs
        or self.disable_adapters
        or self.merged
        or len(adapters) != 1
        or adapters[0] not in self.lora_A
        or getattr(self, "lora_variant", {}).get(adapters[0]) is not None
        or getattr(self, "use_dora", {}).get(adapters[0], False)
        or self.lora_B[adapters[0]].bias is not None
        or not x.is_floating_point()
    ):
        STATS["fallback"] += 1
        return orig(x, *args, **kwargs)
    dev = x.device.type
    if not torch.is_autocast_enabled(dev):
        # No autocast (CPU, fp32 runs): PEFT's own fp32 path is the reference, keep it.
        STATS["fallback"] += 1
        return orig(x, *args, **kwargs)
    adapter = adapters[0]
    dtype = torch.get_autocast_dtype(dev)
    base = self.base_layer
    if x.dtype != dtype:
        x_base, x_lora = _SplitCast.apply(x, dtype)
    else:
        x_base = x_lora = x
    result = base(x_base)
    # bnb returns the caller's dtype, so an fp32 input gets an fp32 sum, exactly as on PEFT.
    if self.__dict__["_unsloth_bnb4"] and result.dtype != x.dtype:
        result = result.to(x.dtype)
    a, b, a_lp, b_lp, scale = _lp_weights(self, adapter, dtype)
    lora = _LoraProduct.apply(self.lora_dropout[adapter](x_lora), a, b, a_lp, b_lp)
    if scale != 1.0:
        lora = lora * scale
    STATS["calls"] += 1
    return result + lora


def _nf4_forward(self, x, *args, **kwargs):
    orig = self.__dict__["_unsloth_orig_forward"]
    cd = self.compute_dtype
    if args or kwargs or cd is None or not x.is_floating_point():
        return orig(x, *args, **kwargs)
    w = self.__dict__["_unsloth_dequant"].get(self, cd)
    if w is None:
        return orig(x, *args, **kwargs)
    bias = self.bias
    out = torch.nn.functional.linear(x.to(cd), w, None if bias is None else bias.to(cd))
    return out.to(x.dtype)


def _lora_linear_classes():
    classes = []
    try:
        from peft.tuners.lora.layer import Linear as LoraLinear
        classes.append(LoraLinear)
    except Exception:  # noqa: BLE001
        pass
    try:
        from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
        classes.append(LoraLinear4bit)
    except Exception:  # noqa: BLE001 -- no bitsandbytes
        pass
    return tuple(classes)


def install_fast_lora(model, dequant_cache: Optional["Nf4DequantCache"] = None) -> int:
    """Route plain LoRA linears (dense or nf4 base) through ``_fast_forward``; returns the count patched."""
    classes = _lora_linear_classes()
    if not classes:
        return 0
    n = 0
    for mod in model.modules():
        if type(mod) not in classes or "_unsloth_orig_forward" in mod.__dict__:
            continue
        base = getattr(mod, "base_layer", None)
        bnb4 = _is_bnb4(base)
        if not bnb4 and type(base) is not torch.nn.Linear:
            continue
        mod.__dict__["_unsloth_orig_forward"] = mod.forward
        mod.__dict__["_unsloth_bnb4"] = bnb4
        mod.forward = types.MethodType(_fast_forward, mod)
        n += 1
    if dequant_cache is not None:
        # Every nf4 linear, LoRA base or not (FLUX.2's untargeted feed-forwards dequantise every call too).
        for mod in model.modules():
            if "_unsloth_orig_forward" in mod.__dict__ or not _is_bnb4(mod):
                continue
            if not isinstance(mod, torch.nn.Linear):
                continue
            mod.__dict__["_unsloth_orig_forward"] = mod.forward
            mod.__dict__["_unsloth_dequant"] = dequant_cache
            mod.forward = types.MethodType(_nf4_forward, mod)
            dequant_cache.layers += 1
    return n


def uninstall_fast_lora(model) -> int:
    n = 0
    for mod in model.modules():
        if "_unsloth_orig_forward" in mod.__dict__:
            del mod.__dict__["forward"]
            for k in ("_unsloth_orig_forward", "_unsloth_bnb4", "_unsloth_dequant", "_unsloth_lp"):
                mod.__dict__.pop(k, None)
            n += 1
    return n


def dequant_budget_bytes(
    free: int,
    reserved: int,
    peak_alloc: int,
    total: int,
    frac: float = _BUDGET_FRAC,
) -> int:
    margin = max(_MARGIN_MIN_BYTES, int(total * _MARGIN_FRAC))
    room = free + reserved - peak_alloc - margin
    return max(0, int(room * frac))


def bucket_activation_scale(canvases) -> float:
    """Largest over smallest canvas area: step 1 may draw the smallest bucket, a later step the largest."""
    if not canvases:
        return 1.0
    areas = [w * h for w, h in canvases]
    return max(areas) / max(1, min(areas))


def plan_dequant_cache(
    cache,
    step_peak: int,
    pre_opt_alloc: int,
    act_scale: float = 1.0,
) -> None:
    """Size ``cache`` after step 1's optimizer step: activations scaled to the largest bucket plus new optimizer state."""
    pinned = None
    override = os.environ.get("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", "").strip()
    if override:
        try:
            pinned = int(float(override) * GIB)
        except ValueError:
            pass
    try:
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    except Exception:  # noqa: BLE001 -- no probe: only a pinned budget caches
        cache.plan(pinned or 0)
        return
    opt_state = max(0, allocated - pre_opt_alloc)
    transient = int(max(0, step_peak - pre_opt_alloc) * max(1.0, act_scale))
    cache.transient = transient
    cache.margin = max(_MARGIN_MIN_BYTES, int(total * _MARGIN_FRAC))
    if pinned is not None:
        cache.plan(pinned)
        return
    need = pre_opt_alloc + transient + opt_state
    cache.plan(dequant_budget_bytes(free = free, reserved = reserved, peak_alloc = need, total = total))


def rebalance_dequant_cache(cache) -> int:
    """Evict if another process took memory since the plan, regrow when it returns. Returns bytes evicted."""
    try:
        free, _total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    except Exception:  # noqa: BLE001 -- no probe: leave the cache as it is
        return 0
    return cache.rebalance(free = free, reserved = reserved, allocated = allocated)


def attach_training_hooks(
    optimizer,
    cache: Optional["Nf4DequantCache"],
    act_scale: float = 1.0,
    on_planned = None,
    on_event = None,
):
    """Every step bumps the LoRA generation; the first plans ``cache``, later ones rebalance it."""
    state = {"pending": cache is not None}

    def pre(_opt, _args, _kwargs):
        if state["pending"]:
            state["peak"] = torch.cuda.max_memory_allocated()
            state["pre"] = torch.cuda.memory_allocated()

    def post(_opt, _args, _kwargs):
        mark_lora_updated()
        if cache is None:
            return
        if state["pending"]:
            state["pending"] = False
            plan_dequant_cache(cache, state["peak"], state["pre"], act_scale)
            if on_planned is not None:
                on_planned(cache.stats())
            return
        evicted = rebalance_dequant_cache(cache)
        if evicted:
            torch.cuda.empty_cache()
            if on_event is not None:
                on_event(
                    "dequant_cache_shrunk",
                    {"evicted_gib": round(evicted / GIB, 3), **cache.stats()},
                )

    return (optimizer.register_step_pre_hook(pre), optimizer.register_step_post_hook(post))


# ``DiffusionTrainingService.lend_vram`` requests arrive via the stop queue; no handler = plain ack.
_VRAM_HANDLER: list = []


def set_vram_request_handler(handler) -> None:
    _VRAM_HANDLER[:] = [handler] if handler is not None else []


def handle_vram_request(kind: str) -> dict:
    """``kind`` is "lend" or "return". Runs between steps, where no graph holds a cache entry."""
    if _VRAM_HANDLER:
        return _VRAM_HANDLER[0](kind)
    return {"kind": kind, "cached_gib": 0.0, "freed_gib": 0.0}


def cache_vram_handler(cache: "Nf4DequantCache"):
    def handle(kind: str) -> dict:
        freed = 0
        if kind == "lend":
            freed = cache.lend()
            if freed and torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif kind == "return":
            cache.unlend()
        return {"kind": kind, "freed_gib": round(freed / GIB, 3), **cache.stats()}

    return handle


class Nf4DequantCache:
    """Dequantised (frozen) nf4 base weights kept while they fit in ``budget`` bytes."""

    def __init__(self, budget: int = 0):
        self.budget = int(budget)
        self.used = 0
        self.entries: dict[int, torch.Tensor] = {}
        self.sizes: dict[int, int] = {}
        self.hits = 0
        self.misses = 0
        self.full = False
        self.layers = 0  # nf4 linears routed through the cache
        self.paused = False  # preview render
        self.lent = False  # another Studio GPU load asked for the room
        self.cap = int(budget)  # the step-1 plan: rebalancing never grows past it
        self.transient = 0  # the next step's activations, from step 1
        self.margin = _MARGIN_MIN_BYTES
        self.extra_reserved = 0  # bytes another planner keeps for itself (see ``reserve``)
        self.shrinks = 0

    def plan(self, budget: int) -> None:
        self.budget = max(0, int(budget))
        self.cap = self.budget
        self.full = self.used >= self.budget

    def reserve(self, nbytes: int) -> None:
        """Bytes held on top of step 1's activations from step 2 on (budgeted checkpointing)."""
        self.extra_reserved = max(0, int(nbytes))

    def _evict(self, nbytes: int) -> int:
        freed = 0
        for key in reversed(list(self.entries)):
            if freed >= nbytes:
                break
            w = self.entries.pop(key)
            freed += self.sizes.pop(key, w.numel() * w.element_size())
        self.used = max(0, self.used - freed)
        return freed

    def rebalance(self, *, free: int, reserved: int, allocated: int) -> int:
        if self.cap <= 0:
            return 0
        room = free + (reserved - allocated) - self.transient - self.extra_reserved - self.margin
        pending = 0 if self.full else max(0, self.budget - self.used)
        if room < pending:
            # Stop growing first (pending entries would take this room), then evict the shortfall.
            self.budget = self.used
            self.full = True
            evicted = self._evict(-room) if room < 0 else 0
            self.budget = self.used
            if evicted:
                self.shrinks += 1
            return evicted
        room -= pending
        if room > _GROW_HYSTERESIS and self.budget < self.cap:
            self.budget = min(self.cap, self.budget + int(room * _BUDGET_FRAC))
            self.full = self.used >= self.budget
        return 0

    def lend(self) -> int:
        freed = self.used
        self.clear()
        self.lent = True
        self.budget = 0
        return freed

    def unlend(self) -> None:
        self.lent = False
        self.full = False

    def get(self, base, dtype: torch.dtype) -> Optional[torch.Tensor]:
        w = self.entries.get(id(base))
        if w is not None and w.dtype == dtype:
            self.hits += 1
            return w
        if self.full or self.budget <= 0 or self.paused or self.lent:
            self.misses += 1
            return None
        weight = base.weight
        shape = weight.quant_state.shape
        nbytes = int(shape[0]) * int(shape[1]) * torch.empty((), dtype = dtype).element_size()
        if self.used + nbytes > self.budget:
            self.full = True
            self.misses += 1
            return None
        try:
            import bitsandbytes.functional as bnbF
            with torch.no_grad():
                w = bnbF.dequantize_4bit(weight.data, weight.quant_state).to(dtype)
        except torch.OutOfMemoryError:
            self.full = True
            self.misses += 1
            return None
        self.entries[id(base)] = w
        self.sizes[id(base)] = nbytes
        self.used += nbytes
        self.misses += 1
        return w

    def clear(self) -> None:
        self.entries.clear()
        self.sizes.clear()
        self.used = 0
        self.full = False

    def suspend(self) -> None:
        self.clear()
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def release(self) -> None:
        self.clear()
        self.budget = 0
        self.cap = 0

    def stats(self) -> dict:
        return {
            "budget_gib": round(self.budget / GIB, 3),
            "cached_gib": round(self.used / GIB, 3),
            "cached_layers": len(self.entries),
            "nf4_layers": self.layers,
            "hits": self.hits,
            "misses": self.misses,
            "shrinks": self.shrinks,
            "lent": self.lent,
        }
