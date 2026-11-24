# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager DiT LoRA forward without PEFT's per-call casts, plus a budgeted dequant cache for nf4 and int8 bases.

Matches PEFT bit for bit per LoRA linear (same GEMMs, roundings and add); a folded ``addmm`` gained nothing.
The int8 entries hold the cast int8 values torchao rebuilds per call (not ``q * scale``), so its GEMM, scale and
bias add run unchanged and the result is bit-identical.
Kill switches: ``UNSLOTH_DIFFUSION_FAST_LORA=0``, ``UNSLOTH_DIFFUSION_DEQUANT_CACHE=0``,
``UNSLOTH_DIFFUSION_INT8_DEQUANT_CACHE=0`` (int8 only); ``UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB=N`` pins the budget.
"""

from __future__ import annotations

import os
import types
from typing import Optional

import torch


def _mem_get_info() -> tuple[int, int]:
    """(free, total) for the cache budget: Studio's trusted reading (Windows ROCm's WDDM free capped), else torch."""
    try:
        from utils.hardware import trusted_mem_get_info
    except ImportError:  # the trainer outside Studio's backend
        return torch.cuda.mem_get_info()
    return trusted_mem_get_info()


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


def int8_dequant_cache_enabled() -> bool:
    return dequant_cache_enabled() and not _env_off("UNSLOTH_DIFFUSION_INT8_DEQUANT_CACHE")


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


def _int8_parts(w):
    """``(qdata, scale)`` of a torchao weight-only per-row int8 weight whose linear is ``mm(x, q.t().to(x.dtype)) *
    scale + bias`` (torchao 0.14-0.17 ``AffineQuantizedTensor`` plain layout, 0.18+ ``Int8Tensor``), else None."""
    name = type(w).__name__
    try:
        if name == "Int8Tensor":
            q, scale = w.qdata, w.scale
            if (
                getattr(w, "act_quant_kwargs", None) is not None
                or getattr(w, "act_pre_scale", None) is not None
                or int(w.block_size[-1]) != int(q.shape[-1])
            ):
                return None
        elif name == "AffineQuantizedTensor":
            impl = w.tensor_impl
            if type(getattr(w, "_layout", None)).__name__ != "PlainLayout":
                return None
            if str(getattr(w, "zero_point_domain", "INT")).rsplit(".", 1)[-1] != "INT":
                return None
            q, scale = impl.int_data, impl.scale
            if tuple(int(b) for b in w.block_size) != (1, int(q.shape[-1])):
                return None
        else:
            return None
    except Exception:  # noqa: BLE001 -- an unknown torchao layout keeps torchao's own forward
        return None
    if q.dtype != torch.int8 or q.dim() != 2 or not isinstance(scale, torch.Tensor):
        return None
    if scale.numel() not in (1, int(q.shape[0])):
        return None
    return q, scale


def _is_int8_wo(mod) -> bool:
    return _int8_parts(getattr(mod, "weight", None)) is not None


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


def _int8_scale(self, dtype: torch.dtype) -> torch.Tensor:
    cached = self.__dict__.get("_unsloth_int8_scale")
    if cached is not None and cached[0] == dtype:
        return cached[1]
    scale = _int8_parts(self.weight)[1].to(dtype).flatten()
    self.__dict__["_unsloth_int8_scale"] = (dtype, scale)
    return scale


def _int8_cached_linear(self, x, w_t):
    """torchao's weight-only int8 linear with the cast weight ``w_t`` (``q.t().to(x.dtype)``) taken from the cache."""
    m = torch.mm(x.reshape(-1, x.shape[-1]), w_t)
    y = m * _int8_scale(self, m.dtype)
    y = y.reshape(*x.shape[:-1], y.shape[-1])
    if self.bias is not None:
        y += self.bias
    return y


def _int8_forward(self, x, *args, **kwargs):
    orig = self.__dict__["_unsloth_orig_forward"]
    if args or kwargs or not x.is_floating_point():
        return orig(x, *args, **kwargs)
    dtype = x.dtype
    dev = x.device.type
    bias = self.bias
    # Only where torchao's GEMM runs in x's dtype: an fp32 input under bf16 autocast keeps torchao.
    if (torch.is_autocast_enabled(dev) and torch.get_autocast_dtype(dev) != dtype) or (
        bias is not None and bias.dtype != dtype
    ):
        return orig(x, *args, **kwargs)
    w_t = self.__dict__["_unsloth_dequant"].get(self, dtype)
    if w_t is None:
        return orig(x, *args, **kwargs)
    return _int8_cached_linear(self, x, w_t)


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


def _is_fp32_island(mod) -> bool:
    # Patching an fp32 island root (diffusion_dit_fp16) would run it under fp16 autocast.
    return bool(mod.__dict__.get("_unsloth_fp32_island", False))


def install_fast_lora(
    model,
    dequant_cache: Optional["Nf4DequantCache"] = None,
    int8: bool = False,
) -> int:
    """Route plain LoRA linears (dense or nf4 base) through ``_fast_forward``; returns the count patched.
    ``int8``: the cache also takes torchao weight-only int8 linears (``int8_dequant_cache_enabled`` gates it)."""
    classes = _lora_linear_classes()
    if not classes:
        return 0
    n = 0
    for mod in model.modules():
        if (
            type(mod) not in classes
            or "_unsloth_orig_forward" in mod.__dict__
            or _is_fp32_island(mod)
        ):
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
        int8 = int8 and int8_dequant_cache_enabled()
        # Every nf4 / int8 linear, LoRA base or not (FLUX.2's untargeted feed-forwards dequantise every call too).
        for mod in model.modules():
            if "_unsloth_orig_forward" in mod.__dict__ or _is_fp32_island(mod):
                continue
            if not isinstance(mod, torch.nn.Linear):
                continue
            if _is_bnb4(mod):
                fwd = _nf4_forward
            elif int8 and _is_int8_wo(mod):
                fwd = _int8_forward
                dequant_cache.int8_layers += 1
            else:
                continue
            mod.__dict__["_unsloth_orig_forward"] = mod.forward
            mod.__dict__["_unsloth_dequant"] = dequant_cache
            mod.forward = types.MethodType(fwd, mod)
            dequant_cache.layers += 1
    return n


def uninstall_fast_lora(model) -> int:
    n = 0
    for mod in model.modules():
        if "_unsloth_orig_forward" in mod.__dict__:
            del mod.__dict__["forward"]
            for k in (
                "_unsloth_orig_forward",
                "_unsloth_bnb4",
                "_unsloth_dequant",
                "_unsloth_lp",
                "_unsloth_int8_scale",
            ):
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
        free, total = _mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    except Exception:  # noqa: BLE001 -- no probe: only a pinned budget caches
        cache.plan(pinned or 0, pinned = pinned is not None)
        return
    opt_state = max(0, allocated - pre_opt_alloc)
    transient = int(max(0, step_peak - pre_opt_alloc) * max(1.0, act_scale))
    cache.transient = transient
    cache.margin = max(_MARGIN_MIN_BYTES, int(total * _MARGIN_FRAC))
    if pinned is not None:
        cache.plan(pinned, pinned = True)
        return
    need = pre_opt_alloc + transient + opt_state
    cache.plan(dequant_budget_bytes(free = free, reserved = reserved, peak_alloc = need, total = total))


def rebalance_dequant_cache(cache) -> int:
    """Evict if another process took memory since the plan, regrow when it returns. Returns bytes evicted."""
    try:
        free, _total = _mem_get_info()
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
    """Dequantised (frozen) nf4 and int8 base weights kept while they fit in ``budget`` bytes."""

    def __init__(self, budget: int = 0):
        self.budget = int(budget)
        self.used = 0
        self.entries: dict[int, torch.Tensor] = {}
        self.sizes: dict[int, int] = {}
        self.hits = 0
        self.misses = 0
        self.full = False
        self.layers = 0  # nf4 + int8 linears routed through the cache
        self.int8_layers = 0
        # None until the first int8 entry is checked against torchao's own forward; False turns int8 caching off.
        self.int8_exact: Optional[bool] = None
        self.paused = False  # preview render
        self.lent = False  # another Studio GPU load asked for the room
        self.cap = int(budget)  # the step-1 plan: rebalancing never grows past it
        self.transient = 0  # the next step's activations, from step 1
        self.margin = _MARGIN_MIN_BYTES
        self.extra_reserved = 0  # bytes another planner keeps for itself (see ``reserve``)
        self.planned = int(budget)  # the step-1 plan before any reservation
        self.pinned = False
        self.shrinks = 0

    def plan(
        self,
        budget: int,
        pinned: bool = False,
    ) -> None:
        self.budget = max(0, int(budget))
        self.cap = self.budget
        self.planned = self.budget
        self.pinned = bool(pinned)
        self.full = self.used >= self.budget

    def reserve(
        self,
        nbytes: int,
        from_plan: bool = False,
    ) -> int:
        """Bytes budgeted checkpointing holds from step 2 on. ``from_plan``: re-plan to the plan less our share of
        them (never for a pinned budget). Returns the bytes evicted."""
        self.extra_reserved = max(0, int(nbytes))
        if not from_plan or self.pinned:
            return 0
        self.cap = max(0, self.planned - int(self.extra_reserved * _BUDGET_FRAC))
        self.budget = min(self.budget, self.cap)
        evicted = self._evict(self.used - self.budget) if self.used > self.budget else 0
        self.full = self.used >= self.budget
        return evicted

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
        int8 = None if _is_bnb4(base) else _int8_parts(weight)
        if int8 is None and not _is_bnb4(base):
            self.misses += 1
            return None
        if int8 is not None and self.int8_exact is False:
            self.misses += 1
            return None
        shape = weight.quant_state.shape if int8 is None else int8[0].shape
        nbytes = int(shape[0]) * int(shape[1]) * torch.empty((), dtype = dtype).element_size()
        if self.used + nbytes > self.budget:
            self.full = True
            self.misses += 1
            return None
        try:
            with torch.no_grad():
                if int8 is None:
                    import bitsandbytes.functional as bnbF
                    w = bnbF.dequantize_4bit(weight.data, weight.quant_state).to(dtype)
                else:
                    # Exactly the tensor torchao builds per call: the transposed int8 view cast to the input dtype.
                    w = int8[0].t().to(dtype)
                    if self.int8_exact is None:
                        self.int8_exact = _int8_matches_torchao(base, w, dtype)
                        if not self.int8_exact:
                            self.misses += 1
                            return None
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
            "nf4_layers": self.layers - self.int8_layers,
            "int8_layers": self.int8_layers,
            "int8_exact": self.int8_exact,
            "hits": self.hits,
            "misses": self.misses,
            "shrinks": self.shrinks,
            "lent": self.lent,
            "reserved_gib": round(self.extra_reserved / GIB, 3),
        }


def _int8_matches_torchao(base, w_t, dtype) -> bool:
    """One small forward through torchao and through the cached path; anything but bit equality (a torchao layout
    this module does not mirror) turns int8 caching off for the run."""
    try:
        with torch.no_grad():
            # Its own generator: the training noise draws must not shift.
            gen = torch.Generator(device = w_t.device).manual_seed(0)
            x = torch.randn(8, w_t.shape[0], device = w_t.device, dtype = dtype, generator = gen)
            ref = base.__dict__["_unsloth_orig_forward"](x)
            got = _int8_cached_linear(base, x, w_t)
        return ref.dtype == got.dtype and ref.shape == got.shape and bool(torch.equal(ref, got))
    except Exception:  # noqa: BLE001 -- an unprovable path stays on torchao
        return False
