# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager DiT LoRA forward without PEFT's per-call casts, plus a budgeted nf4 dequant cache.

PEFT's LoRA linear under bf16 autocast, per call and again in every checkpoint recompute:
casts an fp32 input to bf16 twice (bnb base, then autocast for lora_A), casts the base output
back to fp32, clones it, casts both fp32 LoRA weights, scales and adds in fp32; backward casts
both input grads back to fp32 and adds them. ``install_fast_lora`` swaps that forward for:

* one input cast shared by the base and the LoRA branch;
* bf16 copies of the fp32 masters A and B made once per optimizer step (``mark_lora_updated``
  after each step; the parameters' version counters also catch any other in-place write);
* ``B(A(x))`` in a custom autograd function on those copies whose backward runs PEFT's GEMMs and
  returns fp32 grads to the masters; no clone of the base output, no multiply by a scale of 1.

Same GEMMs, same roundings and the same base + LoRA add as PEFT (an fp32 input still gets PEFT's
fp32 sum): one LoRA linear matches PEFT bit for bit (CPU tests; A100, nf4 and dense bases). A
whole deterministic training run still drifts from PEFT's at bf16-noise level from step 3 on, inside
the run-to-run loss band. Folding the add into one ``addmm`` was measured and gained nothing.

``Nf4DequantCache`` keeps dequantised nf4 base weights (frozen) in VRAM the step does not use,
so the forward, the checkpoint recompute and the backward of every grad-accum micro-batch stop
re-running ``kDequantizeBlockwise`` for the cached layers. The budget is planned after step 1
from free memory and step 1's peak; with no room it caches nothing and memory stays as before.

Eager only: a compiled run already fuses these elementwise ops. Kill switches:
``UNSLOTH_DIFFUSION_FAST_LORA=0`` and ``UNSLOTH_DIFFUSION_DEQUANT_CACHE=0``
(``UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB=N`` pins the budget).
"""

from __future__ import annotations

import os
import types
from typing import Optional

import torch

GIB = 1 << 30
# Room left for the allocator, the next step's transients and anything else on the card.
_MARGIN_MIN_BYTES = 2 * GIB
_MARGIN_FRAC = 0.10
_BUDGET_FRAC = 0.85


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
    """Drop every bf16 master copy: call after each optimizer step and anything else that writes the
    LoRA weights (resume load, EMA swap)."""
    _GEN[0] += 1


def _lp_weights(module, adapter: str, dtype: torch.dtype):
    """bf16 A and B, recast only when a master changed (optimizer step, resume load, EMA swap)."""
    a = module.lora_A[adapter].weight
    b = module.lora_B[adapter].weight
    scale = float(module.scaling[adapter])
    # The fused AdamW writes the masters without bumping their version counters, so the trainer's
    # update generation is the primary key; the version counters still catch any other in-place write.
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
    """One cast of ``x`` serving the base and the LoRA branch (two aliases of one tensor). The backward adds
    the two branch grads in ``x``'s dtype, as PEFT's two separate casts do, so an fp32 input keeps its fp32 sum."""

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
    """Engagement counters since import: fast calls, PEFT fallbacks, bf16 master recasts."""
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
    # An nf4 base reads the dequant cache through its own patched forward (``_nf4_forward``).
    result = base(x_base)
    # bnb hands back the caller's dtype, so an fp32 input (a norm output, Wan's fp32 residual stream) gets an
    # fp32 sum, exactly as on PEFT; a dense base under autocast returns the autocast dtype.
    if self.__dict__["_unsloth_bnb4"] and result.dtype != x.dtype:
        result = result.to(x.dtype)
    a, b, a_lp, b_lp, scale = _lp_weights(self, adapter, dtype)
    lora = _LoraProduct.apply(self.lora_dropout[adapter](x_lora), a, b, a_lp, b_lp)
    if scale != 1.0:
        lora = lora * scale
    STATS["calls"] += 1
    return result + lora


def _nf4_forward(self, x, *args, **kwargs):
    """bnb Linear4bit with its dequantised weight from the cache when it holds (or can take) it."""
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
    """Route every plain LoRA linear (dense nn.Linear or bnb nf4 base) through ``_fast_forward``.
    Returns the number of modules patched; other LoRA layer types keep PEFT's forward."""
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
    """Bytes the cache may hold: what the card has beyond step 1's peak, minus a margin.
    ``free`` is the driver's free memory, ``reserved`` this process's allocator pool and
    ``peak_alloc`` step 1's allocated peak, which the following steps need again."""
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
    """Size ``cache`` right after step 1's optimizer step. ``step_peak`` / ``pre_opt_alloc`` are the allocated
    peak and the allocated bytes just before that step; its activations (peak minus what stays allocated)
    are scaled to the largest bucket and the optimizer state it just created is added."""
    override = os.environ.get("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", "").strip()
    if override:
        try:
            cache.plan(int(float(override) * GIB))
            return
        except ValueError:
            pass
    try:
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
    except Exception:  # noqa: BLE001 -- no probe, no cache
        cache.plan(0)
        return
    opt_state = max(0, allocated - pre_opt_alloc)
    transient = max(0, step_peak - pre_opt_alloc)
    need = pre_opt_alloc + int(transient * max(1.0, act_scale)) + opt_state
    cache.plan(dequant_budget_bytes(free = free, reserved = reserved, peak_alloc = need, total = total))


def attach_training_hooks(
    optimizer,
    cache: Optional["Nf4DequantCache"],
    act_scale: float = 1.0,
    on_planned = None,
):
    """Optimizer hooks: every step bumps the LoRA update generation (the fused AdamW does not bump version
    counters); the first step this process runs also sizes ``cache``. Returns the hook handles."""
    state = {"pending": cache is not None}

    def pre(_opt, _args, _kwargs):
        if state["pending"]:
            state["peak"] = torch.cuda.max_memory_allocated()
            state["pre"] = torch.cuda.memory_allocated()

    def post(_opt, _args, _kwargs):
        mark_lora_updated()
        if state["pending"]:
            state["pending"] = False
            plan_dequant_cache(cache, state["peak"], state["pre"], act_scale)
            if on_planned is not None:
                on_planned(cache.stats())

    return (optimizer.register_step_pre_hook(pre), optimizer.register_step_post_hook(post))


class Nf4DequantCache:
    """Dequantised nf4 base weights kept while they fit in ``budget`` bytes. Frozen weights, so an
    entry stays valid for the whole run; once the budget is full the rest dequantise per call."""

    def __init__(self, budget: int = 0):
        self.budget = int(budget)
        self.used = 0
        self.entries: dict[int, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0
        self.full = False
        self.layers = 0  # nf4 linears routed through the cache
        self.paused = False

    def plan(self, budget: int) -> None:
        self.budget = max(0, int(budget))
        self.full = self.used >= self.budget

    def get(self, base, dtype: torch.dtype) -> Optional[torch.Tensor]:
        w = self.entries.get(id(base))
        if w is not None and w.dtype == dtype:
            self.hits += 1
            return w
        if self.full or self.budget <= 0 or self.paused:
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
        self.used += nbytes
        self.misses += 1
        return w

    def clear(self) -> None:
        self.entries.clear()
        self.used = 0
        self.full = False

    def suspend(self) -> None:
        """Free every entry and cache nothing until ``resume`` (a preview render needs the room)."""
        self.clear()
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def release(self) -> None:
        """Drop every entry and stop caching (previews and the end of the run need the room)."""
        self.clear()
        self.budget = 0

    def stats(self) -> dict:
        return {
            "budget_gib": round(self.budget / GIB, 3),
            "cached_gib": round(self.used / GIB, 3),
            "cached_layers": len(self.entries),
            "nf4_layers": self.layers,
            "hits": self.hits,
            "misses": self.misses,
        }
