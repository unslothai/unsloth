# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 full finetuning: step bfloat16 weights in float32 and round them back stochastically.

bfloat16 keeps 8 mantissa bits, so a weight of magnitude ~0.02 moves in steps of ~1e-4. An AdamW update
at lr 1e-5 is ~1e-5 per element, and round-to-nearest writes most of them back unchanged: on a
Qwen3-0.6B full finetune at lr 1e-5 bfloat16 weights kept 7% of the update float32 masters made.

For the Trainer's AdamW / Adam (torch) and bitsandbytes optimizers (adamw_8bit, paged_adamw_8bit, ...),
each bfloat16 weight is upcast to float32 for its own update (with its gradient), stepped by the same
optimizer, and written back with stochastic rounding (unbiased: E[round(x)] = x), so small updates
survive in expectation. Persistent memory is unchanged: bitsandbytes state does not depend on the
weight dtype, and torch AdamW's moments stay stored in bfloat16 (as bfloat16 AdamW keeps them), also
written back with stochastic rounding. The transient float32 copy is one weight at a time.

On by default for models loaded with full_finetuning = True whose trainable weights are bfloat16, for
optimizers the Trainer builds or receives of the kinds above. UNSLOTH_BF16_STOCHASTIC_ROUNDING=0 turns
it off (round-to-nearest, the previous behaviour). Enabling twice is a no-op.
"""

import os
import types

import torch

__all__ = [
    "enable_bf16_stochastic_rounding",
    "stochastic_round_to_bf16_",
    "bf16_stochastic_rounding_enabled",
]

_MARK = "_unsloth_bf16_stochastic_rounding"
_CHUNK = 1 << 22


def bf16_stochastic_rounding_enabled():
    value = os.environ.get("UNSLOTH_BF16_STOCHASTIC_ROUNDING", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def stochastic_round_to_bf16_(
    dst,
    src,
    generator = None,
):
    """dst (bfloat16) <- src (float32) rounded stochastically: up with probability equal to the
    distance from the lower bfloat16 neighbour in units of the bfloat16 spacing. Inf / NaN are kept.
    Works in chunks so the temporaries stay small."""
    flat_src = src.reshape(-1)
    flat_dst = dst.view(-1)
    for start in range(0, flat_src.numel(), _CHUNK):
        bits = flat_src[start : start + _CHUNK].contiguous().view(torch.int32)
        noise = torch.randint(
            0, 1 << 16, bits.shape, dtype = torch.int32, device = bits.device, generator = generator
        )
        # Adding 16 random low bits then truncating rounds the magnitude up with the right odds
        # (the sign bit is untouched; a carry into the exponent is the correct rounding).
        rounded = (bits + noise) & -65536
        special = (bits & 0x7F800000) == 0x7F800000
        rounded = torch.where(special, bits, rounded)
        flat_dst[start : start + _CHUNK].copy_(rounded.view(torch.float32))
    return dst


def _is_dtensor(p):
    return type(p).__name__ == "DTensor" or hasattr(p, "_local_tensor")


def _tracked(p):
    return (
        p.dtype == torch.bfloat16
        and p.requires_grad
        and not _is_dtensor(p)
        and p.layout == torch.strided
        and p.is_contiguous()
        and p.device.type in ("cuda", "cpu", "xpu")
    )


def _generator(optimizer, device, seed):
    generators = optimizer.__dict__.setdefault("_unsloth_sr_generators", {})
    key = str(device)
    if key not in generators:
        g = torch.Generator(device = device)
        # The same seed on every rank: DDP replicas round identically and stay in sync.
        g.manual_seed(int(seed) + 0x5EED)
        generators[key] = g
    return generators[key]


def _reseed(optimizer, p, seed):
    """Seed from (parameter index, its optimizer step): a resumed run draws what an uninterrupted one would,
    since the step lives in the checkpointed optimizer state."""
    index = optimizer.__dict__.setdefault("_unsloth_sr_index", {})
    i = index.setdefault(id(p), len(index))
    step = (optimizer.state.get(p) or {}).get("step", 0)
    step = int(step.item()) if torch.is_tensor(step) else int(step)
    g = _generator(optimizer, p.device, seed)
    g.manual_seed((int(seed) + 0x5EED + i * 1000003 + step * 2654435761) & 0x7FFFFFFFFFFFFFFF)
    return g


class _Float32Step:
    """Context: one bfloat16 weight (and its gradient) is float32 while the optimizer steps it."""

    def __init__(self, optimizer, p, seed):
        self.optimizer, self.p, self.seed = optimizer, p, seed

    def __enter__(self):
        p = self.p
        self.data, self.grad = p.data, p.grad
        p.grad = None
        p.data = self.data.float()
        p.grad = self.grad.float()
        return p

    def __exit__(self, *exc):
        p = self.p
        stepped = p.data
        p.grad = None
        p.data = self.data
        if exc[0] is None:
            stochastic_round_to_bf16_(self.data, stepped, _reseed(self.optimizer, p, self.seed))
        p.grad = self.grad
        return False


_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def _moments(state, p, dtype):
    # By name: a scalar parameter's fp32 `step` has its shape too.
    return [
        key
        for key in _MOMENT_KEYS
        if torch.is_tensor(state.get(key))
        and state[key].dtype == dtype
        and state[key].shape == p.shape
    ]


def _upcast_state(state, p):
    """torch Adam moments are stored in bfloat16 (as a bfloat16 run keeps them): float32 for the step."""
    for key in _moments(state, p, torch.bfloat16):
        state[key] = state[key].float()


def _store_state(optimizer, state, p, seed):
    """Back to bfloat16 storage after the step, stochastically rounded (unbiased, so a moment decaying
    by less than bfloat16's spacing per step still decays): no memory beyond what bfloat16 AdamW keeps."""
    for key in _moments(state, p, torch.float32):
        value = state[key]
        stored = torch.empty_like(value, dtype = torch.bfloat16)
        stochastic_round_to_bf16_(stored, value, _generator(optimizer, value.device, seed))
        state[key] = stored


def _wrap_bitsandbytes(optimizer, tracked, seed):
    original = optimizer.update_step

    def update_step(self, group, p, gindex, pindex):
        if id(p) not in tracked or p.grad is None or p.dtype != torch.bfloat16:
            return original(group, p, gindex, pindex)
        with _Float32Step(self, p, seed):
            return original(group, p, gindex, pindex)

    optimizer.update_step = types.MethodType(update_step, optimizer)


def _wrap_torch(optimizer, tracked, seed):
    original = optimizer.step

    # A bound method: torch's LR scheduler wraps `optimizer.step.__func__`.
    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        try:
            for group in groups:
                rest = [p for p in group["params"] if id(p) not in tracked]
                mine = [p for p in group["params"] if id(p) in tracked and p.grad is not None]
                if rest:
                    self.param_groups = [dict(group, params = rest)]
                    original()
                for p in mine:
                    self.param_groups = [dict(group, params = [p])]
                    state = self.state.get(p)
                    if state:
                        _upcast_state(state, p)
                    with _Float32Step(self, p, seed):
                        original()
                    _store_state(self, self.state[p], p, seed)
        finally:
            self.param_groups = groups
        return loss

    optimizer.step = types.MethodType(step, optimizer)


def _kind(optimizer):
    try:
        from bitsandbytes.optim.optimizer import Optimizer8bit
        if isinstance(optimizer, Optimizer8bit) and hasattr(optimizer, "update_step"):
            return "bitsandbytes"
    except Exception:
        pass
    if type(optimizer) in (torch.optim.AdamW, torch.optim.Adam):
        return "torch"
    return None


def enable_bf16_stochastic_rounding(
    optimizer,
    seed = 3407,
    verbose = True,
):
    """Step this optimizer's bfloat16 weights in float32 with stochastic rounding back to bfloat16.
    Returns the number of weights it covers (0: nothing to do, unsupported optimizer, or turned off).
    Call before an LR scheduler is built over the optimizer. Idempotent."""
    if optimizer is None or not bf16_stochastic_rounding_enabled():
        return 0
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None and hasattr(inner, "param_groups"):  # accelerate's AcceleratedOptimizer
        optimizer = inner
    if getattr(optimizer, _MARK, None) is not None:
        return getattr(optimizer, _MARK)
    kind = _kind(optimizer)
    if kind is None:
        return 0
    params = [p for group in optimizer.param_groups for p in group["params"] if _tracked(p)]
    if not params:
        return 0
    tracked = {id(p) for p in params}
    if kind == "bitsandbytes":
        _wrap_bitsandbytes(optimizer, tracked, seed)
    else:
        _wrap_torch(optimizer, tracked, seed)
    setattr(optimizer, _MARK, len(params))
    if verbose:
        print(
            f"Unsloth: bfloat16 full finetuning steps {len(params)} bfloat16 weights in float32 and "
            "rounds them back stochastically, so updates smaller than bfloat16's spacing are not lost. "
            "Set UNSLOTH_BF16_STOCHASTIC_ROUNDING=0 to turn this off."
        )
    return len(params)


def _full_finetuning(model):
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        flag = getattr(model, "_unsloth_full_finetuning", None)
        if flag is not None:
            return bool(flag)
        model = getattr(model, "module", None)
    return False


def patch_trainer_bf16_stochastic_rounding(Trainer):
    """Trainer.create_scheduler runs once the optimizer exists (whoever built it) and before the LR
    scheduler wraps its step: enable stochastic rounding there for full finetuning models."""
    original = Trainer.create_scheduler
    if getattr(original, _MARK, False):
        return

    def create_scheduler(
        self,
        num_training_steps,
        optimizer = None,
        *args,
        **kwargs,
    ):
        target = optimizer if optimizer is not None else getattr(self, "optimizer", None)
        if getattr(self, "lr_scheduler", None) is None and _full_finetuning(
            getattr(self, "model", None)
        ):
            try:
                enable_bf16_stochastic_rounding(target, seed = getattr(self.args, "seed", 3407))
            except Exception as exc:  # never block training on this
                print(f"Unsloth: bfloat16 stochastic rounding was not enabled ({exc}).")
        return original(self, num_training_steps, optimizer, *args, **kwargs)

    setattr(create_scheduler, _MARK, True)
    create_scheduler.__wrapped__ = original
    create_scheduler.__doc__ = original.__doc__
    Trainer.create_scheduler = create_scheduler
