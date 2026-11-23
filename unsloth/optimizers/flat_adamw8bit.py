# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bitsandbytes 8-bit Adam / AdamW (optim="adamw_8bit") as one blockwise update per parameter group.

Stock bitsandbytes launches one update per tensor and, from 0.46, synchronizes the device after
each: about 60 us of host time per tensor, 24 ms for the 392 LoRA tensors of Qwen3-0.6B r16, which
is the whole gap between a launch-bound step (block CUDA graphs) and the GPU. Here each group's
gradients and parameters are concatenated, updated by the same bitsandbytes kernel, and copied back.
The kernel quantizes in independent blocks of 256 elements, so when every tensor's size is a
multiple of 256 the result is bitwise the stock one, and the state stays in the stock per-parameter
layout (views into one buffer): checkpoints load into either optimizer. Anything else (paged,
per-parameter overrides such as 32-bit embeddings, 32-bit state, per-tensor state such as
percentile_clipping / max_unorm / non-blockwise 8-bit, missing gradients, CPU) runs the stock step.
FSDP / DeepSpeed trainers and DTensor parameters are left alone (the class is not swapped), since
FSDP2 localizes only the plain bitsandbytes class. UNSLOTH_FLAT_ADAMW8BIT=0 turns it off.
"""

import os

import torch

__all__ = [
    "flat_adamw8bit_enabled",
    "install_flat_adamw8bit",
    "install_flat_adamw8bit_hook",
]

_ENV = "UNSLOTH_FLAT_ADAMW8BIT"
_BLOCK = 256  # bitsandbytes blockwise optimizer block size (Optimizer2State.init_state)
_STATE_KEYS = ("state1", "state2", "absmax1", "absmax2")
_CLASSES = {}


def flat_adamw8bit_enabled():
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def _chunk_elements():
    # Bounds the transient gradient + parameter copies: 1M fp32 elements = 4 MiB each.
    try:
        return max(
            _BLOCK, int(float(os.environ.get(_ENV + "_CHUNK", str(1 << 20)))) // _BLOCK * _BLOCK
        )
    except ValueError:
        return 1 << 20


def _innermost(optimizer):
    while isinstance(getattr(optimizer, "optimizer", None), torch.optim.Optimizer):
        optimizer = optimizer.optimizer
    return optimizer


def _stock_classes():
    try:
        import bitsandbytes.optim as bnb_optim
    except Exception:
        return ()
    names = ("Adam", "Adam8bit", "AdamW", "AdamW8bit")
    return tuple(
        getattr(bnb_optim, n) for n in names if isinstance(getattr(bnb_optim, n, None), type)
    )


def _unsupported_args(optimizer):
    """Why the optimizer's options rule out the fused update (None when they do not). The fused
    update hands bitsandbytes' update_step one concatenated segment with only the blockwise 8-bit
    moments, so every option that keeps per-tensor state, or reduces over the whole tensor, runs
    stock: percentile_clipping < 100 (gnorm_vec, a per-tensor gradient norm history, bnb < 0.50),
    max_unorm > 0 (unorm_vec, a per-tensor update norm), block_wise=False (per-tensor max1 /
    new_max1, bnb < 0.50), 32-bit state, paged state. skip_zeros is elementwise in the kernel but
    is not covered by the parity tests, so it runs stock too."""
    args = getattr(optimizer, "args", None)
    if args is None:
        return "no bitsandbytes args"
    if getattr(optimizer, "is_paged", False):
        return "paged"
    if getattr(args, "optim_bits", 32) != 8:
        return "optim_bits != 8"
    if getattr(args, "percentile_clipping", 100) < 100:
        return "percentile_clipping < 100"
    if getattr(args, "max_unorm", 0.0) > 0.0:
        return "max_unorm > 0"
    if not getattr(args, "block_wise", True):
        return "block_wise = False"
    if getattr(args, "skip_zeros", False):
        return "skip_zeros"
    return None


def _has_dtensor_params(optimizer):
    for group in getattr(optimizer, "param_groups", ()):
        for p in group["params"]:
            if type(p.data) is not torch.Tensor:
                return True  # DTensor (FSDP2 / TP) or another tensor subclass
    return False


def _group_plan(optimizer, gindex, group):
    """Segments of the group for the fused update, or None when the group must run stock."""
    if _unsupported_args(optimizer) is not None:
        return None
    args = optimizer.args
    mng = optimizer.mng
    params = group["params"]
    if not params:
        return None
    device, dtype = params[0].device, params[0].dtype
    if device.type != "cuda":
        return None
    min_8bit = getattr(args, "min_8bit_size", 4096)
    for pindex, p in enumerate(params):
        if (gindex, pindex) in mng.index2config or id(p) in mng.pid2config:
            return None
        if type(p.data) is not torch.Tensor or p.device != device or p.dtype != dtype:
            return None
        n = p.numel()
        if n < min_8bit or n % _BLOCK or not p.is_contiguous():
            return None
    chunk = _chunk_elements()
    segments, start, size = [], 0, 0
    for i, p in enumerate(params):
        if size and size + p.numel() > chunk:
            segments.append((start, i))
            start, size = i, 0
        size += p.numel()
    segments.append((start, len(params)))
    return {"params": list(params), "segments": segments, "flat": None}


def _pack_state(optimizer, plan):
    """Every parameter's 8-bit state as views into one buffer per key (stock shapes, stock keys)."""
    params, states = plan["params"], [optimizer.state[p] for p in plan["params"]]
    total = sum(p.numel() for p in params)
    first = states[0]
    flat = {
        "state1": torch.empty(total, dtype = torch.uint8, device = first["state1"].device),
        "state2": torch.empty(total, dtype = torch.uint8, device = first["state2"].device),
        "absmax1": torch.empty(
            total // _BLOCK, dtype = torch.float32, device = first["absmax1"].device
        ),
        "absmax2": torch.empty(
            total // _BLOCK, dtype = torch.float32, device = first["absmax2"].device
        ),
    }
    offset = 0
    with torch.no_grad():
        for p, state in zip(params, states):
            n = p.numel()
            for key in _STATE_KEYS:
                div = 1 if key.startswith("state") else _BLOCK
                view = flat[key][offset // div : (offset + n) // div]
                view.copy_(state[key].reshape(-1))
                state[key] = view.view(state[key].shape)
            offset += n
    plan["flat"] = flat
    plan["ptrs"] = tuple(states[i]["state1"].data_ptr() for i in (0, -1))


def _ready(optimizer, gindex, group, plan):
    """Initialise or pack the state if needed; False when this step must run stock."""
    params = plan["params"]
    if len(group["params"]) != len(params):
        return False
    for p in params:
        g = p.grad
        if g is None or g.is_sparse or g.dtype != p.dtype or g.device != p.device:
            return False
    states = [optimizer.state[p] for p in params]
    if all(len(s) == 0 for s in states):
        for pindex, p in enumerate(params):
            optimizer.init_state(group, p, gindex, pindex)
        plan["flat"] = None
    elif any(len(s) == 0 for s in states):
        return False
    step = states[0].get("step")
    for s in states:
        if s.get("step") != step or s["state1"].dtype != torch.uint8 or "absmax1" not in s:
            return False
    if plan["flat"] is None or plan["ptrs"] != tuple(
        states[i]["state1"].data_ptr() for i in (0, -1)
    ):
        _pack_state(optimizer, plan)
    return True


def _fused_group(optimizer, gindex, group, plan):
    params, flat = plan["params"], plan["flat"]
    first = optimizer.state[params[0]]
    step = first["step"]
    offset = 0
    for start, end in plan["segments"]:
        seg = params[start:end]
        n = sum(p.numel() for p in seg)
        if len(seg) == 1:
            values, grad = seg[0].detach().reshape(-1), seg[0].grad.reshape(-1)
        else:
            values = torch.cat([p.detach().reshape(-1) for p in seg])
            grad = torch.cat([p.grad.reshape(-1) for p in seg])
        values.grad = grad
        state = {
            "step": step,
            "state1": flat["state1"][offset : offset + n],
            "state2": flat["state2"][offset : offset + n],
            "absmax1": flat["absmax1"][offset // _BLOCK : (offset + n) // _BLOCK],
            "absmax2": flat["absmax2"][offset // _BLOCK : (offset + n) // _BLOCK],
            "qmap1": first["qmap1"],
            "qmap2": first["qmap2"],
        }
        # bitsandbytes' own update_step (its kernel call for the installed version) on the segment.
        optimizer.state[values] = state
        try:
            optimizer.update_step(group, values, gindex, start)
        finally:
            del optimizer.state[values]
            values.grad = None
        if len(seg) > 1:
            torch._foreach_copy_(
                [p.detach().view(-1) for p in seg], list(values.split([p.numel() for p in seg]))
            )
        offset += n
    for p in params:
        optimizer.state[p]["step"] = step + 1


def _flat_class(base):
    cls = _CLASSES.get(base)
    if cls is not None:
        return cls
    # Unhooked: called from inside the hooked step below, so step hooks fire once.
    stock_step = getattr(base.step, "__wrapped__", base.step)

    @torch.no_grad()
    def step(self, closure = None):
        if not flat_adamw8bit_enabled():
            return stock_step(self, closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.initialized:
            # As the stock step: overrides (32-bit embeddings) are only known from here on.
            self.check_overrides()
            self.to_gpu()
            self.initialized = True
            self.__dict__.pop("_unsloth_flat_plans", None)
        plans = self.__dict__.setdefault("_unsloth_flat_plans", {})
        ready = []
        for gindex, group in enumerate(self.param_groups):
            params = group["params"]
            if not params:
                continue  # Trainer keeps an empty no-decay group for LoRA
            key = (gindex, len(params), id(params[0]) if params else 0)
            if key not in plans:
                plans[key] = _group_plan(self, gindex, group)
            plan = plans[key]
            if plan is None or not _ready(self, gindex, group, plan):
                stock_step(self)
                return loss
            ready.append((gindex, group, plan))
        for gindex, group, plan in ready:
            _fused_group(self, gindex, group, plan)
        return loss

    def load_state_dict(self, state_dict):
        self.__dict__.pop("_unsloth_flat_plans", None)
        return base.load_state_dict(self, state_dict)

    def add_param_group(self, param_group):
        self.__dict__.pop("_unsloth_flat_plans", None)
        return base.add_param_group(self, param_group)

    step = torch.optim.Optimizer.profile_hook_step(step)
    step.hooked = True
    cls = type(
        base.__name__,
        (base,),
        {
            "step": step,
            "load_state_dict": load_state_dict,
            "add_param_group": add_param_group,
            "__module__": base.__module__,
            "__qualname__": base.__qualname__,
            "_unsloth_flat_adamw8bit": True,
            "_unsloth_flat_base": base,
        },
    )
    _CLASSES[base] = cls
    return cls


def install_flat_adamw8bit(optimizer):
    """Switch a stock bitsandbytes 8-bit Adam / AdamW to the per-group update; returns the optimizer
    it changed, or None. The class keeps its name and module, so type checks still see bitsandbytes."""
    if not flat_adamw8bit_enabled() or optimizer is None:
        return None
    inner = _innermost(optimizer)
    if getattr(type(inner), "_unsloth_flat_adamw8bit", False):
        return inner
    if type(inner) not in _stock_classes():
        return None
    if _unsupported_args(inner) is not None or _has_dtensor_params(inner):
        return None
    inner.__class__ = _flat_class(type(inner))
    return inner


def _distributed_wrapper(trainer):
    """Whether the trainer shards or wraps the optimizer itself: FSDP (any version, incl. XLA FSDP)
    or DeepSpeed. accelerate's FSDP2 prepare localizes only the plain bitsandbytes class
    (fsdp2_bnb_optim) and DeepSpeed builds its own optimizer, so the class is left stock there."""
    for name in ("is_deepspeed_enabled", "is_fsdp_enabled", "is_fsdp_xla_enabled"):
        if getattr(trainer, name, False):
            return True
    args = getattr(trainer, "args", None)
    if getattr(args, "deepspeed", None) or getattr(args, "fsdp", None):
        return True
    state = getattr(getattr(trainer, "accelerator", None), "state", None)
    if state is not None:
        if getattr(state, "fsdp_plugin", None) is not None:
            return True
        if getattr(state, "deepspeed_plugin", None) is not None:
            return True
        kind = str(getattr(state, "distributed_type", "") or "").upper()
        if "FSDP" in kind or "DEEPSPEED" in kind:
            return True
    for env in ("ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED"):
        if os.environ.get(env, "false").strip().lower() in ("1", "true", "yes", "on"):
            return True
    return False


def install_flat_adamw8bit_hook(trainer):
    """Switch the trainer's optimizer once it is final (inside create_scheduler, before accelerate
    wraps it), as install_optimizer_state_offload does. Not under FSDP / DeepSpeed."""
    if not flat_adamw8bit_enabled() or _distributed_wrapper(trainer):
        return False
    create_scheduler = trainer.create_scheduler
    if getattr(create_scheduler, "_unsloth_flat_adamw8bit", False):
        return True

    def _create_scheduler_flat_adamw8bit(*args, **kwargs):
        optimizer = kwargs.get("optimizer", args[1] if len(args) > 1 else None)
        optimizer = optimizer if optimizer is not None else getattr(trainer, "optimizer", None)
        try:
            if not _distributed_wrapper(trainer):
                install_flat_adamw8bit(optimizer)
        except Exception:
            pass
        return create_scheduler(*args, **kwargs)

    _create_scheduler_flat_adamw8bit._unsloth_flat_adamw8bit = True
    trainer.create_scheduler = _create_scheduler_flat_adamw8bit
    return True
