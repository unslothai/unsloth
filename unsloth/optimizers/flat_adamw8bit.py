# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bitsandbytes 8-bit Adam / AdamW (optim="adamw_8bit") as one blockwise update per chunk of parameters.

Stock bitsandbytes launches one update per tensor (plus, on bnb >= 0.46 off the patched default
stream path, a device sync per tensor). Here consecutive parameters of a group are concatenated into
chunks (UNSLOTH_FLAT_ADAMW8BIT_CHUNK elements, default 1M), each chunk is updated by the same
bitsandbytes kernel through bitsandbytes' own update_step, and copied back. The kernel quantizes in
independent blocks of 256 elements, so when every tensor's size is a multiple of 256 the result is
bitwise the stock one. The state stays in the stock per-parameter layout (views into one buffer per
chunk), so checkpoints load into either optimizer.

The fused step runs only when the whole optimizer is eligible (_live_plans), checked every step
before any state is allocated; otherwise every group takes the stock step. Eligible means: one CUDA
device on its default stream (bitsandbytes launches on the legacy default stream); plain tensors (no
DTensor / FSDP / DeepSpeed); no state offload or instance-level update_step / step wrappers; no
GlobalOptimManager overrides; blockwise 8-bit state with no per-tensor options (percentile_clipping,
max_unorm, skip_zeros); no duplicate or storage-sharing parameters; contiguous parameters whose sizes
are multiples of 256; gradients for every parameter; complete on-device state with int step counters.
UNSLOTH_FLAT_ADAMW8BIT=0 turns it off.
"""

import copyreg
import math
import os
import weakref

import torch

__all__ = [
    "flat_adamw8bit_enabled",
    "install_flat_adamw8bit",
    "install_flat_adamw8bit_hook",
]

_ENV = "UNSLOTH_FLAT_ADAMW8BIT"
_BLOCK = 256  # bitsandbytes blockwise optimizer block size (Optimizer2State.init_state)
_DEFAULT_CHUNK = 1 << 20
_MAX_CHUNK = 1 << 30  # the CUDA dispatcher passes the element count as int32
_STATE_KEYS = ("state1", "state2", "absmax1", "absmax2")
_QMAP_KEYS = ("qmap1", "qmap2")
_TRACKED_KEYS = _STATE_KEYS + _QMAP_KEYS
_STATE_ALLOWED = frozenset(("step",) + _TRACKED_KEYS)
_PLANS = "_unsloth_flat_plans"
_FUSED_STEPS = "_unsloth_flat_fused_steps"
_DEVICE_TYPE = "cuda"  # tests on bitsandbytes >= 0.50 (CPU kernels) switch this to "cpu"
_OOM = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError
_PROFILE_STEP_CODE = torch.optim.Optimizer.profile_hook_step(lambda self: None).__code__
_CLASSES = {}


def flat_adamw8bit_enabled():
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "false", "no", "off", "")


def _truthy(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _offload_requested():
    try:
        from .state_offload import optimizer_state_offload_enabled
    except ImportError:  # loaded outside the package (tests)
        return _truthy("UNSLOTH_OFFLOAD_OPTIMIZER_STATE")
    return optimizer_state_offload_enabled()


def _chunk_elements():
    # Bounds the transient gradient + parameter copies: 1M fp32 elements = 4 MiB each.
    try:
        value = float(os.environ.get(_ENV + "_CHUNK", str(_DEFAULT_CHUNK)))
    except ValueError:
        return _DEFAULT_CHUNK
    if not math.isfinite(value):
        return _DEFAULT_CHUNK
    return min(_MAX_CHUNK, max(_BLOCK, int(value) // _BLOCK * _BLOCK))


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


def _is_profile_step(fn):
    return getattr(fn, "__code__", None) is _PROFILE_STEP_CODE


def _stock_step_for(base):
    """The class step minus torch's profiling wrapper when that is the outermost layer (the fused step
    is wrapped by its own, so hooks fire once). None when a profiling wrapper sits under another
    decorator: peeling it would drop that decorator, keeping it would fire the hooks twice."""
    step = base.step
    if _is_profile_step(step):
        return step.__wrapped__
    fn = step
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
        if _is_profile_step(fn):
            return None
    return step


def _unsupported_args(optimizer):
    """Why the optimizer's options rule out the fused update (None when they do not). A fused chunk
    hands update_step only the blockwise 8-bit moments, so options that keep per-tensor state or
    reduce over a tensor run stock: percentile_clipping < 100 (gnorm_vec, bnb < 0.50), max_unorm > 0
    (unorm_vec), block_wise=False (max1 / new_max1, bnb < 0.50), 32-bit and paged state. skip_zeros
    is elementwise in the kernel but not covered by the parity tests, so it runs stock too."""
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


def _wrapped_instance(optimizer):
    """Why instance-level state rules out the fused update: bf16 stochastic rounding (and anything
    else) replacing update_step / init_state on the instance must see the real parameters, and the
    state offloader and GRPO rollout parking (_unsloth_flat_disabled) move state between steps."""
    if getattr(optimizer, "_unsloth_state_offloader", None) is not None:
        return "state offload"
    disabled = optimizer.__dict__.get("_unsloth_flat_disabled")
    if disabled:
        return str(disabled)
    for name in ("update_step", "init_state", "get_config"):
        if name in vars(optimizer):
            return f"instance {name}"
    return None


def _has_dtensor_params(optimizer):
    for group in getattr(optimizer, "param_groups", ()):
        for p in group["params"]:
            if type(p.data) is not torch.Tensor:
                return True  # DTensor (FSDP2 / TP) or another tensor subclass
    return False


def _install_refusal(optimizer):
    """Why install_flat_adamw8bit leaves this optimizer stock (None: install)."""
    if type(optimizer) not in _stock_classes():
        return "not a stock bitsandbytes Adam / AdamW"
    reason = _unsupported_args(optimizer) or _wrapped_instance(optimizer)
    if reason is not None:
        return reason
    if _has_dtensor_params(optimizer):
        return "DTensor or tensor-subclass parameters"
    devices = {p.device for g in optimizer.param_groups for p in g["params"]}
    if len(devices) != 1 or next(iter(devices)).type != _DEVICE_TYPE:
        return "parameters not on one CUDA device"
    step = vars(optimizer).get("step")
    if step is not None and not getattr(step, "_wrapped_by_lr_sched", False):
        return "instance step wrapper"
    if _stock_step_for(type(optimizer)) is None:
        return "nested step wrapper"
    return None


def _on_default_stream(device):
    return torch.cuda.current_stream(device) == torch.cuda.default_stream(device)


def _segments(params):
    chunk = _chunk_elements()
    segments, start, size = [], 0, 0
    for i, p in enumerate(params):
        if size and size + p.numel() > chunk:
            segments.append((start, i))
            start, size = i, 0
        size += p.numel()
    segments.append((start, len(params)))
    return segments


def _param_ok(p, device, dtype, min_8bit):
    if type(p.data) is not torch.Tensor or p.device != device or p.dtype != dtype:
        return False
    n = p.numel()
    if n < min_8bit or n % _BLOCK or not p.is_contiguous():
        return False
    g = p.grad
    return (
        g is not None
        and type(g) is torch.Tensor
        and not g.is_sparse
        and g.dtype == p.dtype
        and g.device == p.device
        and g.shape == p.shape
    )


def _state_ok(p, state, step):
    """A complete stock blockwise 8-bit state on p's device, with the group's int step counter."""
    if not state.keys() <= _STATE_ALLOWED or len(state) != len(_STATE_ALLOWED):
        return False
    if type(state["step"]) is not int or state["step"] != step:
        return False
    n = p.numel()
    for key in _TRACKED_KEYS:
        t = state[key]
        if type(t) is not torch.Tensor or t.device != p.device:
            return False
        if key.startswith("state"):
            if t.dtype != torch.uint8 or t.shape != p.shape:
                return False
        elif key.startswith("absmax"):
            if t.dtype != torch.float32 or t.numel() != n // _BLOCK or not t.is_contiguous():
                return False
    return True


def _live_plans(optimizer):
    """Read-only check of the whole optimizer, every step, before anything is allocated. Returns
    [(gindex, group, plan, fresh)] for the fused update, or None when every group must run stock."""
    if not flat_adamw8bit_enabled() or _unsupported_args(optimizer) or _wrapped_instance(optimizer):
        return None
    mng = optimizer.mng
    min_8bit = getattr(optimizer.args, "min_8bit_size", 4096)
    plans = optimizer.__dict__.setdefault(_PLANS, {})
    ready, storages, device = [], set(), None
    for gindex, group in enumerate(optimizer.param_groups):
        params = group["params"]
        if not params:
            continue  # Trainer keeps an empty no-decay group for LoRA
        if device is None:
            device = params[0].device
            if device.type != _DEVICE_TYPE or not _on_default_stream(device):
                return None
        dtype = params[0].dtype
        for pindex, p in enumerate(params):
            if not _param_ok(p, device, dtype, min_8bit):
                return None
            if (gindex, pindex) in mng.index2config or id(p) in mng.pid2config:
                return None
            # Duplicates and parameters sharing storage would be gathered before either is written.
            key = p.untyped_storage().data_ptr()
            if key in storages:
                return None
            storages.add(key)
        states = [optimizer.state.get(p) for p in params]
        fresh = not any(states)
        if not fresh:
            if not all(states):
                return None
            step = states[0].get("step")
            if type(step) is not int or not all(
                _state_ok(p, s, step) for p, s in zip(params, states)
            ):
                return None
        plan = plans.get(gindex)
        if (
            plan is None
            or len(plan["params"]) != len(params)
            or any(a is not b for a, b in zip(plan["params"], params))
            or plan["shapes"] != tuple(p.shape for p in params)
        ):
            plan = plans[gindex] = {
                "params": tuple(params),
                "shapes": tuple(p.shape for p in params),
                "segments": _segments(params),
                "packed": {},
            }
        ready.append((gindex, group, plan, fresh))
    return ready or None


def _segment_current(optimizer, plan, si):
    packed = plan["packed"].get(si)
    if packed is None:
        return False
    start, end = plan["segments"][si]
    for p, refs in zip(plan["params"][start:end], packed["refs"]):
        state = optimizer.state[p]
        if any(state[key] is not ref() for key, ref in zip(_TRACKED_KEYS, refs)):
            return False
    return True


def _pack_segment(optimizer, plan, si):
    """One buffer per state key for the chunk, every parameter's state rebound as views into it.
    All four buffers are allocated before anything is rebound: on OOM nothing changes."""
    start, end = plan["segments"][si]
    seg = plan["params"][start:end]
    states = [optimizer.state[p] for p in seg]
    first = states[0]
    for state in states[1:]:
        for key in _QMAP_KEYS:
            if state[key] is not first[key] and not torch.equal(state[key], first[key]):
                return False
    total = sum(p.numel() for p in seg)
    device = seg[0].device
    try:
        flat = {
            "state1": torch.empty(total, dtype = torch.uint8, device = device),
            "state2": torch.empty(total, dtype = torch.uint8, device = device),
            "absmax1": torch.empty(total // _BLOCK, dtype = torch.float32, device = device),
            "absmax2": torch.empty(total // _BLOCK, dtype = torch.float32, device = device),
        }
    except _OOM:
        return False
    views, offset = [], 0
    for p, state in zip(seg, states):
        n = p.numel()
        mine = {}
        for key in _STATE_KEYS:
            div = 1 if key.startswith("state") else _BLOCK
            view = flat[key][offset // div : (offset + n) // div]
            view.copy_(state[key].reshape(-1))
            mine[key] = view.view(state[key].shape)
        views.append(mine)
        offset += n
    for state, mine in zip(states, views):
        state.update(mine)
    plan["packed"][si] = {
        "flat": flat,
        "refs": [tuple(weakref.ref(state[key]) for key in _TRACKED_KEYS) for state in states],
    }
    return True


def _materialize(optimizer, ready):
    """Initialise empty groups (as the stock step would) and pack stale chunks. False: run stock."""
    for gindex, group, plan, fresh in ready:
        if fresh:
            for pindex, p in enumerate(group["params"]):
                optimizer.init_state(group, p, gindex, pindex)
            if not all(_state_ok(p, optimizer.state[p], 0) for p in group["params"]):
                return False
    for gindex, group, plan, fresh in ready:
        for si, (start, end) in enumerate(plan["segments"]):
            if end - start == 1:
                plan["packed"].pop(si, None)
            elif not _segment_current(optimizer, plan, si) and not _pack_segment(
                optimizer, plan, si
            ):
                plan["packed"].pop(si, None)
                return False
    return True


def _fused_group(optimizer, gindex, group, plan):
    params = plan["params"]
    for si, (start, end) in enumerate(plan["segments"]):
        seg = params[start:end]
        if end - start == 1:
            # A lone tensor gains nothing from a copy: bitsandbytes' update on the parameter itself.
            optimizer.update_step(group, seg[0], gindex, start)
            continue
        flat = plan["packed"][si]["flat"]
        first = optimizer.state[seg[0]]
        values = torch.cat([p.detach().reshape(-1) for p in seg])
        values.grad = torch.cat([p.grad.reshape(-1) for p in seg])
        state = {"step": first["step"], **flat, "qmap1": first["qmap1"], "qmap2": first["qmap2"]}
        # bitsandbytes' own update_step (its kernel call for the installed version) on the chunk.
        optimizer.state[values] = state
        try:
            optimizer.update_step(group, values, gindex, start)
        finally:
            del optimizer.state[values]
            values.grad = None
        torch._foreach_copy_(
            [p.detach().view(-1) for p in seg], list(values.split([p.numel() for p in seg]))
        )
        for p in seg:
            optimizer.state[p]["step"] = state["step"]


_ANNOUNCED = False


def _announce(ready):
    global _ANNOUNCED
    if _ANNOUNCED:
        return
    _ANNOUNCED = True
    tensors = sum(len(plan["params"]) for _, _, plan, _ in ready)
    updates = sum(len(plan["segments"]) for _, _, plan, _ in ready)
    print(
        f"Unsloth: 8-bit AdamW steps {tensors} tensors in {updates} fused updates "
        f"({_ENV}=0 turns this off)."
    )


def _flat_class(base):
    cls = _CLASSES.get(base)
    if cls is not None:
        return cls
    stock_step = _stock_step_for(base)
    if stock_step is None:
        return None

    @torch.no_grad()
    def step(self, closure = None):
        if torch.compiler.is_compiling() or not flat_adamw8bit_enabled():
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
        ready = _live_plans(self)
        if ready is None or not _materialize(self, ready):
            stock_step(self)
            return loss
        for gindex, group, plan, _ in ready:
            _fused_group(self, gindex, group, plan)
        steps = self.__dict__[_FUSED_STEPS] = self.__dict__.get(_FUSED_STEPS, 0) + 1
        if steps == 1:
            _announce(ready)
        return loss

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.__dict__.pop(_PLANS, None)
        return base.load_state_dict(self, state_dict, *args, **kwargs)

    def add_param_group(self, param_group):
        self.__dict__.pop(_PLANS, None)
        return base.add_param_group(self, param_group)

    def __reduce_ex__(self, protocol):
        # Pickles as the stock class (the generated one is not importable by name).
        return copyreg._reconstructor, (base, object, None), base.__getstate__(self)

    step = torch.optim.Optimizer.profile_hook_step(step)
    step.hooked = True
    cls = type(
        base.__name__,
        (base,),
        {
            "step": step,
            "load_state_dict": load_state_dict,
            "add_param_group": add_param_group,
            "__reduce_ex__": __reduce_ex__,
            "__module__": base.__module__,
            "__qualname__": base.__qualname__,
            "_unsloth_flat_adamw8bit": True,
            "_unsloth_flat_base": base,
        },
    )
    _CLASSES[base] = cls
    return cls


def _rebind_scheduler_step(optimizer):
    """An LR scheduler built earlier wrapped the old class step on the instance; repoint it."""
    wrapped = vars(optimizer).get("step")
    if wrapped is None or not getattr(wrapped, "_wrapped_by_lr_sched", False):
        return
    opt_ref = weakref.ref(optimizer)
    func = type(optimizer).step

    def wrapper(*args, **kwargs):
        opt = opt_ref()
        opt._opt_called = True
        return func.__get__(opt, opt.__class__)(*args, **kwargs)

    wrapper.__wrapped__ = func
    wrapper._wrapped_by_lr_sched = True
    optimizer.step = wrapper


def install_flat_adamw8bit(optimizer):
    """Switch a stock bitsandbytes 8-bit Adam / AdamW to the chunked update; returns the optimizer
    it changed, or None. The class keeps its name and module, so type checks still see bitsandbytes."""
    if not flat_adamw8bit_enabled() or optimizer is None:
        return None
    inner = _innermost(optimizer)
    if getattr(type(inner), "_unsloth_flat_adamw8bit", False):
        return inner
    if _install_refusal(inner) is not None:
        return None
    cls = _flat_class(type(inner))
    if cls is None:
        return None
    inner.__class__ = cls
    inner.__dict__.pop(_PLANS, None)
    _rebind_scheduler_step(inner)
    return inner


def _native_fsdp2(model):
    for m in (model, getattr(model, "module", None), getattr(model, "_orig_mod", None)):
        if m is not None and getattr(m, "_unsloth_fsdp2_native", False):
            return True
    return False


def _distributed_wrapper(trainer):
    """Whether the trainer shards or wraps the optimizer itself: FSDP (any version, incl. XLA FSDP
    and Unsloth's native FSDP2) or DeepSpeed. FSDP2 localizes only the plain bitsandbytes class
    (fsdp2_bnb_optim) and DeepSpeed builds its own optimizer, so the class is left stock there."""
    for name in ("is_deepspeed_enabled", "is_fsdp_enabled", "is_fsdp_xla_enabled"):
        if getattr(trainer, name, False):
            return True
    if _native_fsdp2(getattr(trainer, "model", None)):
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
    return _truthy("ACCELERATE_USE_FSDP") or _truthy("ACCELERATE_USE_DEEPSPEED")


def install_flat_adamw8bit_hook(trainer):
    """Switch the trainer's optimizer right after create_optimizer builds it: before
    Accelerator.prepare (which, with an fp16 GradScaler, captures and later restores the bound step)
    and before the LR scheduler wraps it. Not under FSDP / DeepSpeed or optimizer state offload."""
    if not flat_adamw8bit_enabled() or _distributed_wrapper(trainer) or _offload_requested():
        return False
    create_optimizer = getattr(trainer, "create_optimizer", None)
    if create_optimizer is None:
        return False
    if getattr(create_optimizer, "_unsloth_flat_adamw8bit", False):
        return True

    def _create_optimizer_flat_adamw8bit(*args, **kwargs):
        result = create_optimizer(*args, **kwargs)
        optimizer = result if isinstance(result, torch.optim.Optimizer) else None
        optimizer = optimizer if optimizer is not None else getattr(trainer, "optimizer", None)
        try:
            if not _distributed_wrapper(trainer) and not _offload_requested():
                install_flat_adamw8bit(optimizer)
        except Exception:
            pass
        return result

    _create_optimizer_flat_adamw8bit._unsloth_flat_adamw8bit = True
    _create_optimizer_flat_adamw8bit.__wrapped__ = create_optimizer
    trainer.create_optimizer = _create_optimizer_flat_adamw8bit
    return True
