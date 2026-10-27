# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bitsandbytes 8-bit (and 32-bit) Adam / AdamW under native FSDP2, stepped on the local shards.

bitsandbytes' kernels take plain tensors, and native FSDP2 parameters are DTensors. Each DTensor
parameter's local shard is an ordinary tensor sharing the parameter's storage, and its gradient's local
shard is the matching ordinary tensor, so bitsandbytes can step the shard in place. The optimizer
keeps the DTensor parameters in its param_groups and its state keyed by them (the LR scheduler, the
Trainer, gradient clipping, GradScaler and torch.distributed.checkpoint all see the same objects); only
init_state / update_step hand bitsandbytes the local shards.

The 8-bit state is quantized per 256-element block of each LOCAL shard, so it is not bitwise equal to
one GPU, where the blocks run over the whole tensor. Which parameters get 8-bit state follows one GPU:
bitsandbytes' `min_8bit_size` rule is applied to the full parameter size, not the shard size.

state_dict() returns the state dequantized to float32, wrapped as DTensors laid out like each parameter,
so the collective full-state save (get_optimizer_state_dict(full_state_dict = True)) gathers a
world-size-independent checkpoint. load_state_dict() re-quantizes it: the inverse is exact when the
blocks are unchanged (same world size), so a resume restores the same 8-bit state.
"""

import contextlib
import functools
import os

import torch

__all__ = [
    "bnb_local_shards_enabled",
    "full_cpu_state_export",
    "is_local_shard_bnb_optimizer",
    "localize_bnb_optimizer",
    "supports_local_shards",
    "rename_optimizer_state_for",
]

_BLOCKSIZE = 256  # bitsandbytes Optimizer2State 8-bit blockwise state block size
_FLAG = "_unsloth_fsdp2_local_shards"
_STATE_KEYS = ("state1", "state2")
# Per-parameter history vectors bitsandbytes keeps next to the moments: gnorm_vec (percentile_clipping
# < 100: the last 100 squared gradient norms, of this rank's shard) and unorm_vec (max_unorm > 0).
_AUX_KEYS = {"gnorm_vec": 100, "unorm_vec": 1}
_TORCH_ADAM_KEYS = {"state1": "exp_avg", "state2": "exp_avg_sq"}


def bnb_local_shards_enabled():
    """UNSLOTH_FSDP2_BNB_LOCAL=0 restores the old behaviour (bitsandbytes Adam switched to torch's)."""
    value = os.environ.get("UNSLOTH_FSDP2_BNB_LOCAL", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def _dtensor_class():
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return None
    return DTensor


def _bnb_base_classes():
    try:
        from bitsandbytes.optim.optimizer import Optimizer2State
    except Exception:
        return None
    return Optimizer2State


def supports_local_shards(optimizer_or_cls):
    """bitsandbytes Adam / AdamW (8-bit blockwise, paged or not, or 32-bit): Optimizer2State with
    optimizer_name 'adam' (AdEMAMix keeps two first moments in one state and is not covered). The class
    check alone when given a class."""
    base = _bnb_base_classes()
    if base is None:
        return False
    cls = optimizer_or_cls if isinstance(optimizer_or_cls, type) else type(optimizer_or_cls)
    if not issubclass(cls, base):
        return False
    if "ademamix" in cls.__name__.lower():
        return False
    if not _plain_bnb_class(cls, base):
        return False
    if isinstance(optimizer_or_cls, type):
        return True
    if getattr(optimizer_or_cls, "optimizer_name", None) != "adam":
        return False
    args = getattr(optimizer_or_cls, "args", None)
    if args is not None and int(getattr(args, "optim_bits", 32)) == 8:
        # 8-bit non-blockwise state (max1 / new_max1) is per tensor: not covered.
        return bool(getattr(args, "block_wise", True))
    return True


def _plain_bnb_class(cls, base):
    """Only bitsandbytes' own Adam / AdamW / PagedAdam(W) classes: every class above Optimizer2State
    comes from bitsandbytes and none replaces step / init_state / update_step. A subclass elsewhere
    (Unsloth's QGaLoreAdamW8bit: optimizer_name 'adam', its own step() projecting whole 2-D weights)
    cannot be stepped shard by shard."""

    def unwrapped(fn):
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return fn

    for klass in cls.__mro__:
        if klass is base:
            return True
        module = str(getattr(klass, "__module__", "") or "")
        if module != "bitsandbytes" and not module.startswith("bitsandbytes."):
            return False
        for name in ("step", "init_state", "update_step"):
            # torch.optim.Optimizer.__init__ rewraps the class's inherited step for profiling
            # (`step.hooked`), so compare the unwrapped function with the base's.
            if name in vars(klass) and unwrapped(vars(klass)[name]) is not unwrapped(
                getattr(base, name)
            ):
                return False
    return False


@functools.lru_cache(maxsize = None)
def bnb_steps_cpu_tensors():
    """Whether this bitsandbytes steps 8-bit blockwise and 32-bit Adam on CPU tensors (0.50 does; older
    releases require GPU inputs). Probed once with one step on a tiny CPU parameter of each."""
    try:
        import bitsandbytes as bnb
        for kwargs in ({"optim_bits": 8, "min_8bit_size": 0}, {"optim_bits": 32}):
            p = torch.nn.Parameter(torch.zeros(256))
            p.grad = torch.ones(256)
            bnb.optim.AdamW([p], lr = 1e-3, **kwargs).step()
            if not torch.isfinite(p).all():
                return False
    except Exception:
        return False
    return True


def is_local_shard_bnb_optimizer(optimizer):
    return getattr(type(optimizer), _FLAG, False) is True


def _local(p, DTensor):
    """The plain local tensor of a DTensor parameter (shares its storage), else None."""
    if DTensor is not None and isinstance(p, DTensor):
        return p._local_tensor
    return None


def _local_grad(grad, DTensor):
    if DTensor is not None and isinstance(grad, DTensor):
        return grad._local_tensor
    return grad


# Blocks per pass when (de)quantizing a checkpointed state: 16384 blocks = 4M elements, so the
# temporaries (a float table and int64 indices per candidate scale) stay a few hundred MB however large
# the shard is. Every block is independent, so the result does not depend on the chunking.
_CHUNK_BLOCKS = 16384


def _dequantize(
    raw,
    key,
    device,
    chunk_blocks = None,
):
    """float32 copy of state `key` on `device`, built chunk by chunk into the one output tensor. Paged
    state reports the CPU device (unified memory)."""
    value = raw[key]
    if value.dtype != torch.uint8:
        return value.to(device = device, dtype = torch.float32)
    index = key[-1]
    code = raw[f"qmap{index}"].to(device = device, dtype = torch.float32)
    absmax = raw[f"absmax{index}"]
    flat_codes = value.reshape(-1)
    n = flat_codes.numel()
    out = torch.empty(n, dtype = torch.float32, device = device)
    step = (chunk_blocks or _CHUNK_BLOCKS) * _BLOCKSIZE
    for start in range(0, n, step):
        end = min(start + step, n)
        codes = flat_codes[start:end].to(device).long()
        scale = absmax[start // _BLOCKSIZE : (end + _BLOCKSIZE - 1) // _BLOCKSIZE]
        scale = scale.to(device = device, dtype = torch.float32).repeat_interleave(_BLOCKSIZE)
        torch.mul(code[codes], scale[: end - start], out = out[start:end])
    return out.reshape(value.shape)


def _nearest_codes(padded, sorted_code, scale):
    """Per block: index into sorted_code of the nearest code * scale for each element, and the block's
    total absolute reconstruction error."""
    table = sorted_code.unsqueeze(0) * scale.unsqueeze(1)  # (blocks, 256), ascending per row
    hi = torch.searchsorted(table, padded.contiguous()).clamp_(max = table.shape[1] - 1)
    lo = (hi - 1).clamp_(min = 0)
    err_lo = (padded - table.gather(1, lo)).abs()
    err_hi = (table.gather(1, hi) - padded).abs()
    pick_lo = err_lo < err_hi
    return torch.where(pick_lo, lo, hi), torch.where(pick_lo, err_lo, err_hi).sum(dim = 1)


def _quantize_into(
    values,
    code,
    out_codes,
    out_absmax,
    chunk_blocks = None,
):
    """bitsandbytes' blockwise quantization of `values` (float32) into `out_codes` (uint8, same shape)
    and `out_absmax` (one float32 per 256 elements), inverting a dequantization exactly. bitsandbytes
    scales a block by its largest magnitude A, and the signed map has no -1.0: when that element is
    negative it is stored as min_code * A. The scale is therefore the largest positive value, or the
    largest |negative| / |min code| (with its float32 neighbours, since that division can round), and
    each element maps to the nearest code * scale; per block the candidate with the smallest
    reconstruction error wins (zero for the original scale). Runs `chunk_blocks` blocks at a time."""
    device = out_codes.device
    code = code.to(device = device, dtype = torch.float32)
    flat = values.reshape(-1)
    n = flat.numel()
    if n == 0:
        return
    sorted_code, order = torch.sort(code)
    min_code = float(sorted_code[0].item())
    codes_out = out_codes.view(-1)
    step = (chunk_blocks or _CHUNK_BLOCKS) * _BLOCKSIZE
    for start in range(0, n, step):
        end = min(start + step, n)
        chunk = flat[start:end].to(device = device, dtype = torch.float32)
        blocks = (end - start + _BLOCKSIZE - 1) // _BLOCKSIZE
        if chunk.numel() != blocks * _BLOCKSIZE:  # the ragged tail block, zero padded
            chunk = torch.nn.functional.pad(chunk, (0, blocks * _BLOCKSIZE - chunk.numel()))
        index, scale = _quantize_blocks(chunk.view(blocks, _BLOCKSIZE), sorted_code, min_code)
        codes_out[start:end].copy_(order[index].reshape(-1)[: end - start].to(torch.uint8))
        out_absmax[start // _BLOCKSIZE : start // _BLOCKSIZE + blocks].copy_(
            scale.to(out_absmax.dtype)
        )


def _quantize_blocks(padded, sorted_code, min_code):
    """(blocks, 256) float32 -> (index into sorted_code per element, scale per block)."""
    positive = padded.clamp(min = 0).amax(dim = 1)
    candidates = [positive]
    negative = None
    if min_code < 0:
        negative = (-padded).clamp(min = 0).amax(dim = 1)
        guess = negative / (-min_code)
        candidates += [
            guess,
            torch.nextafter(guess, torch.full_like(guess, float("inf"))),
            torch.nextafter(guess, torch.zeros_like(guess)),
        ]
    best_scale = best_index = best_err = None
    for scale in candidates:
        index, err = _nearest_codes(padded, sorted_code, scale)
        valid = scale >= positive
        if negative is not None:
            valid = valid & (scale * (-min_code) >= negative)
        err = torch.where(valid, err, torch.full_like(err, float("inf")))
        if best_err is None:
            best_scale, best_index, best_err = scale, index, err
            continue
        better = err < best_err
        best_scale = torch.where(better, scale, best_scale)
        best_index = torch.where(better.unsqueeze(1), index, best_index)
        best_err = torch.where(better, err, best_err)
    return best_index, best_scale


class _LocalShardMixin:
    """Overrides for a bitsandbytes Optimizer2State whose parameters may be DTensors."""

    _unsloth_fsdp2_local_shards = True

    def get_config(self, gindex, pindex, group):
        config = super().get_config(gindex, pindex, group)
        local = _local(group["params"][pindex], _dtensor_class())
        if local is not None:
            full = group["params"][pindex].numel()
            # One GPU picks 8-bit state for numel >= min_8bit_size: decide on the full size.
            config["min_8bit_size"] = 0 if full >= config["min_8bit_size"] else local.numel() + 1
        return config

    def _shard_call(self, method, group, p, gindex, pindex, with_grad):
        DTensor = _dtensor_class()
        local = _local(p, DTensor)
        if local is None:
            return getattr(super(), method)(group, p, gindex, pindex)
        state = self.state.pop(p, {})
        if local.numel() == 0:
            state.setdefault("step", 0)
            if method == "update_step":
                state["step"] += 1
            self.state[p] = state
            return None
        work = local if local.is_contiguous() else local.contiguous()
        alias = work.detach()
        if with_grad:
            alias.grad = _local_grad(p.grad, DTensor).detach()
        self.state[alias] = state
        try:
            return getattr(super(), method)(group, alias, gindex, pindex)
        finally:
            self.state[p] = self.state.pop(alias)
            alias.grad = None
            if work is not local:
                local.copy_(work)

    @torch.no_grad()
    def step(self, closure = None):
        """bitsandbytes' Optimizer8bit.step, but per-parameter device syncs only for paged optimizers
        (async prefetches) or off the default stream."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True
        paged = bool(getattr(self, "is_paged", False))
        fence = paged or _off_default_stream(self)
        p = None
        for gindex, group in enumerate(self.param_groups):
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                if len(self.state[p]) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.prefetch_state(p)
                self.update_step(group, p, gindex, pindex)
                if fence:
                    _sync_device(p)
        if fence and p is not None:
            _sync_device(p)
        return loss

    def init_state(self, group, p, gindex, pindex):
        return self._shard_call("init_state", group, p, gindex, pindex, with_grad = False)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        return self._shard_call("update_step", group, p, gindex, pindex, with_grad = True)

    def prefetch_state(self, p):
        if "state1" not in self.state.get(p, {}):
            return None
        return super().prefetch_state(p)

    def _aux_keys(self, group, gindex, pindex):
        """The history vectors bitsandbytes allocates for this parameter (the same on every rank)."""
        config = self.get_config(gindex, pindex, group)
        keys = []
        if config.get("percentile_clipping", 100) < 100:
            keys.append("gnorm_vec")
        if config.get("max_unorm", 0.0) > 0.0:
            keys.append("unorm_vec")
        return keys

    def _export_aux(self, p, raw, keys, out):
        """Each rank's history vectors for a DTensor parameter, as one Shard(0) DTensor of shape
        (world * n,) over the parameter's mesh, so the full checkpoint holds every rank's vector and a
        same-world-size resume gets its own back; a replicated parameter's vector is plain."""
        DTensor = _dtensor_class()
        local = _local(p, DTensor)
        for key in keys:
            n = _AUX_KEYS[key]
            device = (local if local is not None else p).device
            value = raw.get(key)
            if value is None:  # an empty shard never ran init_state
                value = torch.zeros((n,), dtype = torch.float32, device = device)
            value = value.detach().to(device = device, dtype = torch.float32).reshape(n).clone()
            if local is not None:
                mesh = p.device_mesh
                if mesh.ndim != 1:
                    continue  # not produced by native FSDP2 (1-D mesh); resumes from zeros
                from torch.distributed.tensor import Shard

                value = DTensor.from_local(
                    value, mesh, [Shard(0)], run_check = False, shape = (mesh.size() * n,), stride = (1,)
                )
            out[key] = value

    def _export_state(
        self,
        p,
        raw,
        slot = None,
    ):
        DTensor = _dtensor_class()
        local = _local(p, DTensor)
        out = {"step": int(raw.get("step", 0))}
        if slot is not None:
            group, gindex, pindex = slot
            self._export_aux(p, raw, self._aux_keys(group, gindex, pindex), out)
        for key in _STATE_KEYS:
            if key in raw:
                value = _dequantize(raw, key, (local if local is not None else p).device)
            elif local is not None and local.numel() == 0:
                value = torch.zeros(local.shape, dtype = torch.float32, device = local.device)
            else:
                continue
            if local is not None:
                value = DTensor.from_local(
                    value,
                    p.device_mesh,
                    p.placements,
                    run_check = False,
                    shape = p.shape,
                    stride = p.stride(),
                )
            out[key] = value
        return out

    def state_dict(self):
        """float32 state1 / state2 (DTensors for DTensor parameters) and an int step per parameter."""
        state_dict = torch.optim.Optimizer.state_dict(self)
        slots = [
            (p, (group, gindex, pindex))
            for gindex, group in enumerate(self.param_groups)
            for pindex, p in enumerate(group["params"])
        ]
        to_cpu = getattr(self, "_unsloth_full_cpu_export", False)
        exported = {}
        for index, raw in state_dict["state"].items():
            out = self._export_state(slots[index][0], raw, slots[index][1])
            exported[index] = _gather_to_cpu(out) if to_cpu else out
        state_dict["state"] = exported
        return state_dict

    def _import_state(self, group, p, gindex, pindex, saved):
        DTensor = _dtensor_class()
        self.state[p] = {}
        self.init_state(group, p, gindex, pindex)
        raw = self.state[p]
        step = saved.get("step", 0)
        raw["step"] = int(step.item() if torch.is_tensor(step) else step)
        for key in _STATE_KEYS:
            value = saved.get(key, saved.get(_TORCH_ADAM_KEYS[key]))
            if value is None or key not in raw:
                continue
            value = _local_grad(value, DTensor)
            target = raw[key]
            if value.shape != target.shape:
                raise ValueError(
                    f"Unsloth: optimizer state '{key}' has shape {tuple(value.shape)}, but this rank's "
                    f"shard of the parameter has shape {tuple(target.shape)}."
                )
            if target.dtype == torch.uint8:
                index = key[-1]
                _quantize_into(value, raw[f"qmap{index}"], target, raw[f"absmax{index}"])
            else:
                target.copy_(value.to(device = target.device, dtype = target.dtype))
        for key in _AUX_KEYS:
            value = saved.get(key)
            if value is None or key not in raw:
                continue
            value = _local_grad(value, DTensor)
            if value.numel() == raw[key].numel():  # another world size: the vector starts fresh
                raw[key].copy_(value.reshape(raw[key].shape).to(raw[key].device, raw[key].dtype))

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Inverse of state_dict(); also reads exp_avg / exp_avg_sq (a torch Adam checkpoint)."""
        saved_groups = state_dict["param_groups"]
        torch.optim.Optimizer.load_state_dict(self, {"state": {}, "param_groups": saved_groups})
        slots = [
            (group, p, gindex, pindex)
            for gindex, group in enumerate(self.param_groups)
            for pindex, p in enumerate(group["params"])
        ]
        flat_saved = [saved_id for g in saved_groups for saved_id in g["params"]]
        slot_of = dict(zip(flat_saved, slots))
        with torch.no_grad():
            for saved_id, saved in state_dict["state"].items():
                if saved_id in slot_of:
                    self._import_state(*slot_of[saved_id], saved)


def _gather_to_cpu(entry):
    """Mirrors torch _gather_state_dict + cpu_offload for one parameter: CPU on rank 0, None elsewhere."""
    DTensor = _dtensor_class()
    dist = torch.distributed
    keep = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    out = {}
    for key, value in entry.items():
        if DTensor is not None and isinstance(value, DTensor):
            from torch.distributed.tensor import Replicate

            mesh = value.device_mesh
            if value.device.type != mesh.device_type:
                value = value.to(mesh.device_type)
            value = value.redistribute(
                device_mesh = mesh, placements = [Replicate() for _ in value.placements]
            ).to_local()
            try:
                from torch.distributed._functional_collectives import AsyncCollectiveTensor
            except ImportError:
                AsyncCollectiveTensor = None
            if AsyncCollectiveTensor is not None and isinstance(value, AsyncCollectiveTensor):
                value = value.wait()
        if torch.is_tensor(value):
            value = value.to("cpu") if keep else None
        out[key] = value
    return out


@contextlib.contextmanager
def full_cpu_state_export(optimizer):
    """Gather + offload each parameter's dequantized state as it is exported, instead of holding all
    float32 moments on the device until torch gathers. Every rank must enter it (collectives)."""
    if not is_local_shard_bnb_optimizer(optimizer):
        yield
        return
    optimizer._unsloth_full_cpu_export = True
    try:
        yield
    finally:
        optimizer._unsloth_full_cpu_export = False


def _off_default_stream(optimizer):
    devices = {p.device for group in optimizer.param_groups for p in group["params"]}
    for device in devices:
        if device.type == "cpu":
            continue
        if device.type != "cuda":
            return True
        if torch.cuda.current_stream(device) != torch.cuda.default_stream(device):
            return True
    return False


def _sync_device(p):
    try:
        from bitsandbytes.utils import sync_gpu
    except ImportError:
        sync_gpu = None
    if sync_gpu is not None:
        return sync_gpu(p)
    if p.device.type == "cuda":
        torch.cuda.synchronize()
    elif p.device.type == "xpu":
        torch.xpu.synchronize()


@functools.lru_cache(maxsize = None)
def _local_shard_class(cls):
    namespace = {"__module__": cls.__module__}
    hook_step = getattr(torch.optim.Optimizer, "profile_hook_step", None)
    if hook_step is not None:
        # The mixin's step shadows torch's hooked step: re-wrap so step hooks still run once.
        step = hook_step(_LocalShardMixin.step)
        step.hooked = True
        namespace["step"] = step
    return type(cls.__name__, (_LocalShardMixin, cls), namespace)


def localize_bnb_optimizer(optimizer):
    """Make a bitsandbytes Adam / AdamW step DTensor parameters on their local shards (class swap,
    in place). Returns True when converted (or already converted)."""
    if is_local_shard_bnb_optimizer(optimizer):
        return True
    if not supports_local_shards(optimizer):
        return False
    if any(len(s) for s in optimizer.state.values()):
        raise RuntimeError(
            "Unsloth: native FSDP2 got a bitsandbytes optimizer that already holds state for DTensor "
            "parameters; build it after `shard_model(model)` and before training."
        )
    optimizer.__class__ = _local_shard_class(type(optimizer))
    _rebind_scheduler_step(optimizer)
    return True


def _rebind_scheduler_step(optimizer):
    """A scheduler built before localization wrapped the old class step on the instance; repoint it."""
    wrapped = vars(optimizer).get("step")
    if wrapped is None or not getattr(wrapped, "_wrapped_by_lr_sched", False):
        return
    import weakref

    opt_ref = weakref.ref(optimizer)
    func = type(optimizer).step

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        opt = opt_ref()
        opt._opt_called = True
        return func.__get__(opt, opt.__class__)(*args, **kwargs)

    wrapper._wrapped_by_lr_sched = True
    optimizer.step = wrapper


def rename_optimizer_state_for(optimizer, full_state):
    """Full (FQN-keyed) optimizer state: map torch Adam keys (exp_avg / exp_avg_sq) to the local-shard
    bitsandbytes keys (state1 / state2) or back, to resume across the two optimizers."""
    states = full_state.get("state") if isinstance(full_state, dict) else None
    if not isinstance(states, dict):
        return full_state
    to_bnb = is_local_shard_bnb_optimizer(optimizer)
    to_torch = isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
    if not (to_bnb or to_torch):
        return full_state
    renamed = False
    for entry in states.values():
        if not isinstance(entry, dict):
            continue
        for bnb_key, torch_key in _TORCH_ADAM_KEYS.items():
            if to_bnb and torch_key in entry and bnb_key not in entry:
                entry[bnb_key] = entry.pop(torch_key).float()
                renamed = True
            elif to_torch and bnb_key in entry and torch_key not in entry:
                entry[torch_key] = entry.pop(bnb_key)
                renamed = True
        step = entry.get("step")
        if to_bnb and torch.is_tensor(step):
            entry["step"] = int(step.item())
        elif to_torch and step is not None and not torch.is_tensor(step):
            entry["step"] = torch.tensor(float(step))
    if renamed:
        _align_param_group_keys(optimizer, full_state)
    return full_state


def _align_param_group_keys(optimizer, full_state):
    """Saved param groups from the other optimizer carry its own hyperparameter keys (torch AdamW's
    amsgrad / foreach / ..., bitsandbytes' alpha / t_alpha / ...), which torch's state dict loaders
    refuse. Keep the keys both optimizers have (lr, betas, eps, weight_decay: the saved values), drop
    the saved-only ones and take this optimizer's own value for the rest. Only called for a
    checkpoint of the other optimizer (its state keys were renamed)."""
    saved_groups = full_state.get("param_groups")
    live_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(saved_groups, list) or not live_groups:
        return
    if len(saved_groups) != len(live_groups):
        return  # torch's loader reports the group mismatch itself
    for saved, live in zip(saved_groups, live_groups):
        if not isinstance(saved, dict):
            continue
        for key in [k for k in saved if k != "params" and k not in live]:
            del saved[key]
        for key, value in live.items():
            if key != "params" and key not in saved:
                saved[key] = value
