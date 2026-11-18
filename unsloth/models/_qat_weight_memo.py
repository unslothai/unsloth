# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Fake quantize a frozen QAT base weight once instead of on every forward.

With LoRA + QAT the base weight never changes, yet torchao's weight fake quantizer recomputes
fake_quant(W) on every call: the forward, Unsloth's gradient checkpointing replay and every
gradient accumulation micro batch, each a handful of elementwise passes over the whole base model.
The fake quantizer is a pure function of W (and its own config), so its output is cached per
quantizer and served again while W is the same, unmodified tensor. The cached tensor is the
quantizer's own output, so losses and gradients are bitwise identical.

The cache costs one extra copy of each cached weight, and a run that fits without it must still
fit with it. So nothing is cached before the first optimizer step: that step runs exactly as
without the cache and shows the run's own peak. After every optimizer step the cache may hold only
the memory that stays free at the highest peak seen so far, minus what a full length batch
(batch rows x max_seq_length) would add over the longest batch seen, minus a margin; a cache over
that limit is evicted. Eval mode (model.eval(), Trainer evaluation, for_inference) drops the whole
cache and passes through, so evaluation and generation have the memory they had before. A CUDA out
of memory error anywhere turns the cache off for the rest of the process. UNSLOTH_QAT_WEIGHT_MEMO=1
lifts the limit, =0 turns the cache off. A weight is cached on its second call with the same
tensor, so weights streamed in per forward (offloading) are never held. Skipped: weights that require grad (full
fine-tuning), range learning, tensor subclasses / DTensors, multi-process runs (FSDP reshards
weights), inference-mode calls (whose outputs autograd cannot save) and torch.compile traces: a
compiled region (torch >= 2.11 block compile) keeps fake quantizing inside its graph, unchanged.
PEFT merge / unmerge write the weight through .data (no version bump) and drop the cache."""

import os
import weakref

import torch

__all__ = [
    "install_qat_weight_memo",
    "clear_qat_weight_memo",
    "qat_weight_memo_stats",
]

_ENV = "UNSLOTH_QAT_WEIGHT_MEMO"
_STATS = {
    "installed": 0,
    "hits": 0,
    "stores": 0,
    "over_budget": 0,
    "evictions": 0,
    "eval_drops": 0,
}
# device -> [limit bytes, used bytes]
_BUDGET = {}
# device -> {"main_peak": peak reserved bytes the run needs beside the cache, "cum_peak": last seen
# max_memory_reserved, "max_used": most cache held since the last optimizer step}
_DEVICE = {}
# Process state: armed after the first optimizer step; disabled for good after an out of memory error.
_STATE = {
    "armed": False,
    "disabled": False,
    "steps": 0,
    "live": 0,
    "hook": False,
    "oom_observer": False,
    "tokens": 0,
    "rows": 0,
    "row_probes": 0,
    "max_seq": None,
    "quadratic": False,
}
_MEMOS = weakref.WeakSet()
_MARGIN_MIN = 512 * 1024 * 1024
_MARGIN_FRACTION = 0.05
_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _mode():
    return os.environ.get(_ENV, "auto").strip().lower()


def _multi_process():
    dist = torch.distributed
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _unlimited():
    return _mode() in ("1", "true", "on")


def _bounded(device):
    """Accelerators have a cache limit; host memory does not."""
    return device.type != "cpu"


def _device_stats(device):
    """(free, total, reserved, allocated, peak reserved) bytes of an accelerator, None if unknown."""
    backend = getattr(torch, device.type, None)
    try:
        free, total = backend.mem_get_info(device)
        return (
            free,
            total,
            backend.memory_reserved(device),
            backend.memory_allocated(device),
            backend.max_memory_reserved(device),
        )
    except Exception:
        return None


def _length_scale():
    """How much larger than the largest training batch seen so far a batch may get (>= 1)."""
    seen = _STATE["tokens"]
    if seen <= 0:
        return 2.0
    max_seq = _STATE["max_seq"]
    if not max_seq:
        return 2.0
    possible = max(seen, max(_STATE["rows"], 1) * max_seq)
    scale = possible / seen
    return scale * scale if _STATE["quadratic"] else scale


def _cache_limit(free, total, reserved, allocated, used, main_peak, scale):
    """Bytes the cache may hold: what stays free at the run's own peak, minus the extra a full length
    batch would need (the observed activation swing scaled up), minus a margin."""
    cap = free + reserved  # everything this process can reach now
    base = allocated - used  # model, optimizer state and gradients without the cache
    swing = max(main_peak - base, 0)
    extra = swing * max(scale - 1.0, 0.0)
    margin = max(_MARGIN_MIN, int(total * _MARGIN_FRACTION))
    return max(int(cap - main_peak - extra - margin), 0)


def _evict(device, target):
    """Drop cached weights on device until at most target bytes stay cached."""
    budget = _BUDGET.get(device)
    if budget is None:
        return
    for memo in list(_MEMOS):
        if budget[1] <= target:
            break
        if memo.out is not None and memo.key[2] == device:
            memo.drop()
            _STATS["evictions"] += 1


def _drop_all():
    for memo in list(_MEMOS):
        if memo.out is not None:
            memo.drop()
    _STATE["live"] = 0


def _rebudget():
    if _STATE["disabled"]:
        _drop_all()
        return
    unlimited = _unlimited()
    # Recount from the live caches: a deleted model's caches are gone without a drop().
    used = {}
    live = 0
    for memo in list(_MEMOS):
        if memo.key is None:
            continue
        device = memo.key[2]
        used[device] = used.get(device, 0) + (memo.nbytes if memo.out is not None else 0)
        live += memo.out is not None
    _STATE["live"] = live
    for device in list(_BUDGET):
        if device not in used:
            del _BUDGET[device]
    for device in used:
        budget = _BUDGET.setdefault(device, [0, 0])
        budget[1] = used[device]
        if unlimited or not _bounded(device):
            budget[0] = float("inf")
            continue
        stats = _device_stats(device)
        if stats is None:
            budget[0] = 0
            _evict(device, 0)
            continue
        free, total, reserved, allocated, peak = stats
        state = _DEVICE.setdefault(device, {"main_peak": 0, "cum_peak": 0, "max_used": 0})
        if peak > state["cum_peak"]:
            # A new high water mark since the last step. The cache fills during the first forward after
            # a step (every weight was already seen), before the logits / backward / optimizer peaks,
            # so it is full at the peak; eval mode peaks hold no cache and need no room for one.
            state["main_peak"] = max(state["main_peak"], peak - max(state["max_used"], budget[1]))
        state["cum_peak"] = peak
        limit = _cache_limit(
            free, total, reserved, allocated, budget[1], state["main_peak"], _length_scale()
        )
        budget[0] = limit
        if budget[1] > limit:
            _evict(device, limit)
        state["max_used"] = budget[1]


def _steps_before_arming():
    # UNSLOTH_GC_KEEP_LAYERS plans its checkpoint skips from the first step's memory and applies them
    # from the second step on: arm only once a step has run with that plan in place.
    keep = os.environ.get("UNSLOTH_GC_KEEP_LAYERS", "").strip().lower()
    return 1 if keep in ("", "0") else 2


def _on_optimizer_step(optimizer, args, kwargs):
    _STATE["steps"] += 1
    if not _MEMOS or _mode() in ("0", "false", "off"):
        return
    if not _STATE["armed"] and _STATE["steps"] < _steps_before_arming():
        return
    try:
        _rebudget()
        _STATE["armed"] = True
        _attach_oom_observer()
    except Exception:
        _STATE["armed"] = False


def _on_oom(*args):
    # Runs inside the allocator's out of memory path: only flag it, the cache is dropped on the next call.
    _STATE["disabled"] = True


def _attach_oom_observer():
    if _STATE["oom_observer"]:
        return
    _STATE["oom_observer"] = True
    attach = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
    if attach is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        _STATE["oom_observer"] = False
        return
    try:
        attach(_on_oom)
    except Exception:
        pass


def _record_batch(module, args, kwargs):
    """Forward pre hook on the model: largest training batch (tokens, rows) seen so far."""
    if torch.compiler.is_compiling() or not module.training:
        return
    ids = kwargs.get("input_ids", args[0] if args else None)
    if not isinstance(ids, torch.Tensor):
        ids = kwargs.get("inputs_embeds")
        if not isinstance(ids, torch.Tensor) or ids.dim() < 2:
            return
        shape = ids.shape[:-1]
    else:
        shape = ids.shape
    tokens = shape.numel()
    rows = shape[0] if len(shape) > 1 else 1
    position_ids = kwargs.get("position_ids")
    if rows == 1 and isinstance(position_ids, torch.Tensor) and _STATE["row_probes"] < 8:
        # Padding free / packed rows: count the sequences (position 0 starts each), first steps only.
        _STATE["row_probes"] += 1
        try:
            rows = max(int((position_ids.reshape(-1) == 0).sum()), 1)
        except Exception:
            pass
    _STATE["tokens"] = max(_STATE["tokens"], tokens)
    _STATE["rows"] = max(_STATE["rows"], rows)


class _WeightMemo:
    """Per quantizer cache of fake_quant(W) for an unchanged weight; lives in the quantizer's __dict__."""

    __slots__ = ("storage", "key", "out", "nbytes", "__weakref__")

    def __init__(self):
        self.storage = None
        self.key = None
        self.out = None
        self.nbytes = 0
        _MEMOS.add(self)

    def __getstate__(self):
        return {"cached": False}  # truthy, so unpickling calls __setstate__

    def __setstate__(self, state):
        self.__init__()

    def __deepcopy__(self, memo):
        return _WeightMemo()

    def drop(self):
        if self.out is not None:
            _STATE["live"] -= 1
            device = self.key[2]
            budget = _BUDGET.get(device)
            if budget is not None:
                budget[1] -= self.nbytes
        self.storage = self.key = self.out = None
        self.nbytes = 0


def _weight_key(quantizer, w):
    if (
        type(w) not in _PLAIN_TENSOR_TYPES
        or w.requires_grad
        or w.device.type == "meta"
        or not getattr(quantizer, "enabled", True)
        or getattr(getattr(quantizer, "config", None), "range_learning", False)
        or _multi_process()
    ):
        return None
    device_type = w.device.type
    autocast = torch.is_autocast_enabled(device_type)
    return (
        w.storage_offset(),
        tuple(w.shape),
        w.device,
        w.dtype,
        w.stride(),
        w._version,
        autocast,
        torch.get_autocast_dtype(device_type) if autocast else None,
    )


def _memo_forward(quantizer, w, forward):
    memo = quantizer.__dict__.get("_unsloth_qat_weight_memo")
    if memo is None:
        return forward(quantizer, w)
    mode = _mode()
    if mode in ("0", "false", "off"):
        return forward(quantizer, w)
    if _STATE["disabled"] or not quantizer.training:
        # After an out of memory error, or in eval mode (evaluation, generation): free every cached
        # weight so these phases have the memory they had without the cache, and pass through.
        if _STATE["live"]:
            if not _STATE["disabled"]:
                _STATS["eval_drops"] += 1
            _drop_all()
        return forward(quantizer, w)
    key = _weight_key(quantizer, w)
    if key is None:
        if memo.key is not None:
            memo.drop()
        return forward(quantizer, w)
    storage = w.untyped_storage()
    same = memo.key == key and memo.storage is not None and memo.storage() is storage
    if same and memo.out is not None:
        _STATS["hits"] += 1
        return memo.out
    if torch.is_inference_mode_enabled():
        # Inference tensors cannot be saved for backward later: compute, never cache.
        return forward(quantizer, w)
    out = forward(quantizer, w)
    if not same:
        # First sighting of this tensor: remember it; cache on the next call with the same tensor.
        memo.drop()
        try:
            memo.storage = weakref.ref(storage)
        except TypeError:
            return out
        memo.key = key
        return out
    if type(out) is not torch.Tensor or out.requires_grad:
        return out
    unlimited = mode in ("1", "true", "on")
    if not (_STATE["armed"] or unlimited):
        # Nothing is cached before the first optimizer step has shown the run's own peak.
        return out
    nbytes = out.untyped_storage().nbytes()
    budget = _BUDGET.get(key[2])
    if budget is None:
        budget = _BUDGET[key[2]] = [0, 0]
        if not _bounded(key[2]):
            budget[0] = float("inf")
    if unlimited:
        budget[0] = float("inf")
    if budget[1] + nbytes > budget[0]:
        _STATS["over_budget"] += 1
        return out
    budget[1] += nbytes
    memo.out, memo.nbytes = out, nbytes
    _STATE["live"] += 1
    state = _DEVICE.get(key[2])
    if state is not None and budget[1] > state["max_used"]:
        state["max_used"] = budget[1]
    _STATS["stores"] += 1
    return out


def _patch_quantizer_class(cls):
    """Class-level forward wrapper: one function for every instance, so a torch.compile trace sees the
    original forward (no per-instance guard or recompile) and eager calls of uncached instances pass through."""
    owner = next((c for c in cls.__mro__ if "forward" in c.__dict__), None)
    forward = None if owner is None else owner.__dict__["forward"]
    if forward is None or getattr(forward, "_unsloth_qat_weight_memo", False):
        return

    def memo_forward(self, w):
        if torch.compiler.is_compiling():
            return forward(self, w)
        return _memo_forward(self, w, forward)

    memo_forward._unsloth_qat_weight_memo = True
    memo_forward.__wrapped__ = forward
    memo_forward.__name__ = "forward"
    memo_forward.__qualname__ = getattr(forward, "__qualname__", "forward")
    memo_forward.__doc__ = forward.__doc__
    owner.forward = memo_forward


def _is_adapter_name(fqn):
    return any(part.startswith("lora_") for part in fqn.split("."))


def _memo_of(module):
    quantizer = module._modules.get("weight_fake_quantizer") if module is not None else None
    return None if quantizer is None else quantizer.__dict__.get("_unsloth_qat_weight_memo")


def clear_qat_weight_memo(model):
    """Free every cached fake quantized weight under model (they are rebuilt on the next forwards)."""
    for module in model.modules():
        memo = _memo_of(module)
        if memo is not None:
            memo.drop()


def _clearing(method):
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            # merge / unmerge write the base weight through .data, which bumps no version counter.
            try:
                memo = _memo_of(self.get_base_layer())
            except Exception:
                memo = None
            if memo is not None:
                memo.drop()

    wrapper._unsloth_qat_memo_clearing = True
    wrapper.__wrapped__ = method
    wrapper.__name__ = getattr(method, "__name__", "wrapper")
    wrapper.__doc__ = getattr(method, "__doc__", None)
    return wrapper


def _patch_peft_merge():
    try:
        from peft.tuners.lora import layer as lora_layer
    except Exception:
        return
    classes = {
        cls
        for cls in vars(lora_layer).values()
        if isinstance(cls, type) and issubclass(cls, lora_layer.LoraLayer)
    }
    for cls in classes:
        for name in ("merge", "unmerge"):
            method = cls.__dict__.get(name)
            if callable(method) and not getattr(method, "_unsloth_qat_memo_clearing", False):
                setattr(cls, name, _clearing(method))


def install_qat_weight_memo(model):
    """Cache fake_quant(W) on every non-adapter QAT module whose weight fake quantizer is a plain torchao one. Idempotent."""
    if _mode() in ("0", "false", "off"):
        return 0
    try:
        from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase
    except Exception:
        return 0
    count = 0
    for name, module in model.named_modules():
        quantizer = module._modules.get("weight_fake_quantizer")
        if not isinstance(quantizer, FakeQuantizerBase) or _is_adapter_name(name):
            continue
        _patch_quantizer_class(type(quantizer))
        if not isinstance(quantizer.__dict__.get("_unsloth_qat_weight_memo"), _WeightMemo):
            quantizer.__dict__["_unsloth_qat_weight_memo"] = _WeightMemo()
        count += 1
    if count:
        _patch_peft_merge()
        _install_hooks(model)
    _STATS["installed"] = count
    return count


def _install_hooks(model):
    if not _STATE["hook"]:
        try:
            from torch.optim.optimizer import register_optimizer_step_post_hook
            register_optimizer_step_post_hook(_on_optimizer_step)
            _STATE["hook"] = True
        except Exception:
            pass  # never armed: nothing is cached unless UNSLOTH_QAT_WEIGHT_MEMO=1
    max_seq = None
    for owner in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        value = getattr(owner, "max_seq_length", None) if owner is not None else None
        if isinstance(value, int) and value > 0:
            max_seq = value
            break
    if max_seq:
        _STATE["max_seq"] = max(_STATE["max_seq"] or 0, max_seq)
    config = getattr(model, "config", None)
    if getattr(config, "_attn_implementation", None) == "eager":
        _STATE["quadratic"] = True  # eager attention materializes seq x seq scores
    if not getattr(model, "_unsloth_qat_memo_batch_hook", False):
        try:
            model.register_forward_pre_hook(_record_batch, with_kwargs = True)
            model._unsloth_qat_memo_batch_hook = True
        except Exception:
            pass


def qat_weight_memo_stats():
    stats = dict(_STATS)
    stats["cached_bytes"] = sum(used for _, used in _BUDGET.values())
    stats["armed"] = _STATE["armed"]
    stats["oom_disabled"] = _STATE["disabled"]
    stats["limit_gb"] = {
        str(device): (round(limit / 2**30, 3) if limit != float("inf") else "inf")
        for device, (limit, _) in _BUDGET.items()
    }
    stats["max_batch"] = (_STATE["tokens"], _STATE["rows"], _STATE["max_seq"])
    return stats
