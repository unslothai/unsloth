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

The cache costs one extra copy of each cached weight, so on CUDA / XPU it only fills up to half
of the memory that is free when the first copy is made (UNSLOTH_QAT_WEIGHT_MEMO=1 lifts the cap,
=0 turns the cache off). A weight is cached on its second call with the same tensor, so weights
streamed in per forward (offloading) are never held. Skipped: weights that require grad (full
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
_STATS = {"installed": 0, "hits": 0, "stores": 0, "over_budget": 0}
# device -> [limit bytes, used bytes]
_BUDGET = {}
_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _mode():
    return os.environ.get(_ENV, "auto").strip().lower()


def _multi_process():
    dist = torch.distributed
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _device_limit(device, unlimited):
    if unlimited or device.type == "cpu":
        return float("inf")
    backend = getattr(torch, device.type, None)
    mem_get_info = getattr(backend, "mem_get_info", None)
    if mem_get_info is None:
        return 0
    try:
        free, _ = mem_get_info(device)
        cached = backend.memory_reserved(device) - backend.memory_allocated(device)
    except Exception:
        return 0
    return (free + max(cached, 0)) // 2


class _WeightMemo:
    """Per quantizer cache of fake_quant(W) for an unchanged weight; lives in the quantizer's __dict__."""

    __slots__ = ("storage", "key", "out", "nbytes")

    def __init__(self):
        self.storage = None
        self.key = None
        self.out = None
        self.nbytes = 0

    def __getstate__(self):
        return {"cached": False}  # truthy, so unpickling calls __setstate__

    def __setstate__(self, state):
        self.__init__()

    def __deepcopy__(self, memo):
        return _WeightMemo()

    def drop(self):
        if self.nbytes:
            used = _BUDGET.get(self.key[2])
            if used is not None:
                used[1] -= self.nbytes
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
    nbytes = out.untyped_storage().nbytes()
    budget = _BUDGET.get(key[2])
    if budget is None:
        budget = _BUDGET[key[2]] = [_device_limit(key[2], mode in ("1", "true", "on")), 0]
    if budget[1] + nbytes > budget[0]:
        _STATS["over_budget"] += 1
        return out
    budget[1] += nbytes
    memo.out, memo.nbytes = out, nbytes
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
    _STATS["installed"] = count
    return count


def qat_weight_memo_stats():
    stats = dict(_STATS)
    stats["cached_bytes"] = sum(used for _, used in _BUDGET.values())
    return stats
