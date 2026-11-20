# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Round a frozen LoRA base weight onto its weight-only QAT grid once, in place, and disable its quantizer.

Bitwise equal to per-call fake quant for the trainer, and vLLM (which shares the storage) now samples
the same weights. Disk-based exports are unchanged; in-memory merges now hold the rounded base.
Skipped: non-LoRA Linears, activation / nvfp4 / mxfp8 schemes, trainable, residual-init, DoRA,
subclass, offloaded CPU and memory-shared (tied) weights. UNSLOTH_QAT_ROUND_INPLACE=0 disables.
A rounded weight that is later rewritten (load_state_dict, or any in place write that bumps its version,
or a new tensor) gets its quantizer back on load_state_dict and on the next call here, and is rounded
again when still eligible. Writes through .data (PEFT merge / unmerge, vLLM's fold / restore) are kept.
state_dict() marks a rounded weight in the module's state dict metadata with a checksum of its values (no new
keys, so exports read the same tensors; torch.save keeps it, safetensors drops it), and a load whose weight still
matches that checksum (another prepared model's state_dict) keeps it as is instead of rounding it again."""

import heapq
import os

import torch

__all__ = [
    "round_frozen_qat_weights",
    "qat_round_stats",
]

_ENV = "UNSLOTH_QAT_ROUND_INPLACE"
_FLAG = "_unsloth_qat_rounded"
_RECORD = "_unsloth_qat_round_record"
_HOOK = "_unsloth_qat_round_load_hook"
_SAME = "_unsloth_qat_round_same_load"
_META = "unsloth_qat_rounded"  # state dict metadata key: checksum of this module's rounded weight
_PRIME = (1 << 31) - 1
_CHUNK = 1 << 22
_STATS = {"rounded": 0, "bytes": 0, "skipped": {}}
_PLAIN_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)


def _enabled():
    return os.environ.get(_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def qat_round_stats():
    return {
        "rounded": _STATS["rounded"],
        "bytes": _STATS["bytes"],
        "skipped": dict(_STATS["skipped"]),
    }


def _weight_only_quantizer(quantizer):
    """True for a torchao weight fake quantizer whose output depends only on the weight it is given."""
    try:
        from torchao.quantization.qat.fake_quantizer import (
            Int4WeightFakeQuantizer,
            IntxFakeQuantizer,
        )
    except Exception:
        return False
    cls = type(quantizer)
    if cls is Int4WeightFakeQuantizer:
        return True
    if cls is IntxFakeQuantizer:
        config = quantizer.config
        return bool(getattr(config, "is_dynamic", True)) and not bool(
            getattr(config, "range_learning", False)
        )
    return False


def _span(tensor):
    """[start, end) bytes a strided tensor can touch."""
    if tensor.numel() == 0:
        return None
    extent = 1 + sum(
        (size - 1) * abs(stride) for size, stride in zip(tensor.shape, tensor.stride())
    )
    start = tensor.data_ptr()
    return start, start + extent * tensor.element_size()


def _shared_memory_keys(model):
    """Keys (id(module), name) of every parameter / buffer whose bytes overlap a different one's."""
    spans = []
    for module in model.modules():
        for store in (module._parameters, module._buffers):
            for name, tensor in store.items():
                if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                    continue
                try:
                    span = _span(tensor)
                except Exception:
                    continue
                if span is not None:
                    spans.append((span[0], span[1], tensor.device, (id(module), name)))
    spans.sort(key = lambda x: (str(x[2]), x[0]))
    shared = set()
    active = []
    device = None
    for start, end, dev, key in spans:
        if dev != device:
            active, device = [], dev
        while active and active[0][0] <= start:
            heapq.heappop(active)
        others = [k for _, k in active if k != key]
        if others:
            shared.add(key)
            shared.update(others)
        heapq.heappush(active, (end, key))
    return shared


def _skip(reason):
    _STATS["skipped"][reason] = _STATS["skipped"].get(reason, 0) + 1


def _weight_state(weight):
    """(data_ptr, version) of the weight as rounded; inference tensors carry no version counter."""
    if not isinstance(weight, torch.Tensor):
        return None
    return (weight.data_ptr(), None if weight.is_inference() else weight._version)


def _checksum(weight):
    """Exact, device independent checksum of a tensor's bits: position weighted sum mod 2**31 - 1."""
    with torch.no_grad():
        flat = weight.detach().contiguous().reshape(-1)
        size = flat.element_size()
        if size == 2:
            flat = flat.view(torch.int16)
        elif size == 1:
            flat = flat.view(torch.uint8)
        else:
            # 4 bytes as one int32 word, 8-byte dtypes (float64) as two: the bits, never a value conversion
            # (float64 weights below 1 in magnitude all converted to 0, so every such weight matched).
            flat = flat.view(torch.int32)
        total = 0
        for start in range(0, flat.numel(), _CHUNK):
            part = flat[start : start + _CHUNK].to(torch.int64).remainder_(_PRIME)
            mult = torch.arange(
                start + 1, start + 1 + part.numel(), device = part.device, dtype = torch.int64
            )
            mult.mul_(2654435761).remainder_(_PRIME)
            # Each product < 2**62 and each chunk sums < 2**53: no int64 overflow.
            total += int(part.mul_(mult).remainder_(_PRIME).sum())
        return (str(weight.dtype), tuple(weight.shape), total % _PRIME)


class _RoundRecord:
    """The rounded weight's state. A copy or unpickled model holds a new tensor that is already rounded,
    so the record is reset there and adopts that tensor at its next check instead of flagging a change.
    checksum: (state, _checksum) of the rounded weight, computed when it is rounded; a weight whose state
    changed (a device move: new storage, same values) is still rounded when its values match it."""

    __slots__ = ("state", "checksum")

    def __init__(self, state = None):
        self.state = state
        self.checksum = None

    def __deepcopy__(self, memo):
        return _RoundRecord()

    def __reduce__(self):
        return (_RoundRecord, ())


def _rewritten(quantizer, weight):
    """True if a rounded weight has since been replaced or written in place (a version bump)."""
    record = quantizer.__dict__.get(_RECORD)
    state = _weight_state(weight)
    if not isinstance(record, _RoundRecord) or record.state is None:
        quantizer.__dict__[_RECORD] = _adopted_record(weight, state)
        return False
    if state == record.state and not _unversioned_write(record, weight, state):
        return False
    return not _same_rounded_values(record, weight, state)


def _unversioned_write(record, weight, state):
    """An inference tensor has no version counter: an in place write (a reload under inference mode) keeps its
    (pointer, None) state, so compare its values with the rounding checksum instead. False when it cannot tell."""
    if (
        state is None
        or state[1] is not None
        or not isinstance(record, _RoundRecord)
        or record.checksum is None
    ):
        return False
    try:
        return list(_checksum(weight)) != list(record.checksum[1])
    except Exception:
        return False


def _adopted_record(weight, state):
    """Record for a weight taken as already rounded (a loaded rounded checkpoint, a copied or unpickled model),
    with its checksum, so a later device move of it is still recognised as rounded."""
    record = _RoundRecord(state)
    try:
        record.checksum = (state, _checksum(weight))
    except Exception:
        pass
    return record


def _same_rounded_values(record, weight, state):
    """A weight with new storage or version (a device move: model.cpu(), .to()) that still holds the values
    it was rounded to; adopts the new state when so. False without a rounding checksum to compare."""
    if record.checksum is None:
        return False
    try:
        now = _checksum(weight)
    except Exception:
        return False
    if list(now) != list(record.checksum[1]):
        return False
    record.state = state
    record.checksum = (state, now)
    return True


def _unround(quantizer):
    """Back to per call fake quantization of whatever the weight now holds (the behaviour before rounding)."""
    quantizer.enabled = True
    quantizer.__dict__[_FLAG] = False
    quantizer.__dict__.pop(_RECORD, None)


def _round_into(module, quantizer, weight):
    """Write quantizer(weight) into weight in place and switch the quantizer off; False if shapes differ."""
    with torch.no_grad():
        rounded = quantizer(weight)
        if rounded.dtype != weight.dtype or rounded.shape != weight.shape:
            _skip("output_mismatch")
            return False
        if weight.is_inference():
            with torch.inference_mode():
                weight.copy_(rounded)
        else:
            weight.copy_(rounded)
        del rounded
    quantizer.enabled = False
    quantizer.__dict__[_FLAG] = True
    record = _RoundRecord(_weight_state(weight))
    try:
        record.checksum = (record.state, _checksum(weight))
    except Exception:
        pass
    quantizer.__dict__[_RECORD] = record
    if not module.__dict__.get(_HOOK, False):
        try:
            register_pre = getattr(module, "register_load_state_dict_pre_hook", None)
            if register_pre is not None:  # torch >= 2.5
                register_pre(_before_load)
            else:
                module._register_load_state_dict_pre_hook(_before_load, with_module = True)
            module.register_load_state_dict_post_hook(_after_load)
            register_save = getattr(module, "register_state_dict_post_hook", None)
            if register_save is not None:  # torch >= 2.5
                register_save(_mark_rounded)
            else:
                module._register_state_dict_hook(_mark_rounded)
            module.__dict__[_HOOK] = True
        except Exception:
            pass
    return True


def _recipe(quantizer):
    """The fake quantizer's type and config (dtype, granularity, mapping, ...): which grid its weight was rounded to."""
    return f"{type(quantizer).__name__}:{getattr(quantizer, 'config', None)!r}"


def _mark_rounded(module, state_dict, prefix, local_metadata):
    """state_dict hook on a rounded module: mark its weight as rounded while it still holds the rounded values."""
    quantizer = module._modules.get("weight_fake_quantizer")
    if quantizer is None or not quantizer.__dict__.get(_FLAG, False):
        return
    record = quantizer.__dict__.get(_RECORD)
    weight = module._parameters.get("weight")
    state = _weight_state(weight)
    if state is None or not isinstance(record, _RoundRecord):
        return
    # A reset record (a copy or unpickled model) adopts the tensor it holds, which is already rounded.
    if (
        record.state is not None
        and record.state != state
        and not _same_rounded_values(record, weight, state)
    ):
        return  # rewritten since it was rounded (a device move keeps the rounded values, an edit does not)
    try:
        if record.checksum is None or record.checksum[0] != state:
            record.checksum = (state, _checksum(weight))
        local_metadata[_META] = tuple(record.checksum[1]) + (_recipe(quantizer),)
    except Exception:
        pass


def _before_load(
    module,
    state_dict,
    prefix,
    local_metadata = None,
    *args,
):
    """load_state_dict pre hook on a rounded module: note the weight's state and whether the incoming
    weight is already rounded (marked so by the state_dict that produced it, or the rounded one it holds)."""
    module.__dict__.pop(_SAME, None)
    quantizer = module._modules.get("weight_fake_quantizer")
    if quantizer is None or not quantizer.__dict__.get(_FLAG, False):
        return
    incoming = state_dict.get(prefix + "weight")
    weight = module._parameters.get("weight")
    if not isinstance(incoming, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return  # no weight in this load (adapter only)
    marker = local_metadata.get(_META) if isinstance(local_metadata, dict) else None
    try:
        same = incoming.shape == weight.shape
        if same:
            # The rounded weight state_dict() marked, unchanged since (any edit or replacement changes the
            # checksum; checked before any dtype cast, which the checksum includes), still exact in this
            # module's dtype, or the rounded values this module already holds.
            # The marker also names the fake quantization recipe: values rounded on another grid (an int8
            # model's state_dict loaded into an int4 one) are not rounded for this module.
            marked = isinstance(marker, (tuple, list)) and list(marker) == [
                *_checksum(incoming),
                _recipe(quantizer),
            ]
            converted = incoming.to(device = weight.device, dtype = weight.dtype)
            if marked and converted.dtype != incoming.dtype:
                marked = torch.equal(converted.to(incoming.dtype), incoming.to(converted.device))
            same = marked or torch.equal(converted, weight)
    except Exception:
        same = False
    module.__dict__[_SAME] = (bool(same), _weight_state(weight))


def _after_load(module, incompatible_keys):
    """load_state_dict post hook on a rounded module: a loaded new weight is fake quantized again."""
    info = module.__dict__.pop(_SAME, None)
    quantizer = module._modules.get("weight_fake_quantizer")
    if info is None or quantizer is None or not quantizer.__dict__.get(_FLAG, False):
        return
    same, before = info
    weight = module._parameters.get("weight")
    state = _weight_state(weight)
    if state == before and not _unversioned_write(quantizer.__dict__.get(_RECORD), weight, state):
        return  # not written (for example a failed shape check)
    if same:
        # Rounded values (another prepared model's state_dict, or this model's own): already on the grid. int8 /
        # cactus fake quant is not idempotent, so rounding them again would move them.
        quantizer.__dict__[_RECORD] = _adopted_record(weight, state)
        return
    _unround(quantizer)
    # The same storage written in place keeps every eligibility check that passed before (shape, device,
    # not shared); a new tensor is left to per call fake quant until round_frozen_qat_weights runs again.
    if (
        _enabled()
        and before is not None
        and type(weight) in _PLAIN_TENSOR_TYPES
        and not weight.requires_grad
        and weight.data_ptr() == before[0]
    ):
        _round_into(module, quantizer, weight)


def round_frozen_qat_weights(model):
    """Write fake_quant(W) into every eligible frozen base weight and switch its quantizer off. Idempotent."""
    if not _enabled():
        return 0
    try:
        from ._utils import _is_lora_adapter_name
    except Exception:
        _is_lora_adapter_name = lambda fqn: any(p.startswith("lora_") for p in fqn.split("."))

    on_accelerator = any(p.device.type not in ("cpu", "meta") for p in model.parameters())
    # Residual-init resume rebases the base in place (would give fake_quant(W) + delta); DoRA reads W unquantized.
    residual = {
        f"{name}.base_layer" if name else "base_layer"
        for name, module in model.named_modules()
        if getattr(module, "_unsloth_initial_lora", None)
        or any((getattr(module, "use_dora", None) or {}).values())
    }
    candidates = []
    for name, module in model.named_modules():
        quantizer = module._modules.get("weight_fake_quantizer")
        if quantizer is None:
            continue
        if quantizer.__dict__.get(_FLAG, False):
            if not _rewritten(quantizer, module._parameters.get("weight")):
                continue
            _unround(quantizer)  # rewritten since it was rounded: check and round it again
        if not getattr(quantizer, "enabled", True):
            continue
        # Plain Linears convert from memory, where a rounded int8 / cactus weight would requantize differently.
        if _is_lora_adapter_name(name) or name.rsplit(".", 1)[-1] != "base_layer":
            continue
        if name in residual:
            _skip("residual_init")
            continue
        if module._modules.get("activation_fake_quantizer") is not None:
            _skip("activation_quantized")
            continue
        if not _weight_only_quantizer(quantizer):
            _skip("quantizer_type")
            continue
        weight = module._parameters.get("weight")
        if weight is None or "weight" in vars(module):
            _skip("no_plain_weight")
            continue
        if type(weight) not in _PLAIN_TENSOR_TYPES:
            _skip("tensor_subclass")
            continue
        if weight.requires_grad:
            _skip("trainable")
            continue
        if (
            weight.device.type == "meta"
            or (weight.device.type == "cpu" and on_accelerator)
            or weight.dim() != 2
        ):
            _skip("device")
            continue
        candidates.append((name, module, quantizer, weight))
    if not candidates:
        return 0

    shared = _shared_memory_keys(model)
    count = 0
    nbytes = 0
    for name, module, quantizer, weight in candidates:
        if (id(module), "weight") in shared:
            _skip("shared_memory")
            continue
        if not _round_into(module, quantizer, weight):
            continue
        count += 1
        nbytes += weight.numel() * weight.element_size()

    _STATS["rounded"] += count
    _STATS["bytes"] += nbytes
    if count:
        try:
            from ..precision import record
            record(model, "qat_round_inplace", count)
        except Exception:
            pass
        print(
            f"Unsloth: QAT rounded {count} frozen base weights onto the fake quantization grid once "
            f"({nbytes / 2**30:.2f} GB), so training stops fake quantizing them per forward and "
            f"vLLM rollouts read the same weights. {_ENV}=0 disables this."
        )
    return count
