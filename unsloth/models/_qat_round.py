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

"""Round a frozen QAT base weight onto its fake quantization grid once, in place.

With LoRA + a weight-only `qat_scheme` (int4, int8, cactus) the base weight is frozen, and torchao's
weight fake quantizer is a pure function of it, so every forward recomputes the same fake_quant(W).
Writing fake_quant(W) into W once and switching the quantizer off gives the trainer bitwise the
same weight it computed per call before, with no per-call cost and no extra copy.

It also fixes fast_inference: vLLM shares the base weight storage with the trainer, so before this
vLLM sampled from the unrounded W while the trainer optimised fake_quant(W) (a train / rollout
mismatch nothing corrected by default). After rounding both read the same tensor.

Only the stored base weight changes. Exports that merge LoRA read the base from the checkpoint on
disk (merged_16bit, save_pretrained_torchao, GGUF), so they are unchanged; an in-memory
merge_and_unload or QATConfig(step = "convert") now holds the fake quantized base the model trained
against instead of the unrounded one.

Only the base_layer of a PEFT LoRA layer is rounded. Skipped (keep fake quantizing per call): plain
Linears outside LoRA (lm_head, frozen layers of a full fine-tune), activation schemes (int8-int4, fp8-int4, fp8-fp8, the
input is quantized per call anyway), nvfp4 / mxfp8 (no weight_fake_quantizer), adapters and any
weight that requires grad (full fine-tuning), range learning or static scales, tensor subclasses and
DTensors, CPU weights of a model that also has accelerator weights (an offloaded weight is fake
quantized on the accelerator at forward time; rounding it on the CPU could differ), and weights whose memory another parameter or buffer also uses (a tied
lm_head would round the input embedding too).

Residual inits (LoRA-GA, PiSSA, ...) are skipped: resuming one rebases its frozen base in place. Any other
weight rewritten in place after this (other than vLLM's exact merged rollout fold / restore) is no longer fake
quantized.
UNSLOTH_QAT_ROUND_INPLACE=0 turns it off."""

import heapq
import os

import torch

__all__ = [
    "round_frozen_qat_weights",
    "qat_round_stats",
]

_ENV = "UNSLOTH_QAT_ROUND_INPLACE"
_FLAG = "_unsloth_qat_rounded"
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
        # Static scales and range learning hold state beside the weight; recompute-per-call only.
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
    active = []  # (end, key) of spans that started earlier, on this device
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


def round_frozen_qat_weights(model):
    """Write fake_quant(W) into every eligible frozen base weight and switch its quantizer off. Idempotent."""
    if not _enabled():
        return 0
    try:
        from ._utils import _is_lora_adapter_name
    except Exception:
        _is_lora_adapter_name = lambda fqn: any(p.startswith("lora_") for p in fqn.split("."))

    on_accelerator = any(p.device.type not in ("cpu", "meta") for p in model.parameters())
    # Residual inits (LoRA-GA, PiSSA, ...) keep a snapshot of their initial factors, and resuming such a checkpoint
    # into a re-initialized model rebases the frozen base in place (_native_from_converted). A rounded base with its
    # quantizer off would then train on fake_quant(W) + delta instead of fake_quant(W + delta), so keep those per call.
    residual = {
        f"{name}.base_layer" if name else "base_layer"
        for name, module in model.named_modules()
        if getattr(module, "_unsloth_initial_lora", None)
    }
    candidates = []
    for name, module in model.named_modules():
        quantizer = module._modules.get("weight_fake_quantizer")
        if quantizer is None or quantizer.__dict__.get(_FLAG, False):
            continue
        if not getattr(quantizer, "enabled", True):
            continue
        # Only the frozen base of a PEFT LoRA layer: its exports merge from the checkpoint on disk. A frozen
        # plain Linear (lm_head, a frozen layer of a full fine-tune) is converted from memory, where a rounded
        # int8 / cactus weight would requantize to other codes, so it keeps fake quantizing per call.
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
        with torch.no_grad():
            rounded = quantizer(weight)
            if rounded.dtype != weight.dtype or rounded.shape != weight.shape:
                _skip("output_mismatch")
                continue
            if weight.is_inference():
                with torch.inference_mode():
                    weight.copy_(rounded)
            else:
                weight.copy_(rounded)
            del rounded
        quantizer.enabled = False
        quantizer.__dict__[_FLAG] = True
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
