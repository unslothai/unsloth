# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in run diagnostics, both off (and free) unless their variable is set.

UNSLOTH_MEMORY_REPORT=1: per device, memory held outside torch's caching allocator (CUDA context, NCCL,
vLLM, cuBLAS workspaces, other processes) at model load, vLLM init, the first forward and the first
optimizer step. torch.cuda.max_memory_reserved never sees that memory, so it explains an OOM the
allocator numbers cannot.

UNSLOTH_BATCH_REPORT=N (1 / true = 16): over the first N training micro-batches, one line with the
share of token slots that hold real tokens (padded rows vs packed / padding-free) and the share that
carries a loss. Costs one host sync per sampled micro-batch, none afterwards.
"""

__all__ = [
    "memory_report_enabled",
    "report_memory",
    "install_memory_report",
    "batch_report_size",
    "batch_token_counts",
    "install_batch_report",
]

import os

import torch

_MEMORY_ENV = "UNSLOTH_MEMORY_REPORT"
_BATCH_ENV = "UNSLOTH_BATCH_REPORT"
_DEFAULT_BATCHES = 16
_GB = 1024**3


def _truthy(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def memory_report_enabled():
    return _truthy(os.environ.get(_MEMORY_ENV, "0"))


def _devices():
    """CUDA / ROCm devices this process already uses: mem_get_info would open a context on any other."""
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return []
        current = torch.cuda.current_device()
        return [
            index
            for index in range(torch.cuda.device_count())
            if index == current or torch.cuda.memory_reserved(index) > 0
        ]
    except Exception:
        return []


def memory_snapshot(index):
    """(total, free, reserved, allocated, outside) bytes for one device, or None."""
    try:
        free, total = torch.cuda.mem_get_info(index)
        reserved = torch.cuda.memory_reserved(index)
        allocated = torch.cuda.memory_allocated(index)
    except Exception:
        return None
    return total, free, reserved, allocated, max(total - free - reserved, 0)


def report_memory(stage, owner = None):
    """Print one line per device for `stage`; once per (stage, owner) so repeated calls stay quiet."""
    if not memory_report_enabled():
        return None
    if owner is not None:
        seen = getattr(owner, "_unsloth_memory_reported", None)
        if seen is None:
            seen = set()
            try:
                owner._unsloth_memory_reported = seen
            except Exception:
                pass
        if stage in seen:
            return None
        seen.add(stage)
    lines = []
    for index in _devices():
        snapshot = memory_snapshot(index)
        if snapshot is None:
            continue
        total, free, reserved, allocated, outside = snapshot
        lines.append(
            f"Unsloth: memory at {stage} on cuda:{index}: {total / _GB:.2f} GB total, "
            f"{free / _GB:.2f} GB free, torch reserved {reserved / _GB:.2f} GB "
            f"(allocated {allocated / _GB:.2f} GB), outside torch's allocator {outside / _GB:.2f} GB "
            "(CUDA context, NCCL, vLLM, other processes)."
        )
    for line in lines:
        print(line)
    return lines


def install_memory_report(trainer):
    """First forward through a one-shot hook, first optimizer step through a callback."""
    if not memory_report_enabled() or getattr(trainer, "_unsloth_memory_report_installed", False):
        return False
    model = getattr(trainer, "model", None)
    if not isinstance(model, torch.nn.Module):
        return False
    trainer._unsloth_memory_report_installed = True

    handle = None

    def first_forward(module, args, output):
        try:
            handle.remove()
        except Exception:
            pass
        report_memory("first forward", trainer)

    handle = model.register_forward_hook(first_forward)

    from transformers import TrainerCallback

    class _UnslothMemoryReportCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            report_memory("first optimizer step", trainer)

    trainer.add_callback(_UnslothMemoryReportCallback())
    return True


def batch_report_size():
    value = os.environ.get(_BATCH_ENV, "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return 0
    if value in ("1", "true", "yes", "on"):
        return _DEFAULT_BATCHES
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


def batch_token_counts(inputs):
    """[slots, real, supervised, rows, longest_row] as int64 tensors on the batch device, or None.

    real: attention_mask when present, else the packed sequence lengths, else every slot (a packed
    or padding-free row without either has no padding Unsloth can see). supervised: the shifted
    causal targets labels[..., 1:] != -100 (the loss never sees a row's first label; packed rows
    already mask document starts), else completion_mask[..., 1:], else real."""
    if not hasattr(inputs, "get"):
        return None
    input_ids = inputs.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.dim() == 0:
        return None
    slots = torch.as_tensor(input_ids.numel(), dtype = torch.int64, device = input_ids.device)
    rows = input_ids.shape[0] if input_ids.dim() > 1 else 1
    width = input_ids.shape[-1]
    attention_mask = inputs.get("attention_mask")
    real = None
    if torch.is_tensor(attention_mask) and attention_mask.shape == input_ids.shape:
        real = (attention_mask != 0).sum().to(torch.int64)
    if real is None:
        lengths = inputs.get("packed_seq_lengths")
        if torch.is_tensor(lengths) and lengths.numel() > 0:
            real = torch.minimum(lengths.sum().to(torch.int64).to(slots.device), slots)
    if real is None:
        real = slots
    labels = inputs.get("labels")
    completion_mask = inputs.get("completion_mask")
    if torch.is_tensor(labels) and labels.numel() > 0:
        supervised = (labels[..., 1:] != -100).sum().to(torch.int64)
    elif torch.is_tensor(completion_mask) and completion_mask.numel() > 0:
        supervised = (completion_mask[..., 1:] != 0).sum().to(torch.int64)
    else:
        supervised = real
    return torch.stack(
        [
            slots,
            real.to(slots.device),
            supervised.to(slots.device),
            torch.as_tensor(rows, dtype = torch.int64, device = slots.device),
            torch.as_tensor(width, dtype = torch.int64, device = slots.device),
        ]
    )


def _batch_mode(trainer):
    args = getattr(trainer, "args", None)
    if getattr(args, "packing", False):
        return "packing"
    if getattr(args, "padding_free", False) or getattr(
        getattr(trainer, "data_collator", None), "padding_free", False
    ):
        return "padding-free"
    return "padded rows"


def format_batch_report(totals, batches, mode):
    slots, real, supervised, rows, longest = totals
    slots = max(slots, 1)
    return (
        f"Unsloth: first {batches} micro-batches ({mode}): {real:,} of {slots:,} token slots hold "
        f"real tokens ({100.0 * real / slots:.1f}%), {supervised:,} carry a loss "
        f"({100.0 * supervised / slots:.1f}%); {rows:,} rows, widest {longest:,} tokens."
    )


def install_batch_report(trainer):
    """Wrap trainer.training_step for the first N micro-batches; afterwards it only checks a flag."""
    batches = batch_report_size()
    if batches <= 0 or getattr(trainer, "_unsloth_batch_report_installed", False):
        return False
    original_training_step = getattr(trainer, "training_step", None)
    if original_training_step is None:
        return False
    trainer._unsloth_batch_report_installed = True
    state = {"seen": 0, "totals": [0, 0, 0, 0, 0], "done": False}
    trainer._unsloth_batch_report_state = state

    def training_step(model, inputs, *args, **kwargs):
        if not state["done"]:
            try:
                counts = batch_token_counts(inputs)
                if counts is not None:
                    slots, real, supervised, rows, width = counts.tolist()
                    totals = state["totals"]
                    totals[0] += slots
                    totals[1] += real
                    totals[2] += supervised
                    totals[3] += rows
                    totals[4] = max(totals[4], width)
                    state["seen"] += 1
            except Exception:
                state["done"] = True
            if state["seen"] >= batches and not state["done"]:
                _print_batch_report(trainer, state)
        return original_training_step(model, inputs, *args, **kwargs)

    trainer.training_step = training_step

    from transformers import TrainerCallback

    class _UnslothBatchReportCallback(TrainerCallback):
        # A run shorter than N micro-batches still gets its line.
        def on_train_end(self, args, state_, control, **kwargs):
            if not state["done"] and state["seen"] > 0:
                _print_batch_report(trainer, state)

    trainer.add_callback(_UnslothBatchReportCallback())
    return True


def _print_batch_report(trainer, state):
    state["done"] = True
    is_main = getattr(trainer, "is_world_process_zero", None)
    if is_main is None or is_main():
        state["line"] = format_batch_report(state["totals"], state["seen"], _batch_mode(trainer))
        print(state["line"])
