# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in run diagnostics, free unless their variable is set.

UNSLOTH_MEMORY_REPORT=1: per-device memory outside torch's allocator (CUDA context, NCCL, vLLM, other
processes) at load, vLLM init, first forward and first optimizer step; explains OOMs max_memory_reserved cannot.
On the CPU backend the same stages report the process RSS, its peak and the RAM still available.
UNSLOTH_BATCH_REPORT=N (1 / true = 16): real-token and supervised-token share over the first N micro-batches.
"""

__all__ = [
    "memory_report_enabled",
    "report_memory",
    "cpu_memory_snapshot",
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
        has_context = getattr(torch._C, "_cuda_hasPrimaryContext", None)
        return [
            index
            for index in range(torch.cuda.device_count())
            if torch.cuda.memory_reserved(index) > 0
            or (has_context is not None and has_context(index))
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


def _read_kib(path, key):
    with open(path, encoding = "ascii", errors = "ignore") as handle:
        for line in handle:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return None


def _cgroup_memory_limit():
    """The cgroup v2 memory.max of this process when set, else None (Linux only)."""
    try:
        with open("/proc/self/cgroup", encoding = "ascii") as handle:
            path = next(
                (line.split(":", 2)[2].strip() for line in handle if line.startswith("0::")), None
            )
        if path is None:
            return None
        with open(
            os.path.join("/sys/fs/cgroup", path.lstrip("/"), "memory.max"), encoding = "ascii"
        ) as handle:
            value = handle.read().strip()
        return None if value == "max" else int(value)
    except Exception:
        return None


def cpu_memory_snapshot():
    """(rss, peak rss, available, total) bytes of host RAM for this process; unknown fields are None."""
    rss = peak = available = total = None
    try:
        if os.path.exists("/proc/self/status"):
            rss = _read_kib("/proc/self/status", "VmRSS")
            peak = _read_kib("/proc/self/status", "VmHWM")
            available = _read_kib("/proc/meminfo", "MemAvailable")
            total = _read_kib("/proc/meminfo", "MemTotal")
            limit = _cgroup_memory_limit()
            if limit is not None and total is not None and limit < total:
                total = limit
        else:
            import psutil  # macOS / Windows

            info = psutil.Process().memory_info()
            rss = info.rss
            peak = getattr(info, "peak_wset", None)
            memory = psutil.virtual_memory()
            available, total = memory.available, memory.total
    except Exception:
        pass
    if peak is None:
        try:
            import resource
            import sys

            # ru_maxrss is KiB on Linux, bytes on macOS.
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = maxrss if sys.platform == "darwin" else maxrss * 1024
        except Exception:
            pass
    return rss, peak, available, total


def _cpu_memory_line(stage):
    rss, peak, available, total = cpu_memory_snapshot()
    if rss is None and peak is None:
        return None
    gb = lambda value: "?" if value is None else f"{value / _GB:.2f}"  # noqa: E731
    return (
        f"Unsloth: memory at {stage} on cpu: process RSS {gb(rss)} GB (peak {gb(peak)} GB), "
        f"{gb(available)} GB of {gb(total)} GB RAM available."
    )


def report_memory(
    stage,
    owner = None,
    cpu = False,
):
    """Print one line per device for `stage`; once per (stage, owner) so repeated calls stay quiet. cpu = True (the CPU
    backend) reports host RAM instead of CUDA devices."""
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
    if cpu:
        line = _cpu_memory_line(stage)
        if line is not None:
            print(line)
            lines.append(line)
        return lines
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


def install_memory_report(trainer, cpu = False):
    """First forward through a one-shot hook, first optimizer step through a callback."""
    if not memory_report_enabled() or getattr(trainer, "_unsloth_memory_report_installed", False):
        return False
    model = getattr(trainer, "model", None)
    if not isinstance(model, torch.nn.Module):
        return False
    trainer._unsloth_memory_report_installed = True
    # The GPU route's calls stay exactly report_memory(stage, owner).
    extra = {"cpu": True} if cpu else {}

    handle = None

    def first_forward(module, args, output):
        try:
            handle.remove()
        except Exception:
            pass
        report_memory("first forward", trainer, **extra)

    handle = model.register_forward_hook(first_forward)

    from transformers import TrainerCallback

    class _UnslothMemoryReportCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            report_memory("first optimizer step", trainer, **extra)

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
    """[slots, real, supervised, rows, longest_row] int64 tensors on the batch device, or None.

    supervised uses shifted labels[..., 1:] != -100 (the loss never sees a row's first label)."""
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
