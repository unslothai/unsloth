# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in run diagnostics, free unless their variable is set.

UNSLOTH_MEMORY_REPORT=1: per-device memory outside torch's allocator (CUDA context, NCCL, vLLM, other
processes) at load, vLLM init, first forward and first optimizer step; explains OOMs max_memory_reserved cannot.
On the CPU backend the same stages report the process RSS, its peak and the RAM still available.
UNSLOTH_BATCH_REPORT=N (1 / true = 16): real-token and supervised-token share over the first N micro-batches.
UNSLOTH_SYNC_REPORT=N (1 / true = 16): host-device syncs per optimizer step over the first N steps, by
call site, from torch.cuda.set_sync_debug_mode("warn"); every .item(), nonzero, bool(tensor), D2H copy
and stream synchronize stalls the CPU until the GPU drains, so the top sites are what to remove first.
"""

__all__ = [
    "memory_report_enabled",
    "report_memory",
    "cpu_memory_snapshot",
    "install_memory_report",
    "batch_report_size",
    "batch_token_counts",
    "install_batch_report",
    "sync_report_steps",
    "SyncCounter",
    "install_sync_report",
]

import functools
import os
import traceback
import warnings
from collections import Counter

import torch

_MEMORY_ENV = "UNSLOTH_MEMORY_REPORT"
_BATCH_ENV = "UNSLOTH_BATCH_REPORT"
_SYNC_ENV = "UNSLOTH_SYNC_REPORT"
_SYNC_MESSAGE = "called a synchronizing CUDA operation"
_TOP_SITES = 8
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


def _cgroup_memory(proc_cgroup = "/proc/self/cgroup", root = "/sys/fs/cgroup"):
    """(limit, headroom) bytes from cgroup v2 for this process: the tightest memory.max over its cgroup and every
    ancestor, and the smallest memory.max - memory.current among them. Either is None when unlimited / unreadable."""
    limit = headroom = None
    try:
        with open(proc_cgroup, encoding = "ascii") as handle:
            path = next(
                (line.split(":", 2)[2].strip() for line in handle if line.startswith("0::")), None
            )
        if path is None:
            return None, None
        parts = [part for part in path.split("/") if part]
        for depth in range(len(parts), -1, -1):
            folder = os.path.join(root, *parts[:depth])
            try:
                with open(os.path.join(folder, "memory.max"), encoding = "ascii") as handle:
                    value = handle.read().strip()
            except OSError:
                continue
            if value == "max":
                continue
            value = int(value)
            limit = value if limit is None else min(limit, value)
            try:
                with open(os.path.join(folder, "memory.current"), encoding = "ascii") as handle:
                    left = max(value - int(handle.read().strip()), 0)
            except (OSError, ValueError):
                continue
            headroom = left if headroom is None else min(headroom, left)
    except Exception:
        pass
    return limit, headroom


def cpu_memory_snapshot():
    """(rss, peak rss, available, total) bytes of host RAM for this process; unknown fields are None."""
    rss = peak = available = total = None
    try:
        if os.path.exists("/proc/self/status"):
            rss = _read_kib("/proc/self/status", "VmRSS")
            peak = _read_kib("/proc/self/status", "VmHWM")
            available = _read_kib("/proc/meminfo", "MemAvailable")
            total = _read_kib("/proc/meminfo", "MemTotal")
            limit, headroom = _cgroup_memory()
            if limit is not None and total is not None and limit < total:
                total = limit
            # Host MemAvailable ignores what the cgroup has already used: cap it by the remaining allowance.
            if headroom is not None and available is not None and headroom < available:
                available = headroom
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


def _env_count(name):
    value = os.environ.get(name, "").strip().lower()
    if value in ("", "0", "false", "no", "off"):
        return 0
    if value in ("1", "true", "yes", "on"):
        return _DEFAULT_BATCHES
    try:
        return max(int(value), 0)
    except ValueError:
        return 0


def batch_report_size():
    return _env_count(_BATCH_ENV)


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


def sync_report_steps():
    return _env_count(_SYNC_ENV)


_SKIP_PARTS = (
    os.sep + "torch" + os.sep,
    os.sep + "warnings.py",
    os.sep + "contextlib.py",
)


def _short_path(filename):
    for marker in ("site-packages" + os.sep, "dist-packages" + os.sep):
        if marker in filename:
            return filename.split(marker, 1)[1]
    parts = filename.replace(os.sep, "/").split("/")
    return "/".join(parts[-3:])


def _sync_site(stack = None, depth = 2):
    """`file:line fn` of the innermost frame outside torch, plus up to `depth` outer callers."""
    frames = traceback.extract_stack()[:-1] if stack is None else stack
    outside = [
        f
        for f in frames
        if f.filename != __file__ and not any(part in f.filename for part in _SKIP_PARTS)
    ]
    if not outside:
        frame = frames[-1] if frames else None
        return (
            "unknown"
            if frame is None
            else f"{_short_path(frame.filename)}:{frame.lineno} {frame.name}"
        )
    chain = outside[-1 - depth :][::-1]
    return " <- ".join(f"{_short_path(f.filename)}:{f.lineno} {f.name}" for f in chain)


def _is_sync_filter(entry):
    return entry[0] == "always" and getattr(entry[1], "pattern", None) == _SYNC_MESSAGE


class SyncCounter:
    """Counts CUDA sync-debug warnings by call site while active; one bucket per `step` value.

    Only those warnings are swallowed; every other warning goes to the previous showwarning."""

    def __init__(self):
        self.steps = {}
        self.step = 0
        self.start_step = 0
        self.active = False
        self._previous_mode = None
        self._previous_show = None

    def _show(
        self,
        message,
        category,
        filename,
        lineno,
        file = None,
        line = None,
    ):
        if self.active and _SYNC_MESSAGE in str(message):
            self.steps.setdefault(self.step, Counter())[_sync_site()] += 1
            return
        return self._previous_show(message, category, filename, lineno, file, line)

    def enter(self, step):
        # Every observed step gets a bucket, so a step with no syncs still counts in the average.
        self.step = step
        self.steps.setdefault(step, Counter())

    def start(self, use_cuda = True):
        if self.active:
            return True
        if use_cuda:
            try:
                if not torch.cuda.is_available():
                    return False
                self._previous_mode = torch.cuda.get_sync_debug_mode()
                torch.cuda.set_sync_debug_mode("warn")
            except Exception:
                return False
        # The C++ warning carries the op's own location, so the default filter would print it once.
        warnings.filterwarnings("always", message = _SYNC_MESSAGE)
        self._previous_show = warnings.showwarning
        warnings.showwarning = self._show
        self.active = True
        return True

    def stop(self):
        if not self.active:
            return
        self.active = False
        if self._previous_mode is not None:
            try:
                torch.cuda.set_sync_debug_mode(self._previous_mode)
            except Exception:
                pass
            self._previous_mode = None
        if warnings.showwarning == self._show:
            warnings.showwarning = self._previous_show
        warnings.filters[:] = [f for f in warnings.filters if not _is_sync_filter(f)]

    def summary(self, skip_first = True):
        """(per-step totals, sites Counter over the measured steps, measured step count)."""
        totals = {step: sum(counter.values()) for step, counter in sorted(self.steps.items())}
        measured = [step for step in totals if not (skip_first and step == min(totals))]
        if not measured:
            measured = list(totals)
        sites = Counter()
        for step in measured:
            sites.update(self.steps[step])
        return totals, sites, len(measured)


def format_sync_report(counter, top = _TOP_SITES):
    totals, sites, measured = counter.summary()
    if not any(totals.values()):
        return ["Unsloth: no host-device syncs seen during training."]
    first = min(totals)
    average = sum(sites.values()) / max(measured, 1)
    lines = [
        f"Unsloth: host-device syncs per optimizer step: {average:.1f} on average over {measured} "
        f"step(s) (first step, with compilation and warmup: {totals[first]}). Top call sites:"
    ]
    for site, count in sites.most_common(top):
        lines.append(f"  {count / max(measured, 1):8.1f} / step  {site}")
    return lines


def install_sync_report(trainer):
    """Counts syncs from train begin through the first N optimizer steps, then switches off."""
    steps = sync_report_steps()
    if steps <= 0 or getattr(trainer, "_unsloth_sync_report_installed", False):
        return False
    trainer._unsloth_sync_report_installed = True
    counter = SyncCounter()
    trainer._unsloth_sync_report = counter

    from transformers import TrainerCallback

    def finish():
        if counter.active or (counter.steps and not getattr(counter, "lines", None)):
            counter.stop()
            counter.lines = format_sync_report(counter)
            is_main = getattr(trainer, "is_world_process_zero", None)
            if is_main is None or is_main():
                for line in counter.lines:
                    print(line)

    class _UnslothSyncReportCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            # A new train() call measures its own window; resumed runs start past step 0.
            counter.steps, counter.lines = {}, None
            counter.start_step = int(state.global_step)
            counter.step = counter.start_step + 1
            if not counter.start():
                print("Unsloth: UNSLOTH_SYNC_REPORT needs a CUDA device; skipped.")

        def on_step_begin(self, args, state, control, **kwargs):
            # The bucket is registered when the step runs, so a step with no syncs still counts and a step that never
            # runs (training ending inside the window) does not.
            if counter.active:
                counter.enter(int(state.global_step) + 1)

        def on_step_end(self, args, state, control, **kwargs):
            if not counter.active:
                return
            # The Trainer's logging after this step stays in this step's bucket.
            if int(state.global_step) - counter.start_step >= steps:
                finish()

        def on_train_end(self, args, state, control, **kwargs):
            finish()

    trainer.add_callback(_UnslothSyncReportCallback())

    # on_train_end never runs when train() raises or is interrupted: restore the sync-debug mode and warning hooks
    # there too, so a notebook does not keep counting into an abandoned report.
    original_train = getattr(trainer, "train", None)
    if callable(original_train):

        @functools.wraps(original_train)
        def train(*args, **kwargs):
            try:
                return original_train(*args, **kwargs)
            finally:
                counter.stop()

        trainer.train = train
    return True
