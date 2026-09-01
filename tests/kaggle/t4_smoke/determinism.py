# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Determinism and metric-capture primitives for the Kaggle T4 smoke test.

Self-contained on purpose: the payload ships to a Kaggle kernel as an inlined
notebook with no repo checkout and no network fetch of our sources, so it
cannot import a helper that exists only on the machine that built it.

``enable_full_determinism`` is separate from ``set_all_seeds_fast`` because it
MUST run before ``import torch`` for the cuBLAS workspace setting to take
effect. ``StatisticsCallback`` requires ``logging_steps=1``.
``RepeatingSequentialSampler`` makes the sample sequence a pure function of the
step index.

What these can buy, since ``run_t4_smoke.py``'s assertions depend on it:
run-to-run inside ONE process is bitwise reproducible and is asserted exactly;
across GPU architectures, drivers or library versions it is not, since
reduction order, kernel selection and fp16 vs bf16 all move the low bits, so
those checks are tolerance bands, never equality.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any


# cuBLAS needs a fixed workspace for reproducible GEMM reductions.
# CUDA reads this when the handle is created, on first use after `import torch`;
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def enable_full_determinism() -> None:
    """Set the env vars that only take effect before torch initialises CUDA.

    Call this at the very top of the entry point, before any torch import.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    os.environ["PYTHONHASHSEED"] = "0"
    # A tokenizers worker pool interleaves dataset .map ordering nondeterministically on some versions, and this
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_all_seeds_fast(seed: int = 3407) -> None:
    """Seed every RNG the training loop touches. No algorithm constraints."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic_algorithms(warn_only: bool = True) -> dict:
    """Ask torch for deterministic kernels. Returns what actually took.

    ``warn_only=True`` is deliberate. Unsloth's 4-bit path runs through
    bitsandbytes and fused Triton kernels, some of which register no
    deterministic implementation, so ``warn_only=False`` raises and the smoke
    test dies having proved nothing. Warning instead uses the deterministic
    kernel wherever one exists, and the run-to-run equality assertion is what
    actually verifies the result.
    """
    import torch

    state: dict[str, Any] = {"requested": True, "warn_only": warn_only}
    try:
        torch.use_deterministic_algorithms(True, warn_only = warn_only)
        state["use_deterministic_algorithms"] = True
    except Exception as exc:  # noqa: BLE001
        state["use_deterministic_algorithms"] = False
        state["error"] = f"{type(exc).__name__}: {exc}"[:200]
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        state["cudnn_deterministic"] = True
    except Exception:  # noqa: BLE001
        state["cudnn_deterministic"] = False
    state["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
    return state


def _trainer_callback_base():
    from transformers import TrainerCallback
    return TrainerCallback


class StatisticsCallback(_trainer_callback_base()):  # type: ignore[misc]
    """Accumulate per-step loss / grad_norm / lr into ``.logs``.

    Reads what the Trainer logs rather than recomputing a grad norm from the
    parameters: recomputing measures AFTER the optimizer step and after
    gradients were zeroed, giving a different quantity or zero depending on the
    transformers version. The logged value is the pre-clip norm the trainer
    used.

    Only fires on logged steps, so the caller must set ``logging_steps=1``.
    """

    def __init__(self) -> None:
        self.logs: list[dict] = []

    def on_log(
        self,
        args,
        state,
        control,
        logs = None,
        **kwargs,
    ):  # noqa: ANN001
        if not logs or "loss" not in logs:
            return
        entry = {"step": int(state.global_step), "loss": float(logs["loss"])}
        if logs.get("grad_norm") is not None:
            entry["grad_norm"] = float(logs["grad_norm"])
        if logs.get("learning_rate") is not None:
            entry["learning_rate"] = float(logs["learning_rate"])
        self.logs.append(entry)

    def save_logs(self, path: str) -> None:
        with open(path, "w", encoding = "utf-8") as fh:
            json.dump(self.logs, fh, indent = 2)


def _sampler_base():
    from torch.utils.data import Sampler
    return Sampler


class RepeatingSequentialSampler(_sampler_base()):  # type: ignore[misc]
    """Deterministic, shuffle-free index order.

    Step *i* yields row ``i % dataset_length``, repeated
    ``batch_size * gradient_accumulation_steps`` times: a pure function of the
    step index, independent of RNG state, of dataset length modulo batch size,
    and of which epoch boundary the run lands near.
    """

    def __init__(
        self,
        dataset_length: int,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        max_steps: int | None = None,
    ) -> None:
        self.dataset_length = int(dataset_length)
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.samples_per_step = self.batch_size * self.gradient_accumulation_steps
        steps = int(max_steps) if max_steps else self.dataset_length
        self.total_samples = steps * self.samples_per_step

    def __iter__(self):
        emitted = 0
        step = 0
        while emitted < self.total_samples:
            idx = step % self.dataset_length
            for _ in range(self.samples_per_step):
                if emitted >= self.total_samples:
                    break
                yield idx
                emitted += 1
            step += 1

    def __len__(self) -> int:
        return self.total_samples


def compare_metrics(
    a: list[dict],
    b: list[dict],
    fields: tuple[str, ...] = ("loss", "grad_norm"),
) -> dict:
    """Max absolute deviation between two metric lists, per field.

    ``identical`` is bitwise equality of every compared field, not equality
    within a tolerance: the caller decides what tolerance means.

    Two non-numeric differences count too, both being this comparison's own
    subject matter:

    * A field logged by one run and not the other. Same-length lists carrying
      different keys are two different traces, which ``check_reference`` already
      calls "a change in the SHAPE of what the trainer logged"; the exact
      comparator cannot be laxer than the tolerance band beside it.
    * A moved ``step`` coordinate. The lists are zipped positionally, so a
      shifted, duplicated or reordered step makes every later pairing
      meaningless AND is itself the trainer nondeterminism this exists to catch.
    """
    result: dict[str, Any] = {
        "identical": True,
        "length_a": len(a),
        "length_b": len(b),
        "max_abs_diff": {},
        "first_diff_step": None,
        "step_mismatch": [],
    }
    if len(a) != len(b):
        result["identical"] = False
        result["length_mismatch"] = True
        return result
    for index, (ea, eb) in enumerate(zip(a, b)):
        sa, sb = ea.get("step"), eb.get("step")
        if sa != sb:
            result["step_mismatch"].append({"index": index, "a": sa, "b": sb})
            if result["identical"]:
                result["identical"] = False
                result["first_diff_step"] = sa
    for field in fields:
        worst = 0.0
        for ea, eb in zip(a, b):
            has_a, has_b = field in ea, field in eb
            if not has_a and not has_b:
                continue
            if has_a != has_b:
                if result["identical"]:
                    result["identical"] = False
                    result["first_diff_step"] = (ea if has_a else eb).get("step")
                result.setdefault("one_sided_fields", []).append(
                    {
                        "step": (ea if has_a else eb).get("step"),
                        "field": field,
                        "present_in": "a" if has_a else "b",
                    }
                )
                continue
            va, vb = float(ea[field]), float(eb[field])
            # NaN is legitimate and REPRODUCIBLE here: under fp16 the gradient scaler logs a NaN grad_norm on every
            # overflowing step and skips it, and which step overflows is deterministic.
            na, nb = va != va, vb != vb
            if na or nb:
                if na != nb and result["identical"]:
                    result["identical"] = False
                    result["first_diff_step"] = ea.get("step")
                continue
            # Equal is equal: subtracting is unsafe once a value can be infinite.
            # An fp16 overflow logs as NaN OR as inf (clip_grad_norm_ over an inf gradient returns inf), and abs(inf -
            # inf) is NaN, != 0.0, so two runs overflowing on the same step read as differing with max_abs_diff 0.0.
            if va == vb:
                continue
            diff = abs(va - vb)
            worst = max(worst, diff)
            if diff != 0.0 and result["identical"]:
                result["identical"] = False
                result["first_diff_step"] = ea.get("step")
        result["max_abs_diff"][field] = worst
    return result
