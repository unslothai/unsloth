# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Determinism and metric-capture primitives for the Kaggle T4 smoke test.

Self-contained on purpose. The payload for this test is shipped to a Kaggle
kernel as an inlined notebook with no repo checkout and no network fetch of
our sources, so it cannot import a helper that only exists on the machine
that built it. Everything the payload needs is either in this file or in the
notebook that carries it.

Four primitives, mirroring the workspace debugging utilities they were
modelled on:

``set_all_seeds_fast(seed)``
    Seeds ``random`` / ``numpy`` / ``torch`` (+ CUDA). Fast path: does NOT
    turn on deterministic algorithms.

``enable_full_determinism()``
    The slow path. MUST be called before ``import torch`` for the cuBLAS
    workspace setting to take effect, so it is exposed separately rather
    than folded into the seeding call.

``StatisticsCallback``
    ``transformers.TrainerCallback`` that accumulates one dict per logged
    step into ``.logs`` -- ``step``, ``loss``, ``grad_norm``, ``learning_rate``.
    Requires ``logging_steps=1``.

``RepeatingSequentialSampler``
    Fixed, shuffle-free sampling order. Step *i* draws row ``i % len(dataset)``
    repeated across the whole effective batch, so the sample sequence is a
    pure function of the step index and nothing else.

``compare_metrics``
    Compares two metric lists and reports max absolute deviation per field.

A note on what these can and cannot buy, because the assertions in
``run_t4_smoke.py`` depend on the distinction:

* Run-to-run inside ONE process/session: bitwise reproducible is achievable
  and is asserted exactly.
* Across GPU architectures, driver versions or library versions: bitwise is
  NOT achievable. Reduction order, kernel selection and the fp16 vs bf16
  choice all move the low bits. Cross-environment checks are tolerance
  bands, never equality.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any


# cuBLAS needs a fixed workspace for its GEMM reductions to be reproducible.
# CUDA reads this when the cuBLAS handle is created, which happens on first
# use after `import torch` -- setting it later is silently ignored, which is
# the classic way to believe you have determinism and not have it.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def enable_full_determinism() -> None:
    """Set the env vars that only take effect before torch initialises CUDA.

    Call this at the very top of the entry point, before any torch import.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    os.environ["PYTHONHASHSEED"] = "0"
    # A tokenizers worker pool introduces a nondeterministic interleave in
    # dataset .map ordering under some versions; the dataset here is tiny so
    # there is nothing to gain from it either way.
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

    ``warn_only=True`` is the default and is deliberate. Unsloth's 4-bit path
    goes through bitsandbytes and through fused Triton kernels, and at least
    some of those have no deterministic implementation registered. With
    ``warn_only=False`` torch raises and the smoke test dies having proved
    nothing about the notebook. With ``warn_only=True`` torch uses the
    deterministic kernel wherever one exists and warns where one does not,
    which is strictly better than not asking, and the run-to-run equality
    assertion is what actually verifies the result.
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

    Reads the values the Trainer itself logs rather than recomputing a grad
    norm from the parameters. Recomputing would report a norm measured AFTER
    the optimizer step and after gradients were zeroed, which is either a
    different quantity or zero depending on the transformers version. The
    logged value is the pre-clip norm the trainer used, which is the number
    a regression would actually move.

    Only fires on steps the Trainer logs, so the caller must set
    ``logging_steps=1``.
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
    ``batch_size * gradient_accumulation_steps`` times. The order is a pure
    function of the step index, so it does not depend on the RNG state, on
    the dataset length modulo the batch size, or on which epoch boundary the
    run happens to land near.
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

    Two things that are NOT numeric deviations are still differences here,
    because both are the run-to-run comparison's own subject matter:

    * A field logged by one run and not by the other. Same-length lists whose
      entries carry different keys are two different traces, and no arithmetic
      covers it. ``check_reference`` already calls that "a change in the SHAPE
      of what the trainer logged"; the exact comparator cannot be laxer than
      the tolerance band beside it.
    * A step coordinate that moved. The lists are zipped positionally, so
      values are only comparable while the entries they came from describe the
      same step. A shifted, duplicated or reordered ``step`` makes every later
      pairing meaningless AND is itself the kind of trainer nondeterminism
      this comparison exists to catch, so it is reported rather than zipped
      over.
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
            # NaN is a legitimate, REPRODUCIBLE value here. Under fp16 the
            # gradient scaler reports a NaN grad_norm on any step whose
            # gradients overflowed, and it then skips that step -- which
            # step overflows is itself deterministic. Comparing with
            # abs(a - b) would make every such step register as a
            # difference, because NaN != NaN, and the run-to-run assertion
            # would fail on runs that are in fact identical.
            na, nb = va != va, vb != vb
            if na or nb:
                if na != nb and result["identical"]:
                    result["identical"] = False
                    result["first_diff_step"] = ea.get("step")
                continue
            diff = abs(va - vb)
            worst = max(worst, diff)
            if diff != 0.0 and result["identical"]:
                result["identical"] = False
                result["first_diff_step"] = ea.get("step")
        result["max_abs_diff"][field] = worst
    return result
