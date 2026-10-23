# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Self-contained on purpose: the Kaggle payload is inlined with no repo checkout to import from."""

from __future__ import annotations

import json
import os
import random
from typing import Any


# CUDA reads this at handle creation on first use; setting it later is silently ignored.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def enable_full_determinism() -> None:
    """Sets env vars that only take effect before torch initialises CUDA; call before any torch import."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    os.environ["PYTHONHASHSEED"] = "0"
    # A tokenizers worker pool can make dataset .map ordering nondeterministic.
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
    """warn_only stays True because some bitsandbytes and Triton kernels have no deterministic version."""
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
    """Only fires on logged steps, so the caller must set logging_steps=1; reads the Trainer's grad_norm."""

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
    """Step i yields row i % dataset_length; a pure function of the step index, with no shuffle or RNG."""

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
    """identical is bitwise equality; a field logged by one run only, or a moved step, is a difference."""
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
            # fp16 NaN grad norms are reproducible; NaN != NaN would flag them.
            na, nb = va != va, vb != vb
            if na or nb:
                if na != nb and result["identical"]:
                    result["identical"] = False
                    result["first_diff_step"] = ea.get("step")
                continue
            # Equal first: two infs on the same step would subtract to NaN.
            if va == vb:
                continue
            diff = abs(va - vb)
            worst = max(worst, diff)
            if diff != 0.0 and result["identical"]:
                result["identical"] = False
                result["first_diff_step"] = ea.get("step")
        result["max_abs_diff"][field] = worst
    return result
