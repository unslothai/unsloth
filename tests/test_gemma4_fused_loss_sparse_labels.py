# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Regression tests for unsloth#5230.

Gemma4's multimodal masking (``mm_token_type_ids``) combined with an active
LoRA adapter can produce a batch where almost every label is ``-100``. In that
regime ``unsloth_fused_ce_loss``'s chunked kernel can zero out gradients
entirely instead of computing a correct (if noisy) loss. The fix in
``unsloth/models/llama.py`` adds a guard: once the fraction of valid
(non ``-100``) shifted labels drops below ``GEMMA4_SPARSE_LABEL_MAX_VALID_FRAC``
for a Gemma4 model with an active LoRA adapter and ``mm_token_type_ids``
present, ``RETURN_LOGITS`` is forced ``True`` so the forward pass falls back
to the standard (unfused) cross-entropy path instead of the fused one.

``unsloth/models/llama.py`` unconditionally imports triton-backed kernels
(``fast_cross_entropy_loss``, ``unsloth_fused_ce_loss``), so it cannot be
imported on a GPU-free/triton-free runner. These tests instead extract the
exact guard source text via ``ast`` and execute it in a controlled namespace
with real ``torch`` CPU tensors -- real code, real execution, no triton
required. If the guard block is missing (e.g. on the pre-fix source), the
extraction helper raises, so these tests fail on the unpatched source and
pass once the guard lands (reproduction / base-fails, head-passes)."""

from __future__ import annotations

import ast
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_REPO = Path(__file__).resolve().parent.parent
_LLAMA_PATH = _REPO / "unsloth" / "models" / "llama.py"
_CE_KERNEL_PATH = _REPO / "unsloth" / "kernels" / "cross_entropy_loss.py"

_GUARD_ENV_VAR = "UNSLOTH_GEMMA4_SPARSE_LOSS_GUARD"
_THRESHOLD_NAME = "GEMMA4_SPARSE_LABEL_MAX_VALID_FRAC"
_LORA_HELPER_NAME = "_gemma4_model_has_active_lora"


def _llama_source() -> str:
    return _LLAMA_PATH.read_text(encoding = "utf-8")


def _extract_top_level(source: str, name: str) -> str:
    """Return the source text of the top-level (module-scope) def/assignment
    called `name`. Top-level nodes have clean column offsets, so
    `ast.get_source_segment` needs no dedenting."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets:
                return ast.get_source_segment(source, node)
    raise AssertionError(f"top-level `{name}` not found in {_LLAMA_PATH}")


def _extract_guard_block(source: str) -> str:
    """Return the dedented source of the `if (...): ...` guard block that
    tests `UNSLOTH_GEMMA4_SPARSE_LOSS_GUARD`. Located by scanning `If` nodes
    (outer-first, matching `ast.walk`'s BFS order) for one whose test
    references the guard env var; raises if the guard is absent (pre-fix
    source), which is what makes these tests fail on base / pass on head."""
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.get_source_segment(source, node.test) or ""
        if _GUARD_ENV_VAR in test_src:
            raw = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            return textwrap.dedent(raw)
    raise AssertionError(
        f"guard `if` block referencing {_GUARD_ENV_VAR} not found in {_LLAMA_PATH} "
        f"-- the sparse-label fused-CE guard from unsloth#5230 appears to be missing."
    )


def _build_namespace() -> dict:
    """A namespace with just enough real machinery (`os`, `torch`) plus the
    extracted constant and LoRA-detection helper for the guard block to run."""
    ns: dict = {"os": os, "torch": torch}
    exec(_extract_top_level(_llama_source(), _THRESHOLD_NAME), ns)
    exec(_extract_top_level(_llama_source(), _LORA_HELPER_NAME), ns)
    return ns


class _FakeLoraModule:
    """Mimics a PEFT `LoraLayer`: presence of `lora_A` is what the guard's
    LoRA-detection helper checks for. Supports ``disable_adapters`` and
    ``merged`` to mirror real adapter state flags."""
    def __init__(self, *, disable_adapters: bool = False, merged: bool = False):
        self.lora_A = object()
        self.disable_adapters = disable_adapters
        self.merged = merged


class _FakeModel:
    """Stand-in for `self` inside `_CausalLM_fast_forward`: only needs
    `.config.model_type` and `.modules()` for the guard block / LoRA helper."""
    def __init__(
        self,
        model_type: str,
        has_lora: bool,
        *,
        lora_disabled: bool = False,
        lora_merged: bool = False,
    ) -> None:
        self.config = SimpleNamespace(model_type = model_type)
        self._has_lora = has_lora
        self._lora_disabled = lora_disabled
        self._lora_merged = lora_merged

    def modules(self):
        if not self._has_lora:
            return [self]
        return [
            self,
            _FakeLoraModule(
                disable_adapters = self._lora_disabled,
                merged = self._lora_merged,
            ),
        ]


def _sparse_labels(total: int, valid: int) -> torch.LongTensor:
    """Shifted-label fraction is computed from `labels[..., 1:]`, so build a
    (1, total) label row with exactly `valid` non-`-100` entries among the
    last `total - 1` (shifted) positions."""
    assert 0 <= valid <= total - 1
    labels = torch.full((1, total), -100, dtype = torch.long)
    labels[0, 1 : 1 + valid] = 1
    return labels


def _run_guard(
    *,
    labels: torch.Tensor,
    model_type: str = "gemma4",
    has_mm_token_type_ids: bool = True,
    has_lora: bool = True,
    lora_disabled: bool = False,
    lora_merged: bool = False,
    env_override: str | None = None,
) -> bool:
    """Extract the guard block from the current source and execute it with
    the given inputs; returns the resulting `RETURN_LOGITS` value."""
    ns = _build_namespace()
    ns["RETURN_LOGITS"] = False
    ns["labels"] = labels
    ns["self"] = _FakeModel(
        model_type, has_lora,
        lora_disabled = lora_disabled,
        lora_merged = lora_merged,
    )
    ns["kwargs"] = {"mm_token_type_ids": torch.zeros(1)} if has_mm_token_type_ids else {}

    guard_src = _extract_guard_block(_llama_source())

    prev = os.environ.get(_GUARD_ENV_VAR)
    try:
        if env_override is None:
            os.environ.pop(_GUARD_ENV_VAR, None)
        else:
            os.environ[_GUARD_ENV_VAR] = env_override
        exec(guard_src, ns)
    finally:
        if prev is None:
            os.environ.pop(_GUARD_ENV_VAR, None)
        else:
            os.environ[_GUARD_ENV_VAR] = prev

    return ns["RETURN_LOGITS"]


# ---------------------------------------------------------------------------
# Reproduction / behavioral: base-fails, head-passes.
# ---------------------------------------------------------------------------

def test_gemma4_sparse_label_guard_fires():
    """1 valid label out of 19 shifted positions (~5%) on a gemma4 model with
    an active LoRA adapter and `mm_token_type_ids` must force RETURN_LOGITS
    True, so the forward pass skips `unsloth_fused_ce_loss` and its
    gradient-zeroing failure mode on this input shape. On the unpatched
    source, `_extract_guard_block` raises `AssertionError` since the guard
    doesn't exist yet -- this test fails on base and passes on head."""
    labels = _sparse_labels(total = 20, valid = 1)
    assert _run_guard(labels = labels) is True


# ---------------------------------------------------------------------------
# Behavioral: threshold boundary.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("valid", "total", "expect_fires"),
    [
        (1, 20, True),    # ~5% valid, well under the 15% threshold
        (2, 20, True),    # ~10.5% valid (2/19), still under threshold
        (3, 20, False),   # ~15.8% valid (3/19), at/above threshold
        (10, 20, False),  # ~52.6% valid, well above threshold
    ],
)
def test_valid_label_fraction_below_threshold(valid, total, expect_fires):
    labels = _sparse_labels(total = total, valid = valid)
    assert _run_guard(labels = labels) is expect_fires


# ---------------------------------------------------------------------------
# Behavioral: guard must not fire outside its exact trigger conditions.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("kwargs_override", "reason"),
    [
        ({"labels": _sparse_labels(total = 20, valid = 18)}, "dense labels"),
        ({"model_type": "llama"}, "non-gemma4 model_type"),
        ({"has_mm_token_type_ids": False}, "no mm_token_type_ids"),
        ({"has_lora": False}, "no active LoRA adapter"),
        ({"lora_disabled": True}, "LoRA adapters disabled"),
        ({"lora_merged": True}, "LoRA adapters merged"),
    ],
)
def test_guard_does_not_fire_for_dense_labels_or_non_gemma4(kwargs_override, reason):
    base_kwargs = {"labels": _sparse_labels(total = 20, valid = 1)}
    base_kwargs.update(kwargs_override)
    assert _run_guard(**base_kwargs) is False, f"guard should not fire: {reason}"


# ---------------------------------------------------------------------------
# Behavioral: opt-out env var.
# ---------------------------------------------------------------------------

def test_env_var_disables_guard():
    labels = _sparse_labels(total = 20, valid = 1)
    # Default / explicit "1": guard fires.
    assert _run_guard(labels = labels, env_override = "1") is True
    # Explicit "0": guard is disabled even though the sparse condition holds.
    assert _run_guard(labels = labels, env_override = "0") is False


# ---------------------------------------------------------------------------
# Static: the fallback path the guard routes into actually masks -100 labels.
# ---------------------------------------------------------------------------

def test_cross_entropy_masks_ignored_labels():
    """When the guard fires, RETURN_LOGITS=True routes the forward pass past
    the fused-CE branch into the standard `fast_cross_entropy_loss` path
    (around llama.py's `shift_labels` block), which must mask ignored labels
    correctly -- otherwise the "fix" would just move the bug. Verifies,
    without needing triton: (1) `shift_labels` is boundary-masked with -100
    in llama.py, and (2) `fast_cross_entropy_loss` counts only non -100
    labels for its denominator in cross_entropy_loss.py."""
    llama_source = _llama_source()
    assert "shift_labels[..., -1] = -100" in llama_source, (
        "llama.py must mask the final shifted-label position with -100"
    )
    assert "mask_packed_sequence_boundaries(" in llama_source, (
        "llama.py must mask packed-sequence boundaries in shift_labels"
    )

    ce_source = _CE_KERNEL_PATH.read_text(encoding = "utf-8")
    assert "n_items = torch.count_nonzero(labels != -100)" in ce_source, (
        "fast_cross_entropy_loss must exclude -100 labels from its loss denominator"
    )
