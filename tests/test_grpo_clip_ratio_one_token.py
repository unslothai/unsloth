# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO clip-ratio metrics must respect the completion mask for one-token completions.

`masked_batch_mean` inside the generated `compute_loss` took any width-1 tensor as sequence level
(`importance_sampling_level == "sequence"`) and returned a plain `x.mean()`. With
`max_completion_length = 1` the token-level tensors are also width 1, so fully masked rows stayed
in the denominator and clip_ratio/* was diluted. The helper is lifted from
`unsloth/models/rl_replacements.py` with `ast` so the test tracks the shipped source without
importing unsloth.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _load_masked_batch_mean(completion_mask):
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "masked_batch_mean"
    ]
    assert len(found) == 1, f"expected one masked_batch_mean, found {len(found)}"
    namespace = {
        "completion_mask": completion_mask,
        "completion_token_count": completion_mask.sum().clamp(min = 1.0),
    }
    exec(compile(ast.Module(body = found, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace["masked_batch_mean"]


def test_one_token_completion_excludes_masked_rows():
    completion_mask = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    clipped = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    masked_batch_mean = _load_masked_batch_mean(completion_mask)
    assert masked_batch_mean(clipped).item() == pytest.approx(0.5)


def test_sequence_level_still_plain_mean():
    completion_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    per_sequence = torch.tensor([[1.0], [0.0]])
    masked_batch_mean = _load_masked_batch_mean(completion_mask)
    assert masked_batch_mean(per_sequence).item() == pytest.approx(0.5)


def test_token_level_masked_mean():
    completion_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    clipped = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    masked_batch_mean = _load_masked_batch_mean(completion_mask)
    assert masked_batch_mean(clipped).item() == pytest.approx(2.0 / 3.0)
