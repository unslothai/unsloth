# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO no-grad pass must never put more rows in one forward than the autotune says fit."""

import ast
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
SOURCE = SOURCE_PATH.read_text(encoding = "utf-8")


def _sizing_block():
    outer = next(
        n
        for n in ast.parse(SOURCE).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "grpo_trainer__get_per_token_logps_and_entropies"
    )
    inner = next(
        n
        for n in ast.walk(outer)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_per_token_logps_and_entropies"
    )
    src = ast.get_source_segment(SOURCE, inner)
    start = src.rindex("\n", 0, src.index("if self.args.unsloth_grpo_mini_batch is None:")) + 1
    end = src.index("all_logprobs_list = []", start)
    return textwrap.dedent(src[start:end])


def _rows_per_forward(total_rows, rows_fit):
    scope = {
        "torch": torch,
        "self": SimpleNamespace(
            args = SimpleNamespace(unsloth_grpo_mini_batch = None, unsloth_logit_chunk_multiplier = None)
        ),
        "autotune_batch_and_chunks": lambda *args: (rows_fit, 4),
        "_grpo_collective_forwards": False,
        "input_ids": torch.zeros(total_rows, 8, dtype = torch.long),
        "total_rows": total_rows,
        "seq_len": 8,
        "hidden_dim": 4,
        "vocab_dim": 16,
        "dtype_bytes": 16,
    }
    exec(_sizing_block(), scope)
    # The chunker right below the block: batch_size = ceil(total_samples / B).
    return math.ceil(total_rows / scope["B"])


@pytest.mark.parametrize(
    "total_rows, rows_fit",
    [(8, 5), (16, 9), (16, 15), (7, 4), (3, 2), (2, 4)],
)
def test_split_count_rounds_up_so_the_fit_is_honoured(total_rows, rows_fit):
    assert _rows_per_forward(total_rows, rows_fit) <= rows_fit


@pytest.mark.parametrize("total_rows, rows_fit", [(8, 8), (8, 4), (8, 1), (6, 3)])
def test_plan_is_unchanged_when_the_fit_divides_the_rows(total_rows, rows_fit):
    assert _rows_per_forward(total_rows, rows_fit) == rows_fit


def _validate(splits, generation_batch_size = 8):
    src = (SOURCE_PATH.parent / "rl.py").read_text(encoding = "utf-8")
    start = src.index("        if unsloth_grpo_mini_batch is not None:")
    end = src.index(
        "        self.unsloth_logit_chunk_multiplier = unsloth_logit_chunk_multiplier", start
    )
    config = SimpleNamespace(generation_batch_size = generation_batch_size)
    exec(textwrap.dedent(src[start:end]), {"self": config, "unsloth_grpo_mini_batch": splits})
    return config.unsloth_grpo_mini_batch


@pytest.mark.parametrize("splits", [0, -1, 9])
def test_config_rejects_split_counts_outside_one_to_the_batch(splits):
    # 0 / negative divided by zero or ran no forward in the scoring pass.
    with pytest.raises(ValueError, match = "at least 1"):
        _validate(splits)


@pytest.mark.parametrize("splits", [1, 8])
def test_config_keeps_valid_split_counts(splits):
    assert _validate(splits) == splits


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
