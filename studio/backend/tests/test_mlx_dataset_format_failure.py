# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MLX text path must react to a dataset format failure the way the GPU trainer does.

format_and_template_dataset returns success=False when the chat template raised on every
row. Ignoring that leaves the raw, untemplated dataset in place and trains on it.
"""

import ast
import inspect
import textwrap

import pytest

import core.training.worker as _worker


def _format_block():
    """The `elif format_type:` branch of _run_mlx_training, which formats the dataset."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        branches = [node] + [o for o in node.orelse if isinstance(o, ast.If)]
        for branch in branches:
            if ast.unparse(branch.test) == "format_type":
                return branch
    raise AssertionError("could not locate the `elif format_type:` block")


def _run(
    train_success = True,
    train_warning = None,
    eval_success = True,
    eval_warning = None,
):
    """Execute the real block from _run_mlx_training and report what it did.

    _run_mlx_training only runs on Apple Silicon, so the block is lifted out and executed
    directly, the same technique test_mlx_training_worker_config.py uses.
    """
    block = compile(ast.Module(body = _format_block().body, type_ignores = []), "<fmt>", "exec")
    events = []
    calls = {"n": 0}

    def _fake_format(dataset, **_kwargs):
        calls["n"] += 1
        first = calls["n"] == 1
        return {
            "success": train_success if first else eval_success,
            "dataset": "FORMATTED_TRAIN" if first else "FORMATTED_EVAL",
            "final_format": "chatml_messages",
            "dropped_rows_warning": train_warning if first else eval_warning,
        }

    namespace = dict(vars(_worker))
    namespace.update(
        {
            "format_type": "chatml_messages",
            "dataset": "RAW_TRAIN",
            "eval_dataset": "RAW_EVAL",
            "model_name": "org/model",
            "tokenizer": object(),
            "hf_dataset": "local",
            "custom_format_mapping": None,
            "_fmt_progress": lambda **_kw: None,
            "_send": lambda kind, **kw: events.append((kind, kw)),
            "format_and_template_dataset": _fake_format,
        }
    )

    error = None
    try:
        exec(block, namespace)
    except ValueError as exc:
        error = str(exc)

    return {
        "dataset": namespace.get("dataset"),
        "eval_dataset": namespace.get("eval_dataset"),
        "warnings": [kw["message"] for kind, kw in events if kind == "warning"],
        "error": error,
    }


def test_a_failed_train_dataset_stops_the_run():
    result = _run(train_success = False)

    assert result["error"] is not None
    assert result["dataset"] == "RAW_TRAIN", "the untemplated dataset must not reach training"


def test_a_failed_eval_dataset_stops_the_run():
    result = _run(eval_success = False)

    assert result["error"] is not None
    assert result["eval_dataset"] == "RAW_EVAL"


def test_dropped_row_warnings_reach_the_user_for_both_datasets():
    result = _run(
        train_warning = "Dropped 3 of 8 rows because the chat template failed: boom",
        eval_warning = "Dropped 1 of 4 rows because the chat template failed: boom",
    )

    assert result["error"] is None
    assert result["dataset"] == "FORMATTED_TRAIN"
    assert result["eval_dataset"] == "FORMATTED_EVAL"
    assert result["warnings"] == [
        "Dropped 3 of 8 rows because the chat template failed: boom",
        "Eval dataset: Dropped 1 of 4 rows because the chat template failed: boom",
    ]


def test_a_clean_run_is_unchanged():
    result = _run()

    assert result["error"] is None
    assert result["dataset"] == "FORMATTED_TRAIN"
    assert result["eval_dataset"] == "FORMATTED_EVAL"
    assert result["warnings"] == []
