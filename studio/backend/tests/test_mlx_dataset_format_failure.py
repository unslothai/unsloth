# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ignoring format_and_template_dataset's success=False would train on the raw, untemplated dataset."""

import ast
import inspect
import sys
import textwrap
import types

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
    formatted_eval = "FORMATTED_EVAL",
):
    """Runs the real block lifted from _run_mlx_training, which runs only on Apple Silicon."""
    block = compile(ast.Module(body = _format_block().body, type_ignores = []), "<fmt>", "exec")
    events = []
    calls = {"n": 0}

    def _fake_format(dataset, **_kwargs):
        calls["n"] += 1
        first = calls["n"] == 1
        return {
            "success": train_success if first else eval_success,
            "dataset": "FORMATTED_TRAIN" if first else formatted_eval,
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
            "eval_enabled": True,
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
        "eval_enabled": namespace.get("eval_enabled"),
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


def test_an_eval_split_formatted_to_nothing_disables_evaluation():
    result = _run(formatted_eval = [])

    assert result["error"] is None
    assert result["eval_dataset"] is None
    assert result["eval_enabled"] is False, "must not carve an eval split out of train"
    assert result["warnings"] == [
        "The eval dataset is empty after preprocessing, so this run has no evaluation."
    ]


def _run_mlx_until_model_load(monkeypatch, config, stop_requested):
    """Runs the real _run_mlx_training with MLX stubbed; a model load fails the test."""
    loaded = []
    events = []

    class _FastMLXModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            loaded.append(args)
            raise AssertionError("the model loaded")

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.metal = types.SimpleNamespace(is_available = lambda: False)
    mlx.core = mlx_core
    mlx_utils = types.ModuleType("mlx.utils")
    mlx_utils.tree_flatten = lambda tree: []
    mlx.utils = mlx_utils
    zoo_utils = types.ModuleType("unsloth_zoo.mlx.utils")
    zoo_utils.iter_mlx_lora_modules = lambda model: iter(())
    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _FastMLXModel
    trainer = types.ModuleType("unsloth_zoo.mlx.trainer")
    trainer.MLXTrainer = trainer.MLXTrainingConfig = trainer.train_on_responses_only = object
    s3_dataset = types.ModuleType("core.training.s3_dataset")
    s3_dataset.S3DownloadCancelled = type("S3DownloadCancelled", (Exception,), {})

    def _reject_manifest(*_args, **_kwargs):
        raise ValueError("ambiguous audio reference")

    s3_dataset.prepare_s3_dataset_download = _reject_manifest
    for name, module in {
        "mlx": mlx,
        "mlx.core": mlx_core,
        "mlx.utils": mlx_utils,
        "unsloth_zoo.mlx.utils": zoo_utils,
        "unsloth_zoo.mlx.loader": loader,
        "unsloth_zoo.mlx.trainer": trainer,
        "core.training.s3_dataset": s3_dataset,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(_worker, "_model_load_security_error", lambda *_args: None)
    monkeypatch.setattr(
        _worker,
        "_start_mlx_stop_poller",
        lambda _queue: ([True], stop_requested, [None], lambda: stop_requested[0], None),
    )

    class _Queue:
        def put(self, event):
            events.append(event)

    _worker._run_mlx_training(_Queue(), None, {"model_name": "org/model", **config})
    return loaded, events


def test_mlx_dataset_error_surfaces_before_the_model_loads(monkeypatch):
    with pytest.raises(ValueError, match = "ambiguous audio reference"):
        _run_mlx_until_model_load(monkeypatch, {"s3_config": {"bucket": "b"}}, [False])


def test_mlx_stop_during_dataset_load_skips_the_model_load(monkeypatch):
    stop_requested = [False]

    def _load_and_stop(*_args):
        stop_requested[0] = True
        return ["row"], None

    monkeypatch.setattr(_worker, "_load_hf_train_and_eval_datasets", _load_and_stop)
    loaded, events = _run_mlx_until_model_load(
        monkeypatch, {"hf_dataset": "org/data"}, stop_requested
    )

    assert loaded == []
    assert events[-1]["type"] == "complete"
    assert events[-1]["status_message"] == "Training cancelled"
