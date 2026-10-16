# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The LoftQ variant must initialize the adapter with LoftQ wherever the weights allow it.

PEFT drops loftq_config unless init_lora_weights is "loftq", so Studio sent the config alone and
every LoftQ run trained plain LoRA.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Union
from unittest.mock import Mock

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TRAINER = _BACKEND_ROOT / "core" / "training" / "trainer.py"
_WORKER = _BACKEND_ROOT / "core" / "training" / "worker.py"
_SENTENCE_TRANSFORMER = _BACKEND_ROOT.parents[1] / "unsloth" / "models" / "sentence_transformer.py"


def _functions(
    path,
    *names,
    namespace = None,
):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {} if namespace is None else namespace
    exec(compile(ast.Module(body = body, type_ignores = []), str(path), "exec"), namespace)
    return namespace


def test_loftq_turns_on_the_loftq_initialization():
    kwargs = _functions(_TRAINER, "_loftq_peft_kwargs")["_loftq_peft_kwargs"](True)

    assert kwargs["init_lora_weights"] == "loftq"
    assert kwargs["loftq_config"] == {"loftq_bits": 4, "loftq_iter": 1}


def test_plain_lora_keeps_the_default_initialization():
    loftq_kwargs = _functions(_TRAINER, "_loftq_peft_kwargs")["_loftq_peft_kwargs"]

    assert loftq_kwargs(False) == {"loftq_config": None}


@pytest.mark.parametrize("path", [_TRAINER, _WORKER], ids = lambda p: p.name)
def test_no_get_peft_model_call_passes_loftq_config_alone(path):
    for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get_peft_model"
        ):
            names = {k.arg for k in node.keywords}
            assert (
                "loftq_config" not in names or "init_lora_weights" in names
            ), f"{path.name}:{node.lineno} passes loftq_config without init_lora_weights"


def test_no_worker_call_reads_the_toggle_directly():
    # The persisted toggle alone would rerun LoftQ on a resume saved before LoftQ applied.
    tree = ast.parse(_WORKER.read_text(encoding = "utf-8"))
    helper = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_lora_uses_loftq"
    )
    inside = {id(n) for n in ast.walk(helper)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in inside:
            continue
        for value in [*node.args, *(k.value for k in node.keywords)]:
            for sub in ast.walk(value):
                reads_toggle = (
                    isinstance(sub, ast.Call)
                    and getattr(sub.func, "attr", None) == "get"
                    and any(
                        isinstance(a, ast.Constant) and a.value == "use_loftq" for a in sub.args
                    )
                )
                assert (
                    not reads_toggle
                ), f"worker.py:{node.lineno} passes use_loftq on without _lora_uses_loftq"


def test_the_fast_encoder_lora_config_forwards_the_initialization():
    # Encoder embedding models (MiniLM, BGE, ModernBERT) build their own LoraConfig.
    for node in ast.walk(ast.parse(_SENTENCE_TRANSFORMER.read_text(encoding = "utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LoraConfig":
            names = {k.arg for k in node.keywords}
            assert {"init_lora_weights", "loftq_config"} <= names, f"line {node.lineno}"


def _prepare(monkeypatch, quantized):
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    trainer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainer"
    )
    method = next(
        n
        for n in trainer.body
        if isinstance(n, ast.FunctionDef) and n.name == "prepare_model_for_training"
    )
    helpers = [
        n
        for n in tree.body
        if (
            isinstance(n, ast.FunctionDef)
            and n.name in ("normalize_gradient_checkpointing", "_loftq_peft_kwargs")
        )
        or (
            isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id in ("_UNSET", "LOFTQ_SKIPPED_ON_QUANTIZED")
                for t in n.targets
            )
        )
    ]
    adapter = Mock(return_value = SimpleNamespace(config = object()))
    namespace = {
        "logger": Mock(),
        "Union": Union,
        "FastLanguageModel": SimpleNamespace(get_peft_model = adapter),
        "_has_quantized_weights": lambda model: quantized,
    }
    monkeypatch.setitem(sys.modules, "unsloth", ModuleType("unsloth"))
    exec(
        compile(ast.Module(body = helpers + [method], type_ignores = []), str(_TRAINER), "exec"),
        namespace,
    )
    progress = Mock()
    instance = SimpleNamespace(
        model = SimpleNamespace(config = object()),
        _audio_type = None,
        is_audio_vlm = False,
        is_vlm = False,
        should_stop = False,
        _update_progress = progress,
        _use_gradient_checkpointing = "unsloth",
        _offload_peft_kwargs = lambda: {},
    )
    assert namespace["prepare_model_for_training"](instance, use_loftq = True) is True
    statuses = [c.kwargs.get("status_message") or "" for c in progress.call_args_list]
    return adapter.call_args.kwargs, statuses


def test_loftq_runs_on_dense_weights(monkeypatch):
    kwargs, statuses = _prepare(monkeypatch, quantized = False)

    assert kwargs["init_lora_weights"] == "loftq"
    assert not any("without LoftQ" in s for s in statuses)


def test_a_quantized_load_trains_without_loftq_and_says_so(monkeypatch):
    kwargs, statuses = _prepare(monkeypatch, quantized = True)

    assert "init_lora_weights" not in kwargs
    assert kwargs["loftq_config"] is None
    assert any("without LoftQ" in s for s in statuses)


def _uses_loftq(config):
    worker = _functions(_WORKER, "_lora_uses_loftq", namespace = {"Path": Path})
    return worker["_lora_uses_loftq"](config)


def _checkpoint(folder, init):
    folder.mkdir(parents = True)
    (folder / "adapter_config.json").write_text(
        json.dumps({"init_lora_weights": init}), encoding = "utf-8"
    )
    return str(folder)


def test_a_new_run_follows_the_toggle():
    assert _uses_loftq({"use_loftq": True}) is True
    assert _uses_loftq({"use_loftq": False}) is False


def test_a_resume_reruns_loftq_only_for_a_loftq_checkpoint(tmp_path):
    loftq = _checkpoint(tmp_path / "loftq" / "checkpoint-30", "loftq")
    plain = _checkpoint(tmp_path / "plain" / "checkpoint-30", True)

    assert _uses_loftq({"use_loftq": True, "resume_from_checkpoint": loftq}) is True
    # Saved before Studio applied LoftQ: the checkpoint trained on the original base weights.
    assert _uses_loftq({"use_loftq": True, "resume_from_checkpoint": plain}) is False
    gone = str(tmp_path / "gone")
    assert _uses_loftq({"use_loftq": True, "resume_from_checkpoint": gone}) is False
