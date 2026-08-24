# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


_BACKEND = Path(__file__).resolve().parent.parent
_TRAINER = _BACKEND / "core" / "training" / "trainer.py"
_STRUCTLOG_METHODS = {"debug", "info", "warning", "error", "exception"}


class _StructlogCompatibleLogger:
    """Minimal logger whose methods require structlog's positional event."""

    def __getattr__(self, name):
        if name not in _STRUCTLOG_METHODS:
            raise AttributeError(name)

        def log(event, **_kwargs):
            return event

        return log


def _prepare_model_for_training():
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    trainer_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UnslothTrainer"
    )
    method = next(
        node
        for node in trainer_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_model_for_training"
    )
    namespace = {"logger": _StructlogCompatibleLogger()}
    exec(compile(ast.Module(body = [method], type_ignores = []), str(_TRAINER), "exec"), namespace)
    return namespace[method.name]


def test_audio_vlm_lora_summary_reaches_fast_model_with_structlog_logger(monkeypatch):
    calls = []

    class FastModel:
        @staticmethod
        def get_peft_model(model, **kwargs):
            calls.append((model, kwargs))
            return model

    unsloth = ModuleType("unsloth")
    unsloth.FastModel = FastModel
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)

    trainer = SimpleNamespace(
        model = SimpleNamespace(config = SimpleNamespace()),
        _audio_type = None,
        is_audio_vlm = True,
        is_vlm = False,
        should_stop = False,
        _update_progress = lambda **_kwargs: None,
    )

    assert _prepare_model_for_training()(trainer) is True
    assert len(calls) == 1
    assert calls[0][0] is trainer.model


def test_production_loggers_never_call_structlog_methods_without_an_event():
    violations = []
    for path in _BACKEND.rglob("*.py"):
        if "tests" in path.relative_to(_BACKEND).parts:
            continue
        tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
                and node.func.attr in _STRUCTLOG_METHODS
                and not node.args
                and not any(keyword.arg == "event" for keyword in node.keywords)
            ):
                violations.append(f"{path.relative_to(_BACKEND)}:{node.lineno}")

    assert violations == []
