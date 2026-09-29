# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise audio adapter setup with a real logger without loading GPU models."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Union
from unittest.mock import Mock

import pytest
import structlog


def _module_level_dependencies(tree):
    """Defaults and annotations are evaluated at def time, so exec needs the module-level
    names the signature reads."""
    return [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "normalize_gradient_checkpointing")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_UNSET" for t in node.targets)
        )
    ]


@pytest.mark.parametrize("audio_type", ["csm", "bicodec", "dac", "snac", "audio_vlm"])
def test_audio_lora_setup_accepts_real_structlog(monkeypatch, audio_type):
    source = Path(__file__).resolve().parents[1] / "core/training/trainer.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    trainer = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnslothTrainer"
    )
    method = next(
        n
        for n in trainer.body
        if isinstance(n, ast.FunctionDef) and n.name == "prepare_model_for_training"
    )
    namespace = {"logger": structlog.get_logger(), "Union": Union}
    adapter = Mock(return_value = SimpleNamespace(config = object()))
    fast_model = SimpleNamespace(get_peft_model = adapter)
    module = ModuleType("unsloth")
    module.FastModel = fast_model
    monkeypatch.setitem(sys.modules, "unsloth", module)
    namespace["FastLanguageModel"] = fast_model
    body = _module_level_dependencies(tree) + [method]
    exec(compile(ast.Module(body = body, type_ignores = []), str(source), "exec"), namespace)
    model = SimpleNamespace(config = object())
    progress = Mock()
    instance = SimpleNamespace(
        model = model,
        _audio_type = None if audio_type == "audio_vlm" else audio_type,
        is_audio_vlm = audio_type == "audio_vlm",
        is_vlm = False,
        should_stop = False,
        _update_progress = progress,
        _use_gradient_checkpointing = "unsloth",
    )

    assert namespace["prepare_model_for_training"](instance) is True
    adapter.assert_called_once()
    assert adapter.call_args.args == (model,)
    assert instance.model is adapter.return_value
    progress.assert_called_once_with(status_message = "LoRA adapters configured")
