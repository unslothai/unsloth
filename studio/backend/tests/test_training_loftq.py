# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The LoftQ variant must initialize the adapter with LoftQ, and say so when it cannot.

PEFT drops loftq_config unless init_lora_weights is "loftq", so Studio sent the config alone and
every LoftQ run trained plain LoRA.
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from models.training import TrainingStartRequest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_TRAINER = _BACKEND_ROOT / "core" / "training" / "trainer.py"
_WORKER = _BACKEND_ROOT / "core" / "training" / "worker.py"


def _loftq_peft_kwargs():
    tree = ast.parse(_TRAINER.read_text(encoding = "utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_loftq_peft_kwargs"
    )
    namespace = {}
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(_TRAINER), "exec"), namespace)
    return namespace["_loftq_peft_kwargs"]


def test_loftq_turns_on_the_loftq_initialization():
    kwargs = _loftq_peft_kwargs()(True)

    assert kwargs["init_lora_weights"] == "loftq"
    assert kwargs["loftq_config"] == {"loftq_bits": 4, "loftq_iter": 1}


def test_plain_lora_keeps_the_default_initialization():
    assert _loftq_peft_kwargs()(False) == {"loftq_config": None}


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


@pytest.fixture
def route():
    spec = importlib.util.spec_from_file_location(
        "loftq_training_route", _BACKEND_ROOT / "routes" / "training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._hub_unreachable = lambda: False
    return module


@pytest.fixture
def local_model(tmp_path):
    folder = tmp_path / "qwen2"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"}),
        encoding = "utf-8",
    )
    (folder / "model.safetensors").write_bytes(b"x")
    return str(folder)


async def _inline_to_thread(function, *args, **kwargs):
    return function(*args, **kwargs)


def _start(route, local_model, **overrides):
    started = {}
    backend = SimpleNamespace(
        current_job_id = None,
        is_training_active = lambda: False,
        start_training = lambda **kwargs: started.update(kwargs) or True,
    )
    request = TrainingStartRequest(
        **{
            "model_name": local_model,
            "training_type": "LoRA/QLoRA",
            "format_type": "auto",
            "hf_dataset": "org/data",
            "use_loftq": True,
            **overrides,
        }
    )
    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "_preflight_hf_dataset_request", return_value = None),
        patch.object(route, "_refuse_unauthorized_cached_dataset", return_value = None),
        patch.object(route.asyncio, "to_thread", _inline_to_thread),
        patch(
            "core.training.provenance.effective_training_load_in_4bit",
            side_effect = lambda config, *a, **kw: config["load_in_4bit"],
        ),
    ):
        asyncio.run(route.start_training(request, current_subject = "test-user"))
    return started


def test_loftq_on_a_4bit_load_is_refused_before_the_model_loads(route, local_model):
    with pytest.raises(HTTPException) as refused:
        _start(route, local_model, load_in_4bit = True)

    assert refused.value.status_code == 400
    assert refused.value.detail["code"] == "training_loftq_needs_16bit"


def test_loftq_on_a_16bit_load_starts(route, local_model):
    started = _start(route, local_model, load_in_4bit = False)

    assert started["use_loftq"] is True
    assert started["load_in_4bit"] is False
