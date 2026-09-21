# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from types import SimpleNamespace

import pytest

from core.inference import worker


@pytest.fixture
def outputs(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    return tmp_path / "outputs"


def _model_dir(
    root,
    name,
    config = None,
    adapter = None,
):
    d = root / name
    d.mkdir(parents = True)
    (d / "config.json").write_text(json.dumps(config or {"model_type": "llama"}))
    (d / "model.safetensors").write_bytes(b"")
    if adapter is not None:
        (d / "adapter_config.json").write_text(json.dumps(adapter))
    return str(d)


def _mc(
    path,
    is_lora = False,
    base_model = None,
):
    return SimpleNamespace(is_lora = is_lora, path = path, base_model = base_model)


def test_full_finetune_output_loads_16bit(outputs):
    path = _model_dir(outputs, "unsloth_Qwen3-0.6B_1771227800")
    assert worker._resolve_lora_4bit(_mc(path), True) is False
    assert worker._resolve_lora_4bit(_mc(path), False) is False


def test_quantized_or_foreign_models_keep_4bit(outputs, tmp_path):
    quantized = _model_dir(outputs, "q", {"quantization_config": {"quant_method": "bitsandbytes"}})
    adapter = _model_dir(outputs, "a", adapter = {})
    outside = _model_dir(tmp_path / "exports", "merged")
    for path in (quantized, adapter, outside, "unsloth/Qwen3-0.6B"):
        assert worker._resolve_lora_4bit(_mc(path), True) is True, path
    qlora = _model_dir(outputs, "qlora", adapter = {"unsloth_training_method": "qlora"})
    assert worker._resolve_lora_4bit(_mc(qlora, is_lora = True, base_model = "x"), True) is True
