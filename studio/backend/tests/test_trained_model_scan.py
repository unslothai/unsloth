# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Studio trained-model discovery used by Chat."""

import json
from pathlib import Path
import sys
import types as _types
import importlib


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from unittest.mock import patch

from utils.models.model_config import (
    ModelConfig,
    get_base_model_from_checkpoint,
    get_base_model_from_lora,
    scan_trained_models,
)


def test_scan_trained_models_includes_lora_and_full_finetune_outputs(
    tmp_path: Path, monkeypatch
):
    # resolve_output_dir refuses absolutes outside outputs_root; point it at tmp_path.
    from utils.models import model_config as _mc
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: tmp_path)
    monkeypatch.setattr(_mc, "outputs_root", lambda: tmp_path)

    lora_dir = tmp_path / "unsloth_SmolLM-135M_1775412608"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (lora_dir / "adapter_model.safetensors").write_bytes(b"")

    full_dir = tmp_path / "unsloth_SmolLM-135M_full_1775412609"
    full_dir.mkdir()
    (full_dir / "config.json").write_text(
        json.dumps({"_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (full_dir / "model.safetensors").write_bytes(b"")

    found = {
        name: (path, model_type)
        for name, path, model_type in scan_trained_models(str(tmp_path))
    }

    assert found[lora_dir.name] == (str(lora_dir), "lora")
    assert found[full_dir.name] == (str(full_dir), "merged")


def test_get_base_model_from_checkpoint_falls_back_to_full_finetune_config(
    tmp_path: Path,
):
    (tmp_path / "config.json").write_text(
        json.dumps({"_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert get_base_model_from_checkpoint(str(tmp_path)) == "HuggingFaceTB/SmolLM-135M"


def test_get_base_model_from_lora_rejects_full_finetune_dirs(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert get_base_model_from_lora(str(tmp_path)) is None


@patch("utils.models.model_config.is_audio_input_type", return_value = False)
@patch("utils.models.model_config.detect_audio_type", return_value = None)
@patch("utils.models.model_config.is_vision_model", return_value = False)
def test_model_config_full_finetune_local_path_is_not_lora(
    _mock_vision,
    _mock_audio_type,
    _mock_audio_input,
    tmp_path: Path,
):
    (tmp_path / "config.json").write_text(
        json.dumps({"_name_or_path": "unsloth/Qwen3-4B"})
    )
    (tmp_path / "model.safetensors").write_bytes(b"")

    config = ModelConfig.from_identifier(str(tmp_path))

    assert config is not None
    assert config.is_lora is False
    assert config.base_model is None


def test_scan_trained_loras_aliases_scan_trained_models():
    utils_models = importlib.import_module("utils.models")
    core_module = importlib.import_module("core")

    assert utils_models.scan_trained_loras is utils_models.scan_trained_models
    assert core_module.scan_trained_loras is core_module.scan_trained_models
