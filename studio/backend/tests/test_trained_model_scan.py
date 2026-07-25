# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Unsloth trained-model discovery used by Chat."""

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
    get_base_model_from_lora_identifier,
    scan_trained_models,
)


def test_scan_trained_models_includes_lora_and_full_finetune_outputs(tmp_path: Path, monkeypatch):
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
        name: (path, model_type) for name, path, model_type in scan_trained_models(str(tmp_path))
    }

    assert found[lora_dir.name] == (str(lora_dir), "lora")
    assert found[full_dir.name] == (str(full_dir), "merged")


def test_get_base_model_from_checkpoint_falls_back_to_full_finetune_config(tmp_path: Path):
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


def test_lora_identifier_resolves_local_dir_like_the_local_helper(tmp_path: Path):
    # Local path: behaves like the directory reader, no Hub call.
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    with patch("huggingface_hub.hf_hub_download", side_effect = AssertionError("no Hub call")):
        assert get_base_model_from_lora_identifier(str(tmp_path)) == "HuggingFaceTB/SmolLM-135M"


def test_lora_identifier_resolves_remote_adapter_base(tmp_path: Path):
    # Remote adapter: the identifier helper fetches adapter_config.json from the Hub so
    # the gate can scan the base, where the local helper returns None.
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B-Instruct"}))

    def _dl(
        repo,
        fn,
        token = None,
        cache_dir = None,
    ):
        assert repo == "someone/my-remote-lora"
        assert fn == "adapter_config.json"
        return str(cfg)

    assert get_base_model_from_lora("someone/my-remote-lora") is None  # local-only: misses it
    with patch("huggingface_hub.hf_hub_download", side_effect = _dl):
        base = get_base_model_from_lora_identifier("someone/my-remote-lora")
    assert base == "unsloth/Llama-3.2-1B-Instruct"


def test_lora_identifier_returns_none_for_non_adapter_remote_repo():
    # Non-LoRA remote repo: a 404 on adapter_config.json returns None without retrying.
    from huggingface_hub.utils import EntryNotFoundError

    mock = patch("huggingface_hub.hf_hub_download", side_effect = EntryNotFoundError("404"))
    with mock as m:
        assert get_base_model_from_lora_identifier("unsloth/Llama-3.2-1B-Instruct") is None
    assert m.call_count == 1  # 404 is definitive -> no retry


def test_lora_identifier_retries_transient_then_resolves(tmp_path: Path):
    # A transient error is retried (not treated as "not a LoRA"); the retry resolves the base.
    cfg = tmp_path / "adapter_config.json"
    cfg.write_text(json.dumps({"base_model_name_or_path": "unsloth/Llama-3.2-1B-Instruct"}))
    calls = {"n": 0}

    def _dl(
        repo,
        fn,
        token = None,
        cache_dir = None,
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient network blip")
        return str(cfg)

    with patch("huggingface_hub.hf_hub_download", side_effect = _dl):
        base = get_base_model_from_lora_identifier("someone/remote-lora")
    assert base == "unsloth/Llama-3.2-1B-Instruct"
    assert calls["n"] == 2  # retried once


def test_lora_identifier_persistent_transient_returns_none():
    # Two transient errors -> None, logged at WARNING (a missed base is gated by neither).
    # Assert on the logger directly: robust to the logging backend (structlog vs stub).
    from utils.models import model_config as _mc
    with (
        patch("huggingface_hub.hf_hub_download", side_effect = RuntimeError("down")),
        patch.object(_mc.logger, "warning") as mock_warn,
    ):
        assert get_base_model_from_lora_identifier("someone/remote-lora") is None
    assert any(
        "Could not resolve remote LoRA base" in str(c.args[0]) for c in mock_warn.call_args_list
    )


@patch("utils.models.model_config.is_audio_input_type", return_value = False)
@patch("utils.models.model_config.detect_audio_type", return_value = None)
@patch("utils.models.model_config.is_vision_model", return_value = False)
def test_model_config_full_finetune_local_path_is_not_lora(
    _mock_vision, _mock_audio_type, _mock_audio_input, tmp_path: Path
):
    (tmp_path / "config.json").write_text(json.dumps({"_name_or_path": "unsloth/Qwen3-4B"}))
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
