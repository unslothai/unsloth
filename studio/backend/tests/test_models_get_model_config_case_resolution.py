# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types
from pathlib import Path

# Keep this test runnable in lightweight environments where optional logging
# deps are not installed.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
import utils.models.model_config as model_config_module


def test_get_model_config_resolves_cached_case_before_model_checks(monkeypatch):
    calls: dict[str, str] = {}

    class _DummyModelConfig:
        is_lora = False
        base_model = None

    def _record_load(model_name):
        calls["load_model_defaults"] = model_name
        return {}

    def _record_vision(model_name, hf_token = None):
        calls["is_vision_model"] = model_name
        return False

    def _record_embedding(model_name, hf_token = None):
        calls["is_embedding_model"] = model_name
        return False

    def _record_audio(model_name, hf_token = None):
        calls["detect_audio_type"] = model_name
        return None

    def _record_from_identifier(cls, model_name):
        calls["from_identifier"] = model_name
        return _DummyModelConfig()

    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(
        models_route, "resolve_cached_repo_id_case", lambda _: "Org/Model"
    )
    monkeypatch.setattr(models_route, "load_model_defaults", _record_load)
    monkeypatch.setattr(models_route, "is_vision_model", _record_vision)
    monkeypatch.setattr(models_route, "is_embedding_model", _record_embedding)
    monkeypatch.setattr(model_config_module, "detect_audio_type", _record_audio)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(_record_from_identifier),
    )
    monkeypatch.setattr(models_route, "_get_max_position_embeddings", lambda _: 4096)
    monkeypatch.setattr(models_route, "_get_model_size_bytes", lambda *_args, **_kw: 0)

    result = asyncio.run(
        models_route.get_model_config(
            model_name = "org/model",
            hf_token = None,
            current_subject = "test-subject",
        )
    )

    assert result.model_name == "Org/Model"
    assert calls["load_model_defaults"] == "Org/Model"
    assert calls["is_vision_model"] == "Org/Model"
    assert calls["is_embedding_model"] == "Org/Model"
    assert calls["detect_audio_type"] == "Org/Model"
    assert calls["from_identifier"] == "Org/Model"


def test_get_model_config_treats_local_path_as_local_only(monkeypatch):
    calls: dict[str, object] = {}

    class _DummyModelConfig:
        is_lora = False
        base_model = None

    def _record_vision(model_name, hf_token = None, local_files_only = False):
        calls["is_vision_model"] = (model_name, local_files_only)
        return False

    def _record_embedding(model_name, hf_token = None, local_files_only = False):
        calls["is_embedding_model"] = (model_name, local_files_only)
        return False

    def _record_audio(model_name, hf_token = None, local_files_only = False):
        calls["detect_audio_type"] = (model_name, local_files_only)
        return None

    def _record_from_identifier(cls, model_name, **kwargs):
        calls["from_identifier"] = (model_name, kwargs.get("local_files_only"))
        return _DummyModelConfig()

    def _record_size(model_name, hf_token = None, local_files_only = False):
        calls["model_size"] = (model_name, local_files_only)
        return 0

    def _record_template(model_name, hf_token = None, local_files_only = False):
        calls["chat_template"] = (model_name, local_files_only)
        return None

    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        models_route,
        "_latest_snapshot_from_cache_path",
        lambda *_args: "/tmp/hf/models--org--model/snapshots/main",
    )
    monkeypatch.setattr(models_route, "load_model_defaults", lambda _: {})
    monkeypatch.setattr(models_route, "is_vision_model", _record_vision)
    monkeypatch.setattr(models_route, "is_embedding_model", _record_embedding)
    monkeypatch.setattr(model_config_module, "detect_audio_type", _record_audio)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(_record_from_identifier),
    )
    monkeypatch.setattr(models_route, "_get_max_position_embeddings", lambda _: 4096)
    monkeypatch.setattr(models_route, "_get_model_size_bytes", _record_size)
    monkeypatch.setattr(models_route, "read_model_chat_template", _record_template)

    result = models_route._build_model_details(
        "org/model",
        hf_token = None,
        prefer_local_cache = False,
        local_path = "/tmp/hf/models--org--model",
    )

    snapshot = "/tmp/hf/models--org--model/snapshots/main"
    assert result.model_name == "org/model"
    assert calls["is_vision_model"] == (snapshot, True)
    assert calls["is_embedding_model"] == (snapshot, True)
    assert calls["detect_audio_type"] == (snapshot, True)
    assert calls["from_identifier"] == (snapshot, True)
    assert calls["chat_template"] == (snapshot, True)
    assert calls["model_size"] == (snapshot, True)


def test_get_model_config_local_gguf_uses_fast_metadata_path(
    monkeypatch, tmp_path: Path
):
    calls: dict[str, object] = {}
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")

    def _should_not_run(*_args, **_kwargs):
        raise AssertionError("generic metadata probe should not run for local GGUF")

    def _record_template(model_name, hf_token = None, local_files_only = False):
        calls["chat_template"] = (model_name, local_files_only)
        return "template"

    def _record_size(model_name, hf_token = None, local_files_only = False):
        calls["model_size"] = (model_name, local_files_only)
        return 123

    monkeypatch.setattr(models_route, "is_local_path", lambda _: True)
    monkeypatch.setattr(models_route, "normalize_path", lambda value: value)
    monkeypatch.setattr(models_route, "load_model_defaults", _should_not_run)
    monkeypatch.setattr(models_route, "is_vision_model", _should_not_run)
    monkeypatch.setattr(models_route, "is_embedding_model", _should_not_run)
    monkeypatch.setattr(model_config_module, "detect_audio_type", _should_not_run)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(lambda *_args, **_kwargs: _should_not_run()),
    )
    monkeypatch.setattr(models_route, "detect_mmproj_file", lambda *_args: None)
    monkeypatch.setattr(
        models_route,
        "_read_chat_template_from_tokenizer_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(models_route, "read_model_chat_template", _record_template)
    monkeypatch.setattr(models_route, "_get_model_size_bytes", _record_size)

    result = models_route._build_model_details(
        str(gguf_path),
        hf_token = None,
        model_format = "gguf",
    )

    assert result.is_gguf is True
    assert result.model_format == "gguf"
    assert result.runtime == "llama_cpp"
    assert result.chat_template == "template"
    assert result.model_size_bytes == 123
    assert calls["chat_template"] == (str(gguf_path), True)
    assert calls["model_size"] == (str(gguf_path), True)


def test_check_vision_treats_local_path_as_local_only(monkeypatch):
    calls: dict[str, object] = {}

    def _record_vision(model_name, hf_token = None, local_files_only = False):
        calls["is_vision_model"] = (model_name, local_files_only)
        return True

    monkeypatch.setattr(
        models_route,
        "_model_details_lookup_name",
        lambda *_args: "/tmp/hf/models--org--model/snapshots/main",
    )
    monkeypatch.setattr(models_route, "is_vision_model", _record_vision)

    result = asyncio.run(
        models_route.check_vision_model(
            model_name = "org/model",
            prefer_local_cache = False,
            local_path = "/tmp/hf/models--org--model",
            hf_token = None,
            current_subject = "test-subject",
        )
    )

    assert result.is_vision is True
    assert calls["is_vision_model"] == (
        "/tmp/hf/models--org--model/snapshots/main",
        True,
    )


def test_model_config_metadata_failures_are_logged(monkeypatch):
    records: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []

    class _CaptureLogger:
        def info(self, message, *args, **kwargs):
            records.append(("info", message, args, kwargs))

        def warning(self, message, *args, **kwargs):
            records.append(("warning", message, args, kwargs))

        def debug(self, message, *args, **kwargs):
            records.append(("debug", message, args, kwargs))

    class _AutoConfig:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise ValueError("missing config")

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoConfig = _AutoConfig

    def _raise_from_identifier(cls, model_name, **kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setattr(models_route, "logger", _CaptureLogger())
    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        models_route,
        "_model_details_lookup_name",
        lambda *_args: "/tmp/hf/models--org--model/snapshots/main",
    )
    monkeypatch.setattr(models_route, "load_model_defaults", lambda _: {})
    monkeypatch.setattr(
        models_route,
        "is_vision_model",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        models_route,
        "is_embedding_model",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        model_config_module,
        "detect_audio_type",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(_raise_from_identifier),
    )
    monkeypatch.setattr(models_route, "_get_model_size_bytes", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        models_route,
        "_read_chat_template_from_tokenizer_config",
        lambda *_args, **_kwargs: None,
    )

    result = models_route._build_model_details(
        "org/model",
        hf_token = None,
        prefer_local_cache = True,
        local_path = "/tmp/hf/models--org--model",
        model_format = "safetensors",
    )

    warning_args = [args for level, _msg, args, _kw in records if level == "warning"]
    assert result.max_position_embeddings is None
    assert any(
        args[2] == "/tmp/hf/models--org--model"
        and args[3] == "safetensors"
        and args[4] is True
        and args[5] == "RuntimeError"
        for args in warning_args
    )
    assert any(
        args[2] == "/tmp/hf/models--org--model"
        and args[3] == "safetensors"
        and args[4] is True
        and args[5] == "ValueError"
        for args in warning_args
    )
