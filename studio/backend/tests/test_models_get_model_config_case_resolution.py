# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types

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

    def _record_vision(model_name, hf_token = None, trust_remote_code = False):
        calls["is_vision_model"] = model_name
        calls["trust_remote_code"] = str(trust_remote_code)
        return False

    def _record_embedding(model_name, hf_token = None):
        calls["is_embedding_model"] = model_name
        return False

    def _record_audio(model_name, hf_token = None):
        calls["detect_audio_type"] = model_name
        return None

    def _record_from_identifier(cls, model_name, **_kwargs):
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
    assert calls["trust_remote_code"] == "False"
    assert calls["is_embedding_model"] == "Org/Model"
    assert calls["detect_audio_type"] == "Org/Model"
    assert calls["from_identifier"] == "Org/Model"


def test_get_model_config_reports_yaml_trc_vision_without_probe(monkeypatch):
    class _DummyModelConfig:
        is_lora = False
        base_model = None

    def fail_vision(*_args, **_kwargs):
        raise AssertionError("YAML-known TRC VLM should not probe before opt-in")

    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda value: value)
    monkeypatch.setattr(
        models_route,
        "load_model_defaults",
        lambda _model: {
            "model": {"is_vision": True},
            "inference": {"trust_remote_code": True},
        },
    )
    monkeypatch.setattr(models_route, "is_vision_model", fail_vision)
    monkeypatch.setattr(models_route, "is_embedding_model", lambda *_args, **_kw: False)
    monkeypatch.setattr(model_config_module, "detect_audio_type", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(lambda cls, *_args, **_kwargs: _DummyModelConfig()),
    )
    monkeypatch.setattr(models_route, "_get_max_position_embeddings", lambda _: 4096)
    monkeypatch.setattr(models_route, "_get_model_size_bytes", lambda *_args, **_kw: 0)

    result = asyncio.run(
        models_route.get_model_config(
            model_name = "deepseek-ai/DeepSeek-OCR",
            hf_token = None,
            current_subject = "test-subject",
        )
    )

    assert result.is_vision is True
    assert result.model_type == "vision"


def test_check_vision_reports_yaml_trc_vision_without_probe(monkeypatch):
    def fail_vision(*_args, **_kwargs):
        raise AssertionError("YAML-known TRC VLM should not probe before opt-in")

    monkeypatch.setattr(
        models_route,
        "load_model_defaults",
        lambda _model: {
            "model": {"is_vision": True},
            "inference": {"trust_remote_code": True},
        },
    )
    monkeypatch.setattr(models_route, "is_vision_model", fail_vision)

    result = asyncio.run(
        models_route.check_vision_model(
            model_name = "deepseek-ai/DeepSeek-OCR",
            current_subject = "test-subject",
        )
    )

    assert result.is_vision is True


def test_check_vision_keeps_yaml_trc_vision_after_opt_in(monkeypatch):
    monkeypatch.setattr(
        models_route,
        "load_model_defaults",
        lambda _model: {
            "model": {"is_vision": True},
            "inference": {"trust_remote_code": True},
        },
    )
    def fail_vision(*_args, **_kwargs):
        raise AssertionError("YAML-known TRC VLM should not depend on live probe")

    monkeypatch.setattr(models_route, "is_vision_model", fail_vision)

    result = asyncio.run(
        models_route.check_vision_model(
            model_name = "deepseek-ai/DeepSeek-OCR",
            trust_remote_code = True,
            current_subject = "test-subject",
        )
    )

    assert result.is_vision is True


def test_ocr_defaults_mapping_is_case_insensitive():
    deepseek_defaults = model_config_module.load_model_defaults(
        "deepseek-ai/deepseek-ocr"
    )
    glm_defaults = model_config_module.load_model_defaults("zai-org/glm-ocr")

    assert deepseek_defaults["model"]["is_ocr"] is True
    assert deepseek_defaults["inference"]["trust_remote_code"] is True
    assert glm_defaults["model"]["is_ocr"] is True
    assert glm_defaults["inference"]["trust_remote_code"] is True
