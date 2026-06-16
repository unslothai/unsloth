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
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda _: "Org/Model")
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


def test_get_model_config_autoconfig_fallback_disables_remote_code_for_unsloth(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class _DummyModelConfig:
        is_lora = False
        base_model = None

    class _AutoConfig:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoConfig = _AutoConfig))
    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(models_route, "load_model_defaults", lambda _: {})
    monkeypatch.setattr(models_route, "is_vision_model", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(models_route, "is_embedding_model", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(model_config_module, "detect_audio_type", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(lambda *_args, **_kwargs: _DummyModelConfig()),
    )
    monkeypatch.setattr(models_route, "_get_max_position_embeddings", lambda _config: None)
    monkeypatch.setattr(models_route, "_get_model_size_bytes", lambda *_args, **_kwargs: 0)

    asyncio.run(
        models_route.get_model_config(
            model_name = "unsloth/custom-code",
            hf_token = None,
            current_subject = "test-subject",
        )
    )

    assert calls == [("unsloth/custom-code", {"trust_remote_code": False, "token": None})]


def test_trust_remote_code_requirement_uses_yaml_defaults_only(monkeypatch):
    calls: dict[str, str] = {}

    def _record_load(model_name):
        calls["load_model_defaults"] = model_name
        return {"training": {"trust_remote_code": True}}

    def _unexpected_probe(*_args, **_kwargs):
        raise AssertionError("trust check must not run model capability probes")

    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda _: "Org/Requires")
    monkeypatch.setattr(models_route, "load_model_defaults", _record_load)
    monkeypatch.setattr(models_route, "is_vision_model", _unexpected_probe)
    monkeypatch.setattr(models_route, "is_embedding_model", _unexpected_probe)
    monkeypatch.setattr(
        models_route.ModelConfig,
        "from_identifier",
        classmethod(lambda *_args, **_kwargs: _unexpected_probe()),
    )

    result = asyncio.run(
        models_route.get_model_trust_remote_code_requirement(
            model_name = "org/requires",
            current_subject = "test-subject",
        )
    )

    assert result.model_name == "Org/Requires"
    assert result.requires_trust_remote_code is True
    assert calls["load_model_defaults"] == "Org/Requires"
