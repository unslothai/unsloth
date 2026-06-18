# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
import types

import pytest

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


def _request_without_hf_token():
    return types.SimpleNamespace(query_params = {})


def _request_with_hf_token():
    return types.SimpleNamespace(query_params = {"hf_token": "secret"})


# A YAML-known TRC vision model: is_vision + trust_remote_code come from the
# defaults, so resolution must report vision without a live probe.
_YAML_TRC_VISION_DEFAULTS = {
    "model": {"is_vision": True},
    "inference": {"trust_remote_code": True},
}


def _patch_yaml_trc_vision(monkeypatch):
    """Stub TRC-vision YAML defaults; the live vision probe raises if touched."""

    def fail_vision(*_args, **_kwargs):
        raise AssertionError("YAML-known TRC VLM must not probe before opt-in")

    monkeypatch.setattr(
        models_route,
        "load_model_defaults",
        lambda _model: _YAML_TRC_VISION_DEFAULTS,
    )
    monkeypatch.setattr(models_route, "is_vision_model", fail_vision)


def test_get_model_config_resolves_cached_case_before_model_checks(monkeypatch):
    calls: dict[str, str] = {}

    class _DummyModelConfig:
        is_lora = False
        base_model = None

    def _record_load(model_name):
        calls["load_model_defaults"] = model_name
        return {}

    def _record_vision(
        model_name,
        hf_token = None,
        trust_remote_code = False,
    ):
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
            request = _request_without_hf_token(),
            model_name = "org/model",
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

    _patch_yaml_trc_vision(monkeypatch)
    monkeypatch.setattr(models_route, "is_local_path", lambda _: False)
    monkeypatch.setattr(models_route, "resolve_cached_repo_id_case", lambda value: value)
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
            request = _request_without_hf_token(),
            model_name = "deepseek-ai/DeepSeek-OCR",
            current_subject = "test-subject",
        )
    )

    assert result.is_vision is True
    assert result.model_type == "vision"


def test_check_vision_reports_yaml_trc_vision_without_probe(monkeypatch):
    _patch_yaml_trc_vision(monkeypatch)

    result = asyncio.run(
        models_route.check_vision_model(
            request = _request_without_hf_token(),
            model_name = "deepseek-ai/DeepSeek-OCR",
            current_subject = "test-subject",
        )
    )

    assert result.is_vision is True


def test_check_vision_keeps_yaml_trc_vision_after_opt_in(monkeypatch):
    _patch_yaml_trc_vision(monkeypatch)

    result = asyncio.run(
        models_route.check_vision_model(
            request = _request_without_hf_token(),
            model_name = "deepseek-ai/DeepSeek-OCR",
            trust_remote_code = True,
            current_subject = "test-subject",
        )
    )

    assert result.is_vision is True


@pytest.mark.parametrize(
    "route_fn",
    ["get_model_config", "check_vision_model", "check_embedding_model"],
)
def test_rejects_hf_token_query(route_fn) -> None:
    with pytest.raises(models_route.HTTPException) as exc_info:
        asyncio.run(
            getattr(models_route, route_fn)(
                request = _request_with_hf_token(),
                model_name = "org/model",
                current_subject = "test-subject",
            )
        )

    assert exc_info.value.status_code == 400
    assert "POST JSON" in exc_info.value.detail


def test_ocr_defaults_mapping_is_case_insensitive():
    deepseek_defaults = model_config_module.load_model_defaults("deepseek-ai/deepseek-ocr")
    glm_defaults = model_config_module.load_model_defaults("zai-org/glm-ocr")

    assert deepseek_defaults["model"]["is_ocr"] is True
    assert deepseek_defaults["inference"]["trust_remote_code"] is True
    assert glm_defaults["model"]["is_ocr"] is True
    assert glm_defaults["inference"]["trust_remote_code"] is True


def test_unsloth_ocr_preset_defaults_resolve():
    """The OCR presets point at the unsloth mirrors; each maps to its own
    YAML with the canonical unsloth identifier and the right TRC flag
    (the GLM mirror is native transformers, no custom code)."""
    for ident, trc in [
        ("unsloth/deepseek-ocr", True),
        ("unsloth/DeepSeek-OCR-2", True),
        ("unsloth/glm-ocr", False),
        ("unsloth/PaddleOCR-VL", True),
    ]:
        defaults = model_config_module.load_model_defaults(ident)
        assert defaults["model"]["is_ocr"] is True, ident
        assert defaults["model"]["is_vision"] is True, ident
        assert defaults["model"]["identifier"].startswith("unsloth/"), ident
        assert defaults["inference"].get("trust_remote_code", False) is trc, ident


def test_repo_in_any_hf_cache_matches_case_variant_in_legacy_cache(tmp_path, monkeypatch):
    # A case-variant in a legacy/default cache must read as present (case resolution only
    # covers the active cache; discard deletes case-insensitively, so detection must too,
    # else a decline deletes a pre-existing user repo).
    import utils.paths as paths_pkg
    import huggingface_hub.constants as hf_constants

    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    default = tmp_path / "default"
    for d in (active, legacy, default):
        d.mkdir()
    # Differently-cased entry in the legacy cache only.
    (legacy / "models--Unsloth--Foo").mkdir()

    # No active-cache variant; case resolution is a no-op here.
    monkeypatch.setattr(paths_pkg, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(paths_pkg, "legacy_hf_cache_dir", lambda: legacy)
    monkeypatch.setattr(paths_pkg, "hf_default_cache_dir", lambda: default)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(active))

    assert models_route._repo_in_any_hf_cache("unsloth/foo") is True
    # Absent from every cache -> reported absent.
    assert models_route._repo_in_any_hf_cache("unsloth/not-cached") is False
