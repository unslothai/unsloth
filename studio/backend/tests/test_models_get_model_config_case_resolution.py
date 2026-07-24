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


@pytest.mark.parametrize(
    ("hf_token", "expected_token"),
    (
        (None, False),
        ("  request-token\n", "request-token"),
    ),
)
def test_model_size_metadata_uses_only_explicit_request_token(
    monkeypatch, hf_token, expected_token
):
    captured_tokens = []

    class _Api:
        def __init__(self, *, token):
            captured_tokens.append(token)

        def repo_info(self, *_args, **kwargs):
            captured_tokens.append(kwargs["token"])
            return types.SimpleNamespace(siblings = [])

    monkeypatch.setenv("HF_TOKEN", "operator-secret-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi = _Api))

    assert models_route._get_model_size_bytes("Org/Model", hf_token) is None
    assert captured_tokens == [expected_token, expected_token]


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


def test_repo_in_any_hf_cache_matches_case_variant_in_legacy_cache(tmp_path, monkeypatch):
    # A case-variant in a legacy/default cache must read as present (case resolution only
    # covers the active cache; discard deletes case-insensitively, so detection must too,
    # else a decline deletes a pre-existing user repo).
    import utils.paths as paths_pkg
    import hub.utils.paths as hub_paths

    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    default = tmp_path / "default"
    for d in (active, legacy, default):
        d.mkdir()
    # Differently-cased entry in the legacy cache only.
    (legacy / "models--Unsloth--Foo").mkdir()

    # No active-cache variant; case resolution is a no-op here.
    monkeypatch.setattr(paths_pkg, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(hub_paths, "legacy_hf_cache_dir", lambda: legacy)
    monkeypatch.setattr(hub_paths, "hf_default_cache_dir", lambda: default)
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches",
        lambda: [active],
    )

    assert models_route._repo_in_any_hf_cache("unsloth/foo") is True
    # Absent from every cache -> reported absent.
    assert models_route._repo_in_any_hf_cache("unsloth/not-cached") is False
