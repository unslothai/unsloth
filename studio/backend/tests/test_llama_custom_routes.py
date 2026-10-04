# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom configuration persistence, preflight and request-default boundaries."""

from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.inference import ChatCompletionRequest, LoadRequest, ValidateModelRequest
from routes import inference as routes
from utils import openai_auto_switch_settings as settings
from utils.inference import inference_config as sampling
from core.inference.llama_custom_config import CustomConfigError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_openai_auto_switch import _put, override_store  # noqa: E402, F401


CUSTOM = {"version": 1, "mode": "custom", "ini": "[*]\nnp=1\nc=56000\nfit=off", "section": None}
MANAGED = {"version": 1, "mode": "managed"}
MODEL = "unsloth/example-GGUF:Q4_K_M"


def test_override_roundtrip_and_managed_reset(override_store):
    _put(MODEL, llama_cpp_config = CUSTOM, llama_extra_args = ["--temp", "0.7"])
    row = settings.get_model_override(MODEL)
    assert row["llama_cpp_config"] == CUSTOM
    assert settings.model_override_load_kwargs(row, is_gguf = True) == {"llama_cpp_config": CUSTOM}
    _put(MODEL, llama_cpp_config = MANAGED)
    row = settings.get_model_override(MODEL)
    assert row["llama_cpp_config"] == MANAGED
    assert row["llama_extra_args"] == ["--temp", "0.7"]
    assert settings.model_override_load_kwargs(row, is_gguf = True)["llama_extra_args"] == [
        "--temp",
        "0.7",
    ]


def test_older_client_save_preserves_custom_source(override_store):
    _put(MODEL, llama_cpp_config = CUSTOM)
    _put(MODEL, custom_context_length = 8192)
    assert settings.get_model_override(MODEL)["llama_cpp_config"] == CUSTOM


def test_quant_reset_stops_bare_model_fallback(override_store):
    _put("unsloth/example-GGUF", llama_cpp_config = CUSTOM)
    _put(MODEL, llama_cpp_config = MANAGED)
    _, row = settings.resolve_override_for_load("unsloth/example-GGUF", variant = "Q4_K_M")
    assert row["llama_cpp_config"] == MANAGED


@pytest.mark.parametrize("schema", [LoadRequest, ValidateModelRequest])
def test_source_schema_fails_closed(schema):
    with pytest.raises(ValidationError):
        schema(model_path = "model.gguf", llama_cpp_config = {"version": 2, "mode": "custom"})


def test_corrupt_saved_source_does_not_become_managed():
    with pytest.raises(ValueError):
        settings.normalize_model_override({"llama_cpp_config": {"version": 1, "mode": "custom"}})


def test_source_inheritance_is_model_scoped_and_reset_is_explicit(monkeypatch):
    monkeypatch.setattr(settings, "resolve_override_for_load", lambda *a, **k: (None, {}))
    resident = SimpleNamespace(
        last_load_intent = SimpleNamespace(
            model_identifier = "/models/Original.gguf", hf_variant = None, llama_cpp_config = CUSTOM
        )
    )
    monkeypatch.setattr(routes, "get_llama_cpp_backend", lambda: resident)
    same = routes._resolve_llama_cpp_config(LoadRequest(model_path = "/models/Original.gguf"))
    other = routes._resolve_llama_cpp_config(LoadRequest(model_path = "/models/Other.gguf"))
    reset = routes._resolve_llama_cpp_config(
        LoadRequest(model_path = "/models/Original.gguf", llama_cpp_config = MANAGED)
    )
    assert same.llama_cpp_config == CUSTOM
    assert other.llama_cpp_config is None
    assert reset.llama_cpp_config == MANAGED


def test_custom_ignores_legacy_extras():
    request = LoadRequest(
        model_path = "model.gguf", llama_cpp_config = CUSTOM, llama_extra_args = ["--ctx-size", "128000"]
    )
    assert (
        routes._resolve_inherited_extra_args(
            request, SimpleNamespace(is_gguf = True), "model.gguf", request.llama_extra_args
        )
        == []
    )


@pytest.fixture
def preset_backend(monkeypatch):
    backend = SimpleNamespace(
        model_identifier = "model.gguf",
        llama_cpp_config_summary = {
            "mode": "custom",
            "request_defaults": {"temperature": 0.25, "top_k": 80},
        },
    )
    monkeypatch.setattr(routes, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        sampling, "_recommended_sampling", lambda _: {"temperature": 0.8, "top_k": 20, "top_p": 0.9}
    )
    for field in sampling.SAMPLING_FIELD_NAMES:
        monkeypatch.delenv(sampling._SAMPLING_FIELDS[field][0], raising = False)
    return backend


def _chat(**fields):
    return ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}], **fields)


def test_ini_sampling_ranks_between_client_and_recommendation(preset_backend):
    payload = _chat(top_k = 5)
    routes._fill_recommended_sampling_openai(payload, "model.gguf")
    assert (payload.temperature, payload.top_k, payload.top_p) == (0.25, 5, 0.9)


def test_operator_pin_still_wins(preset_backend, monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "1.5")
    payload = _chat()
    routes._fill_recommended_sampling_openai(payload, "model.gguf")
    assert payload.temperature == 1.5


def test_ini_defaults_never_leak_to_another_model(preset_backend):
    payload = _chat()
    routes._fill_recommended_sampling_openai(payload, "other.gguf")
    assert payload.temperature == 0.8


def test_raw_completions_get_ini_values(preset_backend):
    body = {"prompt": "hi"}
    routes._fill_recommended_sampling_completions(body, "model.gguf")
    assert (body["temperature"], body["top_k"]) == (0.25, 80)


@pytest.mark.asyncio
async def test_preflight_refuses_a_bad_config_before_any_eviction(monkeypatch):
    calls = []

    def prepare(intent):
        calls.append(intent.llama_cpp_config.ini)
        raise CustomConfigError("'bogus' is not an option of the selected llama-server")

    monkeypatch.setattr(
        routes, "get_llama_cpp_backend", lambda: SimpleNamespace(prepare_custom_config = prepare)
    )
    monkeypatch.setattr(routes, "_classify_diffusion_gguf", lambda config: False)
    request = LoadRequest(model_path = "model.gguf", llama_cpp_config = CUSTOM)
    config = SimpleNamespace(is_gguf = True, identifier = "model.gguf")
    with pytest.raises(HTTPException) as caught:
        await routes._preflight_custom_llama_config(request, config)
    assert caught.value.status_code == 400 and "bogus" in caught.value.detail
    assert calls == [CUSTOM["ini"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("caller_sent_custom", [False, True])
async def test_preflight_holds_a_managed_callers_own_config_to_owner_only_paths(
    monkeypatch, managed, caller_sent_custom
):
    compiled = SimpleNamespace(argv = ("--chat-template-file", "/home/owner/secret.jinja"))
    monkeypatch.setattr(
        routes,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(prepare_custom_config = lambda intent: compiled),
    )
    monkeypatch.setattr(routes, "_classify_diffusion_gguf", lambda config: False)
    monkeypatch.setattr(routes.account_access, "managed_account", lambda: managed)
    request = LoadRequest(model_path = "model.gguf", llama_cpp_config = CUSTOM)
    config = SimpleNamespace(is_gguf = True, identifier = "model.gguf")
    if managed and caller_sent_custom:
        with pytest.raises(HTTPException) as caught:
            await routes._preflight_custom_llama_config(
                request, config, caller_sent_custom = caller_sent_custom
            )
        assert caught.value.status_code == 403
        return
    # The owner, or a managed load that inherits the owner's saved override, is unchanged.
    got = await routes._preflight_custom_llama_config(
        request, config, caller_sent_custom = caller_sent_custom
    )
    assert got is compiled


@pytest.mark.asyncio
async def test_preflight_refuses_non_gguf():
    request = LoadRequest(model_path = "org/model", llama_cpp_config = CUSTOM)
    with pytest.raises(HTTPException):
        await routes._preflight_custom_llama_config(request, SimpleNamespace(is_gguf = False))
