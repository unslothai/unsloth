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
from routes.chat_history import ChatInferenceSettings
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
    defaults = {
        "temperature": 0.25,
        "top_k": 180,
        "repeat_penalty": 0.9,
        "frequency_penalty": -0.5,
        "seed": 12,
        "chat_template_kwargs": {"enable_thinking": False, "custom_key": "kept"},
    }
    backend = SimpleNamespace(
        model_identifier = "model.gguf",
        llama_cpp_config_summary = {"mode": "custom", "request_defaults": defaults},
    )
    monkeypatch.setattr(routes, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(
        sampling, "_recommended_sampling", lambda _: {"temperature": 0.8, "top_k": 20}
    )
    for field in sampling.SAMPLING_FIELD_NAMES:
        monkeypatch.delenv(sampling._SAMPLING_FIELDS[field][0], raising = False)
    return backend


def test_preset_beats_auto_seeded_ui_values_and_retains_native_domains(preset_backend):
    payload = ChatCompletionRequest(
        messages = [],
        temperature = 0.8,
        top_k = 20,
        enable_thinking = True,
        sampling_fields_explicit = [],
        seed = 3,
    )
    routes._fill_recommended_sampling_openai(payload, "model.gguf")
    assert (payload.temperature, payload.top_k, payload.repetition_penalty) == (0.25, 180, 0.9)
    assert payload.frequency_penalty == -0.5
    assert payload.enable_thinking is None
    assert payload.seed == 3
    routes._fill_recommended_sampling_openai(payload, "model.gguf")
    assert payload.temperature == 0.25


def test_explicit_zero_and_false_survive_preset(preset_backend):
    payload = ChatCompletionRequest(
        messages = [],
        temperature = 0,
        enable_thinking = False,
        sampling_fields_explicit = ["temperature", "enable_thinking"],
    )
    routes._fill_recommended_sampling_openai(payload, "model.gguf")
    assert payload.temperature == 0
    assert payload.enable_thinking is False


def test_operator_pin_remains_highest(preset_backend, monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.4")
    payload = ChatCompletionRequest(messages = [], temperature = 0.1)
    routes._fill_recommended_sampling_openai(payload, "model.gguf")
    assert payload.temperature == 0.4


def test_raw_completion_defaults_merge_without_losing_limits(preset_backend):
    body = {
        "temperature": 0,
        "max_tokens": 5,
        "stop": ["END"],
        "chat_template_kwargs": {"custom_key": "explicit"},
    }
    routes._fill_recommended_sampling_completions(body, "model.gguf")
    assert body["temperature"] == 0
    assert body["repeat_penalty"] == 0.9
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "custom_key": "explicit"}
    assert (body["max_tokens"], body["stop"]) == (5, ["END"])


def test_defaults_never_leak_to_other_model(preset_backend):
    assert routes._custom_request_defaults("other.gguf") == {}


def test_chat_settings_preserve_sampling_provenance():
    payload = {"temperature": 0.8, "samplingFieldsExplicit": []}
    assert ChatInferenceSettings.model_validate(payload).model_dump(exclude_unset = True) == payload


@pytest.mark.asyncio
async def test_non_gguf_custom_validation_refuses_without_backend_access(monkeypatch):
    monkeypatch.setattr(
        routes, "get_llama_cpp_backend", lambda: pytest.fail("Must not touch resident")
    )
    with pytest.raises(HTTPException) as exc:
        await routes._preflight_custom_llama_config(
            ValidateModelRequest(model_path = "model", llama_cpp_config = CUSTOM),
            SimpleNamespace(is_gguf = False),
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["load", "validate"])
async def test_route_rejects_custom_before_resident_or_gpu_eviction(
    monkeypatch, tmp_path, operation
):
    model = str(tmp_path / "selected.gguf")
    calls = []

    def reject(intent, **kwargs):
        calls.append((intent.model_identifier, kwargs))
        raise CustomConfigError("INI section [*], key 'host': reserved by Studio")

    resident = SimpleNamespace(
        is_loaded = True,
        model_identifier = "resident.gguf",
        prepare_custom_config = reject,
        last_load_intent = None,
    )
    monkeypatch.setattr(routes, "get_llama_cpp_backend", lambda: resident)
    monkeypatch.setattr(
        routes, "get_inference_backend", lambda: SimpleNamespace(active_model_name = None)
    )
    monkeypatch.setattr(
        routes, "_resolve_model_identifier_for_request", lambda *a, **k: (model, model, False)
    )
    monkeypatch.setattr(routes, "resolve_effective_chat_template_override", lambda **k: None)
    config = SimpleNamespace(
        identifier = model,
        is_gguf = True,
        is_vision = False,
        gguf_file = model,
        gguf_mmproj_file = None,
        gguf_variant = "Q4_K_M",
        gguf_hf_repo = None,
    )
    monkeypatch.setattr(routes, "ModelConfig", SimpleNamespace(from_identifier = lambda **k: config))
    monkeypatch.setattr(routes, "_classify_diffusion_gguf", lambda _: False)
    monkeypatch.setattr(
        "core.inference.gpu_arbiter.acquire_for",
        lambda *a, **k: pytest.fail("GPU owner was evicted"),
    )
    request_context = SimpleNamespace(
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
    )
    with pytest.raises(HTTPException) as exc:
        if operation == "load":
            await routes._load_model_impl(
                LoadRequest(model_path = model, gguf_variant = "Q4_K_M", llama_cpp_config = CUSTOM),
                request_context,
                current_subject = "custom-test",
            )
        else:
            await routes.validate_model(
                ValidateModelRequest(
                    model_path = model, gguf_variant = "Q4_K_M", llama_cpp_config = CUSTOM
                ),
                request_context,
                current_subject = "custom-test",
            )
    assert exc.value.status_code == 400
    assert "reserved by Studio" in str(exc.value.detail)
    assert calls == [(model, {"validate_resources": True})]
    assert resident.is_loaded and resident.model_identifier == "resident.gguf"
