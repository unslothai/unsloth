# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Effective sampling resolution: per-model recommendation + operator pins.

Precedence per field: operator UNSLOTH_SAMPLING_* pin -> client explicit value ->
per-model recommendation (load_inference_config) -> static schema default.
"""

import pytest

from utils.inference.inference_config import resolve_effective_sampling, SAMPLING_FIELD_NAMES
from utils.inference import inference_config as ic

_SCHEMA_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.01,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
}


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    # The recommended lookup is lru-cached; clear it so a patched config takes effect.
    ic._recommended_sampling.cache_clear()
    for field in SAMPLING_FIELD_NAMES:
        monkeypatch.delenv(ic._SAMPLING_FIELDS[field][0], raising = False)
    yield
    ic._recommended_sampling.cache_clear()


def _all_omitted():
    return {f: None for f in SAMPLING_FIELD_NAMES}


def _set_recommended(monkeypatch, mapping):
    # _recommended_sampling sources from load_inference_config, the block the Chat UI seeds from.
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: dict(mapping))
    ic._recommended_sampling.cache_clear()


def test_recommended_applies_when_client_omits(monkeypatch):
    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 1.0
    assert eff["top_k"] == 64
    assert eff["min_p"] == 0.0
    assert eff["top_p"] == 0.95


def test_client_explicit_beats_recommended(monkeypatch):
    _set_recommended(monkeypatch, {"temperature": 1.0})
    eff = resolve_effective_sampling("some/model", {**_all_omitted(), "temperature": 0.2})
    assert eff["temperature"] == 0.2


def test_operator_pin_beats_client_and_recommended(monkeypatch):
    _set_recommended(monkeypatch, {"temperature": 1.0})
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")
    eff = resolve_effective_sampling("some/model", {**_all_omitted(), "temperature": 0.2})
    assert eff["temperature"] == 0.9


def test_unknown_model_matches_ui_inference_block(monkeypatch):
    # An unknown model gets what the Chat UI seeds (default.yaml: temp 0.7 / top_k -1), not the schema defaults.
    ui_block = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.01,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: dict(ui_block))
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/unknown-model", _all_omitted())
    assert eff["temperature"] == 0.7
    assert eff["top_k"] == -1
    assert eff["min_p"] == 0.01


def test_empty_recommendation_falls_back_to_schema_defaults(monkeypatch):
    # If load_inference_config yields nothing usable, the resolver falls back to the request schema defaults.
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff == _SCHEMA_DEFAULTS


@pytest.mark.parametrize(
    "model",
    ["unsloth/gemma-4-E4B", "unsloth/Qwen3-4B", "unsloth/Qwen3.5-9B", "someorg/unknown-xyz"],
)
def test_recommendation_matches_ui_source(model):
    # Parity guard against the Chat UI's own source for every field mergeBackendRecommendedInference adopts.
    ic._recommended_sampling.cache_clear()
    ui = ic.load_inference_config(model)
    rec = ic._recommended_sampling(model)
    for f in ic._UI_RECOMMENDED_FIELDS:
        cleaned = ic._clean_sampling_value(f, ui.get(f))
        if cleaned is not None:
            assert rec.get(f) == cleaned, f"{model}:{f} rec={rec.get(f)} ui={ui.get(f)}"


def test_qwen38_reuses_qwen36_sampling_defaults():
    qwen36 = ic.load_inference_config("unsloth/Qwen3.6-27B-GGUF")
    qwen38 = ic.load_inference_config("unsloth/Qwen3.8-27B-GGUF")

    assert qwen38 == qwen36
    assert qwen38 == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "trust_remote_code": False,
    }


def test_repetition_penalty_not_auto_recommended(monkeypatch):
    # The Chat UI never adopts a backend repetition_penalty, so the server must not auto-apply one either.
    monkeypatch.setattr(
        ic, "load_inference_config", lambda mid: {"temperature": 0.7, "repetition_penalty": 1.05}
    )
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/lfm2-model", _all_omitted())
    assert eff["temperature"] == 0.7
    assert eff["repetition_penalty"] == 1.0
    monkeypatch.setenv("UNSLOTH_SAMPLING_REPETITION_PENALTY", "1.05")
    eff2 = resolve_effective_sampling("some/lfm2-model", _all_omitted())
    assert eff2["repetition_penalty"] == 1.05


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.5", 0.5),
        ("abc", None),
        ("9.0", None),
        ("-1", None),
        ("   ", None),
        ("nan", None),
        ("inf", None),
        ("-inf", None),
    ],
)
def test_operator_override_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", raw)
    assert ic._operator_sampling_override("temperature") == expected


def test_out_of_range_recommendation_is_dropped(monkeypatch):
    # A malformed (out-of-range) model recommendation is ignored rather than forwarded to llama-server.
    _set_recommended(monkeypatch, {"temperature": 5.0, "top_k": 64})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 0.6
    assert eff["top_k"] == 64


def test_operator_override_top_k_int_and_range(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "40")
    assert ic._operator_sampling_override("top_k") == 40
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "200")
    assert ic._operator_sampling_override("top_k") is None
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "-1")
    assert ic._operator_sampling_override("top_k") == -1


@pytest.mark.parametrize(
    "field, val",
    [
        ("top_k", 10**400),
        ("top_k", float("nan")),
        ("top_k", float("inf")),
        (
            "temperature",
            10**400,
        ),
    ],
)
def test_clean_sampling_value_rejects_unrepresentable(field, val):
    # An oversized value used to raise OverflowError before the range check could drop it.
    assert ic._clean_sampling_value(field, val) is None


def test_oversized_operator_override_ignored(monkeypatch):
    # A huge integer string parses via int() but overflows float(), and math.isfinite would then 500 the request.
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "9" * 400)
    assert ic._operator_sampling_override("top_k") is None
    _set_recommended(monkeypatch, {})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["top_k"] == 20


def test_oversized_recommendation_ignored(monkeypatch):
    # A malformed per-model recommendation carrying an oversized int must not raise while resolving.
    _set_recommended(monkeypatch, {"temperature": 10**400, "top_k": 64})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 0.6
    assert eff["top_k"] == 64


def test_fill_recommended_sampling_openai_payload(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes.inference import _fill_recommended_sampling_openai

    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})

    payload = ChatCompletionRequest(
        model = "m", messages = [{"role": "user", "content": "hi"}], temperature = 0.2
    )
    _fill_recommended_sampling_openai(payload, "some/model")
    assert payload.temperature == 0.2
    assert payload.top_k == 64
    assert payload.min_p == 0.0
    assert payload.top_p == 0.95


def test_fill_recommended_sampling_openai_operator_pin_overrides_client(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes.inference import _fill_recommended_sampling_openai

    monkeypatch.setattr(ic, "load_model_defaults", lambda mid: {})
    monkeypatch.setattr(ic, "get_family_inference_params", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")

    payload = ChatCompletionRequest(
        model = "m", messages = [{"role": "user", "content": "hi"}], temperature = 0.2
    )
    _fill_recommended_sampling_openai(payload, "some/model")
    assert payload.temperature == 0.9


def test_fill_recommended_sampling_completions_body(monkeypatch):
    # /v1/completions is a raw proxy: an unrecommended, unpinned field is left absent for llama-server.
    from routes.inference import _fill_recommended_sampling_completions

    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})

    body = {"prompt": "hi", "temperature": 0.2}
    _fill_recommended_sampling_completions(body, "some/model")
    assert body["temperature"] == 0.2
    assert body["top_k"] == 64
    assert body["min_p"] == 0.0
    assert "top_p" not in body
    assert "presence_penalty" not in body
    assert "repeat_penalty" not in body


def test_fill_recommended_sampling_completions_operator_pin(monkeypatch):
    # An operator pin overrides the client's raw-body value, written under llama-server's "repeat_penalty".
    from routes.inference import _fill_recommended_sampling_completions

    monkeypatch.setattr(ic, "load_inference_config", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", "0.9")
    monkeypatch.setenv("UNSLOTH_SAMPLING_REPETITION_PENALTY", "1.2")

    body = {"prompt": "hi", "temperature": 0.2, "repeat_penalty": 1.05}
    _fill_recommended_sampling_completions(body, "some/model")
    assert body["temperature"] == 0.9
    assert body["repeat_penalty"] == 1.2
    assert "repetition_penalty" not in body
