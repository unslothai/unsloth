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
    # _recommended_sampling sources from load_inference_config -- the exact block the Chat UI
    # seeds from -- so patch that directly. Fields absent from `mapping` fall to schema defaults.
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: dict(mapping))
    ic._recommended_sampling.cache_clear()


def test_recommended_applies_when_client_omits(monkeypatch):
    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 1.0
    assert eff["top_k"] == 64
    assert eff["min_p"] == 0.0
    # A field with no recommendation keeps the static schema default.
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
    # An unknown model gets the same values the Chat UI would seed (load_inference_config's
    # default.yaml fallback: temp 0.7 / top_k -1), NOT the request schema defaults.
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
    # If load_inference_config yields nothing usable, the resolver falls back to the request
    # schema defaults.
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff == _SCHEMA_DEFAULTS


@pytest.mark.parametrize(
    "model",
    ["unsloth/gemma-4-E4B", "unsloth/Qwen3-4B", "unsloth/Qwen3.5-9B", "someorg/unknown-xyz"],
)
def test_recommendation_matches_ui_source(model):
    # Parity guard: what the server recommends for omitted fields equals the Chat UI's source
    # (load_inference_config) for every field the UI adopts (mergeBackendRecommendedInference).
    ic._recommended_sampling.cache_clear()
    ui = ic.load_inference_config(model)
    rec = ic._recommended_sampling(model)
    for f in ic._UI_RECOMMENDED_FIELDS:
        cleaned = ic._clean_sampling_value(f, ui.get(f))
        if cleaned is not None:
            assert rec.get(f) == cleaned, f"{model}:{f} rec={rec.get(f)} ui={ui.get(f)}"


def test_repetition_penalty_not_auto_recommended(monkeypatch):
    # The Chat UI's mergeBackendRecommendedInference never adopts a backend repetition_penalty
    # (e.g. lfm2's family value 1.05), so the server must not auto-apply one either. It stays at
    # the schema default unless the client sends it or an operator pins it.
    monkeypatch.setattr(
        ic, "load_inference_config", lambda mid: {"temperature": 0.7, "repetition_penalty": 1.05}
    )
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/lfm2-model", _all_omitted())
    assert eff["temperature"] == 0.7  # a UI-adopted field is recommended
    assert eff["repetition_penalty"] == 1.0  # rep is NOT auto-recommended (matches the UI)
    # An operator can still pin it explicitly.
    monkeypatch.setenv("UNSLOTH_SAMPLING_REPETITION_PENALTY", "1.05")
    eff2 = resolve_effective_sampling("some/lfm2-model", _all_omitted())
    assert eff2["repetition_penalty"] == 1.05


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.5", 0.5),
        ("abc", None),  # unparseable
        ("9.0", None),  # above temperature max (2.0)
        ("-1", None),  # below temperature min (0.0)
        ("   ", None),  # blank
        ("nan", None),  # NaN would pass a naive range check
        ("inf", None),  # non-finite
        ("-inf", None),  # non-finite
    ],
)
def test_operator_override_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", raw)
    assert ic._operator_sampling_override("temperature") == expected


def test_out_of_range_recommendation_is_dropped(monkeypatch):
    # A malformed model recommendation (out of range) is ignored, so the request keeps the
    # schema default rather than forwarding a bad value to llama-server.
    _set_recommended(monkeypatch, {"temperature": 5.0, "top_k": 64})
    eff = resolve_effective_sampling("some/model", _all_omitted())
    assert eff["temperature"] == 0.6  # 5.0 is outside [0, 2] -> schema default
    assert eff["top_k"] == 64  # a valid recommendation is still applied


def test_operator_override_top_k_int_and_range(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "40")
    assert ic._operator_sampling_override("top_k") == 40
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "200")  # above max 100
    assert ic._operator_sampling_override("top_k") is None
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "-1")  # min allowed
    assert ic._operator_sampling_override("top_k") == -1


def test_fill_recommended_sampling_openai_payload(monkeypatch):
    from models.inference import ChatCompletionRequest
    from routes.inference import _fill_recommended_sampling_openai

    _set_recommended(monkeypatch, {"temperature": 1.0, "top_k": 64, "min_p": 0.0})

    # Client sent only temperature; top_k / min_p were omitted.
    payload = ChatCompletionRequest(
        model = "m", messages = [{"role": "user", "content": "hi"}], temperature = 0.2
    )
    _fill_recommended_sampling_openai(payload, "some/model")
    assert payload.temperature == 0.2  # explicit client value preserved
    assert payload.top_k == 64  # recommended fills the omitted field
    assert payload.min_p == 0.0
    assert payload.top_p == 0.95  # no recommendation -> schema default unchanged


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
    assert payload.temperature == 0.9  # operator pin wins even over an explicit client value
