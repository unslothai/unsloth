# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Effective sampling resolution: per-model recommendation + operator pins.

Precedence per field: operator UNSLOTH_SAMPLING_* pin -> client explicit value ->
per-model recommendation (load_inference_config) -> static schema default.
"""

import pytest

from utils.inference import resolve_effective_sampling, SAMPLING_FIELD_NAMES
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
    # Recommended sampling is sourced from the model-specific YAML (gated by _has_specific_yaml)
    # or the family JSON, not the generic default.yaml. Exercise the model-YAML tier here.
    monkeypatch.setattr(ic, "_has_specific_yaml", lambda mid: True)
    monkeypatch.setattr(ic, "load_model_defaults", lambda mid: {"inference": dict(mapping)})
    monkeypatch.setattr(ic, "get_family_inference_params", lambda mid: {})
    ic._recommended_sampling.cache_clear()


def test_load_inference_config_includes_repetition_penalty():
    cfg = ic.load_inference_config("unsloth/gemma-4-E4B")
    assert "repetition_penalty" in cfg
    assert isinstance(cfg["repetition_penalty"], (int, float))


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


def test_unknown_model_falls_back_to_schema_defaults(monkeypatch):
    # No model-specific YAML and no family match -> schema defaults, NOT the generic default.yaml
    # that load_model_defaults would otherwise return for an unknown model.
    monkeypatch.setattr(ic, "_has_specific_yaml", lambda mid: False)
    monkeypatch.setattr(ic, "get_family_inference_params", lambda mid: {})
    ic._recommended_sampling.cache_clear()
    eff = resolve_effective_sampling("some/unknown-model", _all_omitted())
    assert eff == _SCHEMA_DEFAULTS


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("0.5", 0.5),
        ("abc", None),  # unparseable
        ("9.0", None),  # above temperature max (2.0)
        ("-1", None),  # below temperature min (0.0)
        ("   ", None),  # blank
    ],
)
def test_operator_override_parsing(monkeypatch, raw, expected):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TEMPERATURE", raw)
    assert ic._operator_sampling_override("temperature") == expected


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
