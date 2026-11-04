# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sampling defaults saved for one model (#10631): a /v1 request that leaves a field unset gets
the saved value, an explicit value and an operator pin still win, and nobody who never saved one
sees any change."""

import asyncio
import math
import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from routes import settings as settings_routes
from utils import openai_auto_switch_settings as auto_switch
from utils.inference import inference_config as ic
from utils.model_sampling_defaults import (
    get_model_sampling_defaults,
    saved_sampling_defaults,
    set_model_sampling_defaults,
)

MODEL = "unsloth/Qwen3.5-9B-GGUF"


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    for field in ic.SAMPLING_FIELD_NAMES:
        monkeypatch.delenv(ic._SAMPLING_FIELDS[field][0], raising = False)
    monkeypatch.setattr(ic, "load_inference_config", lambda mid: {"temperature": 1.0, "top_k": 64})
    ic._recommended_sampling.cache_clear()
    auto_switch._cache.clear()
    yield
    ic._recommended_sampling.cache_clear()
    auto_switch._cache.clear()


def _omitted():
    return {field: None for field in ic.SAMPLING_FIELD_NAMES}


def _put(model_id, sampling):
    return settings_routes.update_model_sampling_defaults_route(
        settings_routes.ModelSamplingDefaultsPayload(model_id = model_id, sampling = sampling),
        "tester",
    )


def test_nothing_saved_resolves_exactly_as_before():
    assert get_model_sampling_defaults() == {}
    assert ic.resolve_effective_sampling(MODEL, _omitted()) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.01,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
    }


def test_saved_defaults_fill_what_the_request_omits():
    _put(MODEL, {"temperature": 0.3, "top_p": 0.8, "repetition_penalty": 1.1})
    eff = ic.resolve_effective_sampling(MODEL, _omitted())
    assert eff["temperature"] == 0.3
    assert eff["top_p"] == 0.8
    assert eff["repetition_penalty"] == 1.1
    # Unsaved fields keep the recommendation, then the schema default.
    assert eff["top_k"] == 64
    assert eff["min_p"] == 0.01


def test_an_explicit_request_value_and_an_operator_pin_still_win(monkeypatch):
    _put(MODEL, {"temperature": 0.3, "top_p": 0.8})
    eff = ic.resolve_effective_sampling(MODEL, {**_omitted(), "temperature": 0.9})
    assert eff["temperature"] == 0.9
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_P", "0.5")
    assert ic.resolve_effective_sampling(MODEL, _omitted())["top_p"] == 0.5


def test_passthrough_fill_sends_only_saved_and_recommended_fields():
    _put(MODEL, {"repetition_penalty": 1.2})
    eff = ic.resolve_effective_sampling(MODEL, _omitted(), fill_defaults = False)
    assert eff == {"temperature": 1.0, "top_k": 64, "repetition_penalty": 1.2}


def test_saved_defaults_beat_the_qwen3_thinking_table():
    model = "unsloth/Qwen3-4B-GGUF"
    _put(model, {"temperature": 0.2})
    assert ic.resolve_effective_sampling(model, _omitted(), thinking = True)["temperature"] == 0.2


def test_a_cached_snapshot_path_finds_the_repo_row():
    _put("unsloth/Foo-GGUF", {"min_p": 0.05})
    path = "/home/u/.cache/huggingface/hub/models--unsloth--Foo-GGUF/snapshots/abc123"
    assert saved_sampling_defaults(path) == {"min_p": 0.05}


def test_a_snapshot_path_save_is_stored_under_the_repo_id():
    old = "/home/u/.cache/huggingface/hub/models--unsloth--Foo-GGUF/snapshots/abc123"
    new = "/home/u/.cache/huggingface/hub/models--unsloth--Foo-GGUF/snapshots/def456"
    response = _put(old + ":Q4_K_M", {"top_k": 40})
    assert response.defaults == {"unsloth/Foo-GGUF": {"top_k": 40}}
    assert response.resolved == {"top_k": 40}
    assert saved_sampling_defaults(new) == {"top_k": 40}


def test_a_quant_qualified_id_reads_the_repo_row():
    _put("unsloth/Foo-GGUF", {"top_k": 40})
    assert saved_sampling_defaults("unsloth/Foo-GGUF:Q4_K_M") == {"top_k": 40}


def test_lookup_folds_case_but_not_ambiguously():
    _put("unsloth/foo-gguf", {"top_k": 40})
    assert saved_sampling_defaults("unsloth/Foo-GGUF") == {"top_k": 40}
    _put("Unsloth/FOO-gguf", {"top_k": 10})
    assert saved_sampling_defaults("UNSLOTH/foo-GGUF") == {}


def test_a_posix_path_never_matches_another_casing():
    _put("/models/Foo.gguf", {"top_k": 40})
    assert saved_sampling_defaults("/models/Foo.gguf") == {"top_k": 40}
    assert saved_sampling_defaults("/models/foo.gguf") == {}


def test_a_windows_path_matches_another_casing_and_slash():
    _put("C:\\Models\\Foo.gguf", {"top_k": 40})
    assert saved_sampling_defaults("c:/models/foo.gguf") == {"top_k": 40}


def test_out_of_range_and_non_finite_values_are_dropped():
    _put(MODEL, {"temperature": 5.0, "top_p": math.nan, "top_k": 30, "min_p": -1})
    assert get_model_sampling_defaults() == {MODEL: {"top_k": 30}}


def test_save_is_idempotent_and_clearing_forgets_the_model():
    first = _put(MODEL, {"temperature": 0.4})
    second = _put(MODEL, {"temperature": 0.4})
    assert first.defaults == second.defaults == {MODEL: {"temperature": 0.4}}
    assert _put(MODEL, {"temperature": None}).defaults == {}
    assert _put(MODEL, None).defaults == {}
    assert ic.resolve_effective_sampling(MODEL, _omitted())["temperature"] == 1.0


def test_an_unreadable_store_never_fails_a_request(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("db gone")

    monkeypatch.setattr("utils.model_sampling_defaults.saved_sampling_defaults", boom, raising = True)
    assert ic.resolve_effective_sampling(MODEL, _omitted())["temperature"] == 1.0


def test_the_sampling_defaults_route_shows_what_a_request_gets():
    from routes import inference as inference_routes

    _put(MODEL, {"temperature": 0.35})
    got = asyncio.run(inference_routes.get_sampling_defaults(model = MODEL, current_subject = "owner"))
    assert got["temperature"] == 0.35


def test_completions_passthrough_body_gets_the_saved_repeat_penalty():
    from routes import inference as inference_routes

    _put(MODEL, {"repetition_penalty": 1.15})
    body = {"prompt": "hi"}
    inference_routes._fill_recommended_sampling_completions(body, MODEL)
    assert body["repeat_penalty"] == 1.15
    explicit = {"prompt": "hi", "repeat_penalty": 1.0}
    inference_routes._fill_recommended_sampling_completions(explicit, MODEL)
    assert explicit["repeat_penalty"] == 1.0
