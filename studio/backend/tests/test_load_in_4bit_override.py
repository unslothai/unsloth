# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The Run settings "Load in 4-bit" switch (#5639): a 16-bit choice is remembered per model and
reaches every load that rebuilds the picker's request, while an untouched model stores nothing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils.openai_auto_switch_settings as settings  # noqa: E402
from models.inference import LoadRequest  # noqa: E402
from test_openai_auto_switch import _mock_override_store, _put  # noqa: E402

DEFAULTS = {"engine": "auto", "engine_precision": "auto", "engine_parallelism": "tensor"}


def test_a_16_bit_choice_is_stored_and_turning_it_back_on_clears_it(monkeypatch):
    _mock_override_store(monkeypatch)
    _put("org/m", load_in_4bit = False, **DEFAULTS)
    assert settings.get_model_override("org/m") == {"load_in_4bit": False}
    _put("org/m", load_in_4bit = True, **DEFAULTS)
    assert "org/m" not in settings.get_model_overrides()


def test_an_older_client_that_never_sends_the_field_stores_nothing(monkeypatch):
    _mock_override_store(monkeypatch)
    _put("org/m", max_seq_length = 8192, **DEFAULTS)
    assert "load_in_4bit" not in settings.get_model_override("org/m")


def test_a_saved_16_bit_choice_reaches_the_load_request():
    kwargs = settings.model_override_load_kwargs({"load_in_4bit": False}, is_gguf = False)
    assert kwargs == {"load_in_4bit": False}
    assert LoadRequest(model_path = "org/m", **kwargs).load_in_4bit is False
    assert LoadRequest(model_path = "org/m").load_in_4bit is True


def test_gguf_and_optional_engines_are_unaffected():
    assert "load_in_4bit" not in settings.model_override_load_kwargs(
        {"load_in_4bit": False}, is_gguf = True
    )
    kwargs = settings.model_override_load_kwargs(
        {"engine": "vllm", "engine_precision": "int4"}, is_gguf = False
    )
    assert kwargs["load_in_4bit"] is False and kwargs["engine_precision"] == "int4"
