# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An all-default "Remember" save must leave no override row: the UI sends engine_precision
"auto" and engine_parallelism "tensor" on every save, and a row holding only those shadowed the
repository row in auto-switch and re-ticked Remember."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils.openai_auto_switch_settings as settings  # noqa: E402
from test_openai_auto_switch import _mock_override_store, _put  # noqa: E402

DEFAULTS = {"engine": "auto", "engine_precision": "auto", "engine_parallelism": "tensor"}


def test_an_all_default_save_leaves_no_row(monkeypatch):
    _mock_override_store(monkeypatch)
    _put("unsloth/Qwen2.5-0.5B-Instruct", **DEFAULTS)
    assert "unsloth/Qwen2.5-0.5B-Instruct" not in settings.get_model_overrides()


def test_a_default_only_variant_no_longer_shadows_the_repository(monkeypatch):
    _mock_override_store(monkeypatch)
    _put("unsloth/Q-GGUF", max_seq_length = 8192, **DEFAULTS)
    _put("unsloth/Q-GGUF:Q4_K_M", **DEFAULTS)
    key, override = settings.resolve_override_for_load("unsloth/Q-GGUF", None, "Q4_K_M")
    assert key == "unsloth/Q-GGUF"
    assert override["max_seq_length"] == 8192


def test_rows_saved_before_the_fix_read_as_unset(monkeypatch):
    store = _mock_override_store(monkeypatch)
    store[settings.MODEL_OVERRIDES_SETTING_KEY] = {
        "unsloth/Q-GGUF": {"max_seq_length": 8192, "engine_precision": "auto"},
        "unsloth/Q-GGUF:Q4_K_M": {"engine_precision": "auto", "engine_parallelism": "tensor"},
    }
    settings._cache.clear()
    assert settings.get_model_overrides() == {"unsloth/Q-GGUF": {"max_seq_length": 8192}}
    key, _ = settings.resolve_override_for_load("unsloth/Q-GGUF", None, "Q4_K_M")
    assert key == "unsloth/Q-GGUF"


def test_a_real_engine_choice_persists_and_resetting_it_clears_it(monkeypatch):
    _mock_override_store(monkeypatch)
    _put("org/m", engine = "vllm", engine_precision = "bf16", engine_parallelism = "pipeline")
    assert settings.get_model_override("org/m") == {
        "engine": "vllm",
        "engine_precision": "bf16",
        "engine_parallelism": "pipeline",
    }
    _put("org/m", **DEFAULTS)
    assert settings.get_model_override("org/m") == {}
