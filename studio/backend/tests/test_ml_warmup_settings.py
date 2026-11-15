# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Selective startup warmup stays opt-in and preserves explicit dependency choices."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import ml_warmup_settings as settings


@pytest.fixture
def preferences(monkeypatch):
    values = {}
    database = types.ModuleType("storage.studio_db")
    database.get_app_setting = lambda key, default = None: values.get(key, default)
    database.upsert_app_settings = values.update
    monkeypatch.setitem(sys.modules, "storage.studio_db", database)
    monkeypatch.delenv("UNSLOTH_STUDIO_DISABLE_TORCH_WARM", raising = False)
    return values


def test_missing_preferences_default_to_no_features(preferences):
    assert settings.get_ml_warmup_features() == frozenset()
    assert settings.get_startup_ml_warmup_features() == frozenset()


@pytest.mark.parametrize(
    "stored", [None, "", "text", True, False, 0, 1, {}, ["unknown"], ["text", 1]]
)
def test_invalid_preferences_default_off(preferences, stored):
    preferences[settings.ML_WARMUP_FEATURES_SETTING_KEY] = stored
    assert settings.get_ml_warmup_features() == frozenset()


def test_unreadable_preferences_default_off(preferences, monkeypatch):
    def fail(*args):
        raise OSError("database unavailable")

    monkeypatch.setattr(sys.modules["storage.studio_db"], "get_app_setting", fail)
    assert settings.get_startup_ml_warmup_features() == frozenset()


def test_selections_are_persisted_deduplicated_in_canonical_order(preferences):
    selected = settings.set_ml_warmup_features(["media", "text", "media", "datasets"])
    assert selected == frozenset({"text", "datasets", "media"})
    assert preferences[settings.ML_WARMUP_FEATURES_SETTING_KEY] == ["text", "datasets", "media"]
    # Required hardware is effective, not an explicit saved choice.
    assert settings.get_startup_ml_warmup_features() == selected | {"hardware"}
    assert "hardware" not in settings.get_ml_warmup_features()


@pytest.mark.parametrize(
    "selected,effective",
    [
        ([], []),
        (["hardware"], ["hardware"]),
        (["datasets"], ["datasets"]),
        (["text"], ["hardware", "text"]),
        (["media"], ["hardware", "media"]),
        (["quantization"], ["hardware", "quantization"]),
        (["media", "quantization"], ["hardware", "media", "quantization"]),
    ],
)
def test_dependencies_add_only_hardware(selected, effective):
    assert settings.effective_ml_warmup_features(selected) == frozenset(effective)


def test_removing_dependent_features_preserves_only_explicit_hardware(preferences):
    settings.set_ml_warmup_features(["text"])
    settings.set_ml_warmup_features([])
    assert settings.get_startup_ml_warmup_features() == frozenset()
    settings.set_ml_warmup_features(["hardware", "text"])
    settings.set_ml_warmup_features(["hardware"])
    assert settings.get_startup_ml_warmup_features() == frozenset({"hardware"})


def test_operator_disable_wins_without_changing_saved_preference(preferences, monkeypatch):
    settings.set_ml_warmup_features(["text", "datasets"])
    monkeypatch.setenv("UNSLOTH_STUDIO_DISABLE_TORCH_WARM", "1")
    assert settings.get_ml_warmup_features() == frozenset({"text", "datasets"})
    assert settings.ml_warmup_disabled_by_env() is True
    assert settings.get_startup_ml_warmup_features() == frozenset()


@pytest.mark.parametrize("invalid", [None, True, "text", 0, 1, {}, ["unknown"], ["text", 1]])
def test_invalid_writes_do_not_change_preference(preferences, invalid):
    settings.set_ml_warmup_features(["datasets"])
    with pytest.raises(ValueError):
        settings.set_ml_warmup_features(invalid)
    assert settings.get_ml_warmup_features() == frozenset({"datasets"})
