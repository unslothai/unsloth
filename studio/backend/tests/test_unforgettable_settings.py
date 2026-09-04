# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import unforgettable_settings


def _install_fake_studio_db(monkeypatch, *, stored = None):
    storage_pkg = types.ModuleType("storage")
    studio_db = types.ModuleType("storage.studio_db")
    values: dict[str, object] = {}
    if stored is not None:
        values[unforgettable_settings.UNFORGETTABLE_SETTING_KEY] = stored

    def get_app_setting(key, fallback = None):
        return values.get(key, fallback)

    def upsert_app_settings(settings):
        values.update(settings)
        return dict(values)

    studio_db.get_app_setting = get_app_setting
    studio_db.upsert_app_settings = upsert_app_settings
    monkeypatch.setitem(sys.modules, "storage", storage_pkg)
    monkeypatch.setitem(sys.modules, "storage.studio_db", studio_db)
    return values


def test_defaults_are_off(monkeypatch, tmp_path):
    _install_fake_studio_db(monkeypatch)
    monkeypatch.setattr(unforgettable_settings, "memory_db_path", lambda: tmp_path / "memory.db")
    monkeypatch.delenv("UNFORGETTABLE_VOTER", raising = False)
    monkeypatch.delenv("UNFORGETTABLE_PLANNER", raising = False)
    monkeypatch.delenv("UNFORGETTABLE_FILTER", raising = False)
    monkeypatch.delenv("UNFORGETTABLE_JUDGE_MODEL", raising = False)
    settings = unforgettable_settings.get_unforgettable_settings()
    assert settings["planner"] == "off"
    assert settings["filter"] == "on"
    assert settings["filter_model"] is None
    assert settings["judge_model"] is None
    assert settings["voter"] == "off"
    assert settings["skip_standing"] is False
    assert settings["namespace"] == "default"


def test_set_and_episode_extras(monkeypatch, tmp_path):
    _install_fake_studio_db(monkeypatch)
    monkeypatch.setattr(unforgettable_settings, "memory_db_path", lambda: tmp_path / "memory.db")
    updated = unforgettable_settings.set_unforgettable_settings(
        {
            "planner": "on",
            "filter": "off",
            "filter_model": "filter-large",
            "judge_model": "judge-large",
            "stakes": "high",
            "confirm_retry": True,
            "adapter_id": "ada-1",
        }
    )
    assert updated["planner"] == "on"
    assert updated["filter"] == "off"
    extras = unforgettable_settings.episode_extras_from_settings(updated)
    assert extras["planner"] == "on"
    assert extras["filter"] == "off"
    assert extras["filter_model"] == "filter-large"
    assert extras["judge_model"] == "judge-large"
    assert extras["stakes"] == "high"
    assert extras["confirm_retry"] is True
    assert extras["adapter_id"] == "ada-1"


def test_twin_plugin_round_trip(monkeypatch, tmp_path):
    _install_fake_studio_db(monkeypatch)
    monkeypatch.setattr(unforgettable_settings, "memory_db_path", lambda: tmp_path / "memory.db")
    updated = unforgettable_settings.set_unforgettable_settings({"twin_plugin": "none"})
    assert updated["twin_plugin"] == "none"
    extras = unforgettable_settings.episode_extras_from_settings(updated)
    assert extras["twin_plugin"] == "none"
    try:
        unforgettable_settings.set_unforgettable_settings({"twin_plugin": "docker"})
    except ValueError as exc:
        assert "Twin plugin" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_rejects_unknown_voter(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    try:
        unforgettable_settings.set_unforgettable_settings({"voter": "maybe"})
    except ValueError as exc:
        assert "Voter" in str(exc)
    else:
        raise AssertionError("expected ValueError")
