# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from routes import pill


@pytest.fixture
def pill_home(tmp_path, monkeypatch):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return tmp_path


def test_settings_defaults_and_partial_update(pill_home):
    settings = pill.get_settings(current_subject = "test-user")
    assert settings == {
        "enabled": False,
        "defaultModel": None,
        "defaultGgufVariant": None,
        "autoLoad": True,
        "excludedApps": [],
    }

    settings = pill.put_settings(
        pill.PillSettingsUpdate(enabled = True, defaultModel = "some/model-GGUF"),
        current_subject = "test-user",
    )
    assert settings["enabled"] is True
    assert settings["defaultModel"] == "some/model-GGUF"

    settings = pill.put_settings(
        pill.PillSettingsUpdate(excludedApps = ["com.apple.Passwords"]),
        current_subject = "test-user",
    )
    assert settings["enabled"] is True
    assert settings["defaultModel"] == "some/model-GGUF"
    assert settings["excludedApps"] == ["com.apple.Passwords"]
