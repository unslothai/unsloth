# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the no-chat-history server policy."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes import chat_history, settings as settings_route
from utils import no_chat_history_settings


def _install_fake_studio_db(monkeypatch, *, stored = None):
    storage_pkg = types.ModuleType("storage")
    studio_db = types.ModuleType("storage.studio_db")
    values: dict[str, object] = {}
    if stored is not None:
        values[no_chat_history_settings.NO_CHAT_HISTORY_SETTING_KEY] = stored

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


def test_no_chat_history_defaults_off_when_setting_missing(monkeypatch):
    monkeypatch.delenv(no_chat_history_settings.NO_CHAT_HISTORY_ENV, raising = False)
    _install_fake_studio_db(monkeypatch)

    assert no_chat_history_settings.get_no_chat_history_enabled() is False


def test_no_chat_history_env_forces_enabled(monkeypatch):
    _install_fake_studio_db(monkeypatch, stored = False)
    monkeypatch.setenv(no_chat_history_settings.NO_CHAT_HISTORY_ENV, "true")

    assert no_chat_history_settings.get_no_chat_history_enabled() is True
    assert no_chat_history_settings.no_chat_history_forced_by_env() is True


def test_settings_route_persists_no_chat_history_toggle(monkeypatch):
    values = _install_fake_studio_db(monkeypatch)
    monkeypatch.delenv(no_chat_history_settings.NO_CHAT_HISTORY_ENV, raising = False)

    response = settings_route.update_no_chat_history(
        settings_route.NoChatHistoryPayload(enabled = True),
        current_subject = "test-user",
    )

    assert response.enabled is True
    assert response.default_enabled is False
    assert response.forced_by_env is False
    assert values[no_chat_history_settings.NO_CHAT_HISTORY_SETTING_KEY] is True


def test_settings_route_rejects_toggle_when_env_locked(monkeypatch):
    _install_fake_studio_db(monkeypatch)
    monkeypatch.setenv(no_chat_history_settings.NO_CHAT_HISTORY_ENV, "1")

    with pytest.raises(Exception):
        settings_route.update_no_chat_history(
            settings_route.NoChatHistoryPayload(enabled = False),
            current_subject = "test-user",
        )


def test_list_threads_returns_empty_when_history_disabled(monkeypatch):
    monkeypatch.setattr(
        chat_history,
        "_chat_history_storage_disabled",
        lambda: True,
    )

    response = chat_history.list_threads(current_subject = "test-user")

    assert response.threads == []


def test_save_thread_rejects_when_history_disabled(monkeypatch):
    monkeypatch.setattr(
        chat_history,
        "_chat_history_storage_disabled",
        lambda: True,
    )

    with pytest.raises(Exception) as exc:
        chat_history.save_thread(
            chat_history.ChatThread(
                id = "thread-1",
                modelType = "base",
                createdAt = 1,
            ),
            current_subject = "test-user",
        )

    assert exc.value.status_code == 403
