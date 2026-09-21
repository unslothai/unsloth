# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from storage import studio_db
from utils import chat_preferences_settings as preferences


def test_missing_preference_defaults_off():
    assert preferences.get_show_model_disclaimer() is False


def test_saved_preference_survives_a_fresh_connection():
    assert preferences.set_show_model_disclaimer(True) is True
    assert preferences.get_show_model_disclaimer() is True


def test_legacy_value_only_seeds_an_empty_server():
    assert preferences.migrate_show_model_disclaimer(True) is True
    assert preferences.migrate_show_model_disclaimer(False) is True
    assert preferences.get_show_model_disclaimer() is True


def test_disabled_or_missing_legacy_value_does_not_claim_the_server_default():
    assert preferences.migrate_show_model_disclaimer(False) is False
    assert preferences.migrate_show_model_disclaimer(None) is False
    assert studio_db.get_app_setting(preferences.MODEL_DISCLAIMER_SETTING_KEY, None) is None
    assert preferences.migrate_show_model_disclaimer(True) is True


def test_route_round_trip_and_migration(monkeypatch):
    from routes import settings

    stored = {"value": None}

    def get_value():
        return False if stored["value"] is None else stored["value"]

    def set_value(value):
        stored["value"] = value
        return value

    def migrate_value(value):
        if stored["value"] is None and value is not None:
            stored["value"] = value
        return get_value()

    monkeypatch.setattr(settings, "get_show_model_disclaimer", get_value)
    monkeypatch.setattr(settings, "set_show_model_disclaimer", set_value)
    monkeypatch.setattr(settings, "migrate_show_model_disclaimer", migrate_value)

    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    client = TestClient(app)

    assert client.get("/chat-preferences").json() == {"show_model_disclaimer": False}
    assert client.post("/chat-preferences/migrate", json = {}).json() == {
        "show_model_disclaimer": False
    }
    assert client.post(
        "/chat-preferences/migrate", json = {"show_model_disclaimer": True}
    ).json() == {"show_model_disclaimer": True}
    assert client.post(
        "/chat-preferences/migrate", json = {"show_model_disclaimer": False}
    ).json() == {"show_model_disclaimer": True}
    assert client.put("/chat-preferences", json = {"show_model_disclaimer": False}).json() == {
        "show_model_disclaimer": False
    }
    assert client.put("/chat-preferences", json = {"show_model_disclaimer": "yes"}).status_code == 422
