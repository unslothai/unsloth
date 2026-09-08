# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The install's download transport setting.

A stored preference, nothing more: the transport an install runs on is unchanged until someone
picks one in Settings > General. These tests pin the default, the validation, and the routes the
settings row reads and writes.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
import storage.studio_db as studio_db
import utils.download_transport_settings as transport_settings


@pytest.fixture
def store(monkeypatch):
    """An in-memory app_settings."""
    values: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: values.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda updates: values.update(updates) or values
    )
    return values


def test_nothing_picked_means_auto(store):
    assert transport_settings.get_download_transport_mode() == "auto"
    # Read-only: seeding a value would be this setting deciding for an install that never
    # opened it.
    assert store == {}


def test_a_stored_choice_is_returned(store):
    store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] = "xet"
    assert transport_settings.get_download_transport_mode() == "xet"


def test_junk_on_disk_reads_as_nothing_picked(store):
    store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] = "carrier-pigeon"
    assert transport_settings.get_download_transport_mode() == "auto"


def test_an_unreadable_db_reports_the_default(monkeypatch):
    """A settings row is not worth a 500, and the default is what the install already ran on."""

    def locked(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_app_setting", locked)
    assert transport_settings.get_download_transport_mode() == "auto"


def test_set_validates_and_persists(store):
    assert transport_settings.set_download_transport_mode("XET ") == "xet"
    assert store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] == "xet"
    for junk in ("ftp", "", None, 3, True):
        with pytest.raises(ValueError):
            transport_settings.set_download_transport_mode(junk)


# ------------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------------


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False)


def test_route_reports_the_mode_and_what_auto_would_do(client):
    body = client.get("/download-transport").json()
    assert body["mode"] == "auto"
    # The row shows this under Auto, so it has to be a transport a download can actually run on.
    assert body["auto_resolves_to"] in {"http", "xet"}
    assert isinstance(body["xet_available"], bool)


def test_route_saves_a_choice(client, store):
    body = client.put("/download-transport", json = {"mode": "http"}).json()
    assert body["mode"] == "http"
    assert store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] == "http"


def test_route_rejects_an_unknown_transport(client):
    assert client.put("/download-transport", json = {"mode": "ftp"}).status_code == 422
