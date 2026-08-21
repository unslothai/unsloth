# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The install's default download transport.

A new install downloads over HTTPS. One that predates that default keeps Auto, the transport it was
already on, so updating never changes how someone's downloads run. These tests pin the seed, what
counts as prior use, and the request path that reads it.
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
from hub.services import download_lifecycle as dl


@pytest.fixture(autouse = True)
def _fresh_seed():
    transport_settings.reset_seed_cache_for_tests()
    yield
    transport_settings.reset_seed_cache_for_tests()


@pytest.fixture
def store(monkeypatch):
    """An in-memory app_settings, plus an install with no history."""
    values: dict = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: values.get(key, fallback)
    )
    monkeypatch.setattr(
        studio_db, "upsert_app_settings", lambda updates: values.update(updates) or values
    )
    monkeypatch.setattr(transport_settings, "_has_prior_studio_use", lambda: False)
    return values


def test_a_new_install_downloads_over_https(store):
    assert transport_settings.get_download_transport_mode() == "http"
    # Seeded, not recomputed: prior use accumulates as the install is used, and re-deciding later
    # would move a fresh install onto Auto behind the user's back.
    assert store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] == "http"


def test_an_install_with_history_keeps_auto(store, monkeypatch):
    monkeypatch.setattr(transport_settings, "_has_prior_studio_use", lambda: True)
    transport_settings.reset_seed_cache_for_tests()
    assert transport_settings.get_download_transport_mode() == "auto"
    assert store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] == "auto"


def test_a_stored_choice_beats_the_seed(store):
    store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] = "xet"
    assert transport_settings.get_download_transport_mode() == "xet"


def test_junk_on_disk_reseeds_rather_than_sticking(store):
    store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] = "carrier-pigeon"
    assert transport_settings.get_download_transport_mode() == "http"


def test_set_validates_and_persists(store):
    assert transport_settings.set_download_transport_mode("XET ") == "xet"
    assert store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] == "xet"
    for junk in ("ftp", "", None, 3, True):
        with pytest.raises(ValueError):
            transport_settings.set_download_transport_mode(junk)


def test_an_unreadable_db_is_not_prior_use(monkeypatch):
    """Grandfathering needs positive evidence: a db that cannot be read is not it."""

    def locked(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_connection", locked)
    assert transport_settings._has_prior_studio_use() is False


def test_a_prior_download_manifest_counts_as_use(monkeypatch, tmp_path):
    from hub.utils import state_dir

    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "some-repo.json").write_text("{}")

    # No db rows at all: the manifest alone has to be enough, since a user who only ever downloaded
    # models writes nothing else.
    def no_db(*_args, **_kwargs):
        raise RuntimeError("no db")

    monkeypatch.setattr(studio_db, "get_connection", no_db)
    monkeypatch.setattr(state_dir, "manifests_dir", lambda **_kwargs: manifests)
    assert transport_settings._has_prior_studio_use() is True


# ------------------------------------------------------------------------------------------
# The download request path
# ------------------------------------------------------------------------------------------


def test_a_request_stating_nothing_takes_the_install_setting(monkeypatch):
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    monkeypatch.setattr(dl, "installed_transport_mode", lambda: "http")
    assert dl.resolve_requested_use_xet(None, None)[0] is False

    monkeypatch.setattr(dl, "installed_transport_mode", lambda: "xet")
    assert dl.resolve_requested_use_xet(None, None)[0] is True

    # Grandfathered installs land on the health verdict, as they did before.
    monkeypatch.setattr(dl, "installed_transport_mode", lambda: "auto")
    monkeypatch.setattr(dl, "resolve_auto_use_xet", lambda: (False, "demoted"))
    assert dl.resolve_requested_use_xet(None, None) == (False, "demoted")


def test_an_explicit_request_still_wins(monkeypatch):
    monkeypatch.setattr(dl, "resolve_effective_use_xet", lambda requested: requested)
    monkeypatch.setattr(dl, "installed_transport_mode", lambda: "http")
    assert dl.resolve_requested_use_xet("xet", None)[0] is True
    # The legacy boolean is a statement too, so it does not fall through to the setting.
    monkeypatch.setattr(dl, "installed_transport_mode", lambda: "xet")
    assert dl.resolve_requested_use_xet(None, False)[0] is False


def test_an_unreadable_setting_downloads_over_https(monkeypatch):
    def boom() -> str:
        raise RuntimeError("db gone")

    monkeypatch.setattr(transport_settings, "get_download_transport_mode", boom)
    assert dl.installed_transport_mode() == "http"


# ------------------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------------------


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(settings.router)
    app.dependency_overrides[settings.get_current_subject] = lambda: "admin"
    return TestClient(app, raise_server_exceptions = False)


def test_route_reports_the_seeded_mode_and_the_new_default(client):
    body = client.get("/download-transport").json()
    assert body["mode"] == "http"
    assert body["default_mode"] == "http"
    assert body["auto_resolves_to"] in {"http", "xet"}
    assert isinstance(body["xet_available"], bool)


def test_route_saves_a_choice(client, store):
    body = client.put("/download-transport", json = {"mode": "auto"}).json()
    assert body["mode"] == "auto"
    assert store[transport_settings.DOWNLOAD_TRANSPORT_SETTING_KEY] == "auto"


def test_route_rejects_an_unknown_transport(client):
    assert client.put("/download-transport", json = {"mode": "ftp"}).status_code == 422
