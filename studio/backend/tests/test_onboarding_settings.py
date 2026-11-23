# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth.authentication import get_current_subject  # noqa: E402
from routes import settings as settings_routes  # noqa: E402
from storage import studio_db  # noqa: E402
from utils.onboarding_settings import _HISTORY_TABLES  # noqa: E402


@pytest.fixture
def settings_db(monkeypatch, tmp_path):
    database = tmp_path / "studio.db"

    def connect():
        connection = sqlite3.connect(database, timeout = 5)
        connection.row_factory = sqlite3.Row
        return connection

    # The real schema, so a renamed history table fails here rather than 500 in production.
    with connect() as connection:
        studio_db._ensure_schema(connection)
    monkeypatch.setattr("storage.studio_db.get_connection", connect)
    return connect


@pytest.fixture
def client(settings_db):
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.include_router(settings_routes.router, prefix = "/api/settings")
    return TestClient(app)


def test_fresh_account_is_unsaved_without_history(client):
    body = client.get("/api/settings/onboarding").json()
    assert body == {"saved": False, "pages": [], "completedAt": None, "hasHistory": False}


@pytest.mark.parametrize("table", _HISTORY_TABLES)
def test_existing_account_reports_history_until_saved(client, settings_db, table):
    with settings_db() as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
        required = [c["name"] for c in columns if c["notnull"] and c["dflt_value"] is None]
        connection.execute(
            f"INSERT INTO {table} ({', '.join(required)}) VALUES ({', '.join('?' * len(required))})",
            ["x"] * len(required),
        )
    assert client.get("/api/settings/onboarding").json()["hasHistory"] is True
    client.put("/api/settings/onboarding", json = {"pages": ["welcome"], "completed": True})
    body = client.get("/api/settings/onboarding").json()
    assert body["saved"] is True
    assert body["hasHistory"] is False


def test_pages_merge_in_order_and_completion_sticks(client):
    client.put("/api/settings/onboarding", json = {"pages": ["welcome", "profile"]})
    body = client.put(
        "/api/settings/onboarding", json = {"pages": ["profile", "look", "look"], "completed": True}
    ).json()
    assert body["pages"] == ["welcome", "profile", "look"]
    first = body["completedAt"]
    assert isinstance(first, int)
    again = client.put("/api/settings/onboarding", json = {"pages": ["device"]}).json()
    assert again["completedAt"] == first
    assert again["pages"][-1] == "device"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"pages": ["Bad Id"]}, id = "invalid_page_id"),
        pytest.param({"pages": ["p"] * 33}, id = "too_many_pages"),
    ],
)
def test_invalid_payloads_are_rejected(client, payload):
    assert client.put("/api/settings/onboarding", json = payload).status_code == 422
