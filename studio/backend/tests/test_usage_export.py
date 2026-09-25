# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time
import pytest
from fastapi.testclient import TestClient
from core.inference.api_monitor import ApiMonitorEntry
from storage.usage_log import record_event


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import storage.studio_db as studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return tmp_path


def _get_client():
    """Create a TestClient with auth overridden to return the given subject."""
    from main import app
    from auth.authentication import get_current_subject

    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    return TestClient(app), app


def _get_client_as(subject: str):
    """Create a TestClient authenticated as the given subject."""
    from main import app
    from auth.authentication import get_current_subject

    app.dependency_overrides[get_current_subject] = lambda: subject
    return TestClient(app), app


def test_usage_export_empty(isolated_db):
    client, app = _get_client()
    try:
        resp_csv = client.get("/api/usage/export?format=csv")
        assert resp_csv.status_code == 200
        assert "id,ts,model" in resp_csv.text

        resp_json = client.get("/api/usage/export?format=json")
        assert resp_json.status_code == 200
        assert resp_json.json() == []
    finally:
        app.dependency_overrides.clear()


def test_usage_summary(isolated_db):
    client, app = _get_client()
    try:
        resp = client.get("/api/usage/summary?granularity=day")
        assert resp.status_code == 200
        data = resp.json()
        assert data["granularity"] == "day"
        assert data["rows"] == []
    finally:
        app.dependency_overrides.clear()


def test_usage_export_with_data(isolated_db):
    """Export endpoints return only the calling user's events."""
    for user in ("unsloth", "other_user"):
        entry = ApiMonitorEntry(
            id = f"evt-{user}",
            endpoint = "/chat",
            method = "POST",
            model = "llama3",
            prompt = "hi",
            status = "completed",
            started_at = time.time(),
            updated_at = time.time(),
            subject = user,
        )
        entry.total_tokens = 50
        record_event(entry)

    client, app = _get_client()
    try:
        # CSV export should only contain unsloth's event
        resp = client.get("/api/usage/export?format=csv")
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert len(lines) == 2  # header + 1 data row
        assert "evt-unsloth" in lines[1]
        assert "evt-other_user" not in resp.text

        # JSON export should only contain unsloth's event
        resp = client.get("/api/usage/export?format=json")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 1
        assert events[0]["id"] == "evt-unsloth"
    finally:
        app.dependency_overrides.clear()


def test_usage_settings_admin_gate(isolated_db):
    """PUT /settings is restricted to the admin user."""
    # Non-admin gets 403
    client, app = _get_client_as("regular_user")
    try:
        resp = client.put(
            "/api/usage/settings",
            json = {"mode": "months", "value": 6},
        )
        assert resp.status_code == 403
    finally:
        app.dependency_overrides.clear()

    # Admin (unsloth) succeeds
    client, app = _get_client_as("unsloth")
    try:
        resp = client.put(
            "/api/usage/settings",
            json = {"mode": "months", "value": 6},
        )
        assert resp.status_code == 200
        assert resp.json()["mode"] == "months"
        assert resp.json()["value"] == 6
    finally:
        app.dependency_overrides.clear()


def test_usage_summary_scoped(isolated_db):
    """GET /summary only returns the calling user's aggregated data."""
    for user in ("unsloth", "other_user"):
        entry = ApiMonitorEntry(
            id = f"sum-{user}",
            endpoint = "/chat",
            method = "POST",
            model = "llama3",
            prompt = "hi",
            status = "completed",
            started_at = time.time(),
            updated_at = time.time(),
            subject = user,
        )
        entry.total_tokens = 100
        entry.prompt_tokens = 40
        entry.completion_tokens = 60
        record_event(entry)

    client, app = _get_client()
    try:
        resp = client.get("/api/usage/summary?granularity=day")
        assert resp.status_code == 200
        rows = resp.json()["rows"]
        assert len(rows) == 1
        assert rows[0]["total_tokens"] == 100
    finally:
        app.dependency_overrides.clear()
