# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import storage.studio_db as studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return tmp_path


def test_usage_export_empty(isolated_db):
    from main import app

    client = TestClient(app)
    app.dependency_overrides = {}
    from auth.authentication import get_current_subject

    app.dependency_overrides[get_current_subject] = lambda: "admin"

    resp_csv = client.get("/api/usage/export?format=csv")
    assert resp_csv.status_code == 200
    assert "id,ts,model" in resp_csv.text

    resp_json = client.get("/api/usage/export?format=json")
    assert resp_json.status_code == 200
    assert resp_json.json() == []


def test_usage_summary(isolated_db):
    from main import app

    client = TestClient(app)
    from auth.authentication import get_current_subject

    app.dependency_overrides[get_current_subject] = lambda: "admin"

    resp = client.get("/api/usage/summary?granularity=day")
    assert resp.status_code == 200
    data = resp.json()
    assert data["granularity"] == "day"
    assert data["rows"] == []
