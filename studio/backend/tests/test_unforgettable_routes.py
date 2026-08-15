# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from auth.authentication import get_current_subject
from routes.unforgettable import router
from unforgettable.store.records import insert_record
from utils import unforgettable_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "memory.db"
    monkeypatch.setattr(unforgettable_settings, "memory_db_path", lambda: db_path)
    import routes.unforgettable as routes_mod

    monkeypatch.setattr(routes_mod, "memory_db_path", lambda: db_path)
    app = FastAPI()
    app.include_router(router, prefix="/api/unforgettable")
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    return TestClient(app), db_path


def test_summary_and_admit(client):
    http, db_path = client
    rec = insert_record(
        kind="claim",
        title="Draft",
        body="maybe",
        provenance="infer",
        status="proposed",
        db_path=db_path,
    )
    summary = http.get("/api/unforgettable/summary").json()
    assert summary["records"]["by_status"]["proposed"] == 1
    listed = http.get("/api/unforgettable/records", params={"status": "proposed"})
    assert listed.status_code == 200
    assert listed.json()["records"][0]["id"] == rec["id"]
    admitted = http.post(f"/api/unforgettable/records/{rec['id']}/admit", json={})
    assert admitted.status_code == 200
    assert admitted.json()["status"] == "active"


def test_admit_active_without_force_conflicts(client):
    http, db_path = client
    rec = insert_record(
        kind="claim",
        title="Live",
        body="stays",
        provenance="world",
        db_path=db_path,
    )
    response = http.post(f"/api/unforgettable/records/{rec['id']}/admit", json={})
    assert response.status_code == 409
    forced = http.post(
        f"/api/unforgettable/records/{rec['id']}/admit", json={"force": True}
    )
    assert forced.status_code == 200


def test_compact_defaults_to_dry_run(client):
    http, db_path = client
    insert_record(
        kind="claim",
        title="Keep",
        body="body",
        provenance="world",
        db_path=db_path,
    )
    report = http.post("/api/unforgettable/compact", json={}).json()
    assert report["dry_run"] is True
