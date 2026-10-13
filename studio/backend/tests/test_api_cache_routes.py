# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The cache settings endpoints: keys in, sizes out, and no path ever accepted."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
import routes.settings as settings_route


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def test_the_inventory_reports_a_size_per_cache(monkeypatch, client):
    monkeypatch.setattr(
        settings_route,
        "cache_inventory",
        lambda refresh = False: {
            "caches": [
                {
                    "key": "uv",
                    "group": "packages",
                    "opt_in": False,
                    "paths": ["/cache/uv"],
                    "size_bytes": 2048,
                    "entry_count": 3,
                    "present": True,
                    "purgeable": True,
                    "blocked_reason": None,
                }
            ],
            "total_bytes": 2048,
            "reclaimable_bytes": 2048,
            "free_bytes": 100,
            "total_disk_bytes": 200,
        },
    )
    body = client.get("/api/settings/caches").json()
    assert body["total_bytes"] == 2048
    assert body["caches"][0]["key"] == "uv"
    assert body["caches"][0]["size_bytes"] == 2048


def test_a_purge_names_caches_by_key(monkeypatch, client):
    asked: list = []

    def fake_purge(keys):
        asked.append(list(keys))
        return {
            "results": [{"key": "uv", "freed_bytes": 10, "removed_entries": 1, "errors": []}],
            "freed_bytes": 10,
            "inventory": {
                "caches": [],
                "total_bytes": 0,
                "reclaimable_bytes": 0,
                "free_bytes": None,
                "total_disk_bytes": None,
            },
        }

    monkeypatch.setattr(settings_route, "purge_caches", fake_purge)
    body = client.post("/api/settings/caches/purge", json = {"keys": ["uv"]}).json()
    assert asked == [["uv"]]
    assert body["freed_bytes"] == 10


def test_a_purge_that_names_a_path_is_rejected(monkeypatch, client):
    called = []
    monkeypatch.setattr(
        settings_route,
        "purge_caches",
        lambda keys: called.append(keys) or (_ for _ in ()).throw(ValueError("boom")),
    )
    # No path field exists to send one through, and a path in the key list is
    # simply not a key.
    response = client.post("/api/settings/caches/purge", json = {"paths": ["/"]})
    assert response.status_code == 422
    assert called == []

    response = client.post("/api/settings/caches/purge", json = {"keys": []})
    assert response.status_code == 422


def test_an_unknown_key_is_a_bad_request(client):
    response = client.post("/api/settings/caches/purge", json = {"keys": ["/"]})
    assert response.status_code == 400
    assert "Unknown cache key" in response.json()["detail"]


def test_an_api_key_caller_cannot_delete_anything(monkeypatch):
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: True
    called = []
    monkeypatch.setattr(settings_route, "purge_caches", lambda keys: called.append(keys))
    remote = TestClient(app)
    response = remote.post("/api/settings/caches/purge", json = {"keys": ["uv"]})
    assert response.status_code == 403
    assert called == []


def test_recheck_asks_for_a_fresh_walk(monkeypatch, client):
    asked: list = []

    def fake_inventory(refresh = False):
        asked.append(refresh)
        return {
            "caches": [],
            "total_bytes": 0,
            "reclaimable_bytes": 0,
            "free_bytes": None,
            "total_disk_bytes": None,
        }

    monkeypatch.setattr(settings_route, "cache_inventory", fake_inventory)
    client.get("/api/settings/caches")
    client.get("/api/settings/caches?refresh=true")
    assert asked == [False, True]


def test_an_api_key_caller_cannot_force_a_rescan(monkeypatch):
    """A forced walk has no memo in front of it and takes seconds per cache.

    It is the interactive Recheck button, so an API key gets the memoised read
    and not the one that occupies an executor thread on demand.
    """
    app = FastAPI()
    app.include_router(settings_route.router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: True
    asked: list = []

    def fake_inventory(refresh = False):
        asked.append(refresh)
        return {
            "caches": [],
            "total_bytes": 0,
            "reclaimable_bytes": 0,
            "free_bytes": None,
            "total_disk_bytes": None,
        }

    monkeypatch.setattr(settings_route, "cache_inventory", fake_inventory)
    remote = TestClient(app)
    assert remote.get("/api/settings/caches?refresh=true").status_code == 403
    assert asked == []
    # ...and the ordinary read still answers it.
    assert remote.get("/api/settings/caches").status_code == 200
    assert asked == [False]
