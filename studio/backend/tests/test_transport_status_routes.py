# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util

import pytest

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("fastapi") is None,
        reason = "fastapi is not installed",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("structlog") is None,
        reason = "structlog is not installed",
    ),
]


def test_model_and_dataset_transport_status_share_response_schema(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    import routes.datasets as dataset_routes
    import routes.models as model_routes

    def has_active_incomplete_blobs(repo_type: str, repo_id: str) -> bool:
        return repo_type == "dataset" and repo_id == "org/dataset"

    def read_active_transport_marker(repo_type: str, repo_id: str) -> str:
        return "xet" if repo_type == "dataset" else "http"

    def is_resumable_partial(repo_type: str, repo_id: str) -> bool:
        return repo_type == "model" and repo_id == "org/model"

    monkeypatch.setattr(
        model_routes.hf_cache_scan,
        "has_active_incomplete_blobs",
        has_active_incomplete_blobs,
    )
    monkeypatch.setattr(
        model_routes.hf_cache_scan,
        "read_active_transport_marker",
        read_active_transport_marker,
    )
    monkeypatch.setattr(
        model_routes.hf_cache_scan,
        "is_resumable_partial",
        is_resumable_partial,
    )

    app = FastAPI()
    app.include_router(model_routes.router, prefix = "/api/models")
    app.include_router(dataset_routes.router, prefix = "/api/datasets")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    client = TestClient(app)
    model = client.get(
        "/api/models/transport-status",
        params = {"repo_id": "org/model"},
    )
    dataset = client.get(
        "/api/datasets/transport-status",
        params = {"repo_id": "org/dataset"},
    )

    assert model.status_code == 200
    assert dataset.status_code == 200
    assert set(model.json()) == set(dataset.json()) == {
        "has_partial",
        "last_transport",
        "resumable",
    }
    assert model.json() == {
        "has_partial": False,
        "last_transport": "http",
        "resumable": True,
    }
    assert dataset.json() == {
        "has_partial": True,
        "last_transport": "xet",
        "resumable": False,
    }


def test_download_transport_capabilities_reports_xet_unavailable(monkeypatch):
    import importlib.util as importlib_util

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    import main as backend_main

    original_find_spec = importlib_util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "hf_xet":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(
        backend_main.hf_cache_scan.importlib.util,
        "find_spec",
        fake_find_spec,
    )

    app = FastAPI()
    app.add_api_route(
        "/api/studio/download-transport-capabilities",
        backend_main.studio_download_transport_capabilities,
        methods = ["GET"],
    )
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    response = TestClient(app).get("/api/studio/download-transport-capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["http"] == {"available": True, "reason": None}
    assert body["xet"]["available"] is False
    assert "hf_xet" in body["xet"]["reason"]


def test_download_routes_reject_unavailable_xet(monkeypatch):
    import importlib.util as importlib_util

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    import routes.datasets as dataset_routes
    import routes.models as model_routes

    original_find_spec = importlib_util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "hf_xet":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(
        model_routes.hf_cache_scan.importlib.util,
        "find_spec",
        fake_find_spec,
    )

    app = FastAPI()
    app.include_router(model_routes.router, prefix = "/api/models")
    app.include_router(dataset_routes.router, prefix = "/api/datasets")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    client = TestClient(app)

    model = client.post(
        "/api/models/download",
        json = {"repo_id": "org/model", "use_xet": True},
    )
    dataset = client.post(
        "/api/datasets/download",
        json = {"repo_id": "org/dataset", "use_xet": True},
    )

    assert model.status_code == 400
    assert dataset.status_code == 400
    assert "hf_xet" in model.json()["detail"]
    assert "hf_xet" in dataset.json()["detail"]
