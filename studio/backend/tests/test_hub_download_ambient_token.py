# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend's own HF_TOKEN is the operator's credential, not a shared service credential.

The Studio UI sends the user's saved token in ``X-Unsloth-HF-Token`` on every hub download, so
only a caller that has none reaches the ambient fallback. A UI session is the installation's
owner and keeps it (Settings hands that session the saved token anyway). An sk-unsloth API key
is the lesser credential -- Settings refuses it the saved token -- so it must not reach private
repos by naming one in a download request instead.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import (
    allow_ambient_hf_token,
    authenticated_via_api_key,
    get_current_subject,
)
from hub.routes import datasets as datasets_routes
from hub.routes import inventory as inventory_routes
from hub.services.datasets import downloads as dataset_downloads
from hub.services.models import downloads as model_downloads


def _client(via_api_key: bool) -> TestClient:
    app = FastAPI()
    app.include_router(inventory_routes.router, prefix = "/api/hub")
    app.include_router(datasets_routes.router, prefix = "/api/hub/datasets")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app)


@pytest.mark.parametrize("via_api_key, expected", [(True, False), (False, True)])
def test_only_a_ui_session_may_borrow_the_backend_token(via_api_key, expected):
    import asyncio

    assert asyncio.run(allow_ambient_hf_token(via_api_key = via_api_key)) is expected


@pytest.mark.parametrize("via_api_key, expected", [(True, False), (False, True)])
def test_model_download_route_gates_the_ambient_token(monkeypatch, via_api_key, expected):
    seen = {}

    async def _fake(body, hf_token = None, *, allow_ambient_token = True):
        seen["repo_id"] = body.repo_id
        seen["allow_ambient_token"] = allow_ambient_token
        return {"job_key": "k", "state": "running", "accepted": True, "generation": 1}

    monkeypatch.setattr(model_downloads, "download_model_response", _fake)

    response = _client(via_api_key).post(
        "/api/hub/download",
        json = {"repo_id": "attacker/private-model"},
        headers = {"Authorization": "Bearer token"},
    )

    assert response.status_code == 202, response.text
    assert seen["repo_id"] == "attacker/private-model"
    assert seen["allow_ambient_token"] is expected


@pytest.mark.parametrize("via_api_key, expected", [(True, False), (False, True)])
def test_dataset_download_route_gates_the_ambient_token(monkeypatch, via_api_key, expected):
    seen = {}

    async def _fake(body, hf_token = None, *, allow_ambient_token = True):
        seen["repo_id"] = body.repo_id
        seen["allow_ambient_token"] = allow_ambient_token
        return {"repo_id": body.repo_id, "state": "running", "accepted": True, "generation": 1}

    monkeypatch.setattr(dataset_downloads, "download_dataset_response", _fake)

    response = _client(via_api_key).post(
        "/api/hub/datasets/download",
        json = {"repo_id": "attacker/private-dataset"},
        headers = {"Authorization": "Bearer token"},
    )

    assert response.status_code == 202, response.text
    assert seen["repo_id"] == "attacker/private-dataset"
    assert seen["allow_ambient_token"] is expected
