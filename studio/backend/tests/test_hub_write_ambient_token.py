# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend's own HF_TOKEN must not be lent to API-key callers on Hub write paths.

Issue #10126: Write endpoints (recipe dataset publishing, model exports pushing to Hub,
and checkpoint loading) must require an explicit HF token from sk-unsloth API key callers
rather than borrowing the server operator's ambient HF_TOKEN.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from routes import export as export_routes
from routes.data_recipe import jobs as data_recipe_jobs_routes
from utils.models.model_config import _detect_audio_from_tokenizer


async def _fake_ensure_export_supported():
    pass


def _create_test_app(via_api_key: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(data_recipe_jobs_routes.router, prefix = "/api/data-recipe")
    app.include_router(export_routes.router, prefix = "/api")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[get_current_credential] = lambda: ("alice", "cred-1")
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return app


# ==============================================================================
# Data Recipe Publish Endpoint Tests
# ==============================================================================


def test_recipe_publish_refuses_api_key_without_token(monkeypatch):
    mock_mgr = MagicMock()
    mock_mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": "/tmp/test-artifacts",
    }
    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mock_mgr)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/data-recipe/jobs/job-123/publish",
        json = {"repo_id": "org/dataset", "description": "Test dataset", "hf_token": None},
    )
    assert response.status_code == 400
    assert (
        "Hugging Face token is required to publish datasets when authenticated via API key"
        in response.json()["detail"]
    )


def test_recipe_publish_allows_api_key_with_explicit_token(monkeypatch):
    mock_mgr = MagicMock()
    mock_mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": "/tmp/test-artifacts",
    }
    seen_token = {}

    def _fake_publish(artifact_path, repo_id, description, hf_token, private):
        seen_token["token"] = hf_token
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mock_mgr)
    monkeypatch.setattr(data_recipe_jobs_routes, "publish_recipe_dataset", _fake_publish)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/data-recipe/jobs/job-123/publish",
        json = {
            "repo_id": "org/dataset",
            "description": "Test dataset",
            "hf_token": "hf_custom_caller_token",
        },
    )
    assert response.status_code == 200
    assert seen_token["token"] == "hf_custom_caller_token"


def test_recipe_publish_allows_ui_session_without_token(monkeypatch):
    mock_mgr = MagicMock()
    mock_mgr.get_status.return_value = {
        "status": "completed",
        "execution_type": "full",
        "artifact_path": "/tmp/test-artifacts",
    }
    seen_token = {}

    def _fake_publish(artifact_path, repo_id, description, hf_token, private):
        seen_token["token"] = hf_token
        return f"https://huggingface.co/datasets/{repo_id}"

    monkeypatch.setattr(data_recipe_jobs_routes, "get_job_manager", lambda: mock_mgr)
    monkeypatch.setattr(data_recipe_jobs_routes, "publish_recipe_dataset", _fake_publish)

    app = _create_test_app(via_api_key = False)
    client = TestClient(app)

    response = client.post(
        "/api/data-recipe/jobs/job-123/publish",
        json = {"repo_id": "org/dataset", "description": "Test dataset", "hf_token": None},
    )
    assert response.status_code == 200
    assert seen_token["token"] is None


# ==============================================================================
# Export Endpoints Tests
# ==============================================================================


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        (
            "/api/export/merged",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/merged-model",
                "hf_token": None,
            },
        ),
        (
            "/api/export/base",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/base-model",
                "hf_token": None,
            },
        ),
        (
            "/api/export/gguf",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/gguf-model",
                "hf_token": None,
                "quantization_method": "q4_k_m",
            },
        ),
        (
            "/api/export/lora",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/lora-model",
                "hf_token": None,
            },
        ),
    ],
)
def test_export_push_to_hub_refuses_api_key_without_token(monkeypatch, endpoint, payload):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(endpoint, json = payload)
    assert response.status_code == 400
    assert (
        "Hugging Face token is required to push to Hub when authenticated via API key"
        in response.json()["detail"]
    )


@pytest.mark.parametrize(
    "endpoint,method_name,payload",
    [
        (
            "/api/export/merged",
            "export_merged_model",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/merged-model",
                "hf_token": "hf_explicit_key_123",
            },
        ),
        (
            "/api/export/base",
            "export_base_model",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/base-model",
                "hf_token": "hf_explicit_key_123",
            },
        ),
        (
            "/api/export/gguf",
            "export_gguf",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/gguf-model",
                "hf_token": "hf_explicit_key_123",
                "quantization_method": "q4_k_m",
            },
        ),
        (
            "/api/export/lora",
            "export_lora_adapter",
            {
                "save_directory": "/tmp/export",
                "push_to_hub": True,
                "repo_id": "org/lora-model",
                "hf_token": "hf_explicit_key_123",
            },
        ),
    ],
)
def test_export_push_to_hub_allows_api_key_with_explicit_token(
    monkeypatch, endpoint, method_name, payload
):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    getattr(mock_backend, method_name).return_value = (True, "Export successful", "/tmp/export")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)
    monkeypatch.setattr(
        export_routes,
        "_export_details",
        lambda *args, **kwargs: {"output_path": "/tmp/export"},
    )

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(endpoint, json = payload)
    assert response.status_code == 200
    call_kwargs = getattr(mock_backend, method_name).call_args.kwargs
    assert call_kwargs["hf_token"] == "hf_explicit_key_123"


def test_load_checkpoint_passes_allow_ambient_false_for_api_key(monkeypatch):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    mock_backend.load_checkpoint.return_value = (True, "Checkpoint loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/load-checkpoint",
        json = {"checkpoint_path": "/tmp/checkpoints/my-model", "hf_token": None},
    )
    assert response.status_code == 200
    call_kwargs = mock_backend.load_checkpoint.call_args.kwargs
    assert call_kwargs["hf_token"] is None
    assert call_kwargs["allow_ambient"] is False


def test_load_checkpoint_passes_allow_ambient_true_for_ui_session(monkeypatch):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    mock_backend.load_checkpoint.return_value = (True, "Checkpoint loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)

    app = _create_test_app(via_api_key = False)
    client = TestClient(app)

    response = client.post(
        "/api/load-checkpoint",
        json = {"checkpoint_path": "/tmp/checkpoints/my-model", "hf_token": None},
    )
    assert response.status_code == 200
    call_kwargs = mock_backend.load_checkpoint.call_args.kwargs
    assert call_kwargs["hf_token"] is None
    assert call_kwargs["allow_ambient"] is True


def test_load_checkpoint_passes_explicit_token_for_api_key(monkeypatch):
    monkeypatch.setattr(export_routes, "_ensure_export_supported", _fake_ensure_export_supported)
    mock_backend = MagicMock()
    mock_backend.load_checkpoint.return_value = (True, "Checkpoint loaded")
    monkeypatch.setattr(export_routes, "get_export_backend", lambda: mock_backend)

    app = _create_test_app(via_api_key = True)
    client = TestClient(app)

    response = client.post(
        "/api/load-checkpoint",
        json = {"checkpoint_path": "/tmp/checkpoints/my-model", "hf_token": "hf_explicit_123"},
    )
    assert response.status_code == 200
    call_kwargs = mock_backend.load_checkpoint.call_args.kwargs
    assert call_kwargs["hf_token"] == "hf_explicit_123"
    assert call_kwargs["allow_ambient"] is False


def test_worker_scrubs_ambient_token_when_allow_ambient_false(monkeypatch):
    import os
    from core.export import worker

    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret_123")
    monkeypatch.setenv("HF_HUB_TOKEN", "hf_hub_secret_456")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_hub_secret_789")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    seen_env = {}

    def _fake_activate(path, token):
        seen_env["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        seen_env["HF_HUB_TOKEN"] = os.environ.get("HF_HUB_TOKEN")
        seen_env["DISABLE_IMPLICIT"] = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        seen_env["passed_token"] = token
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)

    config = {
        "checkpoint_path": "/tmp/model",
        "allow_ambient": False,
        "hf_token": None,
    }

    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = config,
        )

    assert seen_env.get("HF_TOKEN") is None
    assert seen_env.get("HF_HUB_TOKEN") is None
    assert seen_env.get("DISABLE_IMPLICIT") == "1"
    assert seen_env.get("passed_token") is None


def test_worker_preserves_ambient_token_when_allow_ambient_true(monkeypatch):
    import os
    from core.export import worker

    monkeypatch.setenv("HF_TOKEN", "hf_operator_secret_123")
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising = False)

    seen_env = {}

    def _fake_activate(path, token):
        seen_env["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        seen_env["DISABLE_IMPLICIT"] = os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN")
        seen_env["passed_token"] = token
        raise SystemExit(0)

    monkeypatch.setattr(worker, "_activate_transformers_version", _fake_activate)

    config = {
        "checkpoint_path": "/tmp/model",
        "allow_ambient": True,
        "hf_token": None,
    }

    with pytest.raises(SystemExit):
        worker.run_export_process(
            cmd_queue = MagicMock(),
            resp_queue = MagicMock(),
            config = config,
        )

    assert seen_env.get("HF_TOKEN") == "hf_operator_secret_123"
    assert seen_env.get("DISABLE_IMPLICIT") is None
    assert seen_env.get("passed_token") is None
