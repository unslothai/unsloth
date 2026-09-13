# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The backend's own HF_TOKEN is the operator's credential, not a shared service credential.

The Unsloth UI sends the user's saved token in ``X-Unsloth-HF-Token`` on every hub download, so
only a caller that has none reaches the ambient fallback. A UI session is the installation's
owner and keeps it (Settings hands that session the saved token anyway). An sk-unsloth API key
is the lesser credential -- Settings refuses it the saved token -- so it must not reach private
repos by naming one in a download request instead.
"""

import asyncio
import io
import logging

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
from hub.dependencies import get_request_hf_token
from hub.services import download_lifecycle
from hub.services.datasets import downloads as dataset_downloads
from hub.services.models import downloads as model_downloads
from hub.utils import download_registry, state_dir
from routes import models as models_routes


class _Proc:
    pid = 4242

    def __init__(
        self,
        rc,
        stderr = b"",
    ):
        self.rc = rc
        self.stderr = io.BytesIO(stderr)
        self.waited = False

    def poll(self):
        return self.rc if self.waited else None

    def wait(self, timeout = None):
        self.waited = True
        return self.rc

    def kill(self):
        pass


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


def _client(via_api_key: bool) -> TestClient:
    app = FastAPI()
    app.include_router(inventory_routes.router, prefix = "/api/hub")
    app.include_router(datasets_routes.router, prefix = "/api/hub/datasets")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app)


def _models_client(via_api_key: bool) -> TestClient:
    app = FastAPI()
    app.include_router(models_routes.router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: "alice"
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app)


@pytest.mark.parametrize("via_api_key, expected", [(True, False), (False, True)])
def test_only_a_ui_session_may_borrow_the_backend_token(via_api_key, expected):
    assert asyncio.run(allow_ambient_hf_token(via_api_key = via_api_key)) is expected


@pytest.mark.parametrize(
    "hf_token, allow_ambient, expected",
    [
        (None, False, False),
        (None, True, None),
        ("request-token", False, "request-token"),
        (" request-token ", True, "request-token"),
    ],
)
def test_request_metadata_token_keeps_the_caller_boundary(hf_token, allow_ambient, expected):
    resolved = get_request_hf_token(
        hf_token = hf_token,
        allow_ambient_token = allow_ambient,
    )
    assert resolved == expected
    if expected in (None, False):
        assert resolved is expected


@pytest.mark.parametrize("via_api_key, expected", [(True, False), (False, None)])
def test_gguf_metadata_route_does_not_lend_api_keys_the_backend_token(
    monkeypatch, via_api_key, expected
):
    seen = {}

    async def _fake(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["hf_token"] = kwargs["hf_token"]
        return {"repo_id": repo_id, "variants": []}

    monkeypatch.setattr(inventory_routes.gguf_variants, "get_gguf_variants_response", _fake)
    response = _client(via_api_key).get(
        "/api/hub/gguf-variants?repo_id=attacker/private-model",
        headers = {"Authorization": "Bearer token"},
    )

    assert response.status_code == 200, response.text
    assert seen == {"repo_id": "attacker/private-model", "hf_token": expected}


def test_explicit_metadata_token_wins_for_an_api_key(monkeypatch):
    seen = {}

    async def _fake(repo_id, **kwargs):
        seen["hf_token"] = kwargs["hf_token"]
        return {"repo_id": repo_id, "variants": []}

    monkeypatch.setattr(inventory_routes.gguf_variants, "get_gguf_variants_response", _fake)
    response = _client(True).get(
        "/api/hub/gguf-variants?repo_id=owner/private-model",
        headers = {
            "Authorization": "Bearer token",
            "X-Unsloth-HF-Token": "request-token",
        },
    )

    assert response.status_code == 200, response.text
    assert seen["hf_token"] == "request-token"


@pytest.mark.parametrize("via_api_key, expected", [(True, False), (False, None)])
def test_compatibility_progress_route_keeps_the_caller_boundary(monkeypatch, via_api_key, expected):
    seen = {}

    async def _fake(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["hf_token"] = kwargs["hf_token"]
        return {"repo_id": repo_id, "progress": 0.0}

    monkeypatch.setattr(model_downloads, "get_download_progress_response", _fake)
    response = _models_client(via_api_key).get(
        "/api/models/download-progress?repo_id=attacker/private-model",
        headers = {"Authorization": "Bearer token"},
    )

    assert response.status_code == 200, response.text
    assert seen == {"repo_id": "attacker/private-model", "hf_token": expected}


@pytest.mark.parametrize("via_api_key, expected", [(True, False), (False, True)])
def test_model_download_route_gates_the_ambient_token(monkeypatch, via_api_key, expected):
    seen = {}

    async def _fake(
        body,
        hf_token = None,
        *,
        allow_ambient_token = True,
    ):
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

    async def _fake(
        body,
        hf_token = None,
        *,
        allow_ambient_token = True,
    ):
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


def _spawn_env(monkeypatch, hf_token, **kwargs):
    """Run the real spawn_worker against a fake Popen and return the child's environment."""
    captured = {}

    class _Fake:
        pass

    def _fake_popen(*_args, **popen_kwargs):
        captured.update(popen_kwargs["env"])
        return _Fake()

    monkeypatch.setattr(download_lifecycle.subprocess, "Popen", _fake_popen)
    download_lifecycle.spawn_worker(
        ["--repo-id", "attacker/private-model"],
        hf_token,
        use_xet = False,
        **kwargs,
    )
    return captured


def test_an_api_caller_does_not_borrow_the_backend_hf_token(monkeypatch):
    """A caller the route marked as not allowed the ambient token gets an anonymous worker, even
    though the backend process has an HF_TOKEN of its own."""
    monkeypatch.setenv("HF_TOKEN", "operator-secret-token")

    env = _spawn_env(monkeypatch, None, allow_ambient_token = False)

    assert "HF_TOKEN" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_the_ui_still_falls_back_to_the_backend_hf_token(monkeypatch):
    """The other half: a UI session keeps the fallback, so a private repo stays downloadable for
    an install whose token lives in the environment rather than in Settings."""
    monkeypatch.setenv("HF_TOKEN", "operator-secret-token")

    env = _spawn_env(monkeypatch, None, allow_ambient_token = True)

    assert env["HF_TOKEN"] == "operator-secret-token"
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "0"


def test_an_explicit_request_token_wins_over_the_backend_one(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "operator-secret-token")

    env = _spawn_env(monkeypatch, "request-token", allow_ambient_token = True)

    assert env["HF_TOKEN"] == "request-token"


@pytest.mark.parametrize("allow_ambient", [False, True])
def test_http_retry_preserves_ambient_token_policy(
    monkeypatch, tmp_path, cached_hf_login, allow_ambient
):
    """The recovery ladder must carry the token policy: a job started without the ambient token
    must not pick it up when the Xet worker fails and the HTTP one takes over."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", lambda *a, **k: None)
    register_worker = download_lifecycle.register_worker

    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = None,
        blob_hashes = frozenset({"blob"}),
    )[0]
    retried = []
    environments = []
    original_spawn = download_lifecycle.spawn_worker

    def fake_popen(*args, **kwargs):
        environments.append(kwargs["env"])
        return _Proc(0)

    monkeypatch.setattr(download_lifecycle.subprocess, "Popen", fake_popen)

    def fake_spawn(
        _args,
        _token,
        *,
        use_xet,
        allow_ambient_token = True,
        **_kwargs,
    ):
        retried.append(allow_ambient_token)
        return original_spawn(
            _args,
            _token,
            use_xet = use_xet,
            allow_ambient_token = allow_ambient_token,
            **_kwargs,
        )

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn)
    monkeypatch.setattr(download_lifecycle, "register_worker", lambda *a, **k: True)
    assert register_worker(
        registry,
        key,
        _Proc(1, b"xet failed"),
        hf_token = None,
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
        allow_ambient_token = allow_ambient,
    )

    assert retried == [allow_ambient]
    assert len(environments) == 1
    assert environments[0].get("HF_TOKEN") == (cached_hf_login if allow_ambient else None)


@pytest.fixture
def cached_hf_login(monkeypatch, tmp_path):
    from huggingface_hub import constants

    token_path = tmp_path / "token"
    token_path.write_text("hf_test_cached_login\n")
    monkeypatch.setattr(constants, "HF_TOKEN_PATH", str(token_path))
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", False)
    for key in (
        "HF_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ):
        monkeypatch.delenv(key, raising = False)
    return "hf_test_cached_login"


@pytest.mark.parametrize("allow_ambient", [False, True])
@pytest.mark.parametrize("explicit", [None, "hf_test_request"])
@pytest.mark.parametrize("implicit_disabled", [False, True])
def test_cached_login_obeys_caller_and_implicit_auth_policy(
    monkeypatch, cached_hf_login, allow_ambient, explicit, implicit_disabled
):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", implicit_disabled)
    env = _spawn_env(monkeypatch, explicit, allow_ambient_token = allow_ambient)

    expected = explicit or (cached_hf_login if allow_ambient and not implicit_disabled else None)
    assert env.get("HF_TOKEN") == expected
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == ("0" if expected else "1")


def test_environment_token_takes_precedence_over_cached_login(monkeypatch, cached_hf_login):
    monkeypatch.setenv("HF_TOKEN", "hf_test_environment")

    env = _spawn_env(monkeypatch, None, allow_ambient_token = True)

    assert env["HF_TOKEN"] == "hf_test_environment"


def test_forbidden_ambient_token_is_not_resolved(monkeypatch, cached_hf_login):
    import huggingface_hub.utils

    def unexpected_resolution(*args):
        pytest.fail("A restricted caller must not resolve the owner's credentials")

    monkeypatch.setattr(huggingface_hub.utils, "get_token_to_send", unexpected_resolution)
    env = _spawn_env(
        monkeypatch,
        None,
        allow_ambient_token = False,
        cache_env = {
            "HF_TOKEN": "hf_test_environment",
            "HF_HUB_TOKEN": "hf_test_alias",
            "HUGGING_FACE_HUB_TOKEN": "hf_test_alias",
            "HUGGINGFACE_HUB_TOKEN": "hf_test_alias",
            "HUGGINGFACEHUB_API_TOKEN": "hf_test_alias",
        },
    )

    assert not any(
        key in env
        for key in (
            "HF_TOKEN",
            "HF_HUB_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
        )
    )
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


@pytest.mark.parametrize("failure", ["unreadable", "invalid_encoding"])
def test_unusable_cached_login_does_not_block_an_anonymous_worker(
    monkeypatch, cached_hf_login, failure
):
    from pathlib import Path
    from huggingface_hub import constants

    token_path = Path(constants.HF_TOKEN_PATH)
    if failure == "invalid_encoding":
        token_path.write_bytes(b"\x81")
    else:
        original_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == token_path:
                raise PermissionError("cached token is unreadable")
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)

    env = _spawn_env(monkeypatch, None, allow_ambient_token = True)

    assert "HF_TOKEN" not in env
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
