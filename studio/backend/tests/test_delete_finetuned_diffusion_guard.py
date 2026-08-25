# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DELETE /api/models/delete-finetuned must refuse a directory the Images or Video
engine is holding.

Every other guard on that route is chat-only (llama.cpp + the transformers backend), so a
local diffusion model under the storage root -- Images loads any existing local path -- used
to be rmtree'd while a pipeline was still reading it, taking the companion VAE / text encoder
files sd.cpp re-reads on every generation with it. The cached-model delete route already
refuses this; these tests pin the same behaviour on the trained/exported route, which matches
by PATH rather than by repo id.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.models as models_module
from auth.authentication import get_current_subject
from routes.models import router as models_router


class _Backend:
    """Minimal stand-in for the Images engine / Video backend delete-guard surface."""

    def __init__(
        self,
        *,
        loaded = None,
        base = None,
        loading = (),
        extra = (),
    ):
        self._loaded = loaded
        self._base = base
        self._loading = tuple(loading)
        self._extra = tuple(extra)

    def status(self):
        if self._loaded is None:
            return {"loaded": False}
        return {"loaded": True, "repo_id": self._loaded, "base_repo": self._base}

    def loaded_repo_ids(self):
        return self._extra

    def loading_repo_ids(self):
        return self._loading


@pytest.fixture()
def client(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr(models_module, "outputs_root", lambda: outputs)
    # No chat model is resident, so only the diffusion / video guards can refuse.
    monkeypatch.setattr(models_module, "get_inference_backend", lambda: _NoChat())

    app = FastAPI()
    app.include_router(models_router, prefix = "/api/models")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), outputs


class _NoChat:
    active_model_name = None
    loading_models: set = set()


def _model_dir(outputs):
    d = outputs / "my-diffusion-model"
    d.mkdir()
    (d / "model_index.json").write_text("{}", encoding = "utf-8")
    return d


def _delete(client, path):
    return client.request(
        "DELETE",
        "/api/models/delete-finetuned",
        json = {"model_path": str(path), "source": "training"},
    )


def test_refuses_to_delete_a_model_the_images_engine_has_loaded(client, monkeypatch):
    c, outputs = client
    target = _model_dir(outputs)
    monkeypatch.setattr(
        models_module, "_active_diffusion_backend", lambda: _Backend(loaded = str(target))
    )
    monkeypatch.setattr(models_module, "_active_video_backend", lambda: None)

    resp = _delete(c, target)
    assert resp.status_code == 400
    assert "Unload the model" in resp.json()["detail"]
    assert target.exists()


def test_refuses_the_companion_base_and_the_extra_repos_the_engine_reads(client, monkeypatch):
    c, outputs = client
    target = _model_dir(outputs)
    other = outputs / "somewhere-else"
    other.mkdir()
    # The checkpoint is the loaded id; the deleted dir is only the companion base.
    monkeypatch.setattr(
        models_module,
        "_active_diffusion_backend",
        lambda: _Backend(loaded = str(other), base = str(target)),
    )
    monkeypatch.setattr(models_module, "_active_video_backend", lambda: None)
    assert _delete(c, target).status_code == 400

    # Same for a companion the engine reports through loaded_repo_ids (sd.cpp VAE / TE).
    monkeypatch.setattr(
        models_module,
        "_active_diffusion_backend",
        lambda: _Backend(loaded = str(other), extra = (str(target),)),
    )
    assert _delete(c, target).status_code == 400
    assert target.exists()


def test_refuses_while_the_video_backend_is_still_fetching_it(client, monkeypatch):
    c, outputs = client
    target = _model_dir(outputs)
    monkeypatch.setattr(models_module, "_active_diffusion_backend", lambda: None)
    monkeypatch.setattr(
        models_module, "_active_video_backend", lambda: _Backend(loading = (str(target),))
    )

    resp = _delete(c, target)
    assert resp.status_code == 409
    assert "loading" in resp.json()["detail"].lower()
    assert target.exists()


def test_allows_the_delete_when_no_diffusion_or_video_model_holds_it(client, monkeypatch):
    c, outputs = client
    target = _model_dir(outputs)
    monkeypatch.setattr(
        models_module,
        "_active_diffusion_backend",
        lambda: _Backend(loaded = str(outputs / "another-model")),
    )
    monkeypatch.setattr(models_module, "_active_video_backend", lambda: _Backend())

    assert _delete(c, target).status_code == 200
    assert not target.exists()


def test_a_chat_only_install_can_still_delete(client, monkeypatch):
    """No diffusion stack installed: the guard must fail OPEN, not 503 every delete."""
    c, outputs = client
    target = _model_dir(outputs)
    monkeypatch.setattr(models_module, "_active_diffusion_backend", lambda: None)
    monkeypatch.setattr(models_module, "_active_video_backend", lambda: None)

    assert _delete(c, target).status_code == 200
    assert not target.exists()


def test_an_unreadable_engine_state_fails_closed(client, monkeypatch):
    """The engine exists but cannot report its state: refuse rather than risk the rmtree."""
    c, outputs = client
    target = _model_dir(outputs)

    class _Broken:
        def status(self):
            raise RuntimeError("engine wedged")

    monkeypatch.setattr(models_module, "_active_diffusion_backend", lambda: _Broken())
    monkeypatch.setattr(models_module, "_active_video_backend", lambda: None)

    resp = _delete(c, target)
    assert resp.status_code == 503
    assert target.exists()


def test_refuses_the_delete_while_a_diffusion_training_run_is_active(client, monkeypatch):
    # source="training" checked only the LLM trainer, so a delete could rmtree the output directory a live diffusion LoRA run is about to write into.
    import sys
    import types

    c, outputs = client
    target = _model_dir(outputs)
    monkeypatch.setattr(models_module, "_active_diffusion_backend", lambda: None)
    monkeypatch.setattr(models_module, "_active_video_backend", lambda: None)

    stub = types.ModuleType("core.training.diffusion_training_service")
    stub.get_diffusion_training_service = lambda: types.SimpleNamespace(is_active = lambda: True)
    monkeypatch.setitem(sys.modules, "core.training.diffusion_training_service", stub)

    resp = _delete(c, target)
    assert resp.status_code == 409
    assert "diffusion" in resp.json()["detail"].lower()
    assert target.exists()

    # Idle again: the same delete goes through.
    stub.get_diffusion_training_service = lambda: types.SimpleNamespace(is_active = lambda: False)
    assert _delete(c, target).status_code == 200
    assert not target.exists()
