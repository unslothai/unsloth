# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A generation must clear every component of the resident it runs on, and keep owning it."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.inference import gpu_arbiter, image_gallery
from core.inference.diffusion_families import DiffusionModelReplacedError, load_identity
from hub.services.models import account_access as access
from routes import inference, video
from state import active_generations
from utils.account_context import AccountContext, bind_account, reset_account, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")

PUBLIC = {"org/public-model", "org/other-public", "org/public-video"}


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_resident_components", {}, raising = False)
    monkeypatch.setattr(access, "_generation_accounts", {})
    monkeypatch.setattr(access, "repo_is_public", lambda repo_id, *a, **k: repo_id in PUBLIC)
    monkeypatch.setattr(gpu_arbiter, "_owner", "diffusion")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", BOB.account_id)
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[allow_ambient_hf_token] = lambda: False
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(video.router, prefix = "/api/inference")
    return TestClient(app)


def _result(repo_id):
    return {
        "images": [Image.new("RGB", (8, 8))],
        "seed": 1,
        "seeds": [1],
        "repo_id": repo_id,
        "workflow": "txt2img",
    }


def _install_backend(monkeypatch, status, generate):
    from core.inference import diffusion_engine_router

    backend = SimpleNamespace(
        is_loaded = True,
        status = status,
        generate = generate,
        generate_progress = lambda: {
            "active": False,
            "step": 0,
            "total_steps": 0,
            "fraction": 0.0,
            "eta_seconds": None,
        },
        cancel_generate = lambda **kwargs: False,
    )
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    return backend


def test_a_private_base_repo_of_a_shared_resident_is_reauthorized(monkeypatch):
    used = []
    status = {
        "loaded": True,
        "repo_id": "org/public-model",
        "family": "z-image",
        "base_repo": "bob/private-base",
    }
    _install_backend(
        monkeypatch,
        lambda: status,
        lambda **kwargs: (used.append("ran"), _result("org/public-model"))[1],
    )
    with client_for(ALICE) as client:
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "a sloth"}).status_code
            == 404
        )
    assert used == []
    run_as(BOB, access.record_model_grant, "bob/private-base")
    with client_for(BOB) as client:
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "a sloth"}).status_code
            == 200
        )
    assert used == ["ran"]


def test_baked_private_adapters_of_a_shared_resident_are_reauthorized(monkeypatch):
    """The adapters baked in at load time are components too, and status() does not carry them."""
    used = []
    status = {
        "loaded": True,
        "repo_id": "org/public-model",
        "family": "z-image",
        "base_repo": None,
    }
    _install_backend(
        monkeypatch,
        lambda: status,
        lambda **kwargs: (used.append("ran"), _result("org/public-model"))[1],
    )
    run_as(
        BOB, access.note_resident_components, "diffusion", "org/public-model", "bob/private-lora"
    )
    with client_for(ALICE) as client:
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "a sloth"}).status_code
            == 404
        )
    assert used == []
    run_as(BOB, access.note_resident_components, "diffusion", "org/other-public")
    with client_for(ALICE) as client:
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "a sloth"}).status_code
            == 200
        )


def test_video_generation_reauthorizes_a_private_base_repo(monkeypatch):
    from core.inference import video as video_module

    started = []
    _status = lambda: {
        "loaded": True,
        "repo_id": "org/public-video",
        "family": "wan",
        "base_repo": "bob/private-base",
    }
    backend = SimpleNamespace(
        status = _status,
        generation_snapshot = lambda: (_status(), object()),
        begin_generate = lambda **kwargs: started.append("ran"),
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    monkeypatch.setattr(gpu_arbiter, "_owner", "video")
    with client_for(ALICE) as client:
        response = client.post(
            "/api/inference/video/generate",
            json = {"prompt": "a sloth", "width": 320, "height": 320, "num_frames": 17},
        )
    assert response.status_code == 404
    assert started == []


def test_a_video_resident_replaced_after_authorization_is_not_generated_on(monkeypatch):
    """Another account's load commits between the authorized snapshot and the reservation."""
    from core.inference import video as video_module
    from core.inference.video_families import VIDEO_MODEL_CHANGED_MSG

    resident = SimpleNamespace(repo_id = "org/public-video", family = "wan")
    replacement = SimpleNamespace(repo_id = "bob/private-video", family = "wan")
    box = {"state": resident, "reads": 0}
    reserved = []

    def _status_of(state):
        return {
            "loaded": True,
            "repo_id": state.repo_id,
            "family": state.family,
            "base_repo": None,
            "defaults": {
                "num_frames": 17,
                "fps": 16,
                "frame_step": 4,
                "frame_offset": 1,
                "resolution_presets": [[320, 320]],
            },
        }

    def generation_snapshot():
        state = box["state"]
        box["reads"] += 1
        if box["reads"] == 1:
            box["state"] = replacement
        return _status_of(state), state

    def begin_generate(**kwargs):
        expected = kwargs.get("expected_state")
        if expected is not None and expected is not box["state"]:
            raise RuntimeError(VIDEO_MODEL_CHANGED_MSG)
        reserved.append(box["state"].repo_id)
        return {"width": 320, "height": 320, "num_frames": 17, "fps": 16}

    backend = SimpleNamespace(
        status = lambda: _status_of(box["state"]),
        generation_snapshot = generation_snapshot,
        begin_generate = begin_generate,
        generate_progress = lambda: {"active": False},
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    monkeypatch.setattr(gpu_arbiter, "_owner", "video")
    with client_for(ALICE) as client:
        response = client.post(
            "/api/inference/video/generate",
            json = {"prompt": "a sloth", "width": 320, "height": 320, "num_frames": 17},
        )
    assert response.status_code == 404
    assert reserved == []


def test_a_resident_replaced_after_authorization_is_not_generated_on(monkeypatch):
    """Another account's load commits between the status read and generate (#9448)."""
    used = []
    resident = {
        "loaded": True,
        "repo_id": "org/public-model",
        "family": "z-image",
        "base_repo": None,
    }
    replacement = {
        "loaded": True,
        "repo_id": "bob/private-model",
        "family": "z-image",
        "base_repo": None,
    }
    state = {"current": resident, "reads": 0}

    def status():
        current = state["current"]
        state["reads"] += 1
        if state["reads"] == 1:
            state["current"] = replacement
        return current

    def generate(**kwargs):
        current = state["current"]
        loaded = load_identity(current["repo_id"], current["base_repo"], current["family"])
        expected = kwargs.get("expected_load")
        if expected is not None and expected != loaded:
            raise DiffusionModelReplacedError(expected, loaded)
        used.append(current["repo_id"])
        return _result(current["repo_id"])

    _install_backend(monkeypatch, status, generate)
    with client_for(ALICE) as client:
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "a sloth"}).status_code
            == 404
        )
    assert used == []


def test_a_public_replacement_is_regenerated_after_reauthorization(monkeypatch):
    used = []
    resident = {
        "loaded": True,
        "repo_id": "org/public-model",
        "family": "z-image",
        "base_repo": None,
    }
    replacement = {
        "loaded": True,
        "repo_id": "org/other-public",
        "family": "z-image",
        "base_repo": None,
    }
    state = {"current": resident, "reads": 0}

    def status():
        current = state["current"]
        state["reads"] += 1
        if state["reads"] == 1:
            state["current"] = replacement
        return current

    def generate(**kwargs):
        current = state["current"]
        loaded = load_identity(current["repo_id"], current["base_repo"], current["family"])
        expected = kwargs.get("expected_load")
        if expected is not None and expected != loaded:
            raise DiffusionModelReplacedError(expected, loaded)
        used.append(current["repo_id"])
        return _result(current["repo_id"])

    _install_backend(monkeypatch, status, generate)
    with client_for(ALICE) as client:
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "a sloth"}).status_code
            == 200
        )
    assert used == ["org/other-public"]


def test_the_gallery_persist_window_still_belongs_to_the_generating_account(monkeypatch):
    status = {
        "loaded": True,
        "repo_id": "org/public-model",
        "family": "z-image",
        "base_repo": None,
    }
    _install_backend(monkeypatch, lambda: status, lambda **kwargs: _result("org/public-model"))
    persisting = threading.Event()
    release = threading.Event()
    real_save = image_gallery.save

    def blocking_save(image, meta):
        persisting.set()
        release.wait(20)
        return real_save(image, meta)

    monkeypatch.setattr(image_gallery, "save", blocking_save)
    result = {}

    def run():
        with client_for(ALICE) as client:
            result["response"] = client.post(
                "/api/inference/images/generate", json = {"prompt": "a sloth"}
            )

    thread = threading.Thread(target = run)
    thread.start()
    try:
        assert persisting.wait(20)
        with client_for(ALICE) as client:
            progress = client.get("/api/inference/images/generate-progress").json()
            assert progress["active"] is True
        with client_for(BOB) as client:
            assert client.get("/api/inference/images/generate-progress").json() == {
                "loaded": True,
                "yours": False,
            }
    finally:
        release.set()
        thread.join(20)
    assert result["response"].status_code == 200


def test_a_failed_load_leaves_the_previous_resident_with_its_account(monkeypatch):
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_prior_resident_accounts", {})
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    run_as(ALICE, access.note_resident_account, "video", "a/model")
    run_as(BOB, access.note_resident_account, "video", "b/model")
    assert run_as(ALICE, access.resident_hidden, "video", "a/model") is False
    assert run_as(BOB, access.resident_hidden, "video", "a/model") is True
    assert run_as(BOB, access.resident_hidden, "video", "b/model") is False
    assert run_as(ALICE, access.resident_hidden, "video", "b/model") is True


def test_every_generation_access_check_names_its_modality():
    """Baked adapters are recorded per modality, so a check without one skips them."""
    import re
    from pathlib import Path

    routes = Path(__file__).resolve().parents[1] / "routes"
    pattern = re.compile(r"require_media_generation_access,\s*[^,\n]+,\s*\"(video|diffusion)\"")
    for path in routes.glob("*.py"):
        text = path.read_text(encoding = "utf-8")
        calls = text.count("require_media_generation_access,")
        assert len(pattern.findall(text)) == calls, path.name


def test_a_failed_load_does_not_authorize_the_requester_against_the_previous_resident(monkeypatch):
    """The records a load publishes are undone when that load fails with the old build resident.

    Alice loads a shared base with a private adapter baked in. Bob asks for the same base with his
    own adapter and the background load fails, so the engine keeps serving ALICE's pipeline. The
    ownership record names one reference (the model path), which is unchanged, so the prior-owner
    fallback never fires, and the component record was replaced with Bob's adapter set: without the
    rollback Bob clears his own list and generates on Alice's private build."""
    monkeypatch.setattr(access, "_prior_resident_accounts", {})
    monkeypatch.setattr(access, "_uncommitted_resident", {}, raising = False)
    monkeypatch.setattr(access, "_uncommitted_components", {}, raising = False)
    ran = []
    status = {"loaded": True, "repo_id": "org/public-model", "family": "z-image", "base_repo": None}
    _install_backend(
        monkeypatch,
        lambda: status,
        lambda **kwargs: (ran.append("ran"), _result("org/public-model"))[1],
    )
    run_as(ALICE, access.record_model_grant, "alice/private-lora")
    run_as(BOB, access.record_model_grant, "bob/private-lora")
    run_as(ALICE, access.note_resident_account, "diffusion", "org/public-model")
    run_as(
        ALICE,
        access.note_resident_components,
        "diffusion",
        "org/public-model",
        "alice/private-lora",
    )
    monkeypatch.setattr(gpu_arbiter, "_owner_account", ALICE.account_id)
    monkeypatch.setattr(gpu_arbiter, "_prior_account", None)

    # Bob's load route: the arbiter claim, then the two records published right after begin_load.
    run_as(BOB, gpu_arbiter.acquire_for, "diffusion", lambda: None)
    run_as(BOB, access.note_resident_account, "diffusion", "org/public-model")
    run_as(
        BOB,
        access.note_resident_components,
        "diffusion",
        "org/public-model",
        None,
        "bob/private-lora",
    )
    # ... and the background load fails with Alice's pipeline still resident.
    assert run_as(BOB, gpu_arbiter.restore_owner_account, "diffusion") is True
    assert run_as(BOB, access.restore_resident_metadata, "diffusion") is True

    assert access.resident_components(status, "diffusion") == [
        "org/public-model",
        "alice/private-lora",
    ]
    with client_for(BOB) as client:
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "a sloth"}).status_code
            == 404
        )
    assert ran == []
    assert run_as(ALICE, access.resident_hidden, "diffusion", "org/public-model") is False


def test_restoring_residency_records_is_a_noop_once_another_load_took_them(monkeypatch):
    """Mirrors restore_owner_account: a rollback may never displace a newer account's claim."""
    monkeypatch.setattr(access, "_prior_resident_accounts", {})
    monkeypatch.setattr(access, "_uncommitted_resident", {}, raising = False)
    monkeypatch.setattr(access, "_uncommitted_components", {}, raising = False)
    run_as(ALICE, access.note_resident_account, "diffusion", "a/model")
    run_as(ALICE, access.note_resident_components, "diffusion", "a/model", "alice/private-lora")
    run_as(BOB, access.note_resident_account, "diffusion", "b/model")
    run_as(BOB, access.note_resident_components, "diffusion", "b/model", "bob/private-lora")
    assert run_as(ALICE, access.restore_resident_metadata, "diffusion") is False
    assert access._resident_accounts["diffusion"][0] == BOB.account_id
    assert access._resident_components["diffusion"] == ("b/model", frozenset({"bob/private-lora"}))
