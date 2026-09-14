# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU reproduction: an authorized image unload cancels a later foreign generation."""

import sys
import secrets
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio/backend"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import get_current_subject
from core.inference import diffusion_engine_router, gpu_arbiter
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
from hub.services.models import account_access as access
from routes import inference, video
from state import active_generations
from utils.account_context import bind_account, reset_account


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(video.router, prefix = "/api/inference")
    return TestClient(app)


@pytest.mark.parametrize("unloader", ["alice", "unsloth"])
def test_unload_does_not_cancel_foreign_generation_started_after_route_check(
    monkeypatch, accounts, unloader
):
    """Real HTTP handlers and DiffusionBackend slot/teardown; no model or GPU required."""
    backend = DiffusionBackend()
    arrived, proceed, generating, finish = (threading.Event() for _ in range(4))
    cancel = threading.Event()
    responses = {}
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: True)
    for name in ("_resident_accounts", "_generation_accounts", "_generation_holders"):
        monkeypatch.setattr(access, name, {})
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.DIFFUSION)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", accounts[unloader].account_id)
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    # Replace only the model-dependent surface; reservation, cancellation and teardown are real.
    monkeypatch.setattr(
        backend,
        "status",
        lambda: {
            "loaded": True,
            "repo_id": "org/public-model",
            "family": "z-image",
            "base_repo": None,
        },
    )

    def generate(**kwargs):
        with backend._generation_slot(cancel):
            generating.set()
            while not finish.is_set() and not cancel.wait(0.01):
                pass
        raise RuntimeError(DIFFUSION_CANCELLED_MSG)

    real_unload = backend.unload

    def paused_unload(*args, **kwargs):
        arrived.set()
        assert proceed.wait(10), "unload barrier timed out"
        return real_unload(*args, **kwargs)

    monkeypatch.setattr(backend, "generate", generate)
    monkeypatch.setattr(backend, "unload", paused_unload)

    def post(
        account,
        key,
        path,
        body = None,
    ):
        with client_for(account) as client:
            responses[key] = client.post("/api/inference/" + path, json = body)

    unload_thread = threading.Thread(
        target = post, args = (accounts[unloader], "unload", "images/unload")
    )
    generate_thread = threading.Thread(
        target = post,
        args = (accounts["bob"], "generate", "images/generate", {"prompt": "Bob's private prompt"}),
    )
    active_generations.reset_for_tests()
    try:
        assert policy.installation_has_managed_accounts()
        unload_thread.start()
        assert arrived.wait(10), "unload did not pass route ownership checks"
        generate_thread.start()
        assert generating.wait(10), "Bob never took the real generation slot"
        proceed.set()
        unload_thread.join(10)
        assert not unload_thread.is_alive()
        response = responses["unload"]
        assert not cancel.is_set(), f"{unloader}'s unload cancelled Bob's generation: HTTP {response.status_code} {response.text}"
        assert response.status_code == 409
        assert response.json()["error"] == "gpu_busy"
        assert int(response.headers["retry-after"]) > 0
    finally:
        proceed.set()
        finish.set()
        for thread in (unload_thread, generate_thread):
            if thread.ident is not None:
                thread.join(10)
        active_generations.reset_for_tests()


def test_video_unload_does_not_cancel_later_foreign_reservation(monkeypatch, accounts):
    from core.inference import video as video_module

    backend = video_module.VideoBackend()
    arrived, proceed, generating, finish = (threading.Event() for _ in range(4))
    render_done = threading.Event()
    results = {}
    family = SimpleNamespace(
        name = "wan", default_fps = 24, default_num_frames = 49, frame_step = 4, frame_offset = 1
    )
    backend._state = SimpleNamespace(
        family = family, h3_task = None, engine = "diffusers", repo_id = "org/public-video"
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    monkeypatch.setattr(video_module, "validate_video_request_shape", lambda *a, **k: None)
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: True)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", accounts["alice"].account_id)
    for name in ("_resident_accounts", "_generation_accounts", "_generation_holders"):
        monkeypatch.setattr(access, name, {})
    monkeypatch.setattr(backend, "status", lambda: {"loaded": True, "repo_id": "org/public-video"})
    monkeypatch.setattr(backend, "generation_snapshot", lambda: (backend.status(), backend._state))
    monkeypatch.setattr(
        backend, "_resolve_keyframes", lambda *a, **k: (None, None, 512, 512, "t2v")
    )
    monkeypatch.setattr(backend, "_resolve_references", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_resolve_flow_shifts", lambda *a, **k: (None, None))
    monkeypatch.setattr(backend, "_teardown_state_locked", lambda: None)

    def render(**kwargs):
        try:
            with backend._generate_lock:
                generating.set()
                while not finish.is_set() and not backend._active_generate_cancel.wait(0.01):
                    pass
        finally:
            render_done.set()

    monkeypatch.setattr(backend, "_run_generate", render)
    real_unload = backend.unload

    def paused_unload(*args, **kwargs):
        arrived.set()
        assert proceed.wait(10)
        return real_unload(*args, **kwargs)

    monkeypatch.setattr(backend, "unload", paused_unload)

    def unload():
        with client_for(accounts["alice"]) as client:
            results["unload"] = client.post("/api/inference/video/unload")

    thread = threading.Thread(target = unload)
    thread.start()
    try:
        assert arrived.wait(10)
        with client_for(accounts["bob"]) as client:
            response = client.post(
                "/api/inference/video/generate", json = {"prompt": "private", "steps": 5}
            )
        assert response.status_code == 200, response.text
        assert generating.wait(10)
        assert backend.generate_job_account() == accounts["bob"].account_id
        cancel = backend._active_generate_cancel
        proceed.set()
        thread.join(10)
        assert not thread.is_alive()
        assert (
            not cancel.is_set()
        ), f"Alice's unload cancelled Bob's video: HTTP {results['unload'].status_code}"
        assert results["unload"].status_code == 409
        assert results["unload"].json()["error"] == "gpu_busy"
    finally:
        finish.set()
        proceed.set()
        thread.join(10)
        if generating.is_set():
            assert render_done.wait(10)


@pytest.mark.parametrize("kind", ["diffusers", "sd_cpp", "video"])
@pytest.mark.parametrize("managed", [False, True], ids = ["single-user", "managed-install"])
def test_own_unload_still_cancels_and_returns_200(monkeypatch, isolated_auth, kind, managed):
    """The new foreign guard must preserve intentional cancellation of one's own work."""
    from core.inference import video as video_module
    from core.inference.sd_cpp_backend import SdCppDiffusionBackend

    isolated_auth.create_initial_user("unsloth", "password", secrets.token_urlsafe(32))
    if managed:
        for name in ("alice", "bob"):
            isolated_auth.create_initial_user(name, "password", secrets.token_urlsafe(32))
    owner = isolated_auth.get_account("unsloth")
    assert policy.installation_has_managed_accounts() is managed
    backend = {
        "diffusers": DiffusionBackend,
        "sd_cpp": SdCppDiffusionBackend,
        "video": video_module.VideoBackend,
    }[kind]()
    cancel = threading.Event()
    backend._active_generate_cancel = cancel
    if kind == "video":
        backend._generate_job_account = owner.account_id
        monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
        path = "video/unload"
    else:
        backend._active_generate_account = owner.account_id
        monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
        path = "images/unload"
    monkeypatch.setattr(gpu_arbiter, "_owner", "video" if kind == "video" else "diffusion")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", owner.account_id)
    for name in ("_resident_accounts", "_generation_accounts", "_generation_holders"):
        monkeypatch.setattr(access, name, {})
    with client_for(owner) as client:
        response = client.post("/api/inference/" + path)
    assert response.status_code == 200, response.text
    assert response.json()["loaded"] is False
    assert cancel.is_set()
