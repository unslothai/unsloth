# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU reproduction: an authorized image unload cancels a later foreign generation."""

import sys
import secrets
import threading
import time
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
from utils.account_context import bind_account, current_account_id, reset_account

from ..test_diffusion_backend import _FakePipeline
from ..test_diffusion_backend import fake_runtime as _fake_runtime

# Re-exported under its own name because pytest resolves a fixture by NAME, not by reference: the
# tests below take `fake_runtime` as a parameter and never load the import, which reads to
# scripts/verify_import_hoist.py as a hoisted import nothing uses.
fake_runtime = _fake_runtime


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


@pytest.mark.parametrize("unloader", ["alice", "bob", "unsloth"])
@pytest.mark.parametrize(
    "load_method,phase",
    [
        ("begin_load", "preflight"),
        ("begin_load", "prefetch"),
        ("begin_load", "construction"),
        ("load_pipeline", "preflight"),
        ("load_pipeline", "construction"),
    ],
)
def test_cpu_load_cancellation_requires_its_owner(
    monkeypatch, accounts, fake_runtime, tmp_path, unloader, load_method, phase
):
    from core.inference import diffusion as diff_mod

    backend = DiffusionBackend()
    entered, release, response_ready = (threading.Event() for _ in range(3))
    threads, errors, responses = [], [], []
    (tmp_path / "model_index.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    for name in ("_resident_accounts", "_prior_resident_accounts", "_resident_sharers"):
        monkeypatch.setattr(access, name, {})
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(backend, "_estimate_download_bytes", lambda *a, **k: (0, []))
    monkeypatch.setattr(backend, "_prefetch_files", lambda *a, **k: str(tmp_path))
    real_thread = diff_mod.account_thread

    def tracked_thread(**kwargs):
        thread = real_thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(diff_mod, "account_thread", tracked_thread)
    target, name = {
        "preflight": (backend, "validate_load_request"),
        "prefetch": (backend, "_prefetch_files"),
        "construction": (_FakePipeline, "from_pretrained"),
    }[phase]
    original = getattr(target, name)

    def parked(*args, **kwargs):
        entered.set()
        assert release.wait(10), "load barrier timed out"
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, parked)

    def load():
        try:
            getattr(backend, load_method)(
                str(tmp_path), family_override = "z-image", local_files_only = True
            )
        except RuntimeError as exc:
            errors.append(str(exc))

    def unload():
        with client_for(accounts[unloader]) as client:
            responses.append(client.post("/api/inference/images/unload"))
        response_ready.set()

    account_token = bind_account(accounts["alice"])
    try:
        access.note_resident_account("diffusion", str(tmp_path))
        loader = tracked_thread(target = load, daemon = True)
    finally:
        reset_account(account_token)
    ejector = threading.Thread(target = unload, daemon = True)
    loader.start()
    try:
        assert entered.wait(10), errors
        assert backend._state is None
        assert gpu_arbiter.current_owner() is None
        token, cancel = backend._load_token, backend._cancel_event
        ejector.start()
        deadline = time.monotonic() + 5
        while not response_ready.is_set() and not cancel.is_set() and time.monotonic() < deadline:
            response_ready.wait(0.01)
        if unloader == "alice":
            assert cancel.is_set(), "the owner could not cancel its load"
        else:
            assert response_ready.is_set(), "a foreign eject waited on construction"
            assert responses[0].status_code == 409, responses[0].text
            assert responses[0].json()["error"] == "gpu_busy"
            assert not cancel.is_set()
            assert backend._load_token == token
    finally:
        release.set()
        loader.join(10)
        for thread in [*threads, ejector]:
            if thread.ident is not None:
                thread.join(10)
                assert not thread.is_alive()
    if unloader == "alice":
        assert responses[0].status_code == 200, responses[0].text
        assert backend._state is None
        assert all("cancelled" in error for error in errors)
    else:
        assert not errors
        assert backend.is_loaded
    backend.unload()


@pytest.mark.parametrize("phase", ["prefetch", "construction", "resident"])
@pytest.mark.parametrize("unloader", ["alice", "bob", "unsloth"])
def test_pending_caller_cannot_block_load_owner_eject(
    monkeypatch, accounts, fake_runtime, tmp_path, phase, unloader
):
    from core.inference import diffusion as diff_mod

    backend = DiffusionBackend()
    entered, pending, release, response_ready = (threading.Event() for _ in range(4))
    threads, errors, responses = [], [], {}
    (tmp_path / "model_index.json").write_text("{}", encoding = "utf-8")
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    for name in ("_resident_accounts", "_prior_resident_accounts", "_resident_sharers"):
        monkeypatch.setattr(access, name, {})
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(backend, "_estimate_download_bytes", lambda *a, **k: (0, []))
    monkeypatch.setattr(backend, "_prefetch_files", lambda *a, **k: str(tmp_path))
    real_thread = diff_mod.account_thread

    def tracked_thread(**kwargs):
        thread = real_thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr(diff_mod, "account_thread", tracked_thread)

    def start(account, target):
        token = bind_account(accounts[account])
        try:
            tracked_thread(target = target, daemon = True).start()
        finally:
            reset_account(token)

    target, name = (
        (backend, "_prefetch_files") if phase == "prefetch" else (_FakePipeline, "from_pretrained")
    )
    original = getattr(target, name)

    def parked(*args, **kwargs):
        if phase != "resident":
            entered.set()
            assert release.wait(10), "load barrier timed out"
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, parked)
    validate = backend.validate_load_request

    def pending_validation(*args, **kwargs):
        if current_account_id() == accounts["bob"].account_id:
            pending.set()
            assert release.wait(10), "validation barrier timed out"
        return validate(*args, **kwargs)

    monkeypatch.setattr(backend, "validate_load_request", pending_validation)

    def load():
        try:
            backend.begin_load(str(tmp_path), family_override = "z-image", local_files_only = True)
        except RuntimeError as exc:
            errors.append(str(exc))

    def unload(account):
        with client_for(accounts[account]) as client:
            responses[account] = client.post("/api/inference/images/unload")
        response_ready.set()

    try:
        if phase == "resident":
            account_token = bind_account(accounts["alice"])
            try:
                backend.load_pipeline(
                    str(tmp_path), family_override = "z-image", local_files_only = True
                )
                access.note_resident_account("diffusion", str(tmp_path))
            finally:
                reset_account(account_token)
            assert backend.is_loaded
            assert backend._loading is None
        else:
            start("alice", load)
            assert entered.wait(10), errors
            assert backend._loading.account_id == accounts["alice"].account_id
            assert backend._state is None
        assert gpu_arbiter.current_owner() is None
        start("bob", load)
        assert pending.wait(10), errors
        token, cancel = backend._load_token, backend._cancel_event
        start(unloader, lambda: unload(unloader))
        can_eject = unloader == "alice" or (phase == "resident" and accounts[unloader].is_owner)
        if not can_eject:
            assert response_ready.wait(5), "foreign eject waited on construction"
            if phase == "resident":
                assert responses[unloader].status_code == 404, responses[unloader].text
            else:
                assert responses[unloader].status_code == 409, responses[unloader].text
                assert responses[unloader].json()["error"] == "gpu_busy"
            assert not cancel.is_set()
            assert backend._load_token == token
            start("alice", lambda: unload("alice"))
        assert cancel.wait(5), "a pending caller blocked the load owner's eject"
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
            assert not thread.is_alive()
    response = responses[unloader if can_eject else "alice"]
    assert response.status_code == 200, response.text
    assert backend._state is None
    assert backend._loading is None
    assert not backend._load_accounts
    assert errors and all("cancelled" in error for error in errors)


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
