# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A queued image request is not an active generation: the slot holder owns progress and cancel."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.inference import gpu_arbiter
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
from hub.services.models import account_access as access
from routes import inference
from state import active_generations
from utils.account_context import AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_generation_accounts", {})
    if hasattr(access, "_generation_holders"):
        monkeypatch.setattr(access, "_generation_holders", {})
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: True)
    monkeypatch.setattr(gpu_arbiter, "_owner", "diffusion")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
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
    return TestClient(app)


class SharedEngine:
    is_loaded = True

    def __init__(self):
        self.slots = DiffusionBackend()
        self.entered = threading.Semaphore(0)
        self.running = threading.Event()
        self.stop_all = threading.Event()

    def status(self):
        return {
            "loaded": True,
            "repo_id": "org/public-model",
            "family": "z-image",
            "base_repo": None,
        }

    def generate(self, **kwargs):
        cancel = threading.Event()
        self.entered.release()
        with self.slots._generation_slot(cancel):
            self.running.set()
            while not cancel.is_set() and not self.stop_all.is_set():
                cancel.wait(0.02)
        raise RuntimeError(DIFFUSION_CANCELLED_MSG)

    def generate_progress(self):
        return {"active": True, "step": 3, "total_steps": 10, "fraction": 0.3, "eta_seconds": 1.0}

    def cancel_generate(self, expected_account = None):
        return self.slots.cancel_generate()


@pytest.fixture
def engine(monkeypatch):
    from core.inference import diffusion_engine_router

    shared = SharedEngine()
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: shared)
    return shared


def _post_generate(account, engine, results, key):
    def run():
        with client_for(account) as client:
            results[key] = client.post("/api/inference/images/generate", json = {"prompt": "a sloth"})

    thread = threading.Thread(target = run, daemon = True)
    thread.start()
    assert engine.entered.acquire(timeout = 20)
    return thread


def test_a_queued_request_does_not_take_progress_and_cancel_from_the_active_generation(engine):
    results = {}
    alice = _post_generate(ALICE, engine, results, "alice")
    assert engine.running.wait(20)
    bob = _post_generate(BOB, engine, results, "bob")
    try:
        with client_for(BOB) as client:
            assert client.get("/api/inference/images/generate-progress").json() == {
                "loaded": True,
                "yours": False,
            }
            assert client.post("/api/inference/images/generate/cancel").json() == {
                "cancelled": False
            }
        with client_for(ALICE) as client:
            progress = client.get("/api/inference/images/generate-progress").json()
            assert progress["active"] is True and progress["step"] == 3
            assert client.post("/api/inference/images/generate/cancel").json() == {
                "cancelled": True
            }
    finally:
        engine.stop_all.set()
        alice.join(20)
        bob.join(20)
    assert results["alice"].status_code == 409


@pytest.mark.parametrize("engine", ["diffusers", "sd_cpp"])
def test_cancel_rechecks_the_authorized_account_under_the_lock(engine):
    """The route authorizes on the loop and cancels on an executor; a generation that finished and a successor that took the slot in between must not receive the stale cancel."""
    if engine == "diffusers":
        from core.inference.diffusion import DiffusionBackend

        backend = object.__new__(DiffusionBackend)
        backend._generation_cancel_lock = threading.Lock()
        backend._generation_owns_slot = True
    else:
        from core.inference.sd_cpp_backend import SdCppDiffusionBackend
        backend = object.__new__(SdCppDiffusionBackend)
        backend._lock = threading.RLock()
    event = threading.Event()
    backend._active_generate_cancel = event
    backend._active_generate_account = BOB.account_id
    assert backend.cancel_generate(expected_account = ALICE.account_id) is False
    assert not event.is_set()
    assert backend.cancel_generate(expected_account = BOB.account_id) is True
    assert event.is_set()
