# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deactivation does not cancel media generations, so the owner must stay scoped.

Deactivating the last managed account drops the ACTIVE count back to one, but its
in-flight image generation keeps running: ownership tracking has to follow
``account_scope()`` (any managed account exists), not the active-count login mode.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from routes import inference
from state import active_generations
from utils.account_context import OWNER, AccountContext, bind_account, reset_account, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def deactivatable(monkeypatch, tmp_path):
    """A one-managed-account install whose account can be deactivated mid-generation."""
    multi = {"value": True}
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: multi["value"])
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_generation_accounts", {})
    monkeypatch.setattr(access, "_generation_holders", {})
    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: True)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", None)
    active_generations.reset_for_tests()
    yield multi
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


@pytest.fixture
def alice_resident(monkeypatch):
    running = threading.Event()
    cancelled = threading.Event()

    def generate(**kwargs):
        running.set()
        cancelled.wait(20)
        from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

        raise RuntimeError(DIFFUSION_CANCELLED_MSG)

    backend = SimpleNamespace(
        is_loaded = True,
        status = lambda: {
            "loaded": True,
            "repo_id": "org/public-model",
            "family": "z-image",
            "base_repo": None,
        },
        generate = generate,
        generate_progress = lambda: {"active": True, "step": 3, "total": 10},
        cancel_generate = lambda **kwargs: (cancelled.set(), True)[1],
    )
    from core.inference import diffusion_engine_router

    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: backend)
    monkeypatch.setattr(gpu_arbiter, "_owner", "diffusion")
    monkeypatch.setattr(gpu_arbiter, "_owner_account", ALICE.account_id)
    return SimpleNamespace(running = running, cancelled = cancelled)


def _start_alice_generation(alice_resident):
    result = {}

    def run():
        with client_for(ALICE) as client:
            result["response"] = client.post(
                "/api/inference/images/generate", json = {"prompt": "a sloth"}
            )

    thread = threading.Thread(target = run)
    thread.start()
    assert alice_resident.running.wait(20)
    return thread, result


def test_owner_cannot_cancel_a_deactivated_accounts_image_generation(deactivatable, alice_resident):
    thread, result = _start_alice_generation(alice_resident)
    try:
        # The owner deactivates alice mid-generation: the active count drops to one, but
        # set_account_active() only cancels chat generations, so this image job runs on.
        deactivatable["value"] = False
        assert run_as(OWNER, access.generation_is_foreign, "diffusion") is True
        assert run_as(OWNER, access.tracked_generation_account) == OWNER.account_id
        with client_for(OWNER) as client:
            assert client.post("/api/inference/images/generate/cancel").json() == {
                "cancelled": False
            }
        assert not alice_resident.cancelled.is_set()
    finally:
        alice_resident.cancelled.set()
        thread.join(20)
    assert result["response"].status_code == 409


def test_a_deactivated_account_still_owns_its_own_generation(deactivatable):
    token = bind_account(ALICE)
    try:
        with access.media_generation_slot("diffusion"):
            deactivatable["value"] = False
            assert access.generation_is_mine("diffusion") is True
            assert access.generation_is_foreign("diffusion") is False
            assert access.tracked_generation_account() == ALICE.account_id
    finally:
        reset_account(token)
