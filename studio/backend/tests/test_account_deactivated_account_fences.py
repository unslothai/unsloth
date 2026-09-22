# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deactivating the last managed account must not open the foreign-work fences.

set_account_active() only signals a generation; it does not wait for it to unwind, and a
wedged producer can hold its GPU reservation indefinitely. Dropping the ACTIVE count to one
makes installation_is_multi_user() False while the deactivated account's chat generation and
its supervisor task are both still live, so every fence keyed on the login mode opens:
the owner tears the shared backend down underneath that generation, and an owner run reusing
the same client-chosen run id is admitted into a slot the foreign supervisor task still owns.
Both must key on account_scope() / installation_has_managed_accounts() instead.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import allow_ambient_hf_token, get_current_subject
from core.inference import gpu_arbiter
from hub.services.models import account_access as access
from routes import chat_generation_runs as run_routes
from routes import inference
from state import active_generations
from utils.account_context import (
    OWNER,
    AccountContext,
    arun_as,
    bind_account,
    reset_account,
    run_as,
)

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def deactivatable(monkeypatch, tmp_path):
    """A one-managed-account install whose only managed account can be deactivated."""
    multi = {"value": True}
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: multi["value"])
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_generation_accounts", {})
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
    app.include_router(inference.router, prefix = "/api/inference")
    app.include_router(inference.studio_router, prefix = "/api/inference")
    return TestClient(app)


def test_owner_unload_refuses_a_deactivated_accounts_live_generation(deactivatable):
    """Item 3971721488: /unload's resident-control fence and the active-generation
    refusal both go blind once the sole managed account is deactivated."""
    event = threading.Event()
    with run_as(ALICE, active_generations.ActiveGeneration, event, model = "org/alice-private"):
        deactivatable["value"] = False
        assert policy.installation_is_multi_user() is False
        assert policy.installation_has_managed_accounts() is True
        with client_for(OWNER) as client:
            response = client.post(
                "/api/inference/unload",
                json = {"model_path": "org/public", "force_cancel_active": True},
            )
        assert response.status_code == 409, response.text
        assert response.json()["error"] == "gpu_busy"
        assert not event.is_set()


def test_swap_gate_still_sees_a_deactivated_accounts_generation(deactivatable):
    """The refusal-only half of the same gate, called directly: account_scope() counts
    zero owner generations and returns before require_no_foreign_generations()."""
    event = threading.Event()
    with run_as(ALICE, active_generations.ActiveGeneration, event, model = "org/alice-private"):
        deactivatable["value"] = False
        with pytest.raises(HTTPException) as excinfo:
            run_as(
                OWNER,
                lambda: inference._raise_or_cancel_active_generations(
                    force = True, action = "Unloading", cancel = True
                ),
            )
        assert excinfo.value.status_code == 409
        assert excinfo.value.detail["error"] == "gpu_busy"
        assert not event.is_set()


def test_owner_idle_fence_stays_on_for_a_deactivated_accounts_work(deactivatable):
    """require_idle_other_accounts / foreign_work_active are ownership fences, not login mode."""
    event = threading.Event()
    with run_as(ALICE, active_generations.ActiveGeneration, event, model = "org/alice-private"):
        deactivatable["value"] = False
        assert run_as(OWNER, access.foreign_work_active) is True
        with pytest.raises(HTTPException) as excinfo:
            run_as(OWNER, access.require_idle_other_accounts)
        assert excinfo.value.status_code == 409


@pytest.mark.asyncio
async def test_owner_run_cannot_take_a_deactivated_accounts_supervisor_slot(
    deactivatable, monkeypatch
):
    """Item 3971721497: the supervisor keys tasks by bare run id, so an owner run reusing
    a live foreign id commits a queued row that supervisor.start() then silently drops."""
    event = threading.Event()
    with run_as(
        ALICE,
        active_generations.ActiveGeneration,
        event,
        model = "local",
        run_id = "run-1",
        thread_id = "thread-1",
    ):
        deactivatable["value"] = False
        started: list = []
        # A live task under the bare id: start() returns without scheduling anything.
        supervisor = SimpleNamespace(
            _tasks = {"run-1": object()},
            start = lambda run_id, **identity: started.append(run_id),
        )
        request = SimpleNamespace(
            app = SimpleNamespace(state = SimpleNamespace(chat_generation_supervisor = supervisor))
        )
        committed: list = []

        def create_run(**kwargs):
            committed.append(kwargs["run_id"])
            return (
                {
                    "id": "run-1",
                    "status": "queued",
                    "threadId": "thread-1",
                    "requestPayload": {"model": "local"},
                },
                True,
            )

        monkeypatch.setattr(run_routes.db, "create_run", create_run)
        payload = run_routes.CreateChatGenerationRun(
            runId = "run-1",
            threadId = "thread-1",
            userMessageId = "user-1",
            assistantMessageId = "assistant-1",
            requestPayload = {
                "model": "local",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        with pytest.raises(HTTPException) as excinfo:
            await arun_as(
                OWNER,
                run_routes.create_chat_generation_run(payload, request, OWNER.username),
            )
        assert excinfo.value.status_code == 404
        assert committed == []


def test_supervisor_run_id_fence_stays_on_after_deactivation(deactivatable):
    """The gate on its own, so the failure is readable without the route harness."""
    event = threading.Event()
    with run_as(
        ALICE,
        active_generations.ActiveGeneration,
        event,
        model = "local",
        run_id = "run-1",
        thread_id = "thread-1",
    ):
        deactivatable["value"] = False
        with pytest.raises(HTTPException) as excinfo:
            run_as(OWNER, run_routes._require_available_supervisor_run_id, "run-1")
        assert excinfo.value.status_code == 404
