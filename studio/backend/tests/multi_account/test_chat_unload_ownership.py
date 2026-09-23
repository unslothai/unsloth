# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat teardown must drop the CHAT claim, or every other account sees a phantom resident."""

import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studio/backend"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import gpu_arbiter, llama_keepwarm
from hub.services.models import account_access as access
from routes import inference
from utils.account_context import bind_account, reset_account

RESIDENT = "alice/private-gguf"


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(inference.router, prefix="/api/inference")
    return TestClient(app)


class FakeLlama:
    """Only the surface /unload and /status read; teardown flips residency like the real one."""

    def __init__(self):
        self.is_active = True
        self.is_loaded = True
        self.model_identifier = RESIDENT
        self.hf_variant = None
        self.chat_template_override = None
        self.unloaded = False

    def unload_model(self):
        self.is_active = False
        self.is_loaded = False
        self.model_identifier = None
        self.unloaded = True
        return True


@pytest.fixture
def chat_resident(monkeypatch, accounts):
    llama = FakeLlama()
    standard = SimpleNamespace(
        active_model_name=None,
        models={},
        get_loading_model=lambda: None,
        loading_models=(),
    )
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_prior_resident_accounts", {})
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.CHAT)
    monkeypatch.setattr(gpu_arbiter, "_owner_account", accounts["alice"].account_id)
    monkeypatch.setattr(gpu_arbiter, "_prior_account", None)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference, "get_inference_backend", lambda: standard)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(inference, "_probe_llama_cpp_status", lambda _: (False, {}))
    monkeypatch.setattr(inference, "_running_load_attempt", None)
    monkeypatch.setattr(inference, "_pending_load_attempts", {})
    return llama


def bob_status(accounts):
    with client_for(accounts["bob"]) as client:
        return client.get("/api/inference/status")


def test_manual_unload_releases_chat_ownership(chat_resident, accounts):
    with client_for(accounts["alice"]) as client:
        unload = client.post("/api/inference/unload", json={"model_path": RESIDENT})
    assert unload.status_code == 200, unload.text
    assert chat_resident.unloaded, "the fake backend was never torn down"

    response = bob_status(accounts)
    assert response.status_code == 200, response.text
    body = response.json()
    # "yours" is the load-bearing assertion, not "loaded". This route's mask is
    # hidden_chat_status_response(), {"loaded": [], "loading": [], "yours": False}, so an
    # empty "loaded" is what BOTH answers report here and only the "yours" key tells a real
    # empty GPU from a hidden resident. test_only_the_yours_key_separates_the_two_answers
    # pins that, so neither assertion can be dropped as redundant.
    assert "yours" not in body, f"phantom resident after unload: {body}"
    assert body.get("loaded") == [], f"phantom resident after unload: {body}"
    assert body.get("active_model") is None, body
    assert gpu_arbiter.current_owner() is None
    assert gpu_arbiter.owner_account() is None


def arm_the_idle_loop(monkeypatch):
    """Everything the idle loop reads, set so one tick unloads."""
    monkeypatch.setattr(llama_keepwarm, "_is_idle", lambda ttl: True)
    monkeypatch.setattr(llama_keepwarm, "_note_activity", lambda: None)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_auto_unload_idle_seconds", lambda: 1.0
    )
    monkeypatch.setattr("utils.openai_auto_switch_settings.get_auto_unload_api_only", lambda: False)
    monkeypatch.setattr("utils.openai_auto_switch_settings.get_auto_unload_keep_kv", lambda: False)


def test_only_the_yours_key_separates_the_two_answers(chat_resident, accounts):
    """Why both assertions above are needed, and why "loaded" alone would not do.

    With alice's claim still held, bob is supposed to see the mask, and the mask for this
    route reports an EMPTY "loaded" - the same value a genuinely free GPU reports. So a test
    written against "loaded" alone passes just as happily on a phantom resident. Pinning the
    masked body here is what lets the two teardown tests above assert on "yours".
    """
    assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
    assert bob_status(accounts).json() == {"loaded": [], "loading": [], "yours": False}


def test_a_torn_down_backend_does_not_yet_mean_a_released_claim(
    chat_resident, accounts, monkeypatch
):
    """Between the unload and the release, the claim is still alice's and bob sees the mask.

    That interval is the reason the test below waits on the release rather than on the
    teardown: a status read landing inside it gets the masked body, correctly, and a test that
    called that a phantom would be blaming the product for its own timing.

    The window is held open with a hook on the release itself, not by winning a scheduler
    race. Racing would make this depend on undocumented asyncio wakeup order, and worse, it
    would fail against an implementation that made teardown and release atomic - punishing a
    genuine improvement. A hook on release still fires in that implementation, so this test
    keeps asking its question rather than becoming a tripwire.
    """
    arm_the_idle_loop(monkeypatch)

    reached_release = threading.Event()
    may_release = threading.Event()
    real_release = inference.release_chat_gpu_claim

    def held_open():
        reached_release.set()
        assert may_release.wait(timeout=5.0), "the test never let the release proceed"
        return real_release()

    monkeypatch.setattr(inference, "release_chat_gpu_claim", held_open)

    async def tick():
        task = asyncio.ensure_future(llama_keepwarm.idle_unload_loop(poll_seconds=0.01))
        for _ in range(500):
            await asyncio.sleep(0.01)
            if reached_release.is_set():
                break
        assert reached_release.is_set(), "the loop never reached the claim release"

        # Inside the window now, and it stays open until may_release is set.
        assert chat_resident.unloaded, "the backend should already be down here"
        assert gpu_arbiter.current_owner() == gpu_arbiter.CHAT
        assert bob_status(accounts).json() == {"loaded": [], "loading": [], "yours": False}

        may_release.set()
        for _ in range(500):
            await asyncio.sleep(0.01)
            if gpu_arbiter.current_owner() is None:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(tick())

    # And once it closes, the claim is gone and bob gets the real answer.
    assert gpu_arbiter.current_owner() is None
    assert "yours" not in bob_status(accounts).json()


def test_idle_auto_unload_releases_chat_ownership(chat_resident, accounts, monkeypatch):
    arm_the_idle_loop(monkeypatch)

    async def one_tick():
        task = asyncio.ensure_future(llama_keepwarm.idle_unload_loop(poll_seconds=0.01))
        for _ in range(500):
            await asyncio.sleep(0.01)
            # Not chat_resident.unloaded on its own. The loop tears the backend down several
            # awaits before it drops the claim: unload_model, then the reload stash, then
            # clear_resident and release_chat_gpu_claim. Breaking on the teardown lands the
            # status query inside that window, where the claim is still held and bob is
            # correctly handed the masked answer, and the test then reports the mask as a
            # phantom. That is what failed on main in Backend CI (Python 3.13, rest) with the
            # body {"loaded": [], "loading": [], "yours": False}, which is exactly
            # account_access.hidden_chat_status_response(). The release is what this test is
            # about, so the release is what the wait watches.
            if chat_resident.unloaded and gpu_arbiter.current_owner() is None:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(one_tick())
    assert chat_resident.unloaded, "idle loop never tore the backend down"

    response = bob_status(accounts)
    assert response.status_code == 200, response.text
    body = response.json()
    # "yours" is the load-bearing assertion, not "loaded". This route's mask is
    # hidden_chat_status_response(), {"loaded": [], "loading": [], "yours": False}, so an
    # empty "loaded" is what BOTH answers report here and only the "yours" key tells a real
    # empty GPU from a hidden resident. test_only_the_yours_key_separates_the_two_answers
    # pins that, so neither assertion can be dropped as redundant.
    assert "yours" not in body, f"phantom resident after idle unload: {body}"
    assert body.get("loaded") == [], f"phantom resident after idle unload: {body}"
    assert body.get("active_model") is None, body
    assert gpu_arbiter.current_owner() is None
    assert gpu_arbiter.owner_account() is None
