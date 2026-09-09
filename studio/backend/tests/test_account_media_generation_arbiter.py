# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pytest

import core.inference.gpu_arbiter as arb
from auth import policy
from hub.services.models import account_access
from utils.account_context import AccountContext, bind_account, reset_account, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture
def calls(monkeypatch):
    recorded: list[str] = []
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    monkeypatch.setattr(arb, "_prior_account", None)
    monkeypatch.setattr(account_access, "_generation_accounts", {})
    monkeypatch.setattr(account_access, "_generation_holders", {})
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: recorded.append("evict-chat"))
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: recorded.append("evict-diffusion"))
    monkeypatch.setitem(arb._EVICTORS, arb.VIDEO, lambda: recorded.append("evict-video"))
    return recorded


def test_image_generation_blocks_foreign_chat_load(calls):
    run_as(ALICE, arb.acquire_for, arb.DIFFUSION, lambda: None)
    token = bind_account(ALICE)
    try:
        with account_access.media_generation_slot("diffusion"):
            with pytest.raises(arb.GpuBusyForAnotherAccountError):
                run_as(BOB, arb.acquire_for, arb.CHAT, lambda: None)
    finally:
        reset_account(token)
    assert calls == []


def test_video_generation_blocks_foreign_chat_load(calls, monkeypatch):
    from core.inference import video as video_module

    backend = video_module.get_video_backend()
    monkeypatch.setattr(backend, "_generate_job_active", True, raising = False)
    monkeypatch.setattr(backend, "_generate_job_account", ALICE.account_id, raising = False)
    run_as(ALICE, arb.acquire_for, arb.VIDEO, lambda: None)
    with pytest.raises(arb.GpuBusyForAnotherAccountError):
        run_as(BOB, arb.acquire_for, arb.CHAT, lambda: None)
    assert calls == []


def test_own_image_generation_does_not_block_own_load(calls):
    run_as(ALICE, arb.acquire_for, arb.DIFFUSION, lambda: None)
    token = bind_account(ALICE)
    try:
        with account_access.media_generation_slot("diffusion"):
            run_as(ALICE, arb.acquire_for, arb.CHAT, lambda: None)
    finally:
        reset_account(token)
    assert calls == ["evict-diffusion"]
    assert arb.current_owner() == arb.CHAT


def test_single_user_install_still_evicts_during_a_generation(calls, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    arb.acquire_for(arb.DIFFUSION, lambda: None)
    with account_access.media_generation_slot("diffusion"):
        arb.acquire_for(arb.CHAT, lambda: None)
    assert calls == ["evict-diffusion"]
