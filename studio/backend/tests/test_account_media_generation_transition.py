# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The single-user to multi-user transition must not strand an in-flight media generation."""

from __future__ import annotations

import pytest

import core.inference.gpu_arbiter as arb
from auth import policy
from hub.services.models import account_access
from utils.account_context import OWNER, AccountContext, bind_account, reset_account, run_as

BOB = AccountContext("b" * 32, "bob")


@pytest.fixture
def calls(monkeypatch):
    recorded: list[str] = []
    multi = {"value": False}
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: multi["value"])
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_owner_account", None)
    monkeypatch.setattr(arb, "_prior_account", None)
    monkeypatch.setattr(account_access, "_generation_accounts", {})
    monkeypatch.setattr(account_access, "_generation_holders", {})
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: recorded.append("evict-chat"))
    monkeypatch.setitem(arb._EVICTORS, arb.DIFFUSION, lambda: recorded.append("evict-diffusion"))
    monkeypatch.setitem(arb._EVICTORS, arb.VIDEO, lambda: recorded.append("evict-video"))
    return recorded, multi


def test_first_managed_account_cannot_evict_an_owner_generation_started_single_user(calls):
    recorded, multi = calls
    arb.acquire_for(arb.DIFFUSION, lambda: None)
    token = bind_account(OWNER)
    try:
        with account_access.media_generation("diffusion"):
            # The owner creates the first managed account mid-generation.
            multi["value"] = True
            with pytest.raises(arb.GpuBusyForAnotherAccountError):
                run_as(BOB, arb.acquire_for, arb.CHAT, lambda: None)
    finally:
        reset_account(token)
    assert recorded == []
    assert arb.current_owner() == arb.DIFFUSION
    assert account_access._generation_accounts == {}


def test_slot_holder_survives_the_same_transition(calls):
    recorded, multi = calls
    arb.acquire_for(arb.DIFFUSION, lambda: None)
    token = bind_account(OWNER)
    try:
        with account_access.media_generation_slot("diffusion"):
            multi["value"] = True
            with pytest.raises(arb.GpuBusyForAnotherAccountError):
                run_as(BOB, arb.acquire_for, arb.CHAT, lambda: None)
    finally:
        reset_account(token)
    assert recorded == []
    assert account_access._generation_holders == {}


def test_single_user_owner_still_evicts_its_own_generation(calls):
    recorded, _ = calls
    arb.acquire_for(arb.DIFFUSION, lambda: None)
    with account_access.media_generation("diffusion"):
        arb.acquire_for(arb.CHAT, lambda: None)
    assert recorded == ["evict-diffusion"]
