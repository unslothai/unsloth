# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linked-folder scheduling must rotate between accounts instead of restarting at the first one."""

from __future__ import annotations

import pytest

from utils.account_context import AccountContext, current_account

ALICE = AccountContext("alice-job-account", "alice")
BOB = AccountContext("bob-job-account", "bob")


@pytest.fixture
def folder_sync(monkeypatch):
    from core.rag import folder_sync as module

    monkeypatch.setattr(module, "job_accounts", lambda: [ALICE, BOB])
    monkeypatch.setattr(module, "_last_job_account", None, raising = False)
    return lambda next_job: monkeypatch.setattr(module, "_next_job", next_job) or module


def test_folder_sync_claim_rotates_across_accounts(folder_sync):
    """A busy account must not starve the next one; the worker reconciles one folder at a time."""
    queued = {
        ALICE.account_id: ["alice-1", "alice-2", "alice-3"],
        BOB.account_id: ["bob-1"],
    }

    def next_job():
        pending = queued[current_account().account_id]
        job = pending.pop(0) if pending else None
        return (job, f"folder-{job}") if job else None

    module = folder_sync(next_job)
    claimed = [module._next_account_job() for _ in range(3)]
    assert [account for account, _, _ in claimed] == [ALICE, BOB, ALICE]
    assert [job for _, job, _ in claimed] == ["alice-1", "bob-1", "alice-2"]


def test_a_continuous_backlog_does_not_starve_the_next_accounts_pending_sync(folder_sync):
    """Alice keeps queueing successor syncs; Bob's pending job must still be reached."""
    counter = {"n": 0}
    bob_pending = ["bob-1"]

    def next_job():
        if current_account() == ALICE:
            counter["n"] += 1
            return (f"alice-{counter['n']}", "folder-alice")
        return (bob_pending.pop(0), "folder-bob") if bob_pending else None

    module = folder_sync(next_job)
    selected = [module._next_account_job()[0] for _ in range(20)]
    assert BOB in selected, "an account with a continuous backlog starved every other account"
