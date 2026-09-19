# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.inference.api_monitor import ApiMonitor
from utils.account_context import AccountContext, run_as


def _start(monitor, prompt):
    return monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "test-model",
        prompt = prompt,
        subject = "alice",
    )


@pytest.mark.parametrize("length", [0, 359, 360, 361, 11999, 12000, 12001, 1000000])
@pytest.mark.parametrize("character", ["x", "🦥"])
def test_full_prompt_survives_detail_reads(length, character):
    prompt = (character * (length - 6) + "\nEND\t ") if length else ""
    monitor = ApiMonitor()
    entry_id = _start(monitor, prompt)

    for completed in (False, True):
        if completed:
            monitor.finish(entry_id)
        detail = monitor.get(entry_id, subject = "alice")
        assert detail["prompt"] == prompt
        [summary] = monitor.snapshot(include_details = False, subject = "alice")
        assert "prompt" not in summary
        assert len(summary["prompt_preview"]) <= 360
        assert summary["prompt_truncated"] is (len(prompt) > 360)


def test_full_prompt_remains_account_and_subject_scoped():
    monitor = ApiMonitor()
    alice = AccountContext("alice-id", "alice")
    replacement = AccountContext("replacement-id", "alice")
    prompt = "private prompt\n" * 2000
    entry_id = run_as(alice, _start, monitor, prompt)

    assert run_as(alice, monitor.get, entry_id, subject = "alice")["prompt"] == prompt
    assert run_as(alice, monitor.get, entry_id, subject = "bob") is None
    assert run_as(replacement, monitor.get, entry_id, subject = "alice") is None
    assert run_as(replacement, monitor.snapshot, subject = "alice") == []


def test_full_prompt_follows_history_retention_and_clear():
    monitor = ApiMonitor(max_entries = 1)
    prompt = "long prompt\n" * 2000
    running_id = _start(monitor, prompt)
    old_id = _start(monitor, prompt + "old")
    monitor.finish(old_id)
    recent_id = _start(monitor, prompt + "recent")
    monitor.finish(recent_id)

    assert monitor.get(old_id, subject = "alice") is None
    assert monitor.get(running_id, subject = "alice")["prompt"] == prompt
    assert monitor.get(recent_id, subject = "alice")["prompt"] == prompt + "recent"
    monitor.clear(subject = "alice")
    assert monitor.get(recent_id, subject = "alice") is None
    assert monitor.get(running_id, subject = "alice")["prompt"] == prompt
    monitor.finish(running_id)
    monitor.clear(subject = "alice")
    assert monitor.snapshot(subject = "alice") == []
