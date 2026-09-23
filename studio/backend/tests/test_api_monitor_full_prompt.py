# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys

import pytest

import core.inference.api_monitor as monitor_module
from core.inference.api_monitor import ApiMonitor
from utils.account_context import AccountContext, run_as


def _start(monitor, prompt):
    return monitor.start(
        endpoint="/v1/chat/completions",
        method="POST",
        model="test-model",
        prompt=prompt,
        subject="alice",
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
        detail = monitor.get(entry_id, subject="alice")
        assert detail["prompt"] == prompt
        [summary] = monitor.snapshot(include_details=False, subject="alice")
        assert "prompt" not in summary
        assert len(summary["prompt_preview"]) <= 360
        assert summary["prompt_truncated"] is (len(prompt) > 360)


def test_full_prompt_remains_account_and_subject_scoped():
    monitor = ApiMonitor()
    alice = AccountContext("alice-id", "alice")
    replacement = AccountContext("replacement-id", "alice")
    prompt = "private prompt\n" * 2000
    entry_id = run_as(alice, _start, monitor, prompt)

    assert run_as(alice, monitor.get, entry_id, subject="alice")["prompt"] == prompt
    assert run_as(alice, monitor.get, entry_id, subject="bob") is None
    assert run_as(replacement, monitor.get, entry_id, subject="alice") is None
    assert run_as(replacement, monitor.snapshot, subject="alice") == []


def test_full_prompt_follows_history_retention_and_clear():
    monitor = ApiMonitor(max_entries=1)
    prompt = "long prompt\n" * 2000
    running_id = _start(monitor, prompt)
    old_id = _start(monitor, prompt + "old")
    monitor.finish(old_id)
    recent_id = _start(monitor, prompt + "recent")
    monitor.finish(recent_id)

    assert monitor.get(old_id, subject="alice") is None
    assert monitor.get(running_id, subject="alice")["prompt"] == prompt
    assert monitor.get(recent_id, subject="alice")["prompt"] == prompt + "recent"
    monitor.clear(subject="alice")
    assert monitor.get(recent_id, subject="alice") is None
    assert monitor.get(running_id, subject="alice")["prompt"] == prompt
    monitor.finish(running_id)
    monitor.clear(subject="alice")
    assert monitor.snapshot(subject="alice") == []


@pytest.mark.parametrize("character", ["x", "🦥"])
@pytest.mark.parametrize("terminal", [None, "finish", "fail"])
def test_full_prompt_budget_preserves_rows_and_recent_details(monkeypatch, character, terminal):
    prompt = character * 2000
    monkeypatch.setattr(
        monitor_module, "_MAX_PROMPT_BYTES", 2 * sys.getsizeof(prompt), raising=False
    )
    monitor = ApiMonitor()
    ids = []
    for _ in range(3):
        entry_id = _start(monitor, prompt)
        ids.append(entry_id)
        if terminal == "finish":
            monitor.finish(entry_id)
        elif terminal == "fail":
            monitor.fail(entry_id, "context limit")

    oldest = monitor.get(ids[0], subject="alice")
    assert "prompt" not in oldest
    assert oldest["prompt_truncated"] is True
    assert len(oldest["prompt_preview"]) == 360
    for entry_id in ids[1:]:
        assert monitor.get(entry_id, subject="alice")["prompt"] == prompt
    assert len(monitor.snapshot(subject="alice")) == 3
    assert monitor.active_count(subject="alice") == (3 if terminal is None else 0)


def test_oversized_prompt_keeps_preview_without_evicting_smaller_details(monkeypatch):
    prompt = "x" * 2000
    monkeypatch.setattr(monitor_module, "_MAX_PROMPT_BYTES", sys.getsizeof(prompt), raising=False)
    monitor = ApiMonitor()
    retained = _start(monitor, prompt)
    oversized = _start(monitor, prompt * 2)
    monitor.set_reply(oversized, "reply remains available")

    assert monitor.get(retained)["prompt"] == prompt
    detail = monitor.get(oversized)
    assert "prompt" not in detail
    assert detail["prompt_truncated"] is True
    assert detail["reply"] == "reply remains available"
    assert len(detail["prompt_preview"]) == 360


def test_eviction_releases_prompt_budget_for_new_requests(monkeypatch):
    prompt = "x" * 2000
    monkeypatch.setattr(monitor_module, "_MAX_PROMPT_BYTES", sys.getsizeof(prompt), raising=False)
    monitor = ApiMonitor(max_entries=1)
    first = _start(monitor, prompt)
    monitor.finish(first)
    second = _start(monitor, prompt)
    monitor.finish(second)
    assert monitor.get(first) is None
    assert monitor.get(second)["prompt"] == prompt
    monitor.clear()
    third = _start(monitor, prompt)
    assert monitor.get(third)["prompt"] == prompt


def test_reply_refresh_can_omit_immutable_prompt(monkeypatch):
    monkeypatch.setattr(monitor_module.time, "time", lambda: 1_700_000_000.0)
    monitor = ApiMonitor()
    prompt = "long prompt\n" * 2000
    entry_id = _start(monitor, prompt)
    first = monitor.get(entry_id, subject="alice")
    for chunk in ("first", " second", " third"):
        monitor.append_reply(entry_id, chunk)
        refresh = monitor.get(entry_id, subject="alice", include_prompt=False)
        full = monitor.get(entry_id, subject="alice")
        assert "prompt" not in refresh
        assert refresh == {key: value for key, value in full.items() if key != "prompt"}
        assert full["prompt"] == prompt
    assert refresh["reply"] == "first second third"
    assert refresh["updated_at"] > first["updated_at"]
    assert monitor.get(entry_id, subject="bob", include_prompt=False) is None
    monitor.finish(entry_id)
    assert monitor.get(entry_id, include_prompt=False)["status"] == "completed"
