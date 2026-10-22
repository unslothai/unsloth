# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest

from utils import pid_records as records


@pytest.mark.parametrize("delta", [-3600.0, -1.0, 0.0, 1.0, 3600.0])
def test_linux_identity_survives_wall_clock_changes(monkeypatch, delta):
    monkeypatch.setattr(records, "linux_process_identity", lambda pid: "linux:boot:123")
    start = records.recorded_start(["4242", "1000.0", "127.0.0.1", "linux:boot:123"])
    assert records.pid_start_matches(4242, [start], lambda pid: 1000.0 + delta)


@pytest.mark.parametrize("actual", ["linux:next-boot:123", "linux:boot:124"])
def test_reused_pid_is_rejected_even_if_epoch_time_matches(monkeypatch, actual):
    monkeypatch.setattr(records, "linux_process_identity", lambda pid: actual)
    start = records.recorded_start(["4242", "1000.0", "", "linux:boot:123"])
    assert not records.pid_start_matches(4242, [start, None], lambda pid: 1000.0)


def test_a_matching_record_can_corroborate_an_older_stale_one(monkeypatch):
    monkeypatch.setattr(records, "linux_process_identity", lambda pid: "linux:boot:124")
    assert records.pid_start_matches(4242, ["linux:boot:123", "linux:boot:124"])
    # A current legacy writer may reuse the PID of a dead, newer-format backend.
    assert records.pid_start_matches(4242, ["linux:boot:123", 1000.0], lambda pid: 1000.0)


def test_an_unreadable_identity_is_not_evidence_of_pid_reuse(monkeypatch):
    monkeypatch.setattr(records, "linux_process_identity", lambda pid: None)
    assert records.pid_start_matches(4242, ["linux:boot:123"], lambda pid: 9999.0)


@pytest.mark.parametrize(
    "delta, matches", [(-1.0, True), (1.0, True), (-1.01, False), (1.01, False)]
)
def test_legacy_epoch_tolerance_matches_the_desktop(delta, matches):
    assert records.pid_start_matches(4242, [1000.0], lambda pid: 1000.0 + delta) is matches


@pytest.mark.parametrize(
    "extra",
    ["", "linux:boot:garbage", "linux::123", "linux:boot:-1", "linux:boot:18446744073709551616"],
)
def test_invalid_identity_falls_back_to_legacy_time(extra):
    assert records.recorded_start(["4242", "1000.0", "127.0.0.1", extra]) == 1000.0


@pytest.mark.parametrize("lines", [["4242"], ["4242", ""], ["4242", "nan"], ["4242", "inf"]])
def test_unavailable_start_time_stays_unknown(lines):
    assert records.recorded_start(lines) is None


@pytest.mark.skipif(sys.platform != "linux", reason = "Linux procfs identity")
def test_proc_identity_handles_parentheses_and_spaces_in_comm(monkeypatch):
    def read(path, **kwargs):
        if str(path).endswith("/stat"):
            return "4242 (name with ) spaces) S " + "0 " * 18 + "123 0\n"
        return "boot\n"

    monkeypatch.setattr(Path, "read_text", read)
    assert records.linux_process_identity(4242) == "linux:boot:123"


def test_an_unverifiable_legacy_record_cannot_cancel_proven_pid_reuse(monkeypatch):
    monkeypatch.setattr(records, "linux_process_identity", lambda pid: "linux:boot:124")
    assert not records.pid_start_matches(4242, ["linux:boot:123", 1000.0, None], lambda pid: None)


def test_only_a_stable_start_proves_pid_reuse(monkeypatch):
    monkeypatch.setattr(records.sys, "platform", "linux")
    monkeypatch.setattr(records, "linux_process_identity", lambda pid: "linux:boot:124")
    drifted = lambda pid: 4600.0
    assert records.pid_reuse_proven(42, ["linux:boot:123"], drifted)
    assert not records.pid_reuse_proven(42, ["linux:boot:124"], drifted)
    assert not records.pid_reuse_proven(42, [1000.0], drifted)
    assert not records.pid_reuse_proven(42, [None], drifted)
    monkeypatch.setattr(records, "linux_process_identity", lambda pid: None)
    assert not records.pid_reuse_proven(42, ["linux:boot:123"], drifted)
    monkeypatch.setattr(records.sys, "platform", "darwin")
    assert records.pid_reuse_proven(42, [1000.0], drifted)
