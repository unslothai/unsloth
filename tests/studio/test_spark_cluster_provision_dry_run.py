# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A provisioning dry run must not change the peer.

`--rsync-path` is a command rsync runs ON THE PEER before transferring anything, so `--dry-run`
does not suppress it. Wrapping it as `mkdir -p "<parent>" && rsync` therefore created the
directories the dry run was only supposed to report. Every rsync argv the dry run builds is
captured here and checked, rather than asserting on a flag, because the defect was that a flag
did not mean what it looked like it meant.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _cluster():
    spec = importlib.util.spec_from_file_location(
        "spark_cluster_for_dry_run", REPO / "studio" / "spark_cluster.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Recorder:
    """Stands in for subprocess.run, answering `test -d` and recording everything."""

    def __init__(self, *, peer_dirs_exist: bool):
        self.calls: list[list[str]] = []
        self.peer_dirs_exist = peer_dirs_exist

    def __call__(self, cmd, *a, **k):
        self.calls.append(list(cmd))
        code = 0 if (cmd[0] != "ssh" or self.peer_dirs_exist) else 1
        return subprocess.CompletedProcess(cmd, code, stdout = "", stderr = "")

    def rsync_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0] == "rsync"]

    def ssh_calls(self) -> list[list[str]]:
        return [c for c in self.calls if c and c[0] == "ssh"]


def _run(
    cluster,
    monkeypatch,
    tmp_path,
    *,
    dry_run: bool,
    peer_dirs_exist: bool = True,
):
    local = tmp_path / "studio"
    local.mkdir(parents = True, exist_ok = True)
    (local / "marker").write_text("x", encoding = "utf-8")

    monkeypatch.setattr(cluster.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(cluster, "provision_paths", lambda: [(str(local), "studio")])
    monkeypatch.setattr(cluster, "_ssh_user", lambda: "someuser")
    monkeypatch.setattr(
        cluster,
        "peer_gpu_busy",
        lambda ip: {"busy": False, "reason": "", "processes": []},
    )
    monkeypatch.setattr(
        cluster,
        "fast_path_decision",
        lambda ip, no_fast = False: {
            "ok": False,
            "reason": "disabled for this test",
            "local_ip": None,
        },
    )
    recorder = _Recorder(peer_dirs_exist = peer_dirs_exist)
    monkeypatch.setattr(cluster.subprocess, "run", recorder)
    results = cluster.provision_peer("192.0.2.7", dry_run = dry_run, no_fast = True)
    return recorder, results


def test_a_dry_run_never_asks_the_peer_to_create_anything(monkeypatch, tmp_path) -> None:
    cluster = _cluster()
    recorder, _ = _run(cluster, monkeypatch, tmp_path, dry_run = True)

    assert recorder.rsync_calls(), "no rsync was attempted, so this proved nothing"
    for cmd in recorder.calls:
        joined = " ".join(cmd)
        assert "mkdir" not in joined, f"a dry run would run: {joined}"
    for cmd in recorder.rsync_calls():
        assert "--dry-run" in cmd, cmd


def test_a_real_run_still_creates_the_parent(monkeypatch, tmp_path) -> None:
    """No regression: rsync creates only the last component, so a brand-new peer needs this."""
    cluster = _cluster()
    recorder, _ = _run(cluster, monkeypatch, tmp_path, dry_run = False)

    rsyncs = recorder.rsync_calls()
    assert rsyncs, "no rsync was attempted"
    for cmd in rsyncs:
        assert "--dry-run" not in cmd, cmd
        path_arg = cmd[cmd.index("--rsync-path") + 1]
        assert path_arg.startswith("mkdir -p "), path_arg


def test_a_missing_peer_directory_is_reported_not_attempted(monkeypatch, tmp_path) -> None:
    """With nothing on the far side there is nothing to compare against, so the dry run says
    what the real run would create instead of failing on a destination that is not there."""
    cluster = _cluster()
    recorder, results = _run(cluster, monkeypatch, tmp_path, dry_run = True, peer_dirs_exist = False)

    assert not recorder.rsync_calls(), "a dry run tried to transfer into a missing directory"
    assert recorder.ssh_calls(), "the check that replaced the mkdir did not run"
    for cmd in recorder.ssh_calls():
        assert "test" in cmd and "-d" in cmd, cmd
    assert any("would create" in reason for _, reason in results["skipped"]), results["skipped"]
    assert not results["failed"], results["failed"]
