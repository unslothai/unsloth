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


# ── the GPU probe's fail-closed contract ────────────────────────────────────────
# A row that cannot be read is not an absent process. nvidia-smi reports `[N/A]` for
# used_memory in real situations, and skipping such a row and then declaring the peer idle
# inverts the whole point of the probe: provisioning would overwrite a venv the process behind
# that row is running out of.


def _probe(cluster, monkeypatch, stdout: str):
    monkeypatch.setattr(cluster.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd, *a, **k):
        return subprocess.CompletedProcess(cmd, 0, stdout = stdout, stderr = "")

    monkeypatch.setattr(cluster.subprocess, "run", fake_run)
    return cluster.peer_gpu_busy("192.0.2.7")


@pytest.mark.parametrize(
    "row",
    [
        "4242, [N/A]",          # the documented one: nvidia-smi cannot report the memory
        "4242, ",               # empty memory field
        "4242, not-a-number",
        "[N/A], 900 MiB",       # unreadable pid
    ],
)
def test_an_unreadable_process_row_leaves_the_peer_busy(monkeypatch, row: str) -> None:
    cluster = _cluster()
    out = _probe(cluster, monkeypatch, f"pid, used_memory\n{row}\nRC=0\n")
    assert out["busy"] is True, out
    assert out["known"] is False, out
    assert "could not be read" in out["reason"], out


def test_a_clean_idle_peer_is_still_idle(monkeypatch) -> None:
    """No regression: the whole reason provisioning is allowed to run at all."""
    cluster = _cluster()
    out = _probe(cluster, monkeypatch, "pid, used_memory\nRC=0\n")
    assert out["busy"] is False and out["known"] is True, out


def test_a_readable_busy_peer_is_still_busy(monkeypatch) -> None:
    cluster = _cluster()
    out = _probe(cluster, monkeypatch, "pid, used_memory\n4242, 40000 MiB\nRC=0\n")
    assert out["busy"] is True and out["known"] is True, out
    assert out["processes"] == [{"pid": 4242, "used_mib": 40000}], out
