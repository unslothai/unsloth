# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio stop` must stop every server it started.

With one PID file the second launch overwrote the first entry, so stop killed
the newer server, claimed success, and left the older one serving.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _install(
    monkeypatch,
    tmp_path,
    *,
    alive,
    killed = None,
):
    """Point the CLI at tmp_path and fake process liveness."""
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)

    live = set(alive)
    killed = killed if killed is not None else []

    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: pid in live)

    def fake_kill(pid, _sig):
        killed.append(pid)
        live.discard(pid)

    monkeypatch.setattr(studio_mod.os, "kill", fake_kill)
    monkeypatch.setattr(sys, "platform", "linux")
    return studio_mod, live, killed


def _write_pid(tmp_path, name, pid):
    (tmp_path / name).write_text(str(pid), encoding = "utf-8")


def _run_stop(studio_mod):
    import typer as _typer

    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    return CliRunner().invoke(app, ["studio", "stop"])


def test_stop_kills_every_recorded_server(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550, 8600})
    _write_pid(tmp_path, "studio-8901.pid", 8550)
    _write_pid(tmp_path, "studio-8902.pid", 8600)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert sorted(killed) == [8550, 8600]
    assert not list(tmp_path.glob("studio-*.pid"))


def test_stop_does_not_leave_the_older_instance_running(monkeypatch, tmp_path):
    # The reported symptom: stop claimed success while instance A kept serving.
    studio_mod, live, _killed = _install(monkeypatch, tmp_path, alive = {8550, 8600})
    _write_pid(tmp_path, "studio-8901.pid", 8550)
    _write_pid(tmp_path, "studio-8902.pid", 8600)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert live == set()


def test_stop_reads_the_legacy_single_pid_file(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {4242})
    _write_pid(tmp_path, "studio.pid", 4242)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [4242]
    assert not (tmp_path / "studio.pid").exists()


def test_stop_reports_nothing_running_without_pid_files(monkeypatch, tmp_path):
    studio_mod, _live, _killed = _install(monkeypatch, tmp_path, alive = set())

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert "no running unsloth server" in result.output.lower()


def test_stop_cleans_stale_pid_files_without_claiming_a_stop(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = set())
    _write_pid(tmp_path, "studio-8901.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not (tmp_path / "studio-8901.pid").exists()
    assert "stopped" not in result.output.lower()


def test_stop_does_not_claim_a_stop_while_a_server_is_still_alive(monkeypatch, tmp_path):
    # SIGTERM delivered but it never exits: don't claim a stop, keep the file.
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(studio_mod.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert "shutting down" in result.output.lower()
    assert "stopped" not in result.output.lower()
    assert (tmp_path / "studio-8901.pid").exists()


def test_stop_continues_after_one_server_fails_to_stop(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    live = {8550, 8600}
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: pid in live)

    def fake_kill(pid, _sig):
        if pid == 8550:
            raise PermissionError("not permitted")
        live.discard(pid)

    monkeypatch.setattr(studio_mod.os, "kill", fake_kill)
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901.pid", 8550)
    _write_pid(tmp_path, "studio-8902.pid", 8600)

    result = _run_stop(studio_mod)

    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert result.exit_code == 1, combined
    assert 8600 not in live
    assert "8550" in combined


def test_stop_discards_a_corrupt_pid_file(monkeypatch, tmp_path):
    studio_mod, _live, _killed = _install(monkeypatch, tmp_path, alive = set())
    (tmp_path / "studio-8901.pid").write_text("not-a-pid", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "studio-8901.pid").exists()
