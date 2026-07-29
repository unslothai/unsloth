# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio stop` must stop every server it started.

With one PID file the second launch overwrote the first entry, so stop killed
the newer server, claimed success, and left the older one serving.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


# Captured before _install stubs it, for the tests that exercise it.
_REAL_IS_STUDIO_SERVER = _studio()._pid_is_studio_server


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
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)

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
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert sorted(killed) == [8550, 8600]
    assert not list(tmp_path.glob("studio-*.pid"))


def test_stop_does_not_leave_the_older_instance_running(monkeypatch, tmp_path):
    # The reported symptom: stop claimed success while instance A kept serving.
    studio_mod, live, _killed = _install(monkeypatch, tmp_path, alive = {8550, 8600})
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert live == set()


def test_stop_signals_each_server_once(monkeypatch, tmp_path):
    # A server writes its per-port file AND studio.pid. It stays alive while it
    # shuts down gracefully, so a second SIGTERM would hit the SIG_DFL the first
    # one installs and hard-kill it mid-cleanup.
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)
    killed = []
    monkeypatch.setattr(studio_mod.os, "kill", lambda pid, _sig: killed.append(pid))
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [8550]
    assert result.output.lower().count("sent shutdown signal") == 1


def test_stop_removes_every_stale_file_for_one_pid(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = set())
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not list(tmp_path.glob("*.pid"))


def test_stop_does_not_signal_a_reused_pid(monkeypatch, tmp_path):
    # Crash leaves a per-port file behind, the OS hands that PID to something
    # else: stop must drop the record, not SIGTERM an unrelated process.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): False)
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not (tmp_path / "studio-8901-8550.pid").exists()


def test_stop_signals_a_live_server_whose_pid_has_a_stale_record(monkeypatch, tmp_path):
    # Crash leaves studio-8888-8550.pid, the OS reuses 8550 for a new server on
    # another port. The stale timestamp must not veto the live one.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))
    (tmp_path / "studio-8888-8550.pid").write_text("8550\n111.5", encoding = "utf-8")
    (tmp_path / "studio-9000-8550.pid").write_text("8550\n999.0", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [8550]
    assert not list(tmp_path.glob("studio-*.pid"))


def test_a_bare_run_py_command_line_is_not_rejected(monkeypatch):
    # `cd studio/backend && python run.py --port 8901` has no "studio" or "unsloth"
    # in argv. Guessing from the command line deleted its record without stopping it.
    studio_mod = _studio()

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return ["python", "run.py", "--port", "8901"]

        def create_time(self):
            return 111.5

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert studio_mod._pid_is_studio_server(8550) is True


def test_an_untimed_record_is_trusted(monkeypatch):
    # A legacy `python run.py --port 8901` has no telltale argv, and the in-venv
    # path runs in-process. Guessing from the command line rejected real servers.
    studio_mod = _studio()

    assert studio_mod._pid_is_studio_server(8550) is True
    assert studio_mod._pid_is_studio_server(8550, [None]) is True


def test_an_unverifiable_record_is_still_stopped(monkeypatch):
    # psutil is not a base CLI dependency, so the CLI meets timestamped records it
    # cannot check. The old `stop` signalled with no checks at all -- skipping one
    # would leave a live server running, the orphan bug this exists to fix.
    studio_mod = _studio()
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert studio_mod._pid_is_studio_server(8550, [111.5]) is True
    assert studio_mod._pid_is_studio_server(8550, [None]) is True


def test_stop_signals_a_timestamped_record_without_psutil(monkeypatch, tmp_path):
    # Multiple servers on different ports: only the newest is also in studio.pid,
    # so the earlier ones are timestamp-only and must still be stopped.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)
    monkeypatch.setitem(sys.modules, "psutil", None)
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == [8550]
    assert not (tmp_path / "studio-8901-8550.pid").exists()


def test_the_untimed_legacy_record_does_not_cancel_a_timed_one(monkeypatch):
    # Every current server writes BOTH a timed per-port record and an untimed
    # studio.pid, so letting the untimed half win made this check inert exactly
    # where it matters: after a crash and a PID reuse, `stop` SIGTERMed whatever
    # unrelated process had inherited the PID. An untimed record carries no
    # information, so it must not overrule a start time that says "not ours".
    studio_mod = _studio()

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert studio_mod._pid_is_studio_server(8550, [111.5, None]) is False
    assert studio_mod._pid_is_studio_server(8550, [111.5]) is False
    # A matching time still wins over a stale sibling record.
    assert studio_mod._pid_is_studio_server(8550, [111.5, 999.0]) is True
    assert studio_mod._pid_is_studio_server(8550, [None, None]) is True


def test_stop_does_not_signal_a_reused_pid_recorded_in_both_files(monkeypatch, tmp_path):
    # End to end for the case above: a crashed server left studio-8901-8550.pid
    # and studio.pid, and 8550 now belongs to something else entirely.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5\n127.0.0.1", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not list(tmp_path.glob("*.pid"))


def test_pid_identity_check_trusts_the_record_without_psutil(monkeypatch):
    # No psutil: fall back to trusting the record rather than never stopping.
    studio_mod = _studio()
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert studio_mod._pid_is_studio_server(8550) is True


def test_pid_identity_check_uses_the_recorded_start_time(monkeypatch):
    studio_mod = _studio()

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 111.5

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert studio_mod._pid_is_studio_server(8550, [111.5]) is True
    assert studio_mod._pid_is_studio_server(8550, [999.0]) is False


def test_stop_drops_a_record_whose_start_time_no_longer_matches(monkeypatch, tmp_path):
    # The PID was reused: same number, different process.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", _REAL_IS_STUDIO_SERVER)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def create_time(self):
            return 999.0

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not (tmp_path / "studio-8901-8550.pid").exists()
    # Dropped for the start-time mismatch, not because the record looked corrupt.
    assert "invalid pid file" not in result.output.lower()


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
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not (tmp_path / "studio-8901-8550.pid").exists()
    assert "stopped" not in result.output.lower()


def test_stop_does_not_claim_a_stop_while_a_server_is_still_alive(monkeypatch, tmp_path):
    # SIGTERM delivered but it never exits: don't claim a stop, keep the file.
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)
    monkeypatch.setattr(studio_mod.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert "shutting down" in result.output.lower()
    assert "stopped" not in result.output.lower()
    assert (tmp_path / "studio-8901-8550.pid").exists()


def test_stop_continues_after_one_server_fails_to_stop(monkeypatch, tmp_path):
    studio_mod = _studio()
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(studio_mod, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(studio_mod.time, "sleep", lambda _s: None)
    live = {8550, 8600}
    monkeypatch.setattr(studio_mod, "_pid_alive", lambda pid: pid in live)
    monkeypatch.setattr(studio_mod, "_pid_is_studio_server", lambda pid, created_times = (): True)

    def fake_kill(pid, _sig):
        if pid == 8550:
            raise PermissionError("not permitted")
        live.discard(pid)

    monkeypatch.setattr(studio_mod.os, "kill", fake_kill)
    monkeypatch.setattr(sys, "platform", "linux")
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)

    result = _run_stop(studio_mod)

    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert result.exit_code == 1, combined
    assert 8600 not in live
    assert "8550" in combined


def test_stop_never_signals_pid_zero_or_init(monkeypatch, tmp_path):
    # os.kill(0, SIGTERM) hits our whole process group -- the shell and its jobs.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {0, 1})
    _write_pid(tmp_path, "studio-8901-0.pid", 0)
    _write_pid(tmp_path, "studio-8902-1.pid", 1)

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert killed == []
    assert not list(tmp_path.glob("*.pid"))


def test_signal_stop_refuses_pid_zero_or_init(monkeypatch, tmp_path):
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {0, 1})

    assert studio_mod._signal_stop(0) is not None
    assert studio_mod._signal_stop(1) is not None
    assert killed == []


def test_stop_discards_a_corrupt_pid_file(monkeypatch, tmp_path):
    studio_mod, _live, _killed = _install(monkeypatch, tmp_path, alive = set())
    (tmp_path / "studio-8901-8550.pid").write_text("not-a-pid", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "studio-8901-8550.pid").exists()


def test_stop_keeps_a_record_it_cannot_read(monkeypatch, tmp_path):
    # A root-owned record, or one caught mid-write, still belongs to a live
    # server. Deleting it is `stop` manufacturing the orphan it exists to fix.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    path = tmp_path / "studio-8901-8550.pid"
    path.write_text("8550", encoding = "utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    result = _run_stop(studio_mod)

    assert path.exists(), "an unreadable record must not be deleted"
    assert "cannot read" in (result.output + (result.stderr or "")).lower()


def test_stop_does_not_claim_success_when_the_only_record_is_unreadable(monkeypatch, tmp_path):
    # A server started under sudo leaves a record we cannot read. Printing "no
    # running server" and exiting 0 tells the user the opposite of the truth.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550})
    path = tmp_path / "studio-8901-8550.pid"
    path.write_text("8550", encoding = "utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self == path:
            raise PermissionError(13, "Permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    result = _run_stop(studio_mod)

    assert result.exit_code == 1, "an unreachable server is not a successful stop"
    output = result.output + (result.stderr or "")
    assert "no running unsloth server" not in output.lower()
    assert killed == []


def test_stop_reports_failure_when_one_record_is_unreadable_but_another_stops(
    monkeypatch, tmp_path
):
    # Stopping the servers we can see is still a partial result, and exiting 0
    # would hide the one we could not.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8550, 8600})
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)
    hidden = tmp_path / "studio-8902-8600.pid"
    hidden.write_text("8600", encoding = "utf-8")
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self == hidden:
            raise PermissionError(13, "Permission denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)

    result = _run_stop(studio_mod)

    assert killed == [8550], "the readable server must still be stopped"
    assert result.exit_code == 1
    assert hidden.exists()


def test_stop_reaches_every_server_when_one_record_cannot_be_removed(monkeypatch, tmp_path):
    # One undeletable stale record must not end the loop before the live servers.
    studio_mod, _live, killed = _install(monkeypatch, tmp_path, alive = {8600})
    _write_pid(tmp_path, "studio-8901-8550.pid", 8550)  # dead -> stop prunes it
    _write_pid(tmp_path, "studio-8902-8600.pid", 8600)  # live -> stop signals it
    real_unlink = Path.unlink

    def deny(self, *args, **kwargs):
        if self.name == "studio-8901-8550.pid":
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny)

    result = _run_stop(studio_mod)

    assert killed == [8600], "the live server must still be signalled"
    assert result.exit_code == 0, result.output


def test_a_record_whose_pid_is_not_ascii_digits_is_discarded(monkeypatch, tmp_path):
    # A superscript two passes isdigit() but int() rejects it, so that gate alone
    # let a ValueError escape _read_pid_record and abort the whole command.
    studio_mod, _live, _killed = _install(monkeypatch, tmp_path, alive = set())
    (tmp_path / "studio-8901-1.pid").write_text("²", encoding = "utf-8")

    result = _run_stop(studio_mod)

    assert result.exit_code == 0, result.output
    assert not (tmp_path / "studio-8901-1.pid").exists()
