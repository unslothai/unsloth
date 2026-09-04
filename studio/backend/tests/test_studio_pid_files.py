# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-port PID files, so `unsloth studio stop` can find every server.

Imports run.py directly, so run under the Unsloth venv.
"""

from __future__ import annotations

import contextlib
import errno
import os
import socket
import threading
import time
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import run  # noqa: E402
from utils import cache_cleanup  # noqa: E402

# Captured before the autouse fixture stubs them, for the tests that exercise them.
_REAL_IS_STUDIO_BACKEND = run._pid_is_studio_backend
_REAL_PID_ALIVE = run._pid_alive


@pytest.fixture(autouse = True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_studio_root", lambda: tmp_path)
    monkeypatch.setattr(run, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(run, "_OWN_PID_FILE", None)
    monkeypatch.setattr(run, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created_times = (): True)
    # The lock lives in the studio home. A session-scoped conftest fixture
    # already keeps that out of the developer's real one, but this file
    # redirects everything else it touches to tmp_path, so redirect this too.
    monkeypatch.setattr(cache_cleanup, "cache_coordination_dir", lambda: tmp_path)
    monkeypatch.setattr(run, "_OWN_STARTUP_MARKERS", [])
    yield


def _files(tmp_path):
    return sorted(p.name for p in tmp_path.glob("studio-*.pid"))


def _pid_of(path):
    return path.read_text(encoding = "utf-8").splitlines()[0]


def test_write_pid_file_records_port_and_pid(tmp_path):
    run._write_pid_file(8901)

    assert _files(tmp_path) == [f"studio-8901-{os.getpid()}.pid"]
    assert _pid_of(tmp_path / f"studio-8901-{os.getpid()}.pid") == str(os.getpid())


def test_write_pid_file_records_the_start_time(tmp_path):
    # Pins the record to this process, so a reused PID isn't mistaken for it.
    run._write_pid_file(8901)

    record = run._read_pid_record(tmp_path / f"studio-8901-{os.getpid()}.pid")

    assert record[0] == os.getpid()
    assert record[1] == pytest.approx(run._process_create_time(os.getpid()))


def test_write_pid_file_keeps_the_legacy_file_a_bare_pid(tmp_path):
    # An older CLI's `stop` reads studio.pid and expects only digits.
    run._write_pid_file(8901)

    assert (tmp_path / "studio.pid").read_text(encoding = "utf-8") == str(os.getpid())


def test_second_port_does_not_clobber_the_first(tmp_path):
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    run._write_pid_file(8902)

    assert _pid_of(tmp_path / "studio-8901-8550.pid") == "8550"
    assert (tmp_path / f"studio-8902-{os.getpid()}.pid").exists()


def test_same_port_on_two_binds_does_not_clobber(tmp_path):
    # 127.0.0.1:8888 and ::1:8888 can both listen; one file per port would lose one.
    (tmp_path / "studio-8888-8550.pid").write_text("8550", encoding = "utf-8")

    run._write_pid_file(8888)

    assert len(_files(tmp_path)) == 2


def test_remove_pid_file_only_removes_our_own(tmp_path, monkeypatch):
    run._write_pid_file(8901)
    (tmp_path / "studio-8902-8600.pid").write_text("8600", encoding = "utf-8")
    # Nothing to hand the legacy pointer to, so it goes away with us.
    monkeypatch.setattr(run, "_pid_alive", lambda pid: pid == os.getpid())

    run._remove_pid_file()

    assert _files(tmp_path) == ["studio-8902-8600.pid"]
    assert not (tmp_path / "studio.pid").exists()


def test_the_legacy_pointer_moves_to_a_live_sibling(tmp_path):
    # Only one server owns studio.pid. Deleting it on our way out would leave an
    # older CLI, which reads nothing else, unable to stop the sibling still up.
    run._write_pid_file(8901)
    (tmp_path / "studio-8902-8600.pid").write_text("8600", encoding = "utf-8")

    run._remove_pid_file()

    assert (tmp_path / "studio.pid").read_text(encoding = "utf-8").strip() == "8600"


def test_the_legacy_pointer_is_not_handed_to_a_dead_sibling(tmp_path, monkeypatch):
    run._write_pid_file(8901)
    (tmp_path / "studio-8902-8600.pid").write_text("8600", encoding = "utf-8")
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created_times = (): False)

    run._remove_pid_file()

    assert not (tmp_path / "studio.pid").exists()


def test_remove_pid_file_leaves_a_reused_entry_alone(tmp_path):
    run._write_pid_file(8901)
    own = tmp_path / f"studio-8901-{os.getpid()}.pid"
    own.write_text("999999", encoding = "utf-8")

    run._remove_pid_file()

    assert own.read_text(encoding = "utf-8") == "999999"


def test_windows_liveness_does_not_call_every_pid_alive(monkeypatch):
    # os.kill(pid, 0) raises OSError for every pid on Windows, so without the
    # tasklist fallback a stale record would block its port forever.
    import subprocess

    monkeypatch.setattr(run, "_pid_alive", _REAL_PID_ALIVE)
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(stdout = '"python.exe","8550",...')
    )

    assert run._pid_alive(8550) is True
    assert run._pid_alive(9999) is False


def test_windows_liveness_keeps_the_record_when_tasklist_fails(monkeypatch):
    # Unconfirmed must mean keep, matching the CLI's _pid_alive. Pruning a live
    # server's record lets the next launch fall back past it and strand it, which
    # is the bug this file exists to fix; a stale record costs one clear abort.
    import subprocess

    def _boom(*a, **k):
        raise OSError("tasklist missing")

    monkeypatch.setattr(run, "_pid_alive", _REAL_PID_ALIVE)
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(subprocess, "run", _boom)

    assert run._pid_alive(8550) is True


def test_read_pid_record_parses_pid_time_and_address(tmp_path):
    (tmp_path / "r.pid").write_text("8550\n111.5\n127.0.0.1", encoding = "utf-8")

    assert run._read_pid_record(tmp_path / "r.pid") == (8550, 111.5, "127.0.0.1")


def test_read_pid_record_tolerates_a_bare_pid(tmp_path):
    (tmp_path / "r.pid").write_text("8550", encoding = "utf-8")

    assert run._read_pid_record(tmp_path / "r.pid") == (8550, None, None)


def test_read_pid_record_rejects_pid_zero_and_init(tmp_path):
    # kill(0) signals our whole process group.
    (tmp_path / "zero.pid").write_text("0", encoding = "utf-8")
    (tmp_path / "init.pid").write_text("1", encoding = "utf-8")

    assert run._read_pid_record(tmp_path / "zero.pid") is None
    assert run._read_pid_record(tmp_path / "init.pid") is None


def test_read_pid_record_rejects_a_corrupt_file(tmp_path):
    (tmp_path / "r.pid").write_text("not-a-pid", encoding = "utf-8")

    assert run._read_pid_record(tmp_path / "r.pid") is None


def test_graceful_shutdown_drops_the_record_last(monkeypatch):
    # Cleanup can take seconds while the server is still alive. Dropping the record
    # first leaves a retried `stop` or a new launch unable to find it.
    order = []
    monkeypatch.setattr(run, "_remove_pid_file", lambda: order.append("remove_record"))

    class _Server:
        def __setattr__(self, name, value):
            order.append("release_socket")

    run._graceful_shutdown(_Server())

    assert order == ["release_socket", "remove_record"]


def test_own_studio_on_port_is_found_without_psutil(tmp_path, monkeypatch):
    # psutil is optional; a listener scan finds nothing without it, so detection
    # must come from our own records or we silently start a duplicate.
    monkeypatch.setitem(sys.modules, "psutil", None)
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n\n127.0.0.1", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") == 8550


def test_no_record_for_the_port_means_no_own_studio(tmp_path):
    # jupyter-lab on 8888 must keep the fallback, not abort the launch.
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8888, "127.0.0.1") is None


def test_own_studio_on_port_prunes_a_dead_record(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_pid_alive", lambda pid: False)
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") is None
    assert not (tmp_path / "studio-8901-8550.pid").exists()


def test_a_reused_pid_is_not_treated_as_our_studio(tmp_path, monkeypatch):
    # Stale record + the OS handing that PID to something else must not abort.
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created_times = (): False)
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") is None


def test_an_unverifiable_record_still_blocks_a_duplicate(tmp_path, monkeypatch):
    # Can't tell: refusing with a clear message beats a silent second instance.
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created_times = (): True)
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") == 8550


def test_start_time_mismatch_rejects_a_reused_pid(monkeypatch):
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)

    assert run._pid_is_studio_backend(8550, [111.5]) is False
    assert run._pid_is_studio_backend(8550, [999.0]) is True


def test_a_stale_record_does_not_veto_a_live_server_sharing_the_pid(monkeypatch):
    # Crash leaves studio-8888-1234.pid, the OS reuses 1234 for a new server on
    # another port. Keeping only the first timestamp would reject the live one.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)

    assert run._pid_is_studio_backend(1234, [111.5, 999.0]) is True
    assert run._pid_is_studio_backend(1234, [111.5, 222.5]) is False


def test_a_stale_record_on_another_port_does_not_hide_a_live_server(tmp_path, monkeypatch):
    # 1234 was reused: the stale 8888 record must not stop us seeing 9000.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)
    (tmp_path / "studio-8888-1234.pid").write_text("1234\n111.5\n", encoding = "utf-8")
    (tmp_path / "studio-9000-1234.pid").write_text("1234\n999.0\n", encoding = "utf-8")

    assert run._own_studio_on_port(8888, "127.0.0.1") is None
    assert run._own_studio_on_port(9000, "127.0.0.1") == 1234


def test_a_start_time_is_the_only_thing_that_disproves_a_record(monkeypatch):
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)

    assert run._pid_is_studio_backend(8550, [999.0]) is True
    assert run._pid_is_studio_backend(8550, [111.5]) is False


def test_a_bare_run_py_command_line_is_not_rejected(monkeypatch):
    # `cd studio/backend && python run.py --port 8901` has no "studio" or "unsloth"
    # in argv. Guessing from the command line called that "not ours".
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return ["python", "run.py", "--port", "8901"]

        def create_time(self):
            return 111.5

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert run._pid_is_studio_backend(8550) is True


def test_an_untimed_legacy_record_is_trusted(monkeypatch):
    # `python run.py --port 8901` has no telltale argv, so guessing from the
    # command line rejected real servers. Only a start time can disprove one.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)

    assert run._pid_is_studio_backend(8550) is True
    assert run._pid_is_studio_backend(8550, [None]) is True


def test_the_untimed_legacy_record_does_not_cancel_a_timed_one(monkeypatch):
    # Mirrors _pid_is_studio_server in the CLI. An untimed record carries no
    # information, so it must not overrule a start time that says "not ours" --
    # every current server writes one of each, which made the check inert.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)

    assert run._pid_is_studio_backend(8550, [111.5, None]) is False
    assert run._pid_is_studio_backend(8550, [111.5, 999.0]) is True


def test_a_legacy_server_on_the_port_is_recognised(tmp_path, monkeypatch):
    # Pre-upgrade servers wrote only studio.pid. Falling back past one strands it
    # and then overwrites its record.
    monkeypatch.setattr(run, "_get_pid_on_port", lambda p: (8550, "python"))
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") == 8550


def test_a_legacy_record_for_a_different_listener_falls_back(tmp_path, monkeypatch):
    # jupyter holds the port; the legacy server is elsewhere. Keep falling back.
    monkeypatch.setattr(run, "_get_pid_on_port", lambda p: (117, "jupyter-lab"))
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") is None


def test_an_unknowable_listener_treats_the_legacy_record_as_ours(tmp_path, monkeypatch):
    # No psutil: _get_pid_on_port can't say. Refusing beats a silent duplicate.
    monkeypatch.setattr(run, "_get_pid_on_port", lambda p: None)
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") == 8550


def test_a_dead_legacy_record_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(run, "_get_pid_on_port", lambda p: None)
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") is None


def test_a_stale_per_port_record_does_not_mask_a_legacy_server(tmp_path, monkeypatch):
    # Crashed current build left studio-8901-8550.pid; 8550 was then reused by a
    # pre-upgrade server recorded only in studio.pid. The stale record must not
    # count as "port already known" and send us falling back past the live one.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)
    monkeypatch.setattr(run, "_get_pid_on_port", lambda p: (8550, "python"))
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5\n127.0.0.1", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") == 8550


def test_a_current_server_elsewhere_does_not_block_a_foreign_port(tmp_path, monkeypatch):
    # Current builds write studio.pid too. Without psutil the legacy check can't
    # see the listener, so it must not claim our 8901 server holds jupyter's 8888.
    monkeypatch.setattr(run, "_get_pid_on_port", lambda p: None)
    (tmp_path / "studio-8901-5000.pid").write_text("5000\n\n127.0.0.1", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("5000", encoding = "utf-8")

    assert run._own_studio_on_port(8888, "127.0.0.1") is None


def test_a_per_port_record_is_preferred_over_the_legacy_one(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_get_pid_on_port", lambda p: (8550, "python"))
    (tmp_path / "studio-8901-8600.pid").write_text("8600\n\n127.0.0.1", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run._own_studio_on_port(8901, "127.0.0.1") == 8600


def test_our_studio_on_another_bind_address_does_not_abort(tmp_path):
    # Our server holds ::1:8889; binding 127.0.0.1:8889 is not a conflict with us,
    # so fall through to the next port instead of refusing.
    (tmp_path / "studio-8889-8550.pid").write_text("8550\n\n::1", encoding = "utf-8")

    assert run._own_studio_on_port(8889, "127.0.0.1") is None
    assert run._own_studio_on_port(8889, "::1") == 8550


def test_address_matching(tmp_path):
    assert run._addresses_collide("0.0.0.0", "127.0.0.1", 8889) is True
    assert run._addresses_collide("127.0.0.1", "0.0.0.0", 8889) is True
    assert run._addresses_collide("127.0.0.1", "127.0.0.1", 8889) is True
    assert run._addresses_collide("::1", "127.0.0.1", 8889) is False
    # An unrecorded address is unknown, so assume a conflict.
    assert run._addresses_collide(None, "127.0.0.1", 8889) is True


@pytest.mark.parametrize(
    "wildcard",
    ["0", "00", "0.0", "0.0.0", "::0", "0:0:0:0:0:0:0:0", "::ffff:0.0.0.0"],
)
def test_every_wildcard_alias_collides_with_a_specific_bind(wildcard):
    assert run._addresses_collide("127.0.0.1", wildcard, 8889) is True
    assert run._addresses_collide(wildcard, "127.0.0.1", 8889) is True


def test_a_hostname_resolves_the_same_way_the_bind_does(tmp_path):
    # `localhost` and the address _is_port_free actually binds must agree, or a
    # recorded server is missed and a duplicate starts.
    recorded = ",".join(sorted(run._bind_addresses("localhost", 8889)))

    assert run._addresses_collide(recorded, "localhost", 8889) is True


def test_a_hostname_records_every_address_it_resolves_to(tmp_path):
    # `localhost` binds 127.0.0.1 AND ::1. Recording only the first lets a later
    # launch on the other literal miss us and start a duplicate.
    addrs = run._bind_addresses("localhost", 8889)
    recorded = ",".join(sorted(addrs))

    for literal in addrs:
        assert run._addresses_collide(recorded, literal, 8889) is True


def test_port_probe_checks_every_resolved_bind_address(monkeypatch):
    bind_attempts = []
    sockets = []

    class _ProbeSocket:
        def __init__(self, family):
            self.family = family
            self.closed = False
            sockets.append(self)

        def setsockopt(self, *_args):
            pass

        def bind(self, sockaddr):
            bind_attempts.append((self.family, sockaddr))
            if self.family == socket.AF_INET6:
                raise OSError("address already in use")

        def close(self):
            self.closed = True

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("0.0.0.0", 8888)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::", 8888, 0, 0)),
        ],
    )
    monkeypatch.setattr(
        socket,
        "socket",
        lambda family, _socktype, _proto: _ProbeSocket(family),
    )

    assert run._is_port_free("dual-wildcard.test", 8888) is False
    assert bind_attempts == [
        (socket.AF_INET, ("0.0.0.0", 8888)),
        (socket.AF_INET6, ("::", 8888, 0, 0)),
    ]
    assert all(probe.closed for probe in sockets)


def test_a_multi_address_record_matches_either_literal(tmp_path):
    recorded = "127.0.0.1,::1"

    assert run._addresses_collide(recorded, "127.0.0.1", 8889) is True
    assert run._addresses_collide(recorded, "::1", 8889) is True
    assert run._addresses_collide("127.0.0.1", "::1", 8889) is False


def test_fallback_aborts_on_our_own_server_further_up_the_range(tmp_path, monkeypatch):
    # jupyter holds 8888, our server holds 8889: skipping to 8890 is the duplicate.
    (tmp_path / "studio-8889-8550.pid").write_text("8550\n\n127.0.0.1", encoding = "utf-8")
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: p >= 8890)

    with pytest.raises(SystemExit) as excinfo:
        run._find_free_port("127.0.0.1", 8889, avoid_own_studio = True)

    assert excinfo.value.code == 1


def test_fallback_still_skips_foreign_processes(tmp_path, monkeypatch):
    # No record for 8889, so the blocker is not ours: keep falling back.
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: p >= 8890)

    assert run._find_free_port("127.0.0.1", 8889, avoid_own_studio = True) == 8890


def test_the_requested_port_is_kept_when_it_is_free(monkeypatch):
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: True)

    assert run._resolve_port("127.0.0.1", 8888) == 8888


def test_our_own_server_on_the_requested_port_aborts_rather_than_falling_back(
    tmp_path, monkeypatch
):
    # The reported bug: 8888 is ours, so falling back to 8889 is the duplicate
    # that leaves 8888 serving with nothing recording it.
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: p != 8888)
    (tmp_path / "studio-8888-8550.pid").write_text("8550\n\n127.0.0.1", encoding = "utf-8")

    with pytest.raises(SystemExit) as excinfo:
        run._resolve_port("127.0.0.1", 8888)

    assert excinfo.value.code == 1


def test_a_foreign_process_on_the_requested_port_still_falls_back(monkeypatch):
    # jupyter-lab on 8888 must not stop Unsloth starting on 8889.
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: p != 8888)

    assert run._resolve_port("127.0.0.1", 8888) == 8889


def test_a_caller_that_reads_the_port_back_keeps_the_plain_fallback(tmp_path, monkeypatch):
    # api-only callers (the desktop app via TAURI_PORT, `studio run` via
    # app.state.server_port) follow us to the new port, so aborting there only
    # turns a working launch into a crash the desktop app reports as "stopped
    # unexpectedly". Both servers are still recorded, so `stop` finds them.
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: p != 8888)
    (tmp_path / "studio-8888-8550.pid").write_text("8550\n\n127.0.0.1", encoding = "utf-8")

    assert run._resolve_port("127.0.0.1", 8888, avoid_own_studio = False) == 8889


def test_the_recorded_address_is_every_address_the_bind_resolves_to(tmp_path):
    # The only test that runs the writer with a real host. Recording `host`
    # verbatim, or dropping the line, passes every other test here and silently
    # stops matching a launch that spells the same interface differently.
    run._write_pid_file(8901, "localhost")

    record = run._read_pid_record(tmp_path / f"studio-8901-{os.getpid()}.pid")

    assert record[2] is not None, "no bind address recorded"
    assert set(record[2].split(",")) == run._bind_addresses("localhost", 8901)


def test_a_server_started_on_a_hostname_is_found_again_by_ip(tmp_path):
    run._write_pid_file(8901, "localhost")

    for literal in run._bind_addresses("localhost", 8901):
        assert run._own_studio_on_port(8901, literal) == os.getpid()


def test_bind_addresses_keeps_every_family_a_hostname_resolves_to(monkeypatch):
    # Independent oracle: the sibling test derives its expectation from this
    # function's own output, so dropping a family would pass it.
    import socket
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 8889)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 8889, 0, 0)),
        ],
    )

    assert run._bind_addresses("localhost", 8889) == {"127.0.0.1", "::1"}


def test_the_legacy_file_is_written_even_when_the_per_port_record_fails(tmp_path, monkeypatch):
    # A studio root that cannot take a new entry used to leave the server
    # recorded nowhere at all, so the CLI could not stop it. studio.pid is an
    # overwrite of an existing path, so it can still succeed and must be tried.
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding = "utf-8")
    monkeypatch.setattr(
        run, "_pid_file_for_port", lambda port: blocked / f"studio-{port}-{os.getpid()}.pid"
    )

    run._write_pid_file(8901, "127.0.0.1")

    assert (tmp_path / "studio.pid").read_text(encoding = "utf-8") == str(os.getpid())
    assert run._OWN_PID_FILE is None


def test_a_record_whose_pid_is_not_ascii_digits_is_discarded(tmp_path):
    # A superscript two passes isdigit() but int() rejects it, so that gate alone
    # let a ValueError escape into every caller of _read_pid_record.
    (tmp_path / "r.pid").write_text("²", encoding = "utf-8")

    assert run._read_pid_record(tmp_path / "r.pid") is None


def test_the_legacy_file_is_not_taken_from_a_live_server(tmp_path):
    # A pre-upgrade server is recorded in studio.pid and nowhere else, so a
    # second launch overwriting it is exactly what strands it. That is the
    # orphan this file exists to prevent, reached from the other direction.
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    run._write_pid_file(8902, "127.0.0.1")

    assert (tmp_path / "studio.pid").read_text(encoding = "utf-8") == "8550"
    assert (tmp_path / f"studio-8902-{os.getpid()}.pid").exists()


def test_the_legacy_file_is_taken_over_from_a_dead_server(tmp_path, monkeypatch):
    # A stale record must not keep the pointer forever, or an older CLI could
    # never stop anything again.
    monkeypatch.setattr(run, "_pid_alive", lambda pid: False)
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    run._write_pid_file(8902, "127.0.0.1")

    assert (tmp_path / "studio.pid").read_text(encoding = "utf-8") == str(os.getpid())


def test_a_live_sibling_is_found_so_the_shared_cache_survives(tmp_path):
    # The compiled cache is install-tree relative, so a second backend of this
    # install must not wipe it out from under the first.
    (tmp_path / "studio-8888-8550.pid").write_text("8550\n\n127.0.0.1", encoding = "utf-8")

    assert run.live_sibling_backend() == 8550


def test_our_own_record_is_not_a_sibling(tmp_path):
    # _write_pid_file has already run by shutdown, so our own record is there.
    me = os.getpid()
    (tmp_path / f"studio-8888-{me}.pid").write_text(f"{me}\n\n127.0.0.1", encoding = "utf-8")

    assert run.live_sibling_backend() is None


def test_a_dead_record_is_not_a_sibling(tmp_path, monkeypatch):
    # A crashed server leaves its record behind; clearing the cache is right then.
    monkeypatch.setattr(run, "_pid_alive", lambda pid: False)
    (tmp_path / "studio-8888-8550.pid").write_text("8550\n\n127.0.0.1", encoding = "utf-8")

    assert run.live_sibling_backend() is None


def test_a_reused_pid_is_not_a_sibling(tmp_path, monkeypatch):
    # Alive, but not our server: the create_time no longer matches the record.
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created_times = (): False)
    (tmp_path / "studio-8888-8550.pid").write_text("8550\n1.0\n127.0.0.1", encoding = "utf-8")

    assert run.live_sibling_backend() is None


def test_a_sibling_that_is_still_binding_is_found(tmp_path):
    # The window Codex flagged: lifespan startup runs, and would clear the
    # cache, long before uvicorn reports a port for _write_pid_file to record.
    (tmp_path / "studio-starting-8550.marker").write_text("8550\n", encoding = "utf-8")

    assert run.live_sibling_backend() == 8550


def test_a_legacy_only_sibling_is_found(tmp_path):
    # A pre-upgrade server is recorded here and nowhere else, and so is one
    # whose best-effort per-port write failed.
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run.live_sibling_backend() == 8550


def test_a_dead_legacy_record_is_not_a_sibling(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_pid_alive", lambda pid: False)
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run.live_sibling_backend() is None


def test_the_startup_marker_is_written_and_removed(tmp_path):
    run.write_startup_marker()
    marker = tmp_path / f"studio-starting-{os.getpid()}.marker"

    assert marker.is_file()
    # Parsed by the same reader as a per-port record, with a real start time.
    record = run._read_pid_record(marker)
    assert record is not None and record[0] == os.getpid()

    run._remove_startup_marker()

    assert not marker.exists()
    assert run._OWN_STARTUP_MARKERS == []


def test_our_own_startup_marker_is_not_a_sibling(tmp_path):
    run.write_startup_marker()

    assert run.live_sibling_backend() is None

    run._remove_startup_marker()


def test_a_startup_marker_is_invisible_to_the_pid_file_glob(tmp_path):
    # Everything reading PID_FILE_GLOB expects a port in the name; a process
    # that has not bound yet has none, so `stop` and _legacy_heir must skip it.
    (tmp_path / "studio-starting-8550.marker").write_text("8550\n", encoding = "utf-8")

    assert list(tmp_path.glob(run.PID_FILE_GLOB)) == []


def test_the_marker_outlives_the_early_record_drop(tmp_path):
    # _graceful_shutdown drops the records before the server thread is joined,
    # while an in-flight request or a background warm may still be importing
    # from the cache. Going invisible there would let a replacement clear it
    # underneath, so the marker waits for the thread to actually end.
    run.write_startup_marker()
    marker = tmp_path / f"studio-starting-{os.getpid()}.marker"
    run._write_pid_file(8906, "127.0.0.1")

    run._remove_pid_file()

    assert marker.is_file(), "the backend is still serving until the thread ends"
    assert _files(tmp_path) == []

    run._remove_startup_marker()

    assert not marker.exists()


def test_a_bare_legacy_record_cannot_resurrect_a_reused_pid(tmp_path, monkeypatch):
    # studio.pid holds a PID and no start time, and an untimed record is trusted
    # unconditionally. A timestamped record for that same PID, written after it,
    # is proof the server is gone, so the bare one must not override it.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)
    legacy = tmp_path / "studio.pid"
    timed = tmp_path / "studio-8888-8550.pid"
    legacy.write_text("8550", encoding = "utf-8")
    timed.write_text("8550\n111.5\n127.0.0.1", encoding = "utf-8")
    os.utime(legacy, (100.0, 100.0))
    os.utime(timed, (200.0, 200.0))

    assert run.live_sibling_backend() is None


def test_an_older_timed_record_does_not_veto_a_newer_legacy_one(tmp_path, monkeypatch):
    # The other order is a pre-upgrade backend starting on a PID a crashed
    # current-version backend left a record for: it writes studio.pid and
    # nothing else. The older timestamp describes the process that is gone, so
    # letting it veto would clear the compiled cache under a serving backend.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)
    legacy = tmp_path / "studio.pid"
    timed = tmp_path / "studio-8888-8550.pid"
    timed.write_text("8550\n111.5\n127.0.0.1", encoding = "utf-8")
    legacy.write_text("8550", encoding = "utf-8")
    os.utime(timed, (100.0, 100.0))
    os.utime(legacy, (200.0, 200.0))

    assert run.live_sibling_backend() == 8550


def test_a_timed_record_for_another_pid_leaves_the_legacy_one_alone(tmp_path, monkeypatch):
    # Corroboration is per PID: a dead record for a different server says nothing
    # about the pre-upgrade one, which is recorded in studio.pid and nowhere else.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)
    (tmp_path / "studio-8888-8551.pid").write_text("8551\n111.5\n127.0.0.1", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("8550", encoding = "utf-8")

    assert run.live_sibling_backend() == 8550


def test_the_cache_lock_is_exclusive(tmp_path):
    # Two backends racing is the whole point, so the second one must be told the
    # section is taken rather than be let into it.
    with cache_cleanup.compiled_cache_lock() as first:
        assert first == cache_cleanup.LOCK_HELD
        with cache_cleanup.compiled_cache_lock(timeout = 0.05) as second:
            assert second == cache_cleanup.LOCK_BUSY

    with cache_cleanup.compiled_cache_lock() as again:
        assert again == cache_cleanup.LOCK_HELD


def test_the_probe_and_the_clear_happen_inside_one_lock(tmp_path, monkeypatch):
    # The race Codex flagged: our probe finds nobody, a sibling publishes and
    # starts compiling, and then our rmtree deletes what it just wrote.
    events = []

    @contextlib.contextmanager
    def _lock(timeout = None):
        events.append("lock")
        try:
            yield cache_cleanup.LOCK_HELD
        finally:
            events.append("unlock")

    monkeypatch.setattr(cache_cleanup, "compiled_cache_lock", _lock)
    monkeypatch.setattr(
        cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: events.append("clear")
    )

    cache_cleanup.clear_compiled_cache_unless_shared(lambda: events.append("probe"))

    assert events == ["lock", "probe", "clear", "unlock"]


def test_a_busy_cache_lock_keeps_the_cache_without_probing(tmp_path, monkeypatch):
    # Whoever holds it is a sibling by definition, so the answer is already known.
    events = []

    @contextlib.contextmanager
    def _lock(timeout = None):
        yield cache_cleanup.LOCK_BUSY

    monkeypatch.setattr(cache_cleanup, "compiled_cache_lock", _lock)
    monkeypatch.setattr(
        cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: events.append("clear")
    )

    cache_cleanup.clear_compiled_cache_unless_shared(lambda: events.append("probe"))

    assert events == []


def test_a_lock_that_cannot_be_taken_at_all_still_clears(tmp_path, monkeypatch):
    # An unwritable temp dir or a filesystem without flock must not mean the
    # cache is never cleared again on that machine; fall back to the plain probe.
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding = "utf-8")
    monkeypatch.setattr(cache_cleanup, "cache_coordination_dir", lambda: blocked / "lock")
    events = []
    monkeypatch.setattr(
        cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: events.append("clear")
    )

    with cache_cleanup.compiled_cache_lock() as state:
        assert state == cache_cleanup.LOCK_UNAVAILABLE

    cache_cleanup.clear_compiled_cache_unless_shared(lambda: None)

    assert events == ["clear"]


def test_a_live_sibling_keeps_the_compiled_cache(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: events.append("clear")
    )

    cache_cleanup.clear_compiled_cache_unless_shared(lambda: 8550)

    assert events == []


def test_no_sibling_probe_clears_unconditionally(tmp_path, monkeypatch):
    # An embedded app or a test never sets the probe, and the old behaviour stands.
    events = []
    monkeypatch.setattr(
        cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: events.append("clear")
    )

    cache_cleanup.clear_compiled_cache_unless_shared(None)

    assert events == ["clear"]


def test_the_startup_marker_is_published_under_the_cache_lock(tmp_path, monkeypatch):
    # Publication has to be inside the same section as a sibling's probe and
    # clear, or the marker lands in the gap between them and is clear-and-lost.
    marker = tmp_path / f"studio-starting-{os.getpid()}.marker"
    events = []

    @contextlib.contextmanager
    def _lock(timeout = None):
        events.append(("lock", marker.exists()))
        try:
            yield cache_cleanup.LOCK_HELD
        finally:
            events.append(("unlock", marker.exists()))

    monkeypatch.setattr(cache_cleanup, "compiled_cache_lock", _lock)

    run.write_startup_marker()

    assert events == [("lock", False), ("unlock", True)]


def test_a_zombie_backend_is_not_a_live_sibling(tmp_path, monkeypatch):
    # A backend that exited under a non-reaping parent stays a zombie: it
    # answers kill(0) and keeps its start time, so liveness alone reads it as
    # serving and its record would pin the compiled cache forever.
    (tmp_path / "studio-8888-4242.pid").write_text("4242\n111.5\n127.0.0.1", encoding = "utf-8")

    monkeypatch.setattr(run, "_pid_alive", lambda pid: True)
    assert run.live_sibling_backend() == 4242, "the check under test must start from live"

    import utils.process_lifetime as process_lifetime

    monkeypatch.setattr(process_lifetime, "_pid_is_zombie", lambda pid: pid == 4242)

    assert run.live_sibling_backend() is None


def test_a_real_unreaped_child_is_not_a_live_sibling(tmp_path):
    # The same thing without a patch: a killed but unwaited child.
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        child.kill()
        (tmp_path / f"studio-8889-{child.pid}.pid").write_text(f"{child.pid}\n", encoding = "utf-8")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if run.live_sibling_backend() is None:
                break
            time.sleep(0.05)

        assert run.live_sibling_backend() is None, "a zombie was taken for a serving backend"
    finally:
        child.wait()


def test_every_record_call_can_be_repeated(tmp_path):
    # Startup and shutdown both run more than once in a long-lived embedded
    # host, and a crash can leave either half done. Each of these has to be
    # safe to call again on the state the previous call left.
    run.write_startup_marker()
    marker = tmp_path / f"studio-starting-{os.getpid()}.marker"
    published = sorted(p.name for p in tmp_path.iterdir())
    body = marker.read_text(encoding = "utf-8")

    run.write_startup_marker()

    assert sorted(p.name for p in tmp_path.iterdir()) == published, "a second publish added a file"
    assert marker.read_text(encoding = "utf-8") == body
    assert run._OWN_STARTUP_MARKERS == [marker], "tracked twice would be removed twice"

    run._write_pid_file(8888, "127.0.0.1")
    recorded = sorted(p.name for p in tmp_path.iterdir())
    run._write_pid_file(8888, "127.0.0.1")

    assert sorted(p.name for p in tmp_path.iterdir()) == recorded

    run._remove_startup_marker()
    run._remove_startup_marker()
    run._remove_pid_file()
    run._remove_pid_file()

    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) == []
    assert _files(tmp_path) == []
    assert run._OWN_STARTUP_MARKERS == []


def test_reading_the_siblings_changes_nothing_on_disk(tmp_path):
    # The probe runs on every startup and is a read. If it pruned or rewrote
    # anything, one install's startup would edit another's records.
    (tmp_path / "studio-8888-4242.pid").write_text("4242\n111.5\n127.0.0.1", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("4243", encoding = "utf-8")
    before = {p.name: p.read_text(encoding = "utf-8") for p in tmp_path.iterdir()}

    for _ in range(3):
        run.live_sibling_backend()

    assert {p.name: p.read_text(encoding = "utf-8") for p in tmp_path.iterdir()} == before


def test_a_busy_lock_publishes_then_waits_for_the_clear_to_finish(tmp_path, monkeypatch):
    # Publishing unlocked keeps this backend visible to sibling probes, but
    # returning straight away would let the caller import and compile into a
    # cache the holder is still deleting. The marker goes out first, then the
    # wait, so both halves hold.
    marker = tmp_path / f"studio-starting-{os.getpid()}.marker"
    timeouts = []

    @contextlib.contextmanager
    def _lock(timeout = None):
        timeouts.append((timeout, marker.exists()))
        # Busy the first time (the publish), free the second (the wait).
        yield cache_cleanup.LOCK_BUSY if len(timeouts) == 1 else cache_cleanup.LOCK_HELD

    monkeypatch.setattr(cache_cleanup, "compiled_cache_lock", _lock)

    run.write_startup_marker()

    assert timeouts == [
        (30.0, False),
        (run._CACHE_CLEAR_WAIT_SECONDS, True),
    ], "the wait must come after the marker is on disk"


def test_a_lock_taken_at_once_does_not_wait(tmp_path, monkeypatch):
    # Nobody is clearing, so there is nothing to wait for.
    calls = []

    @contextlib.contextmanager
    def _lock(timeout = None):
        calls.append(timeout)
        yield cache_cleanup.LOCK_HELD

    monkeypatch.setattr(cache_cleanup, "compiled_cache_lock", _lock)

    run.write_startup_marker()

    assert calls == [30.0]


def test_a_clear_that_never_finishes_still_lets_the_backend_start(tmp_path, monkeypatch):
    # Refusing to start is worse for the user than starting on a cache that may
    # lose files, so the wait is bounded and startup continues past it.
    @contextlib.contextmanager
    def _lock(timeout = None):
        yield cache_cleanup.LOCK_BUSY

    monkeypatch.setattr(cache_cleanup, "compiled_cache_lock", _lock)
    monkeypatch.setattr(run, "_CACHE_CLEAR_WAIT_SECONDS", 0.0)

    run.write_startup_marker()

    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) != []


def test_a_filesystem_that_cannot_lock_is_not_read_as_contention(tmp_path, monkeypatch):
    # ENOSYS is the lock being unsupported, not a sibling holding it. Retrying it
    # to a timeout and answering busy would keep the cache forever, since busy is
    # read as proof of a sibling.
    def unsupported(fd):
        raise OSError(errno.ENOSYS, "flock not supported")

    monkeypatch.setattr(cache_cleanup, "_try_lock", unsupported)
    events = []
    monkeypatch.setattr(
        cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: events.append("clear")
    )

    with cache_cleanup.compiled_cache_lock(timeout = 30.0) as state:
        assert state == cache_cleanup.LOCK_UNAVAILABLE

    cache_cleanup.clear_compiled_cache_unless_shared(lambda: None)

    assert events == ["clear"]


def test_contention_is_still_retried_to_the_timeout(tmp_path, monkeypatch):
    def busy(fd):
        raise OSError(errno.EWOULDBLOCK, "held")

    monkeypatch.setattr(cache_cleanup, "_try_lock", busy)

    started = time.monotonic()
    with cache_cleanup.compiled_cache_lock(timeout = 0.2) as state:
        assert state == cache_cleanup.LOCK_BUSY
    assert time.monotonic() - started >= 0.2, "contention waits out the budget"


def test_a_startup_that_raises_takes_its_marker_back(tmp_path, monkeypatch):
    # colab.py catches SystemExit and Exception around run_server and keeps the
    # process alive, so no exit hook runs. A marker left behind would answer
    # every later probe as a live backend of this install.
    run.write_startup_marker()
    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) != []

    def explode(*args, **kwargs):
        raise RuntimeError("startup failed")

    monkeypatch.setattr(run, "run_server", run._drops_its_marker_on_failure(explode))

    with pytest.raises(RuntimeError):
        run.run_server()

    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) == []
    assert run._OWN_STARTUP_MARKERS == []


def test_a_startup_that_exits_takes_its_marker_back(tmp_path, monkeypatch):
    # SystemExit is a BaseException, and it is the one colab.py catches.
    run.write_startup_marker()

    def bail(*args, **kwargs):
        raise SystemExit(1)

    monkeypatch.setattr(run, "run_server", run._drops_its_marker_on_failure(bail))

    with pytest.raises(SystemExit):
        run.run_server()

    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) == []


def test_an_interrupt_leaves_the_marker_to_a_live_server_thread(tmp_path, monkeypatch):
    # The KeyboardInterrupt path asks uvicorn to stop and re-raises without
    # joining, so the thread may still be finishing lifespan startup or
    # shutdown. Taking the marker back there makes a backend that is still up
    # invisible to a sibling, which would clear the compiled cache under it.
    run.write_startup_marker()
    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) != []

    stop = threading.Event()
    serving = threading.Thread(target = stop.wait, daemon = True)
    serving.start()
    monkeypatch.setattr(run, "_server_thread", serving)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "run_server", run._drops_its_marker_on_failure(interrupt))
    try:
        with pytest.raises(KeyboardInterrupt):
            run.run_server()

        assert (
            list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) != []
        ), "the sibling can no longer see it"
        assert run._OWN_STARTUP_MARKERS != []
    finally:
        stop.set()
        serving.join(timeout = 5)

    # Once that thread is gone, the same failure does take the marker back:
    # nothing else is going to.
    with pytest.raises(KeyboardInterrupt):
        run.run_server()

    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) == []


def test_a_server_thread_that_dies_after_readiness_drops_its_records(tmp_path, monkeypatch):
    # An embedded host stays alive after the uvicorn thread ends, and a failure
    # in there never reaches run_server's caller. Records left behind would keep
    # validating against the still-live host PID with no backend serving. Both
    # calls are what the thread's finally block makes.
    run.write_startup_marker()
    run._write_pid_file(8905, "127.0.0.1")

    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) != []
    assert _files(tmp_path) != []

    run._remove_startup_marker()
    run._remove_pid_file()

    assert list(tmp_path.glob(run.STARTUP_MARKER_GLOB)) == []
    assert _files(tmp_path) == []


def test_two_cold_starts_keep_rather_than_delete_each_others_modules(tmp_path, monkeypatch):
    # The documented limitation of scoping this back: both keep a cache neither
    # cleaned, which is the safe direction. The failure being replaced is the two
    # of them deleting each other's modules mid-run.
    events = []
    monkeypatch.setattr(
        cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: events.append("clear")
    )

    cache_cleanup.clear_compiled_cache_unless_shared(lambda: 8550)
    cache_cleanup.clear_compiled_cache_unless_shared(lambda: 8551)

    assert events == []

    # ...and a launch that really is alone still clears.
    cache_cleanup.clear_compiled_cache_unless_shared(lambda: None)

    assert events == ["clear"]


def test_a_transient_unlink_failure_is_retried_on_the_spot(tmp_path, monkeypatch):
    # A holder that lets go a moment later (an indexer on Windows) must not cost
    # the marker its removal: waiting for the atexit hook leaves it on disk, and
    # an embedded host that outlives the backend may not reach that for hours.
    run.write_startup_marker()
    marker = tmp_path / f"studio-starting-{os.getpid()}.marker"
    assert run._OWN_STARTUP_MARKERS == [marker]

    real_unlink = Path.unlink
    refused = []

    def refuse_once(self, missing_ok = False):
        if not refused:
            refused.append(self)
            raise OSError("busy")
        return real_unlink(self, missing_ok = missing_ok)

    monkeypatch.setattr(Path, "unlink", refuse_once)
    monkeypatch.setattr(run, "_MARKER_REMOVAL_BACKOFF_SECONDS", 0.0)
    run._remove_startup_marker()

    assert refused, "the first attempt did not run"
    assert run._OWN_STARTUP_MARKERS == []
    assert not marker.exists()


def test_a_marker_that_will_not_delete_stays_retryable(tmp_path, monkeypatch):
    # Once the retries run out, dropping the path loses the only reference to
    # it. The marker would stay on disk, keep matching this live process, and
    # pin the compiled cache with no later cleanup able to retry.
    run.write_startup_marker()
    marker = tmp_path / f"studio-starting-{os.getpid()}.marker"
    assert run._OWN_STARTUP_MARKERS == [marker]

    real_unlink = Path.unlink
    attempts = []

    def refuse_while_locked(self, missing_ok = False):
        attempts.append(self)
        if locked:
            raise OSError("busy")
        return real_unlink(self, missing_ok = missing_ok)

    locked = True
    monkeypatch.setattr(Path, "unlink", refuse_while_locked)
    monkeypatch.setattr(run, "_MARKER_REMOVAL_BACKOFF_SECONDS", 0.0)
    run._remove_startup_marker()

    assert len(attempts) == run._MARKER_REMOVAL_ATTEMPTS, "the retries did not run"
    assert run._OWN_STARTUP_MARKERS == [marker], "still tracked for a retry"
    assert marker.exists()

    locked = False
    run._remove_startup_marker()

    assert run._OWN_STARTUP_MARKERS == []
    assert not marker.exists()
