# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-port PID files, so `unsloth studio stop` can find every server.

Imports run.py directly, so run under the Unsloth venv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import run  # noqa: E402

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
