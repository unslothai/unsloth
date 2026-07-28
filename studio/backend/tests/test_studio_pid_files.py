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

# Captured before the autouse fixture stubs it, for the tests that exercise it.
_REAL_IS_STUDIO_BACKEND = run._pid_is_studio_backend


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


def test_remove_pid_file_only_removes_our_own(tmp_path):
    run._write_pid_file(8901)
    (tmp_path / "studio-8902-8600.pid").write_text("8600", encoding = "utf-8")

    run._remove_pid_file()

    assert _files(tmp_path) == ["studio-8902-8600.pid"]
    assert not (tmp_path / "studio.pid").exists()


def test_remove_pid_file_leaves_a_reused_entry_alone(tmp_path):
    run._write_pid_file(8901)
    own = tmp_path / f"studio-8901-{os.getpid()}.pid"
    own.write_text("999999", encoding = "utf-8")

    run._remove_pid_file()

    assert own.read_text(encoding = "utf-8") == "999999"


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
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created_times = (): None)
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


def test_legacy_records_match_an_in_process_studio(monkeypatch):
    # The in-venv path calls run_server() in-process, so argv is `unsloth studio`
    # with no run.py. Rejecting it would strand a running server.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return ["/root/.unsloth/studio/unsloth_studio/bin/unsloth", "studio", "-p", "8901"]

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert run._pid_is_studio_backend(8550) is True


def test_legacy_records_do_not_match_a_training_run(monkeypatch):
    # No start time recorded: the cmdline check must not accept `unsloth train`.
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)

    class _FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            if self.pid == 8550:
                return ["python", "/pkg/studio/backend/run.py", "--port", "8901"]
            return ["python", "-m", "unsloth", "train", "run.py"]

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process = _FakeProcess))

    assert run._pid_is_studio_backend(8550) is True
    assert run._pid_is_studio_backend(9999) is False


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
