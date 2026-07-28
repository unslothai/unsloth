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
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created = None: True)
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


def test_recorded_records_read_per_port_and_legacy_files(tmp_path):
    (tmp_path / "studio-8901-8550.pid").write_text("8550\n111.5", encoding = "utf-8")
    (tmp_path / "studio-8902-8600.pid").write_text("8600", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("4242", encoding = "utf-8")

    assert run._recorded_studio_records() == {8550: 111.5, 8600: None, 4242: None}


def test_recorded_records_ignore_corrupt_files(tmp_path):
    (tmp_path / "studio-8901-x.pid").write_text("not-a-pid", encoding = "utf-8")

    assert run._recorded_studio_records() == {}


def test_recorded_records_prune_dead_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_pid_alive", lambda pid: False)
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._recorded_studio_records() == {}
    assert not (tmp_path / "studio-8901-8550.pid").exists()


def test_own_studio_blocking_the_port_is_recognised(tmp_path):
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._blocker_is_own_studio([(8550, "python")]) == 8550


def test_a_foreign_blocker_still_falls_back(tmp_path):
    # jupyter-lab on 8888 must keep the fallback, not abort the launch.
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._blocker_is_own_studio([(117, "jupyter-lab")]) is None
    assert run._blocker_is_own_studio([]) is None


def test_our_studio_is_found_behind_a_foreign_listener(tmp_path):
    # One port, two bind addresses: checking only the first listener made the
    # decision depend on psutil's ordering.
    (tmp_path / "studio-8889-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._blocker_is_own_studio([(117, "jupyter-lab"), (8550, "python")]) == 8550


def test_a_reused_pid_is_not_treated_as_our_studio(tmp_path, monkeypatch):
    # Stale record + the OS handing that PID to something else must not abort.
    monkeypatch.setattr(run, "_pid_is_studio_backend", lambda pid, created = None: False)
    (tmp_path / "studio-8901-8550.pid").write_text("8550", encoding = "utf-8")

    assert run._blocker_is_own_studio([(8550, "postgres")]) is None


def test_start_time_mismatch_rejects_a_reused_pid(monkeypatch):
    monkeypatch.setattr(run, "_pid_is_studio_backend", _REAL_IS_STUDIO_BACKEND)
    monkeypatch.setattr(run, "_process_create_time", lambda pid: 999.0)

    assert run._pid_is_studio_backend(8550, created = 111.5) is False
    assert run._pid_is_studio_backend(8550, created = 999.0) is True


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


def test_fallback_aborts_on_our_own_server_further_up_the_range(tmp_path, monkeypatch):
    # jupyter holds 8888, our server holds 8889: skipping to 8890 is the duplicate.
    (tmp_path / "studio-8889-8550.pid").write_text("8550", encoding = "utf-8")
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: p >= 8890)
    monkeypatch.setattr(
        run,
        "_get_pids_on_port",
        lambda p: [(8550, "python")] if p == 8889 else [(117, "jupyter-lab")],
    )

    with pytest.raises(SystemExit) as excinfo:
        run._find_free_port("127.0.0.1", 8889, avoid_own_studio = True)

    assert excinfo.value.code == 1


def test_fallback_still_skips_foreign_processes(monkeypatch):
    monkeypatch.setattr(run, "_is_port_free", lambda host, p: p >= 8890)
    monkeypatch.setattr(run, "_get_pids_on_port", lambda p: [(117, "jupyter-lab")])

    assert run._find_free_port("127.0.0.1", 8889, avoid_own_studio = True) == 8890
