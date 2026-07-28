# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-port PID files, so `unsloth studio stop` can find every server.

Imports run.py directly, so run under the Unsloth venv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import run  # noqa: E402


@pytest.fixture(autouse = True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_studio_root", lambda: tmp_path)
    monkeypatch.setattr(run, "_PID_FILE", tmp_path / "studio.pid")
    monkeypatch.setattr(run, "_OWN_PID_FILE", None)
    yield


def test_write_pid_file_is_per_port(tmp_path):
    run._write_pid_file(8901)

    path = tmp_path / "studio-8901.pid"
    assert path.read_text(encoding = "utf-8") == str(os.getpid())


def test_second_port_does_not_clobber_the_first(tmp_path):
    (tmp_path / "studio-8901.pid").write_text("8550", encoding = "utf-8")

    run._write_pid_file(8902)

    assert (tmp_path / "studio-8901.pid").read_text(encoding = "utf-8") == "8550"
    assert (tmp_path / "studio-8902.pid").read_text(encoding = "utf-8") == str(os.getpid())


def test_remove_pid_file_only_removes_our_own(tmp_path):
    run._write_pid_file(8901)
    (tmp_path / "studio-8902.pid").write_text("8600", encoding = "utf-8")

    run._remove_pid_file()

    assert not (tmp_path / "studio-8901.pid").exists()
    assert (tmp_path / "studio-8902.pid").exists()


def test_remove_pid_file_leaves_a_reused_entry_alone(tmp_path):
    run._write_pid_file(8901)
    (tmp_path / "studio-8901.pid").write_text("999999", encoding = "utf-8")

    run._remove_pid_file()

    assert (tmp_path / "studio-8901.pid").read_text(encoding = "utf-8") == "999999"


def test_recorded_studio_pids_reads_per_port_and_legacy_files(tmp_path):
    (tmp_path / "studio-8901.pid").write_text("8550", encoding = "utf-8")
    (tmp_path / "studio-8902.pid").write_text("8600", encoding = "utf-8")
    (tmp_path / "studio.pid").write_text("4242", encoding = "utf-8")

    assert run._recorded_studio_pids() == {8550, 8600, 4242}


def test_recorded_studio_pids_ignores_corrupt_files(tmp_path):
    (tmp_path / "studio-8901.pid").write_text("not-a-pid", encoding = "utf-8")

    assert run._recorded_studio_pids() == set()


def test_own_studio_blocking_the_port_is_recognised(tmp_path):
    (tmp_path / "studio-8901.pid").write_text("8550", encoding = "utf-8")

    assert run._blocker_is_own_studio((8550, "python")) is True


def test_a_foreign_blocker_still_falls_back(tmp_path):
    # jupyter-lab on 8888 must keep the fallback, not abort the launch.
    (tmp_path / "studio-8901.pid").write_text("8550", encoding = "utf-8")

    assert run._blocker_is_own_studio((117, "jupyter-lab")) is False
    assert run._blocker_is_own_studio(None) is False
