# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic tests for the in-app llama.cpp update orchestration.

No network, no real install: the GitHub release lookup and the installer
subprocess are both monkeypatched. Verifies detection (update_available) and
the apply flow (job lifecycle, installer invocation, post-swap re-read).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.llama_cpp_freshness as freshness  # noqa: E402
import utils.llama_cpp_update as upd  # noqa: E402

MARKER = "UNSLOTH_PREBUILT_INFO.json"


def _write_install(
    dir_: Path,
    tag: str,
    repo: str = "unslothai/llama.cpp",
) -> str:
    """Create a fake prebuilt install tree and return the llama-server path."""
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("#!/bin/sh\necho stub\n")
    (dir_ / MARKER).write_text(
        json.dumps(
            {
                "tag": tag,
                "release_tag": tag,
                "published_repo": repo,
                "installed_at_utc": "2020-01-01T00:00:00Z",
                "bundle_profile": "cuda13-newer",
                "runtime_line": "cuda13",
            }
        )
    )
    return str(binary)


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch):
    freshness.reset_caches()
    upd._reset_job_for_tests()
    # Never hit the network in these tests.
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    yield
    freshness.reset_caches()
    upd._reset_job_for_tests()


def test_status_no_marker(monkeypatch, tmp_path):
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")  # no marker file alongside
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    st = upd.get_update_status()
    assert st["supported"] is False
    assert st["update_available"] is False
    assert st["installed_tag"] is None


def test_status_update_available(monkeypatch, tmp_path):
    binary = _write_install(tmp_path, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    st = upd.get_update_status(force_refresh = True)
    assert st["supported"] is True
    assert st["installed_tag"] == "b9493"
    assert st["latest_tag"] == "b9518"
    assert st["update_available"] is True


def test_status_up_to_date(monkeypatch, tmp_path):
    binary = _write_install(tmp_path, "b9518")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    st = upd.get_update_status(force_refresh = True)
    assert st["installed_tag"] == "b9518"
    assert st["latest_tag"] == "b9518"
    assert st["update_available"] is False


def test_start_update_no_marker_refuses(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("stub")  # no marker
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "no_prebuilt_marker"


def test_start_update_happy_path(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    captured = {}

    class _Proc:
        returncode = 0
        stdout = "installed"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Simulate the installer writing a new marker with the latest tag.
        _write_install(install_dir, "b9518")
        return _Proc()

    monkeypatch.setattr(upd.subprocess, "run", _fake_run)

    res = upd.start_update()
    assert res["started"] is True
    assert res["job"]["from_tag"] == "b9493"

    # Wait for the background worker.
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "success", job
    assert job["to_tag"] == "b9518"
    # Installer was invoked with the resolved install dir + latest + repo.
    assert "--install-dir" in captured["cmd"]
    assert str(install_dir) in captured["cmd"]
    assert "--llama-tag" in captured["cmd"] and "latest" in captured["cmd"]
    assert "unslothai/llama.cpp" in captured["cmd"]


def test_start_update_installer_failure_reports_error(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")

    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "boom: network error"

    monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **kw: _Proc())

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "error"
    assert "boom" in (job["error"] or "")
