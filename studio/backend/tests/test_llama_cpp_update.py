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
from types import ModuleType

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
    asset: str | None = None,
) -> str:
    """Create a fake prebuilt install tree and return the llama-server path.

    ``asset`` is the bundle filename recorded in the marker; omit it to model an
    older marker that predates asset-based ROCm forwarding (backward compat)."""
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("#!/bin/sh\necho stub\n")
    marker = {
        "tag": tag,
        "release_tag": tag,
        "published_repo": repo,
        "installed_at_utc": "2020-01-01T00:00:00Z",
        "bundle_profile": "cuda13-newer",
        "runtime_line": "cuda13",
    }
    if asset is not None:
        marker["asset"] = asset
    (dir_ / MARKER).write_text(json.dumps(marker))
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


# --- installer-argument construction (mirrors the post-#5963 setup scripts) ---


def test_rocm_install_args_lemonade_gfx():
    # Lemonade HIP app bundle: gfx family lives in the asset name.
    assert upd._rocm_install_args("app-b9585-linux-x64-rocm-gfx110X.tar.gz") == [
        "--rocm-gfx",
        "gfx110x",
    ]
    assert upd._rocm_install_args("app-b9585-windows-x64-rocm-gfx1150.zip") == [
        "--rocm-gfx",
        "gfx1150",
    ]


def test_rocm_install_args_fork_version_bundle():
    # Fork ROCm bundles encode a ROCm version, not a gfx -> forward --has-rocm.
    assert upd._rocm_install_args("llama-b9334-bin-ubuntu-rocm-6.4-x64.tar.gz") == ["--has-rocm"]


def test_rocm_install_args_windows_hip():
    assert upd._rocm_install_args("llama-b9334-bin-win-hip-radeon-x64.zip") == ["--has-rocm"]


def test_rocm_install_args_non_rocm_and_missing():
    assert upd._rocm_install_args("llama-b9334-bin-ubuntu-x64.tar.gz") == []
    assert upd._rocm_install_args("app-b9585-linux-x64-cuda13.tar.gz") == []
    assert upd._rocm_install_args(None) == []


def _capture_install_cmd(
    monkeypatch,
    tmp_path,
    *,
    tag = "b9493",
    repo = "unslothai/llama.cpp",
    asset = None,
    latest = "b9518",
) -> list:
    """Run start_update() with the installer subprocess stubbed; return the argv."""
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, tag, repo = repo, asset = asset)
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: latest)

    captured = {}

    class _Proc:
        returncode = 0
        stdout = "installed"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        _write_install(install_dir, latest, repo = repo, asset = asset)
        return _Proc()

    monkeypatch.setattr(upd.subprocess, "run", _fake_run)

    res = upd.start_update()
    assert res["started"] is True, res
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    return captured.get("cmd", [])


def test_install_cmd_rocm_marker_forwards_gfx(monkeypatch, tmp_path):
    cmd = _capture_install_cmd(
        monkeypatch, tmp_path, asset = "app-b9585-linux-x64-rocm-gfx110X.tar.gz"
    )
    assert "--rocm-gfx" in cmd
    assert cmd[cmd.index("--rocm-gfx") + 1] == "gfx110x"
    assert "--has-rocm" not in cmd
    assert "--cpu-fallback" not in cmd
    assert "--simple-policy" not in cmd
    assert "--published-repo" in cmd and "unslothai/llama.cpp" in cmd


def test_install_cmd_fork_rocm_marker_forwards_has_rocm(monkeypatch, tmp_path):
    cmd = _capture_install_cmd(
        monkeypatch, tmp_path, asset = "llama-b9334-bin-ubuntu-rocm-6.4-x64.tar.gz"
    )
    assert "--has-rocm" in cmd
    assert "--rocm-gfx" not in cmd


def test_install_cmd_ggml_cpu_marker_has_no_cpu_fallback(monkeypatch, tmp_path):
    # CPU installs come from ggml-org. Re-running into the same install-dir/repo
    # reproduces the same CPU bundle; --cpu-fallback (which force-drops GPU
    # detection) is reserved for setup.sh's arm64 rescue and must not appear here.
    cmd = _capture_install_cmd(
        monkeypatch,
        tmp_path,
        repo = "ggml-org/llama.cpp",
        asset = "llama-b9334-bin-ubuntu-x64.tar.gz",
    )
    assert "--cpu-fallback" not in cmd
    assert "--rocm-gfx" not in cmd
    assert "--has-rocm" not in cmd
    assert "--simple-policy" not in cmd
    assert "--published-repo" in cmd and "ggml-org/llama.cpp" in cmd


def test_install_cmd_cuda_marker_minimal_and_backward_compatible(monkeypatch, tmp_path):
    # Marker without an asset field (older install): no ROCm flags, no crash, and
    # never the obsolete --simple-policy that #5963 removed from setup.
    cmd = _capture_install_cmd(monkeypatch, tmp_path, asset = None)
    assert "--simple-policy" not in cmd
    assert "--rocm-gfx" not in cmd
    assert "--has-rocm" not in cmd
    assert "--cpu-fallback" not in cmd


# --- refusal + maintenance-state coordination ---


def test_start_update_already_running_refuses(monkeypatch, tmp_path):
    binary = _write_install(tmp_path / "llama.cpp", "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    with upd._job_lock:
        upd._job.update(state = upd._JOB_RUNNING)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "already_running"


def test_start_update_installer_missing_refuses(monkeypatch, tmp_path):
    binary = _write_install(tmp_path / "llama.cpp", "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: None)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "installer_missing"


class _FakeBackend:
    """Minimal stand-in for LlamaCppBackend's update-coordination surface."""

    def __init__(self):
        import threading

        self._serial_load_lock = threading.Lock()
        self._llama_update_in_progress = False
        self.is_active = True
        self.unloaded = False

    def unload_model(self):
        self.unloaded = True


def _inject_backend(monkeypatch, backend):
    routes_pkg = ModuleType("routes")
    routes_pkg.__path__ = []
    inference_mod = ModuleType("routes.inference")
    inference_mod.get_llama_cpp_backend = lambda: backend
    monkeypatch.setitem(sys.modules, "routes", routes_pkg)
    monkeypatch.setitem(sys.modules, "routes.inference", inference_mod)


def test_update_sets_maintenance_flag_and_unloads(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    backend = _FakeBackend()
    _inject_backend(monkeypatch, backend)

    seen = {}

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        # The maintenance flag must be set while the installer runs.
        seen["flag_during_install"] = backend._llama_update_in_progress
        _write_install(install_dir, "b9518")
        return _Proc()

    monkeypatch.setattr(upd.subprocess, "run", _fake_run)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)

    assert backend.unloaded is True
    assert seen.get("flag_during_install") is True
    # Cleared in the finally so model loads work again after the swap.
    assert backend._llama_update_in_progress is False


def test_update_clears_maintenance_flag_on_installer_failure(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")

    backend = _FakeBackend()
    _inject_backend(monkeypatch, backend)

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **kw: _Proc())

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert upd.get_update_status()["job"]["state"] == "error"
    assert backend._llama_update_in_progress is False


def test_update_fails_open_when_backend_unavailable(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    def _raise():
        raise RuntimeError("no backend")

    inference_mod = ModuleType("routes.inference")
    inference_mod.get_llama_cpp_backend = lambda: _raise()
    routes_pkg = ModuleType("routes")
    routes_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "routes", routes_pkg)
    monkeypatch.setitem(sys.modules, "routes.inference", inference_mod)

    class _Proc:
        returncode = 0
        stdout = "ok"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        _write_install(install_dir, "b9518")
        return _Proc()

    monkeypatch.setattr(upd.subprocess, "run", _fake_run)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "success", job
