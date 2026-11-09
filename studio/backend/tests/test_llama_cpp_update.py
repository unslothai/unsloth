# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic llama.cpp update tests: the GitHub release lookup and installer subprocess are faked."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.llama_cpp_freshness as freshness  # noqa: E402
import utils.llama_cpp_update as upd  # noqa: E402
import utils.process_lifetime as process_lifetime  # noqa: E402


class _Proc:
    returncode = 0
    stdout = "installed"
    stderr = ""


MARKER = "UNSLOTH_PREBUILT_INFO.json"


class _FakeInstallerPopen:
    """Stands in for the streamed installer process in _run_update."""

    def __init__(
        self,
        cmd,
        *,
        returncode = 0,
        lines = None,
        on_start = None,
        captured_kwargs = None,
        **kwargs,
    ):
        if captured_kwargs is not None:
            captured_kwargs.update(kwargs)
        if on_start is not None:
            on_start(list(cmd))
        self.pid = 515151
        self.returncode = returncode
        self.stdout = iter(lines or [])

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        pass


# Bound before any patch, so a delegated spawn reaches the real one.
_REAL_POPEN = subprocess.Popen


def _installer_command(cmd) -> bool:
    """True for installs only; the script's --resolve-prebuilt/--resolve-backends probes must not match."""
    parts = [str(part) for part in cmd]
    if not any("install_llama_prebuilt" in part for part in parts):
        return False
    return not any(part.startswith("--resolve-") for part in parts)


def _patch_installer_popen(
    monkeypatch,
    *,
    returncode = 0,
    lines = None,
    on_start = None,
    captured_kwargs = None,
    spawned = None,
):
    """Patches Popen for the installer only, delegating the rest so the pid identity lookup keeps
    working."""

    def _popen(cmd, **kw):
        if spawned is not None:
            spawned.append(list(cmd))
        if not _installer_command(cmd):
            return _REAL_POPEN(cmd, **kw)
        return _FakeInstallerPopen(
            cmd,
            returncode = returncode,
            lines = lines,
            on_start = on_start,
            captured_kwargs = captured_kwargs,
            **kw,
        )

    monkeypatch.setattr(upd.subprocess, "Popen", _popen)


def _write_install(
    dir_: Path,
    tag: str,
    repo: str = "unslothai/llama.cpp",
    asset: str | None = None,
    release_tag: str | None = None,
    force_cpu: bool | None = None,
    llama_backend: str | None = None,
    include_llama_backend: bool = False,
) -> str:
    """Create a fake prebuilt install and return the llama-server path."""
    bin_dir = dir_ / "build" / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    binary = bin_dir / "llama-server"
    binary.write_text("#!/bin/sh\necho stub\n")
    marker = {
        "tag": tag,
        "release_tag": release_tag or tag,
        "published_repo": repo,
        "installed_at_utc": "2020-01-01T00:00:00Z",
        "bundle_profile": "cuda13-newer",
        "runtime_line": "cuda13",
    }
    if asset is not None:
        marker["asset"] = asset
    if force_cpu is not None:
        marker["force_cpu"] = force_cpu
    if include_llama_backend or llama_backend is not None:
        marker["llama_backend"] = llama_backend
    (dir_ / MARKER).write_text(json.dumps(marker))
    return str(binary)


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch, tmp_path):
    freshness.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    upd._backends_memo.clear()
    # Isolate the freshness disk cache so tests never write the real ~/.unsloth cache.
    monkeypatch.setattr(freshness, "_cache_dir", lambda: tmp_path / ".freshness_cache")
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    monkeypatch.setattr(upd, "_whisper_chain_status", lambda **kwargs: None)
    yield
    freshness.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    upd._backends_memo.clear()


def _no_prebuilt(monkeypatch):
    """Stub the host prebuilt probe to 'none available' (no source-build offer)."""
    monkeypatch.setattr(upd, "_resolve_prebuilt_for_host", lambda *, force_refresh = False: None)


def _prebuilt(
    monkeypatch,
    *,
    repo = "unslothai/llama.cpp",
    release_tag = "b9585",
    llama_tag = None,
    asset = None,
):
    """Stub the host prebuilt probe to report an available prebuilt."""
    payload = {
        "prebuilt_available": True,
        "repo": repo,
        "release_tag": release_tag,
        "llama_tag": llama_tag or release_tag,
        "asset": asset or f"llama-{release_tag}-bin-macos-arm64.tar.gz",
        "install_kind": "macos-arm64",
    }
    monkeypatch.setattr(upd, "_resolve_prebuilt_for_host", lambda *, force_refresh = False: payload)


def test_status_no_marker_no_prebuilt(monkeypatch, tmp_path):
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _no_prebuilt(monkeypatch)
    st = upd.get_update_status()
    assert st["supported"] is False
    assert st["update_available"] is False
    assert st["installed_tag"] is None


def test_status_source_build_offers_prebuilt(monkeypatch, tmp_path):
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch, release_tag = "b9585")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: None)
    st = upd.get_update_status()
    assert st["supported"] is True
    assert st["update_available"] is True
    assert st["source_build"] is True
    assert st["latest_tag"] == "b9585"
    assert st["published_repo"] == "unslothai/llama.cpp"


def test_status_source_build_compares_llama_tag(monkeypatch, tmp_path):
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch, release_tag = "v1.0", llama_tag = "b9457")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: 9000)
    st = upd.get_update_status()
    assert st["latest_tag"] == "b9457"
    assert st["update_available"] is True


def test_status_source_build_pinned_binary_not_offered(monkeypatch, tmp_path):
    binary = tmp_path / "custom" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(binary))
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch)
    st = upd.get_update_status()
    assert st["supported"] is False
    assert st["update_available"] is False


def test_llama_install_root_pinned_returns_none(monkeypatch, tmp_path):
    binary = tmp_path / "custom" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(binary))
    assert upd._llama_install_root(str(binary)) is None


def test_status_source_build_suppressed_when_newer(monkeypatch, tmp_path):
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch, release_tag = "b10360")

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = "version: 0.1.1-dev (build 10470, commit abc1234)\n"

    monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **kw: _Proc())
    st = upd.get_update_status()
    assert st["supported"] is True
    assert st["update_available"] is False
    assert st["installed_tag"] == "b10470"


def test_status_source_build_offers_same_base_mix(monkeypatch, tmp_path):
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch, release_tag = "b9596-mix-e6f2453", llama_tag = "b9596")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: 9596)
    st = upd.get_update_status()
    assert st["supported"] is True
    assert st["update_available"] is True
    assert st["source_build"] is True
    assert st["installed_tag"] == "b9596"
    assert st["latest_tag"] == "b9596-mix-e6f2453"


def test_status_source_build_same_base_bare_not_offered(monkeypatch, tmp_path):
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch, release_tag = "b9596", llama_tag = "b9596")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: 9596)
    st = upd.get_update_status()
    assert st["update_available"] is False
    assert st["latest_tag"] == "b9596"


def test_status_source_build_skips_probe_while_job_runs(monkeypatch, tmp_path):
    # Status polls must not exec the binary mid-swap (can fail os.replace on Windows).
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    probes = {"resolve": 0, "version": 0}

    def _count_resolve(*, force_refresh = False):
        probes["resolve"] += 1
        return None

    def _count_version(b):
        probes["version"] += 1
        return None

    monkeypatch.setattr(upd, "_resolve_prebuilt_for_host", _count_resolve)
    monkeypatch.setattr(upd, "_installed_build_number", _count_version)
    with upd._job_lock:
        upd._job["state"] = upd._JOB_RUNNING
    st = upd.get_update_status()
    assert st["job"]["state"] == "running"
    assert probes == {"resolve": 0, "version": 0}


def test_status_source_build_skips_probe_when_update_checks_disabled(monkeypatch, tmp_path):
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))

    def _resolve(*, force_refresh = False):
        raise AssertionError("probed for a prebuilt despite UNSLOTH_DISABLE_UPDATE_CHECK=1")

    monkeypatch.setattr(upd, "_resolve_prebuilt_for_host", _resolve)
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is False
    assert st["source_build"] is False


def test_installed_version_skips_probe_while_job_runs(monkeypatch, tmp_path):
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    probed = {"n": 0}

    def _count_version(b):
        probed["n"] += 1
        return 9585

    monkeypatch.setattr(upd, "_installed_build_number", _count_version)

    with upd._job_lock:
        upd._job["state"] = upd._JOB_RUNNING
    assert upd.get_installed_llama_version() is None
    assert probed["n"] == 0

    upd._reset_job_for_tests()
    assert upd.get_installed_llama_version() == "b9585"
    assert probed["n"] == 1


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


def test_start_update_no_marker_no_prebuilt_refuses(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    _no_prebuilt(monkeypatch)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "no_prebuilt_available"


def test_start_update_refuses_when_update_checks_disabled(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")

    def _network(*args, **kwargs):
        raise AssertionError("planned an update despite UNSLOTH_DISABLE_UPDATE_CHECK=1")

    monkeypatch.setattr(upd, "_plan_llama_phase", _network)
    monkeypatch.setattr(upd, "_pending_backend_migration", _network)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", _network)
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "update_checks_disabled"


def test_start_update_source_build_installs_prebuilt(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = install_dir / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    _prebuilt(
        monkeypatch, repo = "unslothai/llama.cpp", asset = "app-b9585-linux-x64-rocm-gfx110X.tar.gz"
    )

    captured = {}

    def _fake_run(cmd, **kwargs):
        cmd = list(cmd)
        assert "--version" in cmd
        return _Proc()

    def _on_start(cmd):
        captured["cmd"] = cmd
        _write_install(install_dir, "b9585")

    monkeypatch.setattr(upd.subprocess, "run", _fake_run)
    _patch_installer_popen(monkeypatch, on_start = _on_start)

    res = upd.start_update()
    assert res["started"] is True, res
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    cmd = captured["cmd"]
    assert "--install-dir" in cmd and str(install_dir) in cmd
    assert "--published-repo" in cmd and "unslothai/llama.cpp" in cmd
    assert "--llama-tag" in cmd and "latest" in cmd
    assert cmd[cmd.index("--rocm-gfx") + 1] == "gfx110x"
    assert "--simple-policy" not in cmd and "--cpu-fallback" not in cmd
    assert "--published-release-tag" not in cmd


def test_start_update_happy_path(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    captured = {}

    def _on_start(cmd):
        captured["cmd"] = cmd
        _write_install(install_dir, "b9518")

    popen_kwargs: dict = {}
    _patch_installer_popen(
        monkeypatch,
        lines = [
            "[llama-prebuilt] resolving release\n",
            "Downloading llama.zip:  35.0% (12.0 MiB/35.0 MiB) at 9.0 MiB/s\n",
            "Downloading llama.zip:  80.0% (28.0 MiB/35.0 MiB) at 9.0 MiB/s\n",
        ],
        on_start = _on_start,
        captured_kwargs = popen_kwargs,
    )

    res = upd.start_update()
    assert res["started"] is True
    assert res["job"]["from_tag"] == "b9493"
    assert res["job"]["progress"] == 0.0

    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "success", job
    assert job["to_tag"] == "b9518"
    assert job["reload_required"] is False
    assert "--install-dir" in captured["cmd"]
    assert str(install_dir) in captured["cmd"]
    assert "--llama-tag" in captured["cmd"] and "latest" in captured["cmd"]
    assert "unslothai/llama.cpp" in captured["cmd"]
    assert job["progress"] == 1.0
    assert popen_kwargs["env"]["UNSLOTH_PROGRESS_PERCENT_STEP"] == "5"


def test_a_second_spawn_during_the_update_does_not_overwrite_the_installer_argv(
    monkeypatch, tmp_path
):
    """Forces the ps fallback for the pid identity lookup, since CI runs this file on Linux only."""
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    monkeypatch.setattr(process_lifetime.sys, "platform", "darwin", raising = False)

    captured: dict = {}
    spawned: list = []

    def _on_start(cmd):
        captured["cmd"] = cmd
        _write_install(install_dir, "b9518")

    _patch_installer_popen(monkeypatch, on_start = _on_start, spawned = spawned)

    job = _run_start_update_to_completion()
    assert job["state"] == "success", job
    # The ps spawn must complete: a partial Popen stand-in disables the lookup under test.
    assert any(cmd and cmd[0] == "ps" for cmd in spawned), spawned
    assert process_lifetime._pid_identity(os.getpid()), "the ps lookup did not complete"
    assert "--install-dir" in captured["cmd"], captured["cmd"]
    assert str(install_dir) in captured["cmd"], captured["cmd"]


@pytest.mark.parametrize(
    "marker_fields, expected_choice",
    [
        ({"asset": "llama-b9493-bin-ubuntu-vulkan-x64.tar.gz"}, "vulkan"),
        (
            {
                "asset": "llama-b9493-bin-ubuntu-vulkan-x64.tar.gz",
                "llama_backend": None,
                "include_llama_backend": True,
            },
            "auto",
        ),
        (
            {"asset": "llama-b9493-bin-ubuntu-vulkan-x64.tar.gz", "llama_backend": "auto"},
            "auto",
        ),
        (
            {"asset": "llama-b9493-bin-ubuntu-vulkan-x64.tar.gz", "llama_backend": "vulkan"},
            "vulkan",
        ),
        ({"asset": "llama-b9493-bin-ubuntu-x64.tar.gz", "force_cpu": True}, "cpu"),
        ({"asset": "llama-b9493-bin-ubuntu-x64.tar.gz", "force_cpu": False}, "auto"),
        ({"asset": "llama-b9493-bin-ubuntu-x64.tar.gz"}, "auto"),
    ],
)
def test_update_leaves_the_recorded_choice_to_the_installer(
    monkeypatch, tmp_path, marker_fields, expected_choice
):
    """An update passes no backend override; install_llama_prebuilt.py reads the choice from its marker."""
    monkeypatch.delenv("UNSLOTH_FORCE_VULKAN", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_BACKEND", raising = False)
    install_dir = tmp_path / "llama.cpp"
    fields = dict(marker_fields)
    binary = _write_install(install_dir, "b9493", repo = "ggml-org/llama.cpp", **fields)
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    captured: dict = {}
    popen_kwargs: dict = {}

    def _on_start(cmd):
        captured["cmd"] = cmd
        _write_install(install_dir, "b9518", repo = "ggml-org/llama.cpp", **fields)

    _patch_installer_popen(
        monkeypatch,
        lines = ["installed\n"],
        on_start = _on_start,
        captured_kwargs = popen_kwargs,
    )

    assert upd.start_update()["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "success", job

    assert "--llama-backend" not in captured["cmd"]
    assert "--force-cpu" not in captured["cmd"]
    assert "--cpu-fallback" not in captured["cmd"]
    assert "UNSLOTH_LLAMA_CPP_BACKEND" not in popen_kwargs["env"]
    assert "UNSLOTH_FORCE_VULKAN" not in popen_kwargs["env"]

    _studio = _BACKEND.parent
    if str(_studio) not in sys.path:
        sys.path.insert(0, str(_studio))
    import install_llama_prebuilt as ilp

    assert ilp.effective_backend_request(None, install_dir = install_dir) == (
        expected_choice,
        False,
    )


def test_start_update_reports_full_release_tag(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9595")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(
        freshness,
        "_fetch_latest_release_tag",
        lambda repo, timeout = 5.0: "b9596-mix-e6f2453",
    )

    def _on_start(cmd):
        _write_install(install_dir, "b9596", release_tag = "b9596-mix-e6f2453")

    _patch_installer_popen(monkeypatch, on_start = _on_start)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "success", job
    assert job["to_tag"] == "b9596-mix-e6f2453"
    assert "Updated llama.cpp to b9596-mix-e6f2453." in job["message"]


def test_an_update_that_kept_the_existing_install_does_not_claim_a_new_release(
    monkeypatch, tmp_path
):
    """Exit 0 can keep the existing install after a transient failure, so success is judged by the tag."""
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9595", release_tag = "b9595-mix-aaaaaaa")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(
        freshness,
        "_fetch_latest_release_tag",
        lambda repo, timeout = 5.0: "b9596-mix-e6f2453",
    )

    monkeypatch.setattr(upd.sys, "platform", "darwin")

    _patch_installer_popen(monkeypatch, on_start = lambda cmd: None)

    job = _run_start_update_to_completion()
    assert job["state"] == "success", job
    assert job["to_tag"] == "b9595-mix-aaaaaaa"
    assert "Updated llama.cpp to" not in job["message"], job["message"]
    assert "up to date" not in job["message"], job["message"]
    assert "could not be updated" in job["message"], job["message"]
    assert "Try again later" in job["message"], job["message"]


def _run_start_update_to_completion():
    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            return job
        time.sleep(0.05)
    return upd.get_update_status()["job"]


def test_start_update_pinned_tag_mismatch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9595")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(
        freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9601-mix-a0e2906"
    )
    _patch_installer_popen(
        monkeypatch,
        on_start = lambda cmd: _write_install(install_dir, "b9500", release_tag = "b9500-mix-deadbee"),
    )
    job = _run_start_update_to_completion()
    assert job["state"] == "error", job
    assert "b9601-mix-a0e2906" in (job["error"] or "")


def test_start_update_pinned_reroute_to_other_repo_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "linux")
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9595", repo = "unslothai/llama.cpp")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(
        freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9601-mix-a0e2906"
    )
    _patch_installer_popen(
        monkeypatch,
        on_start = lambda cmd: _write_install(install_dir, "b9601", repo = "ggml-org/llama.cpp"),
    )
    job = _run_start_update_to_completion()
    assert job["state"] == "success", job


def test_start_update_installer_failure_reports_error(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    _patch_installer_popen(monkeypatch, returncode = 2, lines = ["boom: network error\n"])

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


def test_start_update_rate_limit_reports_actionable_error(monkeypatch, tmp_path):
    monkeypatch.delenv("GH_TOKEN", raising = False)
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    lines = [
        "[llama-prebuilt] prebuilt fallback reason: failed to inspect published "
        "releases in unslothai/llama.cpp: GitHub API returned 403 for "
        "https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b9518; "
        "set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits\n",
        "windows_runtime_dirs=C:\\\\very\\\\long\\\\path\n",
    ]
    _patch_installer_popen(monkeypatch, returncode = 2, lines = lines)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "error"
    assert "GH_TOKEN" in (job["error"] or "")
    assert "windows_runtime_dirs" not in (job["error"] or "")


def test_start_update_reports_a_github_429_as_a_rate_limit(monkeypatch, tmp_path):
    monkeypatch.delenv("GH_TOKEN", raising = False)
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    lines = [
        "[llama-prebuilt] prebuilt fallback reason: failed to inspect published "
        "releases in unslothai/llama.cpp: GitHub API returned 429 for "
        "https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b9518; "
        "set GH_TOKEN or GITHUB_TOKEN to avoid GitHub API rate limits\n",
    ]
    _patch_installer_popen(monkeypatch, returncode = 2, lines = lines)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "error"
    assert "GH_TOKEN" in (job["error"] or "")


def test_start_update_tells_an_authenticated_run_to_wait_for_the_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("GH_TOKEN", "x")
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    lines = [
        "[llama-prebuilt] prebuilt fallback reason: failed to inspect published "
        "releases in unslothai/llama.cpp: GitHub API returned 429 for "
        "https://api.github.com/repos/unslothai/llama.cpp/releases/tags/b9518\n",
    ]
    _patch_installer_popen(monkeypatch, returncode = 2, lines = lines)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "error"
    error = job["error"] or ""
    assert "rate-limiting" in error
    assert "GH_TOKEN" not in error
    assert "Wait for the limit to reset" in error


def test_start_update_does_not_blame_github_for_a_hugging_face_rate_limit(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    lines = [
        "[llama-prebuilt] prebuilt fallback reason: validation model unavailable: "
        "429 Client Error: Too Many Requests for url "
        "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf "
        "(Rate limit reached)\n",
    ]
    _patch_installer_popen(monkeypatch, returncode = 2, lines = lines)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "error"
    error = job["error"] or ""
    assert "GH_TOKEN" not in error
    assert "validation model unavailable" in error


def test_rocm_install_args_gfx_family():
    assert upd._rocm_install_args("app-b9585-linux-x64-rocm-gfx110X.tar.gz") == [
        "--rocm-gfx",
        "gfx110x",
    ]
    assert upd._rocm_install_args("app-b9585-windows-x64-rocm-gfx1150.zip") == [
        "--rocm-gfx",
        "gfx1150",
    ]


def test_rocm_install_args_fork_version_bundle():
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

    def _fake_run(cmd, **kwargs):
        cmd = list(cmd)
        assert "--version" in cmd
        return _Proc()

    def _on_start(cmd):
        captured["cmd"] = cmd
        _write_install(install_dir, latest, repo = repo, asset = asset)

    monkeypatch.setattr(upd.subprocess, "run", _fake_run)
    _patch_installer_popen(monkeypatch, on_start = _on_start)

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
    assert "--force-cpu" not in cmd
    assert "--simple-policy" not in cmd
    assert "--published-repo" in cmd and "unslothai/llama.cpp" in cmd


def test_install_cmd_fork_rocm_marker_forwards_has_rocm(monkeypatch, tmp_path):
    cmd = _capture_install_cmd(
        monkeypatch, tmp_path, asset = "llama-b9334-bin-ubuntu-rocm-6.4-x64.tar.gz"
    )
    assert "--has-rocm" in cmd
    assert "--rocm-gfx" not in cmd


def test_install_cmd_ggml_cpu_marker_has_no_cpu_fallback(monkeypatch, tmp_path):
    cmd = _capture_install_cmd(
        monkeypatch,
        tmp_path,
        repo = "ggml-org/llama.cpp",
        asset = "llama-b9334-bin-ubuntu-x64.tar.gz",
    )
    assert "--force-cpu" not in cmd
    assert "--rocm-gfx" not in cmd
    assert "--has-rocm" not in cmd
    assert "--simple-policy" not in cmd
    assert "--published-repo" in cmd and "ggml-org/llama.cpp" in cmd


def test_install_cmd_cuda_marker_minimal_and_backward_compatible(monkeypatch, tmp_path):
    cmd = _capture_install_cmd(monkeypatch, tmp_path, asset = None)
    assert "--simple-policy" not in cmd
    assert "--rocm-gfx" not in cmd
    assert "--has-rocm" not in cmd
    assert "--force-cpu" not in cmd


def test_install_cmd_pins_offered_release_tag(monkeypatch, tmp_path):
    # Pin the release the banner offered; the installer's own 'latest' can lag.
    monkeypatch.setattr(sys, "platform", "linux")
    cmd = _capture_install_cmd(monkeypatch, tmp_path, latest = "b9601-mix-a0e2906")
    assert cmd[cmd.index("--published-release-tag") + 1] == "b9601-mix-a0e2906"


def test_install_cmd_pins_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    cmd = _capture_install_cmd(monkeypatch, tmp_path)
    assert cmd[cmd.index("--published-release-tag") + 1] == "b9518"


def test_install_cmd_does_not_pin_on_macos(monkeypatch, tmp_path):
    # A pinned tag disables the walk-back macOS needs for host-compatible prebuilts.
    monkeypatch.setattr(sys, "platform", "darwin")
    cmd = _capture_install_cmd(monkeypatch, tmp_path)
    assert "--published-release-tag" not in cmd
    assert "--llama-tag" in cmd and "latest" in cmd


def test_start_update_already_running_refuses(monkeypatch, tmp_path):
    binary = _write_install(tmp_path / "llama.cpp", "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    with upd._job_lock:
        upd._job.update(state = upd._JOB_RUNNING)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "already_running"
    assert res["message"] == "Another llama.cpp install is already running."


def test_start_update_installer_missing_refuses(monkeypatch, tmp_path):
    binary = _write_install(tmp_path / "llama.cpp", "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: None)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "installer_missing"


class _FakeBackend:
    """Fake backend for update coordination."""

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

    def _on_start(cmd):
        seen["flag_during_install"] = backend._llama_update_in_progress
        _write_install(install_dir, "b9518")

    _patch_installer_popen(monkeypatch, on_start = _on_start)

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)

    assert backend.unloaded is True
    assert upd.get_update_status()["job"]["reload_required"] is True
    assert seen.get("flag_during_install") is True
    assert backend._llama_update_in_progress is False


def test_update_drains_and_marks_each_kept_model_before_swapping(monkeypatch, tmp_path):
    import core.inference.model_slots as model_slots

    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    _inject_backend(monkeypatch, _FakeBackend())
    kept = _FakeBackend()
    monkeypatch.setattr(model_slots, "slots", [SimpleNamespace(llama = kept)])
    seen = {}

    def unload_llama_slots(strict = False):
        seen["lock_free"] = kept._serial_load_lock.acquire(blocking = False)
        seen["flag"] = kept._llama_update_in_progress
        return 1

    monkeypatch.setattr(model_slots, "unload_llama_slots", unload_llama_slots)
    _patch_installer_popen(monkeypatch, on_start = lambda cmd: _write_install(install_dir, "b9518"))
    assert upd.start_update()["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert seen == {"lock_free": True, "flag": True}
    assert kept._llama_update_in_progress is False


def test_update_stops_when_a_kept_model_would_not_unload(monkeypatch, tmp_path):
    import core.inference.model_slots as model_slots

    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    _inject_backend(monkeypatch, _FakeBackend())
    monkeypatch.setattr(model_slots, "slots", [])

    def stuck(strict = False):
        assert strict
        raise RuntimeError("Could not unload 1 model(s) kept alongside")

    monkeypatch.setattr(model_slots, "unload_llama_slots", stuck)
    spawned = []
    _patch_installer_popen(monkeypatch, spawned = spawned)
    assert upd.start_update()["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert upd.get_update_status()["job"]["state"] == "error"
    assert not any("install_llama_prebuilt" in " ".join(map(str, c)) for c in spawned)


def test_update_clears_maintenance_flag_on_installer_failure(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    backend = _FakeBackend()
    _inject_backend(monkeypatch, backend)

    _patch_installer_popen(monkeypatch, returncode = 1, lines = ["boom\n"])

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        if upd.get_update_status()["job"]["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert upd.get_update_status()["job"]["state"] == "error"
    assert upd.get_update_status()["job"]["reload_required"] is True
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

    _patch_installer_popen(monkeypatch, on_start = lambda cmd: _write_install(install_dir, "b9518"))

    res = upd.start_update()
    assert res["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "success", job


def test_resolve_prebuilt_parses_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    calls = {"n": 0}

    class _Proc:
        returncode = 0
        stdout = (
            '{"prebuilt_available": true, "repo": "unslothai/llama.cpp", "release_tag": "b9585"}'
        )
        stderr = "[llama-prebuilt] some log\n"

    def _fake_run(cmd, **kwargs):
        calls["n"] += 1
        assert "--resolve-prebuilt" in cmd
        return _Proc()

    monkeypatch.setattr(upd.subprocess, "run", _fake_run)
    res = upd._resolve_prebuilt_for_host()
    assert res["prebuilt_available"] is True and res["release_tag"] == "b9585"
    upd._resolve_prebuilt_for_host()
    assert calls["n"] == 1


def test_resolve_prebuilt_fails_open(monkeypatch, tmp_path):
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")

    def _boom(cmd, **kwargs):
        raise OSError("subprocess failed")

    monkeypatch.setattr(upd.subprocess, "run", _boom)
    assert upd._resolve_prebuilt_for_host() is None

    class _Proc:
        returncode = 0
        stdout = '{"prebuilt_available": false}'
        stderr = ""

    monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **kw: _Proc())
    assert upd._resolve_prebuilt_for_host() == {"prebuilt_available": False}


def test_installed_build_number(monkeypatch):
    def _ver(text):
        class _Proc:
            returncode = 0
            stdout = ""
            stderr = text

        monkeypatch.setattr(upd.subprocess, "run", lambda cmd, **kw: _Proc())
        return upd._installed_build_number("/bin/llama-server")

    assert _ver("version: 0.1.1-dev (build 10470, commit abc1234)\nbuilt with clang\n") == 10470
    assert _ver("version: 0.1.1 (build 10470, commit abc1234)\n") == 10470
    assert _ver("version: 9585 (abc1234)\nbuilt with clang\n") == 9585
    assert _ver("version: 0.1.1-dev (build 0, commit unknown)\n") is None
    assert _ver("version: 0.1.1-dev (build 1, commit deadbee)\n") is None
    assert _ver("version: 1 (deadbee)\n") is None
    assert _ver("version: 2.0.0\n") is None
    assert _ver("no version here") is None
    assert upd._installed_build_number(None) is None


def test_llama_install_root_finds_llama_cpp_ancestor(monkeypatch, tmp_path):
    root = tmp_path / "llama.cpp"
    binary = root / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    assert upd._llama_install_root(str(binary)) == root


def test_llama_install_root_unmanaged_path_returns_none(monkeypatch, tmp_path):
    binary = tmp_path / "usr" / "local" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    assert upd._llama_install_root(str(binary)) is None


def test_llama_install_root_unsloth_env_dir(monkeypatch, tmp_path):
    root = tmp_path / "vendor" / "llama"
    binary = root / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    assert upd._llama_install_root(str(binary)) == root


def test_llama_install_root_ignores_inactive_env_root(monkeypatch, tmp_path):
    inactive = tmp_path / "custom-empty"
    inactive.mkdir()
    active = tmp_path / "llama.cpp"
    binary = active / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(inactive))
    assert upd._llama_install_root(str(binary)) == active


def test_llama_install_root_refuses_pinned_checkout_under_llama_cpp(monkeypatch, tmp_path):
    root = tmp_path / "my-project" / "llama.cpp"
    binary = root / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(binary))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    assert upd._llama_install_root(str(binary)) is None


def test_start_update_source_build_refuses_when_newer(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    binary = install_dir / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    _prebuilt(monkeypatch, release_tag = "b9518")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: 9600)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "up_to_date"


def test_status_not_offered_on_mix_latest(monkeypatch, tmp_path):
    binary = _write_install(tmp_path / "llama.cpp", "b9596", release_tag = "b9596-mix-e6f2453")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(
        freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9596-mix-e6f2453"
    )
    st = upd.get_update_status()
    assert st["update_available"] is False
    assert st["installed_tag"] == "b9596"
    assert st["latest_tag"] == "b9596-mix-e6f2453"


def test_status_not_offered_when_latest_lags(monkeypatch, tmp_path):
    binary = _write_install(tmp_path / "llama.cpp", "b9585")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    st = upd.get_update_status()
    assert st["update_available"] is False


def test_start_update_marked_refuses_when_not_behind(monkeypatch, tmp_path):
    binary = _write_install(tmp_path / "llama.cpp", "b9596", release_tag = "b9596-mix-e6f2453")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(
        freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9596-mix-e6f2453"
    )
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "up_to_date"


def test_status_update_available_includes_size(monkeypatch, tmp_path):
    binary = _write_install(tmp_path, "b9493", asset = "app-b9493-linux-x64-cuda13-newer.tar.gz")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    monkeypatch.setattr(
        freshness,
        "latest_release_assets",
        lambda repo, *, force_refresh = False: {
            "app-b9518-linux-x64-cuda13-newer.tar.gz": 88_000_000
        },
    )
    st = upd.get_update_status(force_refresh = True)
    assert st["update_available"] is True
    assert st["update_size_bytes"] == 88_000_000


def test_status_source_build_includes_update_size(monkeypatch, tmp_path):
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(
        monkeypatch,
        repo = "unslothai/llama.cpp",
        release_tag = "b9585",
        asset = "app-b9585-linux-x64-cpu.tar.gz",
    )
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: None)
    monkeypatch.setattr(
        upd,
        "latest_release_assets",
        lambda repo, *, force_refresh = False: (
            {"app-b9585-linux-x64-cpu.tar.gz": 77_000_000}
            if repo == "unslothai/llama.cpp"
            else None
        ),
    )
    st = upd.get_update_status()
    assert st["source_build"] is True
    assert st["update_available"] is True
    assert st["update_size_bytes"] == 77_000_000


def test_update_changelog_uses_full_installed_release_identity(monkeypatch, tmp_path):
    binary = _write_install(
        tmp_path,
        "b10698",
        release_tag = "b10698-mix-old",
    )
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(
        freshness,
        "_fetch_latest_release_tag",
        lambda repo, timeout = 5.0: "b10715-mix-new",
    )
    seen = {}

    def fake_changelog(
        repo,
        installed,
        latest,
        *,
        force_refresh = False,
    ):
        seen.update(
            repo = repo,
            installed = installed,
            latest = latest,
            force_refresh = force_refresh,
        )
        return {
            "changes": [{"summary": "New model", "links": []}],
            "total_changes": 1,
            "truncated": False,
            "release_url": "https://github.com/unslothai/llama.cpp/releases/tag/new",
        }

    monkeypatch.setattr(upd, "changelog_for_update", fake_changelog)

    result = upd.get_update_changelog(force_refresh = True)

    assert result["matched"] is True
    assert result["installed_tag"] == "b10698"
    assert result["latest_tag"] == "b10715-mix-new"
    assert result["changes"][0]["summary"] == "New model"
    assert seen == {
        "repo": "unslothai/llama.cpp",
        "installed": "b10698-mix-old",
        "latest": "b10715-mix-new",
        "force_refresh": True,
    }


def test_update_changelog_retry_keeps_the_banner_target(monkeypatch, tmp_path):
    binary = _write_install(tmp_path, "b10698", release_tag = "b10698-mix-old")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    published = {"tag": "b10715-mix-new"}
    monkeypatch.setattr(
        freshness,
        "_fetch_latest_release_tag",
        lambda repo, timeout = 5.0: published["tag"],
    )
    assert freshness.latest_published_release("unslothai/llama.cpp") == "b10715-mix-new"
    published["tag"] = "b10800-mix-newer"
    monkeypatch.setattr(
        upd,
        "changelog_for_update",
        lambda *_args, **_kwargs: {
            "changes": [{"summary": "New model", "links": []}],
            "total_changes": 1,
            "truncated": False,
            "release_url": "https://github.com/unslothai/llama.cpp/releases/tag/new",
        },
    )

    result = upd.get_update_changelog(force_refresh = True)

    assert result["latest_tag"] == "b10715-mix-new"
    assert result["matched"] is True


def test_update_changelog_answers_the_pair_the_caller_asked_about(monkeypatch, tmp_path):
    binary = _write_install(tmp_path, "b10698", release_tag = "b10698-mix-old")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(
        freshness,
        "_fetch_latest_release_tag",
        lambda repo, timeout = 5.0: "b10800-mix-newer",
    )
    seen = {}

    def fake_changelog(
        repo,
        installed,
        latest,
        *,
        force_refresh = False,
    ):
        seen.update(installed = installed, latest = latest)
        return {
            "changes": [],
            "total_changes": 0,
            "truncated": False,
            "release_url": "https://github.com/unslothai/llama.cpp/releases/tag/x",
        }

    monkeypatch.setattr(upd, "changelog_for_update", fake_changelog)

    result = upd.get_update_changelog(installed_tag = "b10698", latest_tag = "b10715-mix-new")

    assert result["latest_tag"] == "b10715-mix-new"
    assert seen["latest"] == "b10715-mix-new"
    assert result["matched"] is True

    monkeypatch.setattr(upd, "changelog_for_update", lambda *_a, **_k: None)
    failed = upd.get_update_changelog(installed_tag = "b10698", latest_tag = "b10715-mix-new")

    assert failed["release_url"].endswith("b10715-mix-new")


def test_update_changelog_ignores_a_pair_from_a_different_install(monkeypatch, tmp_path):
    binary = _write_install(tmp_path, "b10698", release_tag = "b10698-mix-old")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(
        freshness,
        "_fetch_latest_release_tag",
        lambda repo, timeout = 5.0: "b10800-mix-newer",
    )
    monkeypatch.setattr(
        upd,
        "changelog_for_update",
        lambda *_args, **_kwargs: {
            "changes": [],
            "total_changes": 0,
            "truncated": False,
            "release_url": "https://github.com/unslothai/llama.cpp/releases/tag/x",
        },
    )

    result = upd.get_update_changelog(installed_tag = "b9000", latest_tag = "b10715-mix-new")

    assert result["installed_tag"] == "b10698"
    assert result["latest_tag"] == "b10800-mix-newer"
