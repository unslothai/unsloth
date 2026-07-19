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
        self.returncode = returncode
        self.stdout = iter(lines or [])

    def wait(self):
        return self.returncode

    def kill(self):
        pass


def _patch_installer_popen(
    monkeypatch,
    *,
    returncode = 0,
    lines = None,
    on_start = None,
    captured_kwargs = None,
):
    monkeypatch.setattr(
        upd.subprocess,
        "Popen",
        lambda cmd, **kw: _FakeInstallerPopen(
            cmd,
            returncode = returncode,
            lines = lines,
            on_start = on_start,
            captured_kwargs = captured_kwargs,
            **kw,
        ),
    )


def _write_install(
    dir_: Path,
    tag: str,
    repo: str = "unslothai/llama.cpp",
    asset: str | None = None,
    release_tag: str | None = None,
    install_kind: str | None = None,
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
    if install_kind is not None:
        marker["install_kind"] = install_kind
    (dir_ / MARKER).write_text(json.dumps(marker))
    return str(binary)


@pytest.fixture(autouse = True)
def _clean_state(monkeypatch, tmp_path):
    freshness.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()
    # Isolate the freshness disk cache so the suite never writes the real
    # ~/.unsloth cache (the default when storage_roots can't be imported).
    monkeypatch.setattr(freshness, "_cache_dir", lambda: tmp_path / ".freshness_cache")
    # Deterministic markerless paths: no host-pinned binary, no custom dir.
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    # Never hit the network in these tests.
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    yield
    freshness.reset_caches()
    upd._reset_job_for_tests()
    upd._resolve_memo.clear()


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
    # No marker AND no prebuilt available for the host -> unsupported (the genuine
    # source-build-with-nothing-to-offer case).
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")  # no marker file alongside
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _no_prebuilt(monkeypatch)
    st = upd.get_update_status()
    assert st["supported"] is False
    assert st["update_available"] is False
    assert st["installed_tag"] is None


def test_status_source_build_offers_prebuilt(monkeypatch, tmp_path):
    # Markerless source build with a prebuilt now available for the host: surface
    # the update. Unknown installed version (source build) is treated as behind.
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
    # release_tag may be a fork wrapper (v1.0); compare/display the upstream
    # llama_tag (b9457) so a source build is not wrongly judged newer.
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch, release_tag = "v1.0", llama_tag = "b9457")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: 9000)
    st = upd.get_update_status()
    assert st["latest_tag"] == "b9457"  # not the wrapper tag
    assert st["update_available"] is True  # 9000 < 9457


def test_status_source_build_pinned_binary_not_offered(monkeypatch, tmp_path):
    # LLAMA_SERVER_PATH pins a custom binary outside any llama.cpp dir; an apply
    # could not take effect, so the button must not surface.
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
    # A source build already newer than the latest prebuilt is not nagged.
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    _prebuilt(monkeypatch, release_tag = "b9518")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: 9600)
    st = upd.get_update_status()
    assert st["supported"] is True
    assert st["update_available"] is False
    assert st["installed_tag"] == "b9600"


def test_status_source_build_offers_same_base_mix(monkeypatch, tmp_path):
    # The reported banner bug: a source build at the same upstream base as a new
    # Unsloth prebuilt that adds a mix-<sha> suffix. The base build numbers match
    # (9596 == 9596) but the mix carries extra patches the source build lacks, so
    # the update must still surface -- mirroring the marker path's is_behind.
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
    # Same base, but the prebuilt is a bare rebuild (no mix suffix): nothing extra
    # to gain, so do not nag.
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
    # While the updater swaps the tree, status polls must not exec the binary
    # being replaced (on Windows that exec can fail the installer's os.replace);
    # the 3s poller only consumes job progress.
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


def test_installed_version_skips_probe_while_job_runs(monkeypatch, tmp_path):
    # Markerless build: get_installed_llama_version falls back to exec'ing
    # `llama-server --version`. While the updater swaps the tree that exec can
    # fail the installer's os.replace on Windows, so the About-panel probe must
    # be skipped (return None) exactly like get_update_status's source probe.
    binary = tmp_path / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")  # markerless: no UNSLOTH_PREBUILT_INFO.json
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    probed = {"n": 0}

    def _count_version(b):
        probed["n"] += 1
        return 9585

    monkeypatch.setattr(upd, "_installed_build_number", _count_version)

    with upd._job_lock:
        upd._job["state"] = upd._JOB_RUNNING
    assert upd.get_installed_llama_version() is None
    assert probed["n"] == 0  # never exec'd the binary mid-swap

    upd._reset_job_for_tests()  # back to idle -> probe runs
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
    binary.write_text("stub")  # no marker
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    _no_prebuilt(monkeypatch)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "no_prebuilt_available"


def test_start_update_source_build_installs_prebuilt(monkeypatch, tmp_path):
    # Markerless install + available prebuilt: install in place into the resolved
    # root, with the asset-derived ROCm forwarding and the resolved repo.
    install_dir = tmp_path / "llama.cpp"
    binary = install_dir / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")  # no marker
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    _prebuilt(
        monkeypatch, repo = "unslothai/llama.cpp", asset = "app-b9585-linux-x64-rocm-gfx110X.tar.gz"
    )

    captured = {}

    class _Proc:
        returncode = 0
        stdout = "installed"
        stderr = ""

    def _fake_run(cmd, **kwargs):
        cmd = list(cmd)
        assert "--version" in cmd  # only status polls still use run()
        return _Proc()

    def _on_start(cmd):
        captured["cmd"] = cmd
        _write_install(install_dir, "b9585")  # installer writes the marker

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
    # No pin: source-build detection and the unpinned apply share the same
    # "latest" resolver, so they already agree.
    assert "--published-release-tag" not in cmd


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

    def _on_start(cmd):
        captured["cmd"] = cmd
        # Simulate the installer writing a new marker with the latest tag.
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


def test_start_update_preserves_vulkan_via_env(monkeypatch, tmp_path):
    # A Vulkan install (marker asset carries 'vulkan') must re-assert
    # UNSLOTH_FORCE_VULKAN on update, or detect_host on a GPU box re-routes to
    # CUDA/ROCm and silently replaces the Vulkan build.
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(
        install_dir,
        "b9493",
        repo = "ggml-org/llama.cpp",
        asset = "llama-b9493-bin-ubuntu-vulkan-x64.tar.gz",
    )
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    def _on_start(cmd):
        _write_install(
            install_dir,
            "b9518",
            repo = "ggml-org/llama.cpp",
            asset = "llama-b9518-bin-ubuntu-vulkan-x64.tar.gz",
        )

    popen_kwargs: dict = {}
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
    assert popen_kwargs["env"]["UNSLOTH_FORCE_VULKAN"] == "1"


@pytest.mark.parametrize(
    "install_kind, asset, expect_flag",
    [
        ("linux-cpu", "llama-b9493-bin-ubuntu-x64.tar.gz", True),
        ("windows-cpu", "llama-b9493-bin-win-cpu-x64.zip", True),
        ("linux-arm64", "llama-b9493-bin-ubuntu-arm64.tar.gz", True),
        ("windows-arm64", "llama-b9493-bin-win-cpu-arm64.zip", True),
        ("linux-vulkan", "llama-b9493-bin-ubuntu-vulkan-x64.tar.gz", False),
        ("linux-cuda", "llama-b9493-bin-ubuntu-cuda-x64.tar.gz", False),
        # Legacy markers without install_kind keep the pre-#6097 heal-to-GPU behaviour
        # (no forced --cpu-fallback); see test_install_cmd_ggml_cpu_marker_has_no_cpu_fallback.
        (None, "llama-b9493-bin-ubuntu-x64.tar.gz", False),
    ],
)
def test_start_update_cpu_fallback_preserved_by_kind(
    monkeypatch, tmp_path, install_kind, asset, expect_flag
):
    # A CPU install (x86_64 *-cpu or arm64 *-arm64) must re-assert --cpu-fallback on
    # update, or detect_host on a GPU host re-routes to a GPU/source build and
    # reintroduces the crash (#7213). Keyed off install_kind; legacy markers with no
    # install_kind deliberately do not force CPU (#6097 heal-to-GPU).
    install_dir = tmp_path / "llama.cpp"
    binary = _write_install(install_dir, "b9493", asset = asset, install_kind = install_kind)
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")

    captured: dict = {}

    def _on_start(cmd):
        captured["cmd"] = cmd
        _write_install(install_dir, "b9518", asset = asset, install_kind = install_kind)

    _patch_installer_popen(monkeypatch, lines = ["installed\n"], on_start = _on_start)

    assert upd.start_update()["started"] is True
    deadline = time.time() + 10
    while time.time() < deadline:
        job = upd.get_update_status()["job"]
        if job["state"] in ("success", "error"):
            break
        time.sleep(0.05)
    assert job["state"] == "success", job
    assert ("--cpu-fallback" in captured["cmd"]) is expect_flag


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
    # Installer stays on the pinned repo but produces a different tag -> it
    # ignored the pin (the silent mismatch this pin exists to prevent). Fail loud.
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
    # A Vulkan/Intel host reroutes fork->upstream and drops the pin, installing a
    # different-repo tag. Legitimate: the pin check must not flag the repo switch.
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


# --- installer-argument construction (mirrors the post-#5963 setup scripts) ---


def test_rocm_install_args_gfx_family():
    # Per-gfx ROCm bundle: gfx family lives in the asset name.
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
        cmd = list(cmd)
        assert "--version" in cmd  # only status polls still use run()
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
    # Legacy CPU installs recorded a ggml-org marker (new installs use the fork).
    # Re-running into the same install-dir/repo reproduces the same CPU bundle;
    # --cpu-fallback (which force-drops GPU detection) is reserved for setup.sh's
    # arm64 rescue and must not appear here.
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


def test_install_cmd_pins_offered_release_tag(monkeypatch, tmp_path):
    # Apply must install exactly the release the banner offered. The installer's
    # own "latest" comes from commit-date-ordered sources, which can lag the
    # published_at-newest tag detection picked; unpinned, that lag makes Update
    # reinstall the current build while the banner never clears.
    monkeypatch.setattr(sys, "platform", "linux")
    cmd = _capture_install_cmd(monkeypatch, tmp_path, latest = "b9601-mix-a0e2906")
    # The full release identity is pinned, not the bare upstream base.
    assert cmd[cmd.index("--published-release-tag") + 1] == "b9601-mix-a0e2906"


def test_install_cmd_pins_on_windows(monkeypatch, tmp_path):
    # The darwin exemption must not leak to other platforms.
    monkeypatch.setattr(sys, "platform", "win32")
    cmd = _capture_install_cmd(monkeypatch, tmp_path)
    assert cmd[cmd.index("--published-release-tag") + 1] == "b9518"


def test_install_cmd_does_not_pin_on_macos(monkeypatch, tmp_path):
    # A pinned tag disables the installer's older-release walk-back, which macOS
    # needs to skip prebuilts built for a newer macOS than the host.
    monkeypatch.setattr(sys, "platform", "darwin")
    cmd = _capture_install_cmd(monkeypatch, tmp_path)
    assert "--published-release-tag" not in cmd
    assert "--llama-tag" in cmd and "latest" in cmd


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


# --- markerless helper units ---


def test_resolve_prebuilt_parses_and_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    calls = {"n": 0}

    class _Proc:
        returncode = 0
        # stderr noise plus the JSON line on stdout (installer logs to stderr).
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
    # Second call is memoized (no second subprocess).
    upd._resolve_prebuilt_for_host()
    assert calls["n"] == 1


def test_resolve_prebuilt_fails_open(monkeypatch, tmp_path):
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")

    def _boom(cmd, **kwargs):
        raise OSError("subprocess failed")

    monkeypatch.setattr(upd.subprocess, "run", _boom)
    assert upd._resolve_prebuilt_for_host() is None
    # Failures are not cached: a later success is observed.

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

    assert _ver("version: 9585 (abc1234)\nbuilt with clang\n") == 9585
    assert _ver("version: 1 (deadbee)\n") is None  # source build without tags
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
    # A binary on PATH (no marker, no env pin, no llama.cpp ancestor) is foreign:
    # installing elsewhere would not replace it, so report no manageable root.
    binary = tmp_path / "usr" / "local" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    assert upd._llama_install_root(str(binary)) is None


def test_llama_install_root_unsloth_env_dir(monkeypatch, tmp_path):
    # UNSLOTH_LLAMA_CPP_PATH dir holding the active binary is the managed root.
    root = tmp_path / "vendor" / "llama"
    binary = root / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    assert upd._llama_install_root(str(binary)) == root


def test_llama_install_root_ignores_inactive_env_root(monkeypatch, tmp_path):
    # UNSLOTH_LLAMA_CPP_PATH set but the active binary is not under it: do not
    # target the stale env root, resolve from the binary's own llama.cpp tree.
    inactive = tmp_path / "custom-empty"
    inactive.mkdir()
    active = tmp_path / "llama.cpp"
    binary = active / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(inactive))
    assert upd._llama_install_root(str(binary)) == active


def test_llama_install_root_refuses_pinned_checkout_under_llama_cpp(monkeypatch, tmp_path):
    # The LLAMA_SERVER_PATH pin guard must run before the ancestor scan, or a
    # user's own llama.cpp checkout could be handed to the installer.
    root = tmp_path / "my-project" / "llama.cpp"
    binary = root / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(binary))
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    assert upd._llama_install_root(str(binary)) is None


def test_start_update_source_build_refuses_when_newer(monkeypatch, tmp_path):
    # A direct POST on a source build already newer than the prebuilt must not
    # downgrade it; start_update mirrors the detection suppression.
    install_dir = tmp_path / "llama.cpp"
    binary = install_dir / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")  # no marker
    monkeypatch.setattr(upd, "_find_binary", lambda: str(binary))
    monkeypatch.setattr(upd, "_installer_script", lambda: tmp_path / "install_llama_prebuilt.py")
    _prebuilt(monkeypatch, release_tag = "b9518")
    monkeypatch.setattr(upd, "_installed_build_number", lambda b: 9600)
    res = upd.start_update()
    assert res["started"] is False
    assert res["reason"] == "up_to_date"


# --- mix-tag detection + apply guard (the reported banner bug) ---


def test_status_not_offered_on_mix_latest(monkeypatch, tmp_path):
    # Installed the mix latest; GitHub latest is that same full tag -> no banner.
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
    # A lagging latest (older build than installed) must never be offered.
    binary = _write_install(tmp_path / "llama.cpp", "b9585")
    monkeypatch.setattr(upd, "_find_binary", lambda: binary)
    monkeypatch.setattr(freshness, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    st = upd.get_update_status()
    assert st["update_available"] is False


def test_start_update_marked_refuses_when_not_behind(monkeypatch, tmp_path):
    # A direct POST / stale banner must not reinstall when already on the latest.
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
    # Marker (prebuilt) update path attaches the download size of the asset the
    # banner would fetch.
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
    # #6338 P3: a source build offered a prebuilt must carry the asset size too.
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("stub")  # no marker -> source build
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
