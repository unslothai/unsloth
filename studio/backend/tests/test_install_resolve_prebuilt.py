# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install_llama_prebuilt.py: host->repo mapping and the --resolve-prebuilt mode.

These back the in-app update for source-build (markerless) installs: the backend
asks the installer whether an official prebuilt exists for this host without
downloading. Network and host detection are stubbed; no GPU or internet needed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

ilp = importlib.import_module("install_llama_prebuilt")

if not hasattr(ilp, "published_repo_for_host") or not hasattr(
    ilp, "resolve_simple_install_release_plans"
):
    pytest.skip("PR symbols not present - check branch", allow_module_level = True)

FORK = ilp.DEFAULT_PUBLISHED_REPO  # unslothai/llama.cpp
UPSTREAM = ilp.UPSTREAM_REPO  # ggml-org/llama.cpp


def _host(**kw):
    base = dict(
        system = "Linux",
        machine = "x86_64",
        is_windows = False,
        is_linux = False,
        is_macos = False,
        is_x86_64 = False,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = False,
        rocm_gfx_target = None,
        macos_version = None,
    )
    base.update(kw)
    return ilp.HostInfo(**base)


def test_published_repo_for_host():
    # CPU-only Linux (x64 and arm64) -> ggml-org upstream.
    assert ilp.published_repo_for_host(_host(is_linux = True, is_x86_64 = True)) == UPSTREAM
    assert (
        ilp.published_repo_for_host(_host(is_linux = True, is_arm64 = True, machine = "aarch64"))
        == UPSTREAM
    )
    # GPU Linux -> fork.
    assert (
        ilp.published_repo_for_host(_host(is_linux = True, is_x86_64 = True, has_usable_nvidia = True))
        == FORK
    )
    assert ilp.published_repo_for_host(_host(is_linux = True, is_x86_64 = True, has_rocm = True)) == FORK
    # CPU-only Windows -> ggml-org (setup.ps1: the fork ships no win-cpu bundle).
    assert (
        ilp.published_repo_for_host(_host(system = "Windows", is_windows = True, is_x86_64 = True))
        == UPSTREAM
    )
    # GPU Windows -> fork.
    assert (
        ilp.published_repo_for_host(
            _host(system = "Windows", is_windows = True, is_x86_64 = True, has_usable_nvidia = True)
        )
        == FORK
    )
    # macOS -> fork regardless of GPU (ggml-org macOS bundles need too-new macOS).
    assert (
        ilp.published_repo_for_host(
            _host(system = "Darwin", is_macos = True, is_arm64 = True, machine = "arm64")
        )
        == FORK
    )
    # Linux with AMD tooling but no probed GPU -> fork (setup.sh routes on tooling).
    assert (
        ilp.published_repo_for_host(
            _host(is_linux = True, is_x86_64 = True), linux_amd_tooling_present = True
        )
        == FORK
    )
    # The tooling hint is Linux-only: Windows CPU stays on ggml-org.
    assert (
        ilp.published_repo_for_host(
            _host(system = "Windows", is_windows = True, is_x86_64 = True),
            linux_amd_tooling_present = True,
        )
        == UPSTREAM
    )


def _run_resolve(monkeypatch, capsys, plans_or_exc):
    monkeypatch.setattr(
        ilp,
        "detect_host",
        lambda: _host(system = "Darwin", is_macos = True, is_arm64 = True, machine = "arm64"),
    )

    def _resolver(tag, host, repo, published_release_tag):
        if isinstance(plans_or_exc, Exception):
            raise plans_or_exc
        return ("b9585", plans_or_exc)

    monkeypatch.setattr(ilp, "resolve_simple_install_release_plans", _resolver)
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_llama_prebuilt.py", "--resolve-prebuilt", "latest", "--output-format", "json"],
    )
    rc = ilp.main()
    assert rc == ilp.EXIT_SUCCESS
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_resolve_prebuilt_available(monkeypatch, capsys):
    plan = SimpleNamespace(
        release_tag = "b9585",
        llama_tag = "b9585",
        attempts = [
            SimpleNamespace(name = "llama-b9585-bin-macos-arm64.tar.gz", install_kind = "macos-arm64")
        ],
    )
    out = _run_resolve(monkeypatch, capsys, [plan])
    assert out["prebuilt_available"] is True
    assert out["repo"] == FORK
    assert out["release_tag"] == "b9585"
    assert out["asset"] == "llama-b9585-bin-macos-arm64.tar.gz"
    assert out["install_kind"] == "macos-arm64"


def test_resolve_prebuilt_unavailable(monkeypatch, capsys):
    out = _run_resolve(monkeypatch, capsys, ilp.PrebuiltFallback("no macOS asset"))
    assert out["prebuilt_available"] is False
    assert out["repo"] == FORK


def test_resolve_prebuilt_linux_amd_tooling_routes_to_fork(monkeypatch, capsys):
    # CPU-probed Linux host but rocminfo on PATH: the dispatch must route to the
    # fork so a HIP source build is not offered an upstream CPU prebuilt.
    monkeypatch.setattr(ilp, "detect_host", lambda: _host(is_linux = True, is_x86_64 = True))
    monkeypatch.setattr(ilp.shutil, "which", lambda tool: tool == "rocminfo")
    seen = {}

    def _resolver(tag, host, repo, published_release_tag):
        seen["repo"] = repo
        raise ilp.PrebuiltFallback("no asset")

    monkeypatch.setattr(ilp, "resolve_simple_install_release_plans", _resolver)
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_llama_prebuilt.py", "--resolve-prebuilt", "latest", "--output-format", "json"],
    )
    assert ilp.main() == ilp.EXIT_SUCCESS
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert seen["repo"] == FORK
    assert out["repo"] == FORK
