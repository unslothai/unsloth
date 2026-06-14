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

if not hasattr(ilp, "resolve_simple_install_release_plans"):
    pytest.skip("PR symbols not present - check branch", allow_module_level = True)

FORK = ilp.DEFAULT_PUBLISHED_REPO  # unslothai/llama.cpp


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


def _run_resolve_capture_host(monkeypatch, capsys):
    """Drive --resolve-prebuilt and return the host the resolver was handed."""
    seen = {}

    def _resolver(tag, host, repo, published_release_tag):
        seen["repo"] = repo
        seen["host"] = host
        raise ilp.PrebuiltFallback("no asset")

    monkeypatch.setattr(ilp, "resolve_simple_install_release_plans", _resolver)
    monkeypatch.setattr(
        sys,
        "argv",
        ["install_llama_prebuilt.py", "--resolve-prebuilt", "latest", "--output-format", "json"],
    )
    assert ilp.main() == ilp.EXIT_SUCCESS
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    return seen, out


def test_resolve_prebuilt_cpu_linux_routes_to_fork(monkeypatch, capsys):
    # CPU-only Linux host (no GPU, no ROCm tooling): the dispatch routes to the
    # fork, which now ships the CPU prebuilt, and the host is left CPU-only.
    monkeypatch.setattr(ilp, "detect_host", lambda: _host(is_linux = True, is_x86_64 = True))
    monkeypatch.setattr(ilp.shutil, "which", lambda tool: None)
    seen, out = _run_resolve_capture_host(monkeypatch, capsys)
    assert seen["repo"] == FORK
    assert out["repo"] == FORK
    assert seen["host"].has_rocm is False


@pytest.mark.parametrize(
    "os_kwargs",
    [
        {"is_linux": True, "is_x86_64": True},
        {"system": "Windows", "is_windows": True, "is_x86_64": True, "machine": "AMD64"},
    ],
)
def test_resolve_prebuilt_rocm_tooling_host_not_offered_cpu(monkeypatch, capsys, os_kwargs):
    # A Linux or Windows host whose runtime probe could not confirm ROCm (has_rocm
    # False) but exposes ROCm tooling (e.g. hipconfig) must be treated as ROCm so
    # the probe does not offer the CPU bundle over a possible HIP source build.
    monkeypatch.setattr(ilp, "detect_host", lambda: _host(**os_kwargs))
    monkeypatch.setattr(ilp.shutil, "which", lambda tool: "/opt/rocm/bin/hipconfig" if tool == "hipconfig" else None)
    seen, out = _run_resolve_capture_host(monkeypatch, capsys)
    assert seen["repo"] == FORK
    assert seen["host"].has_rocm is True
    # The gfx-less ROCm host yields no covered bundle -> the probe reports the
    # headline promise: no prebuilt available (source build), never a CPU bundle.
    assert out["prebuilt_available"] is False
