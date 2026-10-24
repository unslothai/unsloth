# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux gfx90c (Renoir / Cezanne / Lucienne / Barcelo iGPU) alone, with a readable arch.

rocminfo reported gfx90c, which was in no uncovered list, so get_torch_index_url took the
generic pytorch.org ROCm tag: a multi-GB torch with no gfx90c kernels. Beside a discrete card
(#7776) the discrete card still gets ROCm torch.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(PACKAGE_ROOT / "tests" / "_shared"))

import test_missing_kernel_arch_routing_9396 as routing  # noqa: E402
import test_uncovered_amd_read_arch_linux_8529 as read_arch  # noqa: E402
import test_unsupported_arch_routing_guards_8529 as guards  # noqa: E402

stack_mod = routing.stack_mod
_INSTALL_SH = PACKAGE_ROOT / "install.sh"
_POSIX = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None, reason = "POSIX shell only"
)

# Trimmed from a real Ryzen 7 5700G rocminfo: the GPU agent names gfx90c twice.
_ROCMINFO_GFX90C = """\
*******
Agent 2
*******
  Name:                    gfx90c
  Marketing Name:          AMD Radeon Graphics
  Device Type:             GPU
      Name:                    amdgcn-amd-amdhsa--gfx90c:xnack-
"""
_ROCMINFO_GFX1100 = """\
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
      Name:                    amdgcn-amd-amdhsa--gfx1100
"""


@pytest.fixture(autouse = True)
def _reset_probe(monkeypatch):
    for name in (
        "HSA_OVERRIDE_GFX_VERSION",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "UNSLOTH_ROCM_GFX_ARCH",
    ):
        monkeypatch.delenv(name, raising = False)
    stack_mod._invalidate_torch_runtime_probe()
    yield
    stack_mod._invalidate_torch_runtime_probe()


def _rocminfo_harness(monkeypatch, tmp_path, rocminfo: str):
    """The real _probe_amd_gfx_arch over a fake rocminfo on PATH, instead of the arch stub."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "rocminfo.txt").write_text(rocminfo, encoding = "utf-8")
    fake = bin_dir / "rocminfo"
    fake.write_text(f"#!/bin/sh\ncat '{bin_dir / 'rocminfo.txt'}'\n", encoding = "utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    src = _INSTALL_SH.read_text(encoding = "utf-8")
    real = guards._INDEX_URL_STUBS
    stub = (
        "_probe_amd_gfx_arch() { printf '%s\\n' \"${UNSLOTH_ROCM_GFX_ARCH:-}\" "
        "| tr '[:upper:]' '[:lower:]'; }"
    )
    assert stub in real, "probe stub not found"
    monkeypatch.setattr(
        guards,
        "_INDEX_URL_STUBS",
        real.replace(
            stub,
            "_ensure_rocm_probe_env() { :; }\n_kfd_gfx_targets() { return 1; }\n"
            + guards._sh_function_body(src, "_probe_amd_gfx_arch"),
        ),
    )


# ── install.sh get_torch_index_url ──────────────────────────────────────────


@_POSIX
@pytest.mark.parametrize("tag", ["rocm7.2", "rocm6.4"])
def test_install_sh_lone_gfx90c_takes_cpu_with_the_vulkan_hint(monkeypatch, tmp_path, tag):
    _rocminfo_harness(monkeypatch, tmp_path, _ROCMINFO_GFX90C)
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = tag)
    assert url.endswith("/cpu"), (url, err)
    assert "AMD GPU detected (gfx90c) -- Unsloth has no ROCm PyTorch wheels" in err, err
    assert "export UNSLOTH_LLAMA_CPP_BACKEND=vulkan and re-run this installer" in err, err
    # TheRock rates gfx90c build-passing only and current torch there has no device-gfx90c extra.
    assert "TheRock" not in err, err
    # The torch 2.11 ROCm wheels carry gfx900 kernels, which run gfx90c under this override.
    assert "export HSA_OVERRIDE_GFX_VERSION=9.0.0" in err, err


@_POSIX
def test_install_sh_gfx90c_beside_a_dgpu_keeps_the_generic_tag(monkeypatch, tmp_path):
    _rocminfo_harness(monkeypatch, tmp_path, _ROCMINFO_GFX90C + _ROCMINFO_GFX1100)
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = "rocm7.2")
    assert url.endswith("/rocm7.2"), (url, err)
    assert "no ROCm PyTorch wheels" not in err, err


@_POSIX
def test_install_sh_gfx90c_under_a_gfx900_spoof_keeps_rocm(monkeypatch, tmp_path):
    """HSA_OVERRIDE_GFX_VERSION=9.0.0 is the usual Renoir workaround; the generic wheel has gfx900."""
    _rocminfo_harness(monkeypatch, tmp_path, _ROCMINFO_GFX90C)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "9.0.0")
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = "rocm7.2")
    assert url.endswith("/rocm7.2"), (url, err)


# ── install_python_stack._ensure_rocm_torch ─────────────────────────────────


@pytest.mark.parametrize("ver", [(6, 4), (7, 2)])
def test_python_lone_gfx90c_gets_no_kernel_less_rocm_torch(ver):
    calls = routing._run_install(gfx_devices = ("gfx90c",), rocm_version = ver)
    assert "download.pytorch.org" not in calls and "repo.amd.com" not in calls, calls
    assert "AMD gfx90c: no ROCm PyTorch wheel" in routing._run_install.printed


def test_python_gfx90c_beside_a_dgpu_keeps_the_dgpu_route():
    calls = routing._run_install(gfx_devices = ("gfx90c", "gfx1100"), rocm_version = (7, 2))
    assert "download.pytorch.org/whl/rocm7.2" in calls, calls


def test_python_gfx90c_spoofed_to_gfx900_keeps_rocm():
    calls = routing._run_install(
        gfx_devices = ("gfx90c",),
        rocm_version = (7, 2),
        env = {"HSA_OVERRIDE_GFX_VERSION": "9.0.0"},
    )
    assert "download.pytorch.org/whl/rocm7.2" in calls, calls


# ── re-running install.sh keeps an existing ROCm build on gfx90c ─────────────


@_POSIX
def test_install_sh_rerun_keeps_an_existing_rocm_build_on_gfx90c(monkeypatch, tmp_path):
    # The update path: install.sh recreates the venv; the old one ran a ROCm torch (with the
    # gfx900 override at runtime), so it must not be demoted to CPU.
    _rocminfo_harness(monkeypatch, tmp_path, _ROCMINFO_GFX90C)
    monkeypatch.setenv("_PREV_TORCH_VER", "2.11.0+rocm7.2")
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = "rocm7.2")
    assert url.endswith("/rocm7.2"), (url, err)
    assert "Keeping ROCm PyTorch on gfx90c" in err, err
    assert "installing CPU PyTorch" not in err, err


@_POSIX
@pytest.mark.parametrize("prev", ["2.11.0+cpu", "2.11.0", ""])
def test_install_sh_a_non_rocm_previous_torch_still_takes_cpu_on_gfx90c(monkeypatch, tmp_path, prev):
    _rocminfo_harness(monkeypatch, tmp_path, _ROCMINFO_GFX90C)
    monkeypatch.setenv("_PREV_TORCH_VER", prev)
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = "rocm7.2")
    assert url.endswith("/cpu"), (url, err)


@_POSIX
def test_install_sh_an_existing_rocm_build_does_not_keep_rocm_for_other_uncovered_arches(monkeypatch, tmp_path):
    rocminfo = _ROCMINFO_GFX90C.replace("gfx90c", "gfx1010")
    _rocminfo_harness(monkeypatch, tmp_path, rocminfo)
    monkeypatch.setenv("_PREV_TORCH_VER", "2.11.0+rocm7.2")
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = "rocm7.2")
    assert url.endswith("/cpu"), (url, err)


# ── the gfx900-override hint is gfx90c's alone ──────────────────────────────

_OVERRIDE_HINT = "export HSA_OVERRIDE_GFX_VERSION=9.0.0"


@_POSIX
def test_install_sh_other_uncovered_arches_get_no_gfx900_hint(monkeypatch, tmp_path):
    rocminfo = _ROCMINFO_GFX90C.replace("gfx90c", "gfx1010").replace("AMD Radeon Graphics", "AMD Radeon RX 5700 XT")
    _rocminfo_harness(monkeypatch, tmp_path, rocminfo)
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = "rocm7.2")
    assert url.endswith("/cpu"), (url, err)
    assert _OVERRIDE_HINT not in err, err


@_POSIX
@pytest.mark.parametrize(
    "gfx,hint", [("gfx90c", True), ("gfx1010", False), ("gfx803", False)]
)
def test_setup_sh_names_the_gfx900_override_for_gfx90c_only(gfx, hint):
    out = read_arch._setup_report(gfx, gfx, "")
    assert f"AMD GPU detected ({gfx}) -- no ROCm PyTorch wheels" in out, out
    assert (_OVERRIDE_HINT in out) is hint, out


@pytest.mark.parametrize("gfx,hint", [("gfx90c", True), ("gfx1010", False)])
def test_python_names_the_gfx900_override_for_gfx90c_only(gfx, hint):
    routing._run_install(gfx_devices = (gfx,), rocm_version = (7, 2))
    assert (_OVERRIDE_HINT in routing._run_install.printed) is hint, routing._run_install.printed
