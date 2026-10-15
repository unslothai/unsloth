# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux AMD torch routing for two host shapes no table handled.

Strix APU (gfx1150-gfx1152) listed ahead of a discrete RDNA card: the device-0 pick sent torch
to the Strix-only per-arch index, whose wheels carry no kernels for the discrete card. A lone
gfx1153 (Krackan Point 2) was in no Linux table, so it got the generic ROCm wheel (no gfx1153
kernels) with no warning (#11815, #9932).
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
import test_rocm_support as rocm_support  # noqa: E402
import test_unsupported_arch_routing_guards_8529 as guards  # noqa: E402

stack_mod = routing.stack_mod
_INSTALL_SH = PACKAGE_ROOT / "install.sh"
_POSIX = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None, reason = "POSIX shell only"
)
_STRIX_INDEX = "repo.amd.com/rocm/whl/gfx1151"


@pytest.fixture(autouse = True)
def _reset_probe(monkeypatch):
    for name in (
        "HSA_OVERRIDE_GFX_VERSION",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "UNSLOTH_ROCM_GFX_ARCH",
    ):
        monkeypatch.delenv(name, raising = False)
    stack_mod._invalidate_torch_runtime_probe()
    yield
    stack_mod._invalidate_torch_runtime_probe()


def _sh_routing_leaf(arches, env = ""):
    """install.sh's routing block on a rocminfo stub; the index leaf it leaves selected."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    start = source.find('case "$_torch_index_leaf" in\n    rocm[0-9]*)')
    end = source.find("\nfi  # _torch_index_pinned guard", start)
    assert start >= 0 and end >= 0
    script = (
        "set -euo pipefail\n"
        + rocm_support._extract_sh_function_body(source, "_rocm_leaf_below")
        + "\n"
        + rocm_support._gpu_record_helpers(source)
        + "\n"
        + 'TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm7.1"\n'
        + '_torch_index_leaf="rocm7.1"\n'
        + "_torch_index_pinned=false\nSKIP_TORCH=false\n_amd_gpu_radeon=false\n"
        + "_gfx_rocm64_target=false\n"
        + "unset HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES\n"
        + "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_PYTORCH_MIRROR\n"
        + "unset UNSLOTH_AMD_ROCM_MIRROR\n"
        + "rocminfo() { printf '"
        + "".join(f"Name: {a}\\n" for a in arches)
        + "'; }\n"
        + "".join(f"export {a}\n" for a in env.split())
        + source[start:end]
        + '\nprintf "URL:%s\\n" "$TORCH_INDEX_URL"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, timeout = 60)
    assert out.returncode == 0, out.stderr
    url = out.stdout.strip().rsplit("URL:", 1)[-1].rstrip("/")
    return url.rsplit("/", 1)[-1].lower()


# ── Strix APU + discrete card ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "devices,expected",
    [
        (("gfx1151", "gfx1100"), "gfx1100"),
        (("gfx1151", "gfx1201"), "gfx1201"),
        (("gfx1150", "gfx1101"), "gfx1101"),
    ],
)
def test_python_target_skips_a_leading_strix_apu(devices, expected):
    calls = routing._run_install(gfx_devices = devices, rocm_version = (7, 1))
    assert _STRIX_INDEX not in calls and "whl/gfx1150" not in calls, calls
    assert f"installing for {expected}" in routing._run_install.printed


def test_python_strix_plus_rdna3_installs_the_generic_wheel():
    calls = routing._run_install(gfx_devices = ("gfx1151", "gfx1100"), rocm_version = (7, 1))
    assert "download.pytorch.org/whl/rocm7.1" in calls, calls


def test_python_strix_plus_rdna4_installs_the_rdna4_family():
    calls = routing._run_install(gfx_devices = ("gfx1151", "gfx1201"), rocm_version = (7, 1))
    assert "repo.amd.com/rocm/whl/gfx120X-all" in calls, calls


@pytest.mark.parametrize(
    "devices,env",
    [
        (("gfx1151",), {}),
        (("gfx1151", "gfx1100"), {"HIP_VISIBLE_DEVICES": "0"}),
        (("gfx1151", "gfx1100"), {"UNSLOTH_ROCM_GFX_ARCH": "gfx1151"}),
    ],
)
def test_python_lone_or_selected_strix_keeps_its_index(devices, env):
    calls = routing._run_install(gfx_devices = devices, rocm_version = (7, 1), env = env)
    assert _STRIX_INDEX in calls, calls


@_POSIX
@pytest.mark.parametrize(
    "arches,env,expected",
    [
        (("gfx1151", "gfx1100"), "", "rocm7.1"),
        (("gfx1151", "gfx1201"), "", "gfx120x-all"),
        (("gfx1152", "gfx1102"), "", "rocm7.1"),
    ],
)
def test_install_sh_skips_a_leading_strix_apu(arches, env, expected):
    assert _sh_routing_leaf(arches, env) == expected


@_POSIX
@pytest.mark.parametrize(
    "arches,env",
    [
        (("gfx1151",), ""),
        (("gfx1151", "gfx1100"), "HIP_VISIBLE_DEVICES=0"),
        (("gfx1151", "gfx1100"), "CUDA_VISIBLE_DEVICES=0"),
        (("gfx1151", "gfx1100"), "UNSLOTH_ROCM_GFX_ARCH=gfx1151"),
    ],
)
def test_install_sh_lone_or_selected_strix_keeps_its_index(arches, env):
    assert _sh_routing_leaf(arches, env) == "gfx1151"


@_POSIX
def test_install_sh_strix_reroute_names_the_discrete_choice():
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    assert '"$_runtime_gfx" torch)' in source
    assert '"$_gpu_disp_gfx" torch)' in source


# ── lone gfx1153 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ver", [(6, 4), (7, 2)])
def test_python_lone_gfx1153_gets_no_kernel_less_rocm_torch(ver):
    calls = routing._run_install(gfx_devices = ("gfx1153",), rocm_version = ver)
    assert "download.pytorch.org" not in calls and "repo.amd.com" not in calls, calls
    assert "AMD gfx1153: no ROCm PyTorch wheel" in routing._run_install.printed


def test_python_gfx1153_beside_a_dgpu_keeps_the_dgpu_route():
    calls = routing._run_install(gfx_devices = ("gfx1153", "gfx1100"), rocm_version = (7, 2))
    assert "download.pytorch.org/whl/rocm7.2" in calls, calls


def test_python_gfx1153_spoofed_to_a_covered_arch_keeps_rocm():
    calls = routing._run_install(
        gfx_devices = ("gfx1153",),
        rocm_version = (7, 2),
        env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    assert "download.pytorch.org/whl/rocm7.2" in calls, calls


@_POSIX
@pytest.mark.parametrize("tag", ["rocm7.2", "rocm6.4", ""])
def test_install_sh_lone_gfx1153_warns_and_takes_cpu(tag):
    url, err = guards._run_sh_get_torch_index_url("gfx1153", rocm_tag = tag)
    assert url.endswith("/cpu"), (url, err)
    assert "AMD GPU detected (gfx1153) -- Unsloth has no ROCm PyTorch wheels" in err, err


@_POSIX
def test_install_sh_gfx1153_beside_a_dgpu_keeps_the_generic_tag():
    url, err = guards._run_sh_get_torch_index_url("gfx1153\ngfx1100", rocm_tag = "rocm7.2")
    assert url.endswith("/rocm7.2"), (url, err)


@_POSIX
def test_install_sh_names_the_multiarch_device_extra_for_gfx1153():
    body = rocm_support._extract_sh_function_body(
        _INSTALL_SH.read_text(encoding = "utf-8"), "_therock_device_extra_for_gfx"
    )
    out = subprocess.run(
        ["bash", "-c", body + "\n_therock_device_extra_for_gfx gfx1153 || echo none"],
        capture_output = True,
        text = True,
        timeout = 30,
    )
    assert out.stdout.strip() == "device-gfx1153", out


# ── gfx1153 inferred without a runtime ──────────────────────────────────────

# Krackan Point 2 names the broader Krackan patterns also match; gfx1152 hosts stay gfx1152.
_KRACKAN2_TEXT = [
    "model name : AMD Ryzen AI 5 330 w/ Radeon 820M\n",
    "Advanced Micro Devices, Inc. [AMD/ATI] Krackan Point 2 [Radeon Graphics]\n",
]
_KRACKAN_TEXT = "model name : AMD Ryzen AI 7 350 w/ Radeon 860M\n"


@pytest.mark.parametrize("text", _KRACKAN2_TEXT)
def test_python_cpuinfo_names_gfx1153_before_gfx1152(text, monkeypatch):
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: text)
    assert stack_mod._linux_amd_gfx_from_cpuinfo() == "gfx1153"
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: _KRACKAN_TEXT)
    assert stack_mod._linux_amd_gfx_from_cpuinfo() == "gfx1152"


def test_python_lspci_names_gfx1153_before_gfx1152(monkeypatch):
    line = "c4:00.0 VGA compatible controller [0300]: " + _KRACKAN2_TEXT[1]
    monkeypatch.setattr(stack_mod.shutil, "which", lambda _n: "/usr/bin/lspci")
    monkeypatch.setattr(
        stack_mod.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, line, "")
    )
    assert stack_mod._linux_amd_gfx_from_lspci() == "gfx1153"


def _sh_infer(
    tmp_path,
    cpuinfo,
    lspci = "",
):
    """install.sh _infer_linux_amd_gfx_arch on a stub cpuinfo, PCI evidence and lspci."""
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    body = rocm_support._extract_sh_function_body(source, "_infer_linux_amd_gfx_arch")
    (tmp_path / "cpuinfo").write_text(cpuinfo, encoding = "utf-8")
    body = body.replace("/proc/cpuinfo", str(tmp_path / "cpuinfo"))
    body = body.replace("/proc/version", "/dev/null")
    script = (
        "_amd_gpu_present_via_pci() { return 0; }\n"
        + rocm_support._extract_sh_function_body(source, "_infer_amd_gfx_arch_from_gpu_name")
        + "\n"
        + f"lspci() {{ printf '%s\\n' '{lspci}'; }}\n"
        + "unset UNSLOTH_ROCM_GFX_ARCH\n"
        + body
        + "\n_infer_linux_amd_gfx_arch || echo none\n"
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, timeout = 30)
    return out.stdout.strip().splitlines()[-1], out.stderr


@_POSIX
def test_install_sh_cpuinfo_names_gfx1153_before_gfx1152(tmp_path):
    assert _sh_infer(tmp_path, _KRACKAN2_TEXT[0])[0] == "gfx1153"
    assert _sh_infer(tmp_path, _KRACKAN_TEXT)[0] == "gfx1152"


@_POSIX
def test_install_sh_lspci_names_gfx1153_before_gfx1152(tmp_path):
    line = "c4:00.0 VGA compatible controller [0300]: " + _KRACKAN2_TEXT[1].strip()
    assert _sh_infer(tmp_path, "model name : AMD\n", lspci = line)[0] == "gfx1153"


@_POSIX
def test_install_sh_inferred_gfx1153_warns_and_takes_cpu(monkeypatch):
    # No rocminfo/amd-smi reading: the arch comes only from _infer_linux_amd_gfx_arch.
    stubs = guards._INDEX_URL_STUBS
    probe = next(ln for ln in stubs.splitlines() if ln.startswith("_probe_amd_gfx_arch()"))
    monkeypatch.setattr(
        guards, "_INDEX_URL_STUBS", stubs.replace(probe, "_probe_amd_gfx_arch() { :; }")
    )
    url, err = guards._run_sh_get_torch_index_url("gfx1153")
    assert url.endswith("/cpu"), (url, err)
    assert "AMD GPU detected (gfx1153) -- Unsloth has no ROCm PyTorch wheels" in err, err
