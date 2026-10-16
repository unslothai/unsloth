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


# ── Strix deferral only where the discrete wheel installs and runs ─────────


def test_python_strix_keeps_its_index_when_the_rocm_version_is_unreadable():
    # Versionless: no generic tag can serve the dGPU, so the Strix route main installed stays.
    calls = routing._run_install(gfx_devices = ("gfx1151", "gfx1100"), rocm_version = None)
    assert _STRIX_INDEX in calls, calls


@pytest.mark.parametrize("devices", [("gfx1151", "gfx1100"), ("gfx1151", "gfx1201")])
def test_python_strix_keeps_its_index_under_an_hsa_override(devices):
    # The override spoofs every agent, the dGPU included, so dGPU wheels would never load.
    calls = routing._run_install(
        gfx_devices = devices, rocm_version = (7, 1), env = {"HSA_OVERRIDE_GFX_VERSION": "11.5.1"}
    )
    assert "gfx120X-all" not in calls and "whl/rocm7.1" not in calls, calls


def test_python_inferred_install_uses_the_deferred_discrete_card():
    calls = routing._run_install(
        gfx_devices = ("gfx1151", "gfx1201"),
        inferred = "gfx1151",
        rocm_version = (7, 1),
        rocm_gpu_visible = False,
    )
    assert "repo.amd.com/rocm/whl/gfx120X-all" in calls and _STRIX_INDEX not in calls, calls


@_POSIX
@pytest.mark.parametrize("hsa,expected", [("", "gfx1201"), ("11.5.1", "gfx1151")])
def test_install_sh_strix_keeps_its_index_under_an_hsa_override(hsa, expected):
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    script = (
        rocm_support._gpu_record_helpers(source)
        + "\nunset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES\n"
        + (
            f"export HSA_OVERRIDE_GFX_VERSION={hsa}\n"
            if hsa
            else "unset HSA_OVERRIDE_GFX_VERSION\n"
        )
        + "_amd_prefer_discrete_gfx \"$(printf 'gfx1151\\ngfx1201')\" gfx1151 torch\n"
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, timeout = 60)
    assert out.stdout.strip() == expected, out


@_POSIX
def test_install_sh_strix_handoff_clears_the_radeon_downloader():
    # rocminfo's "Radeon 8060S" set the flag; the dGPU's generic index must not be overridden.
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    start = source.find('case "$_torch_index_leaf" in\n    rocm[0-9]*)')
    end = source.find("\nfi  # _torch_index_pinned guard", start)
    script = (
        "set -euo pipefail\n"
        + rocm_support._extract_sh_function_body(source, "_rocm_leaf_below")
        + "\n"
        + rocm_support._gpu_record_helpers(source)
        + "\n"
        + 'TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm7.1"\n'
        + '_torch_index_leaf="rocm7.1"\n'
        + "_torch_index_pinned=false\nSKIP_TORCH=false\n_amd_gpu_radeon=true\n"
        + "_gfx_rocm64_target=false\n"
        + "unset HSA_OVERRIDE_GFX_VERSION HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES\n"
        + "unset CUDA_VISIBLE_DEVICES UNSLOTH_ROCM_GFX_ARCH UNSLOTH_PYTORCH_MIRROR\n"
        + "unset UNSLOTH_AMD_ROCM_MIRROR\n"
        + "rocminfo() { printf 'Name: gfx1151\\nName: gfx1030\\n'; }\n"
        + source[start:end]
        + '\nprintf "RADEON:%s\\n" "$_amd_gpu_radeon"\n'
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, timeout = 60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().rsplit("RADEON:", 1)[-1] == "false", out.stdout


@_POSIX
def test_install_sh_final_summary_keeps_the_inferred_gfx1153():
    source = _INSTALL_SH.read_text(encoding = "utf-8")
    start = source.find("elif _has_amd_rocm_gpu; then\n    # The arch read fine")
    end = source.find('elif [ "$_torch_index_leaf" = xpu ]; then', start)
    assert start >= 0 and end >= 0
    script = (
        rocm_support._extract_sh_function_body(source, "_hsa_override_gfx_arch")
        + "\nstep() { printf 'STEP:%s\\n' \"$2\"; }\n"
        + "_has_amd_rocm_gpu() { return 0; }\n_probe_amd_gfx_arch() { :; }\n"
        + "_infer_linux_amd_gfx_arch() { echo gfx1153; }\n"
        + "_torch_index_pinned=false\nunset HSA_OVERRIDE_GFX_VERSION UNSLOTH_ROCM_GFX_ARCH\n"
        + "if false; then :\n"
        + source[start:end]
        + "fi\n"
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, timeout = 60)
    assert "AMD GPU (gfx1153, no ROCm PyTorch wheels -- CPU fallback)" in out.stdout, out


# ── Codex round: generic-only dGPU, Krackan Point 2 peer, HIP guidance ──────

_RX7900 = "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX]"


def test_python_inferred_strix_defers_to_a_generic_only_dgpu():
    # gfx942 has no AMD per-arch index: the generic route installs for it, not the Strix wheel.
    calls = routing._run_install(
        gfx_devices = ("gfx1151", "gfx942"),
        inferred = "gfx1151",
        rocm_version = (7, 1),
        rocm_gpu_visible = False,
    )
    assert "download.pytorch.org/whl/rocm7.1" in calls and _STRIX_INDEX not in calls, calls


@pytest.mark.parametrize(
    "lspci,expected", [(_RX7900, "gfx1100"), (_KRACKAN2_TEXT[1].strip(), "gfx1153")]
)
def test_python_krackan2_cpu_defers_to_a_discrete_peer(lspci, expected, monkeypatch):
    monkeypatch.setattr(stack_mod, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack_mod, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: _KRACKAN2_TEXT[0])
    monkeypatch.setattr(stack_mod.shutil, "which", lambda _n: "/usr/bin/lspci")
    line = lspci if lspci.startswith("0") else "c4:00.0 VGA compatible controller [0300]: " + lspci
    monkeypatch.setattr(
        stack_mod.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, line, "")
    )
    assert stack_mod._infer_linux_amd_gfx_arch() == expected


@_POSIX
@pytest.mark.parametrize(
    "lspci,expected",
    [
        (_RX7900, "gfx1100"),
        ("c4:00.0 VGA compatible controller [0300]: " + _KRACKAN2_TEXT[1].strip(), "gfx1153"),
        ("", "gfx1153"),
    ],
)
def test_install_sh_krackan2_cpu_defers_to_a_discrete_peer(tmp_path, lspci, expected):
    assert _sh_infer(tmp_path, _KRACKAN2_TEXT[0], lspci = lspci)[0] == expected


@_POSIX
@pytest.mark.parametrize(
    "arches,vulkan", [("gfx1153", False), ("gfx1010", True), ("gfx1010\ngfx1153", True)]
)
def test_install_sh_gfx1153_keeps_its_hip_guidance(arches, vulkan):
    url, err = guards._run_sh_get_torch_index_url(arches, rocm_tag = "rocm7.2")
    assert url.endswith("/cpu"), (url, err)
    assert ("UNSLOTH_LLAMA_CPP_BACKEND=vulkan" in err) is vulkan, err
    assert ("llama.cpp HIP build" in err) is (not vulkan), err


def test_setup_sh_gfx1153_keeps_its_hip_guidance():
    source = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    block = source[source.index('step "gpu" "AMD GPU detected ($_setup_unsup_gfx)') :]
    block = block[: block.index('elif [ -n "$_setup_mkt" ]')]
    assert 'if [ "$_setup_unsup_gfx" = gfx1153 ]' in block and "llama.cpp HIP build" in block


def _setup_unsupported_lookup(gpu_name, lspci_lines, tmp_path):
    """setup.sh's runtime-less unsupported lookup, with its covered-peer guard and a scripted lspci."""
    import os
    import shutil
    import subprocess

    from test_rdna1_unsupported_message_8529 import _SETUP_SH, _sh_function_body

    src = _SETUP_SH.read_text(encoding = "utf-8")
    body = "\n".join(
        _sh_function_body(src, name)
        for name in (
            "_setup_supported_gfx_from_name",
            "_setup_unsupported_gfx_from_name",
            "_setup_unsupported_gfx_any",
        )
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("sh", "grep", "cat", "sed", "tr", "awk", "printf"):
        found = shutil.which(tool)
        if found:
            (bin_dir / tool).symlink_to(found)
    if lspci_lines is not None:
        fake = bin_dir / "lspci"
        printed = "\n".join(lspci_lines)
        fake.write_text(f'#!/bin/sh\ncat <<"LSPCI_EOF"\n{printed}\nLSPCI_EOF\n', encoding = "utf-8")
        fake.chmod(0o755)
    out = subprocess.run(
        ["sh", "-c", f'{body}\n_setup_unsupported_gfx_any "$1" || true\n', "sh", gpu_name],
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        text = True,
        timeout = 30,
        env = dict(os.environ, PATH = str(bin_dir)),
    )
    return out.stdout.strip()


_K2_LINE = "c5:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Krackan Point 2 [1002:1114]"
_NAVI31_LINE = (
    "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
    "Navi 31 [Radeon RX 7900 XTX] [1002:744c]"
)


@pytest.mark.parametrize(
    "gpu_name,lspci,expected",
    [
        ("", [_K2_LINE], "gfx1153"),
        ("AMD Radeon 820M", None, "gfx1153"),
        ("", [_K2_LINE, _NAVI31_LINE], ""),
    ],
)
def test_setup_sh_runtime_less_summary_names_gfx1153(gpu_name, lspci, expected, tmp_path):
    # No rocminfo / amd-smi: the summary still warns for a lone Krackan Point 2, never beside a covered peer.
    assert _setup_unsupported_lookup(gpu_name, lspci, tmp_path) == expected
