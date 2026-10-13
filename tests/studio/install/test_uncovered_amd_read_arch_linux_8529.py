# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux RDNA 1 / Polaris with a READABLE arch and ROCm version (#8529).

With ROCm installed, rocminfo reports gfx1010 and every installer used to pick the generic
pytorch.org ROCm tag for it, a multi-GB torch with no gfx1010 kernels. Only the unreadable-arch
path reached the "no ROCm PyTorch wheels" message.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from unittest.mock import patch

_HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(PACKAGE_ROOT / "tests" / "_shared"))

import test_amd_fastpath_probe as fastpath  # noqa: E402
import test_missing_kernel_arch_routing_9396 as routing  # noqa: E402
import test_unsupported_arch_routing_guards_8529 as guards  # noqa: E402

stack_mod = routing.stack_mod
_INSTALL_SH = PACKAGE_ROOT / "install.sh"
_SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"

_UNCOVERED = ("gfx1010", "gfx1011", "gfx1012", "gfx803")
_POSIX = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None, reason = "POSIX shell only"
)


@pytest.fixture(autouse = True)
def _reset_probe(monkeypatch):
    # The shell harnesses inherit os.environ; a test sets these itself.
    for name in ("HSA_OVERRIDE_GFX_VERSION", "ROCR_VISIBLE_DEVICES", "UNSLOTH_ROCM_GFX_ARCH"):
        monkeypatch.delenv(name, raising = False)
    stack_mod._invalidate_torch_runtime_probe()
    yield
    stack_mod._invalidate_torch_runtime_probe()


# ── the list ────────────────────────────────────────────────────────────────


def test_no_uncovered_arch_has_a_wheel_route():
    assert set(_UNCOVERED) == set(stack_mod._LINUX_UNCOVERED_AMD_GFX)
    for gfx in _UNCOVERED:
        assert not stack_mod._gfx_has_a_wheel_route(gfx), gfx
        with patch.object(stack_mod, "IS_WINDOWS", False):
            assert stack_mod._amd_arch_index_url(gfx) is None, gfx


def test_the_shell_copies_name_the_same_arches():
    want = "gfx(" + "|".join(sorted(g[3:] for g in _UNCOVERED)) + ")"
    for path in (_INSTALL_SH, _SETUP_SH):
        found = re.findall(r"\$1 !~ /\^(gfx\([0-9|]+\))\$/", path.read_text(encoding = "utf-8"))
        assert found, f"{path.name}: uncovered-arch filter not found"
        for f in found:
            got = "gfx(" + "|".join(sorted(f[4:-1].split("|"))) + ")"
            assert got == want, f"{path.name} names {f}, python names {sorted(_UNCOVERED)}"


@pytest.mark.parametrize(
    "lists,expected",
    [
        ((["gfx1010"],), "gfx1010"),
        ((["gfx1010:xnack-", "GFX1010"], ["gfx1010"]), "gfx1010"),
        ((["gfx1010", "gfx803"],), "gfx1010, gfx803"),
        ((["gfx1010", "gfx1100"],), None),
        ((["gfx1010"], ["gfx1030"]), None),
        ((["gfx1030"],), None),
        (([],), None),
        ((["", None],), None),
    ],
)
def test_only_an_all_uncovered_host_answers(lists, expected):
    assert stack_mod._linux_host_only_uncovered_amd(*lists) == expected


# ── install_python_stack._ensure_rocm_torch ─────────────────────────────────


@pytest.mark.parametrize("gfx", _UNCOVERED)
@pytest.mark.parametrize("ver", [(6, 4), (7, 1)])
def test_a_lone_uncovered_card_gets_no_rocm_torch(gfx, ver):
    calls = routing._run_install(gfx_devices = (gfx,), rocm_version = ver)
    assert "download.pytorch.org" not in calls and "repo.amd.com" not in calls, calls
    assert "bitsandbytes" not in calls, calls
    assert f"AMD {gfx}: no ROCm PyTorch wheel" in routing._run_install.printed


@pytest.mark.parametrize("peer", ["gfx1100", "gfx1030"])
def test_a_covered_peer_keeps_the_generic_install(peer):
    calls = routing._run_install(gfx_devices = ("gfx1010", peer), rocm_version = (6, 4))
    assert "download.pytorch.org/whl/rocm6.4" in calls, calls


def test_a_covered_card_alone_is_unchanged():
    calls = routing._run_install(gfx_devices = ("gfx1030",), rocm_version = (6, 4))
    assert "download.pytorch.org/whl/rocm6.4" in calls, calls


@pytest.mark.parametrize(
    "devices,hsa,generic",
    [
        # amd-smi / KFD read the physical gfx1010; the runtime runs the spoofed arch.
        (("gfx1010",), "10.3.0", True),
        (("gfx1030",), "10.3.0", True),
        # A spoof onto another uncovered arch still has no wheels (RX 5500 posing as gfx1010).
        (("gfx1012",), "10.1.0", False),
        # Unreadable: ROCr ignores it and runs the physical arch.
        (("gfx1010",), "ten", False),
    ],
)
def test_an_hsa_spoof_is_judged_by_its_target(devices, hsa, generic):
    calls = routing._run_install(
        gfx_devices = devices, rocm_version = (6, 4), env = {"HSA_OVERRIDE_GFX_VERSION": hsa}
    )
    assert ("download.pytorch.org/whl/rocm6.4" in calls) is generic, calls


@pytest.mark.parametrize("hsa,needs", [("10.3.0", True), ("10.1.0", False)])
def test_an_hsa_spoof_decides_the_update_preflight(monkeypatch, hsa, needs):
    fastpath._host(
        monkeypatch,
        torch = ("2.10.0+cpu", ""),
        gfx = ("gfx1010",),
        env = {"HSA_OVERRIDE_GFX_VERSION": hsa},
    )
    assert fastpath._needs_pass() is needs


def test_an_existing_rocm_build_is_left_alone():
    calls = routing._run_install(
        gfx_devices = ("gfx1010",),
        rocm_version = (7, 1),
        torch_probe = routing._ROCM_GENERIC_TORCH,
        torch_owns_rocm = False,
    )
    assert "download.pytorch.org" not in calls, calls
    assert "no ROCm PyTorch wheel" not in routing._run_install.printed


def test_an_explicit_rocm_pin_still_installs():
    pin = "https://download.pytorch.org/whl/rocm6.4"
    calls = routing._run_install(
        gfx_devices = ("gfx1010",),
        rocm_version = (6, 4),
        env = {"UNSLOTH_TORCH_INDEX_URL": pin},
    )
    assert pin in calls, calls


# ── install_python_stack._amd_torch_needs_dependency_pass ────────────────────


@pytest.mark.parametrize("gfx", _UNCOVERED)
def test_the_update_preflight_stops_forcing_a_pass(monkeypatch, gfx):
    fastpath._host(monkeypatch, torch = ("2.10.0+cpu", ""), gfx = (gfx,))
    assert fastpath._needs_pass() is False


def test_the_update_preflight_still_forces_a_covered_card(monkeypatch):
    fastpath._host(monkeypatch, torch = ("2.10.0+cpu", ""), gfx = ("gfx1030",))
    assert fastpath._needs_pass() is True


def test_the_update_preflight_still_forces_a_mixed_host(monkeypatch):
    fastpath._host(monkeypatch, torch = ("2.10.0+cpu", ""), gfx = ("gfx1010", "gfx1100"))
    assert fastpath._needs_pass() is True


# ── install.sh get_torch_index_url ──────────────────────────────────────────


@_POSIX
@pytest.mark.parametrize("arch", ["gfx1010", "gfx1010:xnack-", "GFX1012", "gfx1011", "gfx803"])
@pytest.mark.parametrize("tag", ["rocm6.4", "rocm7.0", ""])
def test_install_sh_sends_a_read_uncovered_arch_to_cpu(arch, tag):
    url, err = guards._run_sh_get_torch_index_url(arch, rocm_tag = tag)
    assert url.endswith("/cpu"), (url, err)
    bare = arch.split(":")[0].lower()
    assert f"AMD GPU detected ({bare}) -- Unsloth has no ROCm PyTorch wheels" in err, err
    assert "Install the ROCm/HIP SDK" not in err, err


@_POSIX
@pytest.mark.parametrize("arch", ["gfx1010\ngfx1100", "gfx1030\ngfx1010", "gfx1030"])
def test_install_sh_keeps_the_generic_tag_with_a_covered_card(arch):
    url, err = guards._run_sh_get_torch_index_url(arch, rocm_tag = "rocm6.4")
    assert url.endswith("/rocm6.4"), (url, err)
    assert "no ROCm PyTorch wheels" not in err, err


# ── setup.sh GPU report ─────────────────────────────────────────────────────


def _setup_report(
    gfx: str,
    gfx_all: str,
    mkt: str = "",
    torch_version: str = "",
) -> str:
    src = _SETUP_SH.read_text(encoding = "utf-8")
    venv = tempfile.mkdtemp(dir = os.environ.get("TMPDIR"))
    if torch_version:
        tv = Path(venv) / "lib" / "python3.12" / "site-packages" / "torch"
        tv.mkdir(parents = True)
        hip = "'7.1.0'" if "rocm" in torch_version else "None"
        (tv / "version.py").write_text(
            f"__version__ = '{torch_version}'\nhip: Optional[str] = {hip}\n", encoding = "utf-8"
        )
    start = src.index("    # An arch READ as uncovered on every probed GPU")
    end = src.index('        step "gpu" "AMD ROCm"\n    fi\n', start) + len(
        '        step "gpu" "AMD ROCm"\n    fi\n'
    )
    funcs = "\n".join(
        guards._sh_function_body(src, n)
        for n in ("_setup_supported_gfx_from_name", "_setup_unsupported_gfx_from_name")
    )
    script = (
        'step() { echo "STEP: $2"; }\nsubstep() { echo "  $1"; }\n'
        '_setup_unsupported_gfx_any() { _setup_unsupported_gfx_from_name "$1"; }\n'
        f"{funcs}\n_setup_gfx='{gfx}'\n_setup_gfx_all='{gfx_all}'\n_setup_mkt='{mkt}'\n"
        f"VENV_DIR='{venv}'\n" + src[start:end]
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_TORCH_INDEX")}
    out = subprocess.run(
        ["bash", "-c", script], stdout = subprocess.PIPE, text = True, timeout = 30, env = env
    )
    shutil.rmtree(venv, ignore_errors = True)
    return out.stdout


@_POSIX
@pytest.mark.parametrize(
    "gfx,gfx_all,mkt",
    [
        ("gfx1010", "gfx1010", "AMD Radeon RX 5700 XT"),
        ("gfx1010", "gfx1010\ngfx1010", ""),
        ("gfx803", "gfx803", "Radeon RX 580 Series"),
    ],
)
def test_setup_sh_reports_a_read_uncovered_arch(gfx, gfx_all, mkt):
    out = _setup_report(gfx, gfx_all, mkt)
    assert f"STEP: AMD GPU detected ({gfx}) -- no ROCm PyTorch wheels" in out, out
    assert "torch stays CPU-only" in out, out


@_POSIX
@pytest.mark.parametrize(
    "gfx,gfx_all,mkt,step",
    [
        ("gfx1010", "gfx1010\ngfx1100", "AMD Radeon RX 5700 XT", "AMD Radeon RX 5700 XT (gfx1010)"),
        ("gfx1100", "gfx1100", "AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX (gfx1100)"),
        ("gfx1030", "gfx1030", "", "AMD ROCm (gfx1030)"),
        ("", "", "AMD Radeon RX 5700 XT", "AMD GPU detected (gfx1010) -- no ROCm PyTorch wheels"),
        ("", "", "", "AMD ROCm"),
    ],
)
def test_setup_sh_other_reports_are_unchanged(gfx, gfx_all, mkt, step):
    out = _setup_report(gfx, gfx_all, mkt)
    assert f"STEP: {step}" in out, out


@_POSIX
@pytest.mark.parametrize(
    "gfx,gfx_all",
    [("gfx1010", "gfx1100"), ("gfx1010", "gfx1010\ngfx1100"), ("gfx803", "gfx1030")],
)
def test_setup_sh_a_declared_uncovered_arch_ignores_the_physical_peer(monkeypatch, gfx, gfx_all):
    """install.sh and _runtime_gfx_target route on UNSLOTH_ROCM_GFX_ARCH alone."""
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", gfx)
    out = _setup_report(gfx, gfx_all, "AMD Radeon RX 7900 XTX")
    assert f"STEP: AMD GPU detected ({gfx}) -- no ROCm PyTorch wheels" in out, out


@_POSIX
def test_setup_sh_a_declared_covered_arch_is_unchanged(monkeypatch):
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1030")
    out = _setup_report("gfx1030", "gfx1010", "")
    assert "STEP: AMD ROCm (gfx1030)" in out, out


# ── install.sh final GPU summary ────────────────────────────────────────────


def _install_summary_gpu_step(probe: str, pinned: bool = False) -> str:
    src = _INSTALL_SH.read_text(encoding = "utf-8")
    start = src.index("    # The arch read fine and get_torch_index_url sent it to CPU")
    tail = '        step "gpu" "AMD GPU (no usable ROCm -- CPU fallback)" "$C_WARN"\n    fi\n'
    end = src.index(tail, start) + len(tail)
    script = (
        'step() { echo "STEP: $2"; }\n_has_amd_rocm_gpu() { return 0; }\n'
        f"_probe_amd_gfx_arch() {{ printf '%s\\n' '{probe}'; }}\n"
        f"_torch_index_pinned={'true' if pinned else 'false'}\n_torch_index_leaf=cpu\n"
        + guards._sh_function_body(src, "_hsa_override_gfx_arch")
        + "\nif false; then :\nelif _has_amd_rocm_gpu; then\n"
        + src[start:end]
        + "fi\n"
    )
    out = subprocess.run(["sh", "-c", script], stdout = subprocess.PIPE, text = True, timeout = 30)
    return out.stdout


@_POSIX
@pytest.mark.parametrize("probe", ["gfx1010", "gfx1010:xnack-\ngfx1010", "gfx803"])
def test_install_summary_names_the_uncovered_arch(probe):
    bare = probe.split(":")[0].split("\n")[0]
    out = _install_summary_gpu_step(probe)
    assert f"STEP: AMD GPU ({bare}, no ROCm PyTorch wheels -- CPU fallback)" in out, out
    assert "no usable ROCm" not in out, out


@_POSIX
@pytest.mark.parametrize("probe", ["", "gfx1010\ngfx1100", "gfx1030"])
def test_install_summary_other_hosts_are_unchanged(probe):
    assert "STEP: AMD GPU (no usable ROCm -- CPU fallback)" in _install_summary_gpu_step(probe)


@_POSIX
def test_install_summary_a_pin_still_wins():
    out = _install_summary_gpu_step("gfx1010", pinned = True)
    assert "STEP: AMD GPU (torch index pinned: cpu)" in out, out


def _amd_smi_probe_harness(monkeypatch, physical: str):
    """The stub probe answers like amd-smi: the physical arch, whatever the spoof says."""
    real = guards._INDEX_URL_STUBS
    monkeypatch.setattr(
        guards,
        "_INDEX_URL_STUBS",
        real.replace(
            "_probe_amd_gfx_arch() { printf '%s\\n' \"${UNSLOTH_ROCM_GFX_ARCH:-}\" "
            "| tr '[:upper:]' '[:lower:]'; }",
            f"_probe_amd_gfx_arch() {{ echo {physical}; }}",
        ),
    )
    assert guards._INDEX_URL_STUBS != real, "probe stub not found"


@_POSIX
@pytest.mark.parametrize(
    "physical,hsa,tail",
    [("gfx1010", "10.3.0", "/rocm6.4"), ("gfx1012", "10.1.0", "/cpu"), ("gfx1010", "ten", "/cpu")],
)
def test_install_sh_an_hsa_spoof_is_judged_by_its_target(monkeypatch, physical, hsa, tail):
    _amd_smi_probe_harness(monkeypatch, physical)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", hsa)
    url, err = guards._run_sh_get_torch_index_url("", rocm_tag = "rocm6.4")
    assert url.endswith(tail), (url, err)


@_POSIX
def test_install_sh_the_warning_names_the_spoof_escape(monkeypatch):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising = False)
    _url, err = guards._run_sh_get_torch_index_url("gfx1010", rocm_tag = "rocm6.4")
    assert "HSA_OVERRIDE_GFX_VERSION, export it and re-run" in err, err


@_POSIX
def test_setup_sh_a_spoof_onto_an_uncovered_arch_still_warns(monkeypatch):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.1.0")
    out = _setup_report("gfx1012", "gfx1012", "AMD Radeon RX 5500 XT")
    assert "STEP: AMD GPU detected (gfx1010) -- no ROCm PyTorch wheels" in out, out


@_POSIX
@pytest.mark.parametrize(
    "env",
    [{"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}, {"ROCR_VISIBLE_DEVICES": "1"}],
    ids = ["hsa-spoof", "rocr-mask"],
)
def test_setup_sh_a_spoof_or_rocr_mask_keeps_the_old_report(monkeypatch, env):
    """rocminfo in setup.sh is ROCR-filtered, so a masked covered peer is invisible there."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    out = _setup_report("gfx1010", "gfx1010", "AMD Radeon RX 5700 XT")
    assert "STEP: AMD Radeon RX 5700 XT (gfx1010)" in out, out


@_POSIX
def test_setup_sh_a_declared_arch_outranks_the_rocr_mask(monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1010")
    out = _setup_report("gfx1010", "gfx1010", "")
    assert "STEP: AMD GPU detected (gfx1010) -- no ROCm PyTorch wheels" in out, out


@_POSIX
@pytest.mark.parametrize(
    "hsa,step",
    [
        ("10.3.0", "AMD GPU (no usable ROCm -- CPU fallback)"),
        ("10.1.0", "AMD GPU (gfx1010, no ROCm PyTorch wheels -- CPU fallback)"),
    ],
)
def test_install_summary_an_hsa_spoof_is_judged_by_its_target(monkeypatch, hsa, step):
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", hsa)
    assert f"STEP: {step}" in _install_summary_gpu_step("gfx1012")


@_POSIX
@pytest.mark.parametrize("torch_version", ["2.10.0a0+rocm7.10.0", "2.9.1+rocm6.4"])
def test_setup_sh_a_kept_rocm_torch_is_not_called_cpu_only(torch_version):
    """studio update keeps an existing ROCm build (TheRock, hand-installed)."""
    out = _setup_report("gfx1010", "gfx1010", "AMD Radeon RX 5700 XT", torch_version)
    assert "STEP: AMD Radeon RX 5700 XT (gfx1010)" in out, out
    assert "CPU-only" not in out, out


@_POSIX
def test_setup_sh_a_cpu_torch_still_gets_the_warning():
    out = _setup_report("gfx1010", "gfx1010", "AMD Radeon RX 5700 XT", "2.10.0+cpu")
    assert "STEP: AMD GPU detected (gfx1010) -- no ROCm PyTorch wheels" in out, out
