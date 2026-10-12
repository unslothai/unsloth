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
def _reset_probe():
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


def test_an_hsa_spoof_routes_as_the_spoofed_arch():
    """rocminfo under HSA_OVERRIDE_GFX_VERSION=10.3.0 reports gfx1030, the user's choice."""
    calls = routing._run_install(
        gfx_devices = ("gfx1030",),
        rocm_version = (6, 4),
        env = {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"},
    )
    assert "download.pytorch.org/whl/rocm6.4" in calls, calls


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
) -> str:
    src = _SETUP_SH.read_text(encoding = "utf-8")
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
        + src[start:end]
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH_TORCH_INDEX")}
    out = subprocess.run(
        ["bash", "-c", script], stdout = subprocess.PIPE, text = True, timeout = 30, env = env
    )
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
