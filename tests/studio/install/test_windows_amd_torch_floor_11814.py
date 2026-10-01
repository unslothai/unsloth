# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""unslothai/unsloth#11814: every Windows AMD family gets the torch 2.11 floor.

install.ps1 keeps a venv's existing torch release on reinstall (Get-PreviousTorchPin)
unless the arch has a ROCm floor to veto it. With the floor on five arches only, a
gfx103X-all venv from an earlier install stayed on 2.10.0+rocm7.13.0, whose _grouped_mm
access-violates on an RX 6500 XT and takes `import unsloth` down with it. What this file
pins: the Windows-routed RDNA arches all carry the 2.11 trio in the python repair map, the
two PowerShell installers carry the same keys with the same specs, and the 2.11-allowlist
leaves agree with the family map for those arches.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_STACK_PY = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"


def _load_stack_module():
    spec = importlib.util.spec_from_file_location(
        "studio_install_python_stack_floor_11814", _STACK_PY
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()

# The families whose per-arch index has Windows wheels and a sub-2.11 build with the bug.
_FLOORED_FAMILIES = {"gfx103X-all", "gfx110X-all", "gfx1150", "gfx1151", "gfx1152", "gfx120X-all"}
_RDNA_WINDOWS_ARCHES = sorted(
    gfx for gfx, fam in stack_mod._GFX_TO_AMD_INDEX_ARCH.items() if fam in _FLOORED_FAMILIES
)
_TRIO = stack_mod._ROCM_TORCH_PKG_SPECS["rocm7.2"]


def _ps_map(path, name):
    src = path.read_text(encoding = "utf-8")
    m = re.search(r"\$" + name + r"\s*=\s*@\{(.*?)\n\s*\}", src, re.DOTALL)
    assert m, f"{path.name}: ${name} not found"
    return dict(re.findall(r'"(gfx[0-9a-z]+)"\s*=\s*"([^"]+)"', m.group(1)))


class TestPythonRepairMap:
    def test_the_arches_under_test_are_real(self):
        assert "gfx1034" in _RDNA_WINDOWS_ARCHES and "gfx1103" in _RDNA_WINDOWS_ARCHES

    @pytest.mark.parametrize("gfx", _RDNA_WINDOWS_ARCHES)
    def test_every_windows_rdna_arch_pins_the_211_trio(self, gfx):
        assert stack_mod._WINDOWS_ROCM_TORCH_PKG_SPECS.get(gfx) == _TRIO, f"{gfx} has no 2.11 floor"

    def test_the_trio_is_the_211_floor(self):
        torch_spec, vision_spec, audio_spec = _TRIO
        assert torch_spec.startswith("torch>=2.11.0,<2.12.0")
        assert vision_spec.startswith("torchvision>=0.26.0,<0.27.0")
        assert audio_spec.startswith("torchaudio>=2.11.0,<2.12.0")

    @pytest.mark.parametrize("gfx", _RDNA_WINDOWS_ARCHES)
    def test_the_family_leaf_is_in_the_211_allowlist(self, gfx):
        """The stale/mismatch checks key on the family leaf, the install on the arch: a
        floor in one and not the other reinstalls every update or never repairs."""
        leaf = stack_mod._GFX_TO_AMD_INDEX_ARCH[gfx].lower()
        assert (
            leaf in stack_mod._ROCM_GFX_TORCH211_LEAVES
        ), f"{gfx} -> {leaf} not in the 2.11 allowlist"

    def test_cdna_stays_bare(self):
        """gfx908 / gfx90a: no Windows wheels, unmeasured; Linux reaches them via the
        floored rocm7.2 index. Deliberately not in the allowlist."""
        assert "gfx908" not in stack_mod._ROCM_GFX_TORCH211_LEAVES
        assert "gfx90a" not in stack_mod._ROCM_GFX_TORCH211_LEAVES
        assert "gfx908" not in stack_mod._WINDOWS_ROCM_TORCH_PKG_SPECS


class TestPowerShellMirrorsTheFloor:
    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    def test_same_keys_and_specs_in_all_three_maps(self, path):
        torch_map = _ps_map(path, "torchFloorMap")
        vision_map = _ps_map(path, "torchvisionFloorMap")
        audio_map = _ps_map(path, "torchaudioFloorMap")
        expected = set(stack_mod._WINDOWS_ROCM_TORCH_PKG_SPECS)
        assert (
            set(torch_map) == expected
        ), f"{path.name}: torchFloorMap keys {sorted(set(torch_map) ^ expected)}"
        assert set(vision_map) == expected and set(audio_map) == expected
        for gfx in expected:
            assert (torch_map[gfx], vision_map[gfx], audio_map[gfx]) == _TRIO, f"{path.name}: {gfx}"

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    def test_the_211_allowlist_names_the_same_leaves(self, path):
        src = path.read_text(encoding = "utf-8")
        m = re.search(r"@\('gfx120x-all'[^)]*\)", src)
        assert m, f"{path.name}: 2.11 allowlist literal not found"
        leaves = set(re.findall(r"'([^']+)'", m.group(0)))
        assert leaves == set(stack_mod._ROCM_GFX_TORCH211_LEAVES), f"{path.name}: {sorted(leaves)}"
