# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""unslothai/unsloth#11815: every Windows RDNA arch installs from AMD's stable multi-arch index.

What this file pins: the arch set is exactly RDNA 1 through 4 minus gfx1033; every one of
them resolves to the multi-arch index with the pinned trio by default; a family-layout
mirror keeps the per-family route for arches that have one and changes nothing for RDNA 1;
a multi-arch mirror wins over a family mirror; gfx1033 and CDNA keep their family; the two
PowerShell installers carry the same arch set and setup.ps1 counts every arch in it as
having wheels.
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
        "studio_install_python_stack_multiarch_all", _STACK_PY
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()

_EXPECTED = {
    "gfx1010",
    "gfx1011",
    "gfx1012",
    "gfx1030",
    "gfx1031",
    "gfx1032",
    "gfx1034",
    "gfx1035",
    "gfx1036",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
    "gfx1200",
    "gfx1201",
}
_MULTIARCH = "https://repo.amd.com/rocm/whl-multi-arch/"


@pytest.fixture
def stock(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ROCM_WINDOWS_MIRROR", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR", raising = False)
    monkeypatch.setattr(stack_mod, "_ROCM_WINDOWS_INDEX_BASE", "https://repo.amd.com/rocm/whl")
    monkeypatch.setattr(
        stack_mod, "_ROCM_WINDOWS_MULTIARCH_INDEX_BASE", "https://repo.amd.com/rocm/whl-multi-arch"
    )
    return monkeypatch


class TestTheArchSet:
    def test_is_exactly_rdna_one_to_four_minus_van_gogh(self):
        assert set(stack_mod._WINDOWS_MULTIARCH_GFX) == _EXPECTED

    def test_van_gogh_and_cdna_are_out(self):
        for gfx in ("gfx1033", "gfx908", "gfx90a"):
            assert gfx not in stack_mod._WINDOWS_MULTIARCH_GFX
        assert "gfx1033" in stack_mod._ROCM_MISCOMPUTING_GFX


class TestDefaultRoute:
    @pytest.mark.parametrize("gfx", sorted(_EXPECTED))
    def test_resolves_to_the_multiarch_index_with_the_pinned_trio(self, gfx, stock):
        assert stack_mod._windows_rocm_index_url(gfx) == _MULTIARCH
        assert stack_mod._windows_rocm_index_url(gfx.upper() + ":xnack-") == _MULTIARCH
        tag = stack_mod._ROCM_MULTIARCH_TAG
        assert stack_mod._windows_rocm_torch_pkg_specs(gfx) == (
            f"torch[device-{gfx}]=={stack_mod._ROCM_MULTIARCH_TORCH_VERSION}+{tag}",
            f"torchvision=={stack_mod._ROCM_MULTIARCH_TORCHVISION_VERSION}+{tag}",
            f"torchaudio=={stack_mod._ROCM_MULTIARCH_TORCHAUDIO_VERSION}+{tag}",
        )

    def test_van_gogh_and_cdna_keep_their_family(self, stock):
        assert (
            stack_mod._windows_rocm_index_url("gfx1033")
            == "https://repo.amd.com/rocm/whl/gfx103X-all/"
        )
        assert (
            stack_mod._windows_rocm_index_url("gfx90a") == "https://repo.amd.com/rocm/whl/gfx90a/"
        )
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1033") == (
            "torch",
            "torchvision",
            "torchaudio",
        )

    def test_unknown_stays_none(self, stock):
        assert stack_mod._windows_rocm_index_url("gfx9999") is None
        assert stack_mod._windows_rocm_index_url(None) is None


class TestMirrors:
    def test_a_family_mirror_keeps_the_family_route_where_one_exists(self, stock):
        stock.setenv("UNSLOTH_ROCM_WINDOWS_MIRROR", "https://mirror.example/whl")
        stock.setattr(stack_mod, "_ROCM_WINDOWS_INDEX_BASE", "https://mirror.example/whl")
        assert (
            stack_mod._windows_rocm_index_url("gfx1034")
            == "https://mirror.example/whl/gfx103X-all/"
        )
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1034") == (
            "torch",
            "torchvision",
            "torchaudio",
        )
        assert (
            stack_mod._windows_rocm_index_url("gfx1201")
            == "https://mirror.example/whl/gfx120X-all/"
        )
        assert (
            stack_mod._windows_rocm_torch_pkg_specs("gfx1201")
            == stack_mod._WINDOWS_ROCM_TORCH_PKG_SPECS["gfx1201"]
        )
        # No family to fall back to: RDNA 1 and gfx1153 stay on the multi-arch index.
        assert stack_mod._windows_rocm_index_url("gfx1010") == _MULTIARCH
        assert stack_mod._windows_rocm_index_url("gfx1153") == _MULTIARCH

    def test_a_multiarch_mirror_wins_over_a_family_mirror(self, stock):
        stock.setenv("UNSLOTH_ROCM_WINDOWS_MIRROR", "https://mirror.example/whl")
        stock.setattr(stack_mod, "_ROCM_WINDOWS_INDEX_BASE", "https://mirror.example/whl")
        stock.setenv(
            "UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR", "https://mirror.example/whl-multi-arch"
        )
        stock.setattr(
            stack_mod, "_ROCM_WINDOWS_MULTIARCH_INDEX_BASE", "https://mirror.example/whl-multi-arch"
        )
        assert (
            stack_mod._windows_rocm_index_url("gfx1034") == "https://mirror.example/whl-multi-arch/"
        )

    def test_a_patched_family_base_counts_as_a_family_mirror(self, stock):
        """Tests and callers that set _ROCM_WINDOWS_INDEX_BASE directly expect the family
        leaf under it (the constant is read once at import from the env)."""
        stock.setattr(stack_mod, "_ROCM_WINDOWS_INDEX_BASE", "https://mirror.example/whl/")
        assert stack_mod._windows_rocm_index_url("gfx1151") == "https://mirror.example/whl/gfx1151/"


class TestPowerShellAgrees:
    @staticmethod
    def _list(path):
        src = path.read_text(encoding = "utf-8")
        block = src[src.index("$multiArchGfx = @(") :]
        block = block[: block.index(")") + 1]
        return set(re.findall(r'"(gfx[0-9a-z]+)"', block))

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    def test_same_arch_set(self, path):
        assert self._list(path) == _EXPECTED, f"{path.name}: {sorted(self._list(path) ^ _EXPECTED)}"

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    def test_family_mirror_rule_is_mirrored(self, path):
        src = path.read_text(encoding = "utf-8")
        assert (
            "$_familyMirrorPinned = [bool]($env:UNSLOTH_ROCM_WINDOWS_MIRROR) -and -not [bool]($env:UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR)"
            in src
        )
        assert "-and -not ($archFamily -and $_familyMirrorPinned)" in src
        assert "is RDNA 1 --" not in src

    def test_setup_counts_every_multiarch_arch_as_having_wheels(self):
        src = _SETUP_PS1.read_text(encoding = "utf-8")
        block = src[src.index("$_rocmWheelArches = @(") :]
        block = block[: block.index("\n)") + 2]
        listed = set(re.findall(r'"(gfx[0-9a-z]+)"', block))
        missing = (_EXPECTED - {"gfx1010", "gfx1011", "gfx1012"}) - listed
        assert not missing, f"setup.ps1 $_rocmWheelArches lacks {sorted(missing)}"
