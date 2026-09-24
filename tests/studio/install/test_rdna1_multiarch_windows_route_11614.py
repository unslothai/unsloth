# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""RDNA 1 on Windows installs from AMD's multi-arch index (unslothai/unsloth#11614).

An RX 5700 XT (gfx1010) has no family on repo.amd.com/rocm/whl, so until now the Windows
installers put it on CPU torch and said so (#8529). AMD's multi-arch index
(repo.amd.com/rocm/whl-multi-arch) carries per-card kernel packs; there
`torch[device-gfx1010]` resolves torch plus amd-torch-device-gfx1010. What this file pins
down about that route:

* the resolvers send gfx1010 / gfx1011 / gfx1012 to that index, in every spelling hipinfo
  or a user can produce, and to nothing else;
* the package specs are PINNED to one release tag (the index serves several, and the
  newest sits on the Windows torch ceiling), with torchvision and torchaudio on that tag;
* the Windows GPU name tables resolve the RDNA 1 marketing names to their arch, and the
  "not covered" table no longer claims them;
* the PowerShell installers carry the same three arches and the same tag, so a bump in one
  place cannot leave the other installing a different build.
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
_HARDWARE_PY = PACKAGE_ROOT / "studio" / "backend" / "utils" / "hardware" / "hardware.py"


def _load_stack_module():
    spec = importlib.util.spec_from_file_location(
        "studio_install_python_stack_rdna1_route", _STACK_PY
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


stack_mod = _load_stack_module()

_RDNA1 = ["gfx1010", "gfx1011", "gfx1012"]
_SPELLINGS = _RDNA1 + [a.upper() for a in _RDNA1] + [f"{a}:xnack-" for a in _RDNA1]


class TestPackageSpecs:
    @pytest.mark.parametrize("arch", _SPELLINGS)
    def test_specs_are_pinned_to_one_tag_with_the_device_extra(self, arch):
        torch_spec, vision_spec, audio_spec = stack_mod._windows_rocm_torch_pkg_specs(arch)
        bare = arch.lower().split(":")[0]
        tag = stack_mod._ROCM_MULTIARCH_TAG
        assert (
            torch_spec == f"torch[device-{bare}]=={stack_mod._ROCM_MULTIARCH_TORCH_VERSION}+{tag}"
        )
        assert vision_spec == f"torchvision=={stack_mod._ROCM_MULTIARCH_TORCHVISION_VERSION}+{tag}"
        assert audio_spec == f"torchaudio=={stack_mod._ROCM_MULTIARCH_TORCHAUDIO_VERSION}+{tag}"

    def test_the_pin_is_a_release_tag_not_a_nightly(self):
        """A nightly tag (rocmX.Y.ZaYYYYMMDD) moves under users; the stable index is dated
        by release only."""
        assert re.fullmatch(r"rocm\d+\.\d+\.\d+", stack_mod._ROCM_MULTIARCH_TAG)

    def test_the_pin_sits_inside_the_windows_torch_window(self):
        """install.ps1 applies torch<2.12.0 everywhere else on Windows; the multi-arch index
        also serves 2.12.0, so the pin must stay the newest release under that ceiling."""
        major, minor, _ = (int(x) for x in stack_mod._ROCM_MULTIARCH_TORCH_VERSION.split("."))
        assert (major, minor) < (2, 12)
        assert (
            stack_mod._ROCM_MULTIARCH_TORCHAUDIO_VERSION == stack_mod._ROCM_MULTIARCH_TORCH_VERSION
        )

    def test_other_arches_keep_their_specs(self):
        assert (
            stack_mod._windows_rocm_torch_pkg_specs("gfx1201")
            == stack_mod._WINDOWS_ROCM_TORCH_PKG_SPECS["gfx1201"]
        )
        assert stack_mod._windows_rocm_torch_pkg_specs("gfx1034") == (
            "torch",
            "torchvision",
            "torchaudio",
        )
        assert stack_mod._windows_rocm_torch_pkg_specs(None) == (
            "torch",
            "torchvision",
            "torchaudio",
        )


class TestIndexResolution:
    @pytest.mark.parametrize("arch", _SPELLINGS)
    def test_rdna1_resolves_to_the_multiarch_index(self, arch, monkeypatch):
        monkeypatch.delenv("UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR", raising = False)
        assert (
            stack_mod._windows_rocm_index_url(arch) == stack_mod._ROCM_WINDOWS_MULTIARCH_INDEX_BASE
        )

    def test_the_default_base_is_amds_stable_multiarch_index(self):
        assert (
            stack_mod._ROCM_WINDOWS_MULTIARCH_INDEX_BASE
            == "https://repo.amd.com/rocm/whl-multi-arch"
        )

    def test_rdna2_still_resolves_to_its_family(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_ROCM_WINDOWS_MIRROR", raising = False)
        monkeypatch.setattr(stack_mod, "_ROCM_WINDOWS_INDEX_BASE", "https://repo.amd.com/rocm/whl")
        assert (
            stack_mod._windows_rocm_index_url("gfx1034")
            == "https://repo.amd.com/rocm/whl/gfx103X-all/"
        )

    def test_polaris_still_resolves_to_nothing(self):
        assert stack_mod._windows_rocm_index_url("gfx803") is None

    @pytest.mark.parametrize("arch", _RDNA1)
    def test_the_two_card_picker_counts_rdna1_as_having_wheels(self, arch):
        """_dedup_pick decides "has wheels" by asking the resolver, so an RX 5700 XT next to
        an iGPU is no longer deposed to CPU torch."""
        assert stack_mod._is_windows_multiarch_gfx(arch)
        assert stack_mod._windows_rocm_index_url(arch) is not None


_RDNA1_NAMES = [
    ("AMD Radeon RX 5700 XT", "gfx1010"),
    ("AMD Radeon RX 5600 XT", "gfx1010"),
    ("AMD Radeon Pro W5700", "gfx1010"),
    ("AMD Radeon Pro V520", "gfx1011"),
    ("AMD Radeon RX 5500 XT", "gfx1012"),
    ("AMD Radeon RX 5300M", "gfx1012"),
]


class TestNameTables:
    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_python_table_resolves_rdna1(self, name, expected):
        assert stack_mod._gfx_arch_from_gpu_name(name) == expected
        assert stack_mod._unsupported_gfx_arch_from_gpu_name(name) is None

    @pytest.mark.parametrize(
        "name", ["AMD Radeon RX 570", "AMD Radeon RX 580", "AMD Radeon RX 550"]
    )
    def test_polaris_is_not_swallowed_by_the_rdna1_rows(self, name):
        assert stack_mod._gfx_arch_from_gpu_name(name) is None

    @staticmethod
    def _ps_rows(path, header):
        src = path.read_text(encoding = "utf-8")
        start = src.index(header)
        i = src.index("(", start)
        depth = 0
        while True:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        return re.findall(
            r'@\{\s*P\s*=\s*"([^"]+)"\s*;\s*A\s*=\s*"(gfx[0-9a-z]+)"\s*\}', src[start : i + 1]
        )

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_powershell_tables_agree(self, path, name, expected):
        supported = self._ps_rows(path, "$nameArchTable = @(")
        unsupported = self._ps_rows(path, "$unsupportedNameArchTable = @(")
        hit = next((a for p, a in supported if re.search(p, name, re.IGNORECASE)), None)
        assert hit == expected, f"{path.name}: {name!r} -> {hit!r}"
        assert not any(
            re.search(p, name, re.IGNORECASE) for p, _ in unsupported
        ), f"{path.name} still calls {name!r} unsupported"

    @pytest.mark.parametrize("name,expected", _RDNA1_NAMES)
    def test_backend_table_agrees(self, name, expected):
        import ast

        tree = ast.parse(_HARDWARE_PY.read_text(encoding = "utf-8"))
        rows = None
        for node in tree.body:
            targets = (
                [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
            )
            if any(getattr(t, "id", "") == "_GPU_NAME_GFX_TABLE" for t in targets):
                rows = ast.literal_eval(node.value)
        assert rows, "_GPU_NAME_GFX_TABLE not found"
        hit = next((a for p, a in rows if re.search(p, name, re.IGNORECASE)), None)
        assert hit == expected


class TestPowerShellMirrorsThePin:
    """install.ps1 and setup.ps1 install torch themselves (the python stack only repairs),
    so they carry the same route. Read as text: the parity is the point."""

    @pytest.mark.parametrize("path", [_INSTALL_PS1, _SETUP_PS1], ids = lambda p: p.name)
    def test_same_arches_and_same_tag(self, path):
        src = path.read_text(encoding = "utf-8")
        assert "$multiArchGfx = @(" in src, f"{path.name}: no $multiArchGfx list"
        block = src[src.index("$multiArchGfx = @(") :]
        block = block[: block.index(")") + 1]
        archs = set(re.findall(r'"(gfx[0-9a-z]+)"', block))
        assert archs == set(stack_mod._WINDOWS_MULTIARCH_GFX), f"{path.name}: {archs}"
        assert (
            f'"{stack_mod._ROCM_MULTIARCH_TAG}"' in src
        ), f"{path.name}: tag differs from install_python_stack.py"
        assert f'"{stack_mod._ROCM_MULTIARCH_TORCH_VERSION}"' in src
        assert f'"{stack_mod._ROCM_MULTIARCH_TORCHVISION_VERSION}"' in src
        assert (
            f'$MultiArchTorchaudioVersion = "{stack_mod._ROCM_MULTIARCH_TORCHAUDIO_VERSION}"' in src
        )
        assert "https://repo.amd.com/rocm/whl-multi-arch" in src
        assert "nightly.repo.amd.com" not in src, f"{path.name} still names the nightly index"


class TestTheWindowsRepairSiteRunsForRdna1:
    """The `studio update` repair in _ensure_rocm_torch reaches the multi-arch trio for a
    gfx1010 host whose venv holds a CPU torch. Exercised end to end because the message
    there once called _bare_gfx(), a name the function later rebinds as a local, and no
    test ran that line."""

    def test_a_gfx1010_host_on_cpu_torch_installs_the_multiarch_trio(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        _mark = stack_mod._TORCH_PROBE_MARKER
        probe = MagicMock(returncode = 0, stdout = _mark + "2.10.0+cpu||" + chr(10))
        pip_try = MagicMock(return_value = True)
        monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
        monkeypatch.delenv("UNSLOTH_ROCM_WINDOWS_MULTIARCH_MIRROR", raising = False)
        with (
            patch.object(stack_mod, "IS_WINDOWS", True),
            patch.object(stack_mod, "IS_MACOS", False),
            patch.object(stack_mod, "_TORCH_BACKEND", ""),
            patch.object(stack_mod, "_explicit_rocm_torch_index_url", return_value = None),
            patch.object(stack_mod, "_explicit_unknown_family_torch_index_url", return_value = None),
            patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = False),
            patch.object(stack_mod, "_detect_windows_gfx_arch", return_value = "gfx1010"),
            patch.object(stack_mod, "_installed_rocm_wheel_family", return_value = None),
            patch.object(stack_mod, "_install_bnb_windows_rocm", return_value = True),
            patch.object(stack_mod, "pip_install_try", pip_try),
            patch("subprocess.run", return_value = probe),
        ):
            stack_mod._ensure_rocm_torch()
        assert pip_try.call_count == 1
        args = " ".join(str(a) for a in pip_try.call_args.args)
        assert "whl-multi-arch" in args
        assert (
            f"torch[device-gfx1010]=={stack_mod._ROCM_MULTIARCH_TORCH_VERSION}+{stack_mod._ROCM_MULTIARCH_TAG}"
            in args
        )
