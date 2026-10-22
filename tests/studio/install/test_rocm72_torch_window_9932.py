# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generic rocm7.2 index admits torch 2.11-2.14 (#9932) in the update path too; the AMD
per-arch leaves and the Windows pins stay on 2.11, and a rocm7.2 pin accepts the whole window."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "install_python_stack_9932", REPO / "studio" / "install_python_stack.py"
)
stack = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stack
_spec.loader.exec_module(stack)

_TRIO_211 = (
    "torch>=2.11.0,<2.12.0",
    "torchvision>=0.26.0,<0.27.0",
    "torchaudio>=2.11.0,<2.12.0",
)


def _install_sh_case(leaf_pattern: str) -> tuple[str, str, str]:
    """The (torch, torchvision, torchaudio) install.sh sets in the arm named by leaf_pattern."""
    src = (REPO / "install.sh").read_text(encoding = "utf-8")
    block = src[src.index("# rocm7.2 and per-gfx indexes ship torch 2.11.0") :]
    block = block[: block.index("\nesac\n")]
    arm = block[block.index(f"    {leaf_pattern})") :]
    arm = arm[: arm.index(";;")]
    return tuple(
        re.search(rf'{name}_CONSTRAINT="([^"]+)"', arm).group(1)
        for name in ("TORCH", "TORCHVISION", "TORCHAUDIO")
    )


def test_generic_rocm72_spec_matches_install_sh():
    assert stack._ROCM_TORCH_PKG_SPECS["rocm7.2"] == _install_sh_case("rocm7.2")
    assert stack._ROCM_TORCH_PKG_SPECS["rocm7.2"] == (
        "torch>=2.11.0,<2.15.0",
        "torchvision>=0.26.0,<0.30.0",
        "torchaudio>=2.11.0,<2.12.0",
    )


def test_per_arch_spec_matches_install_sh_and_stays_211():
    assert stack._ROCM_TORCH211_PKG_SPEC == _TRIO_211
    assert (
        _install_sh_case("gfx120x-all|gfx1151|gfx1150|gfx1152|gfx103x-all|gfx110x-all") == _TRIO_211
    )


def test_windows_pins_unchanged():
    assert set(stack._WINDOWS_ROCM_TORCH_PKG_SPECS.values()) == {_TRIO_211}


def test_ceiling_constant_matches_spec():
    hi = re.search(r"<(\d+)\.(\d+)\.0", stack._ROCM_TORCH_PKG_SPECS["rocm7.2"][0])
    assert (int(hi.group(1)), int(hi.group(2))) == stack._ROCM72_TORCH_CEILING


@pytest.mark.parametrize(
    "installed, mismatch",
    [
        ("2.11.0+rocm7.2", False),
        ("2.12.1+rocm7.2", False),
        ("2.14.1+rocm7.2", False),
        ("2.10.0+rocm7.2", True),
        ("2.15.0+rocm7.2", True),
        ("2.14.1+rocm7.1", True),
        ("2.14.1", True),
    ],
)
def test_rocm72_pin_accepts_the_window(installed, mismatch):
    assert (
        stack._rocm_pin_family_mismatch("https://download.pytorch.org/whl/rocm7.2", installed)
        is mismatch
    )
