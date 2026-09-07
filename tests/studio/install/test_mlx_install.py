# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MLX installs must honor the platform, install mode, and supported versions."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement

import install_python_stack as stack


class _BeforeExtras(Exception):
    pass


def _run_to_extras(monkeypatch, *, platform, skip_base, no_torch):
    monkeypatch.setenv("SKIP_STUDIO_BASE", "1" if skip_base else "0")
    for name in ("STUDIO_LOCAL_REPO", "STUDIO_PACKAGE_NAME", "UNSLOTH_CI_SOURCE_OVERLAY"):
        monkeypatch.delenv(name, raising = False)
    for name, value in {
        "IS_WINDOWS": platform == "windows",
        "IS_MACOS": platform.startswith("macos"),
        "IS_MAC_ARM": platform == "macos_arm",
        "NO_TORCH": no_torch,
        "_rocm_windows_torch_installed": False,
    }.items():
        monkeypatch.setattr(stack, name, value)
    for name, value in {
        "_bootstrap_uv": True,
        "_shared_base_requirements": None,
        "_repair_duplicate_core_metadata": True,
        "_repair_damaged_core_payload": True,
        "_bitsandbytes_installed": False,
        "_has_usable_nvidia_gpu": True,
        "_ensure_cuda_torch": None,
        "_ensure_rocm_torch": None,
        "_ensure_xpu_torch": None,
        "_ensure_cpu_torch": None,
        "_ensure_xpu_triton": None,
        "run": None,
    }.items():
        monkeypatch.setattr(stack, name, Mock(return_value = value))
    monkeypatch.setattr(stack.install_manifest, "remove_manifest", Mock(return_value = True))
    monkeypatch.setattr(stack.install_manifest, "set_no_torch_marker", Mock())
    install = Mock()
    monkeypatch.setattr(stack, "pip_install", install)
    progress = stack._progress

    def stop_before_extras(label):
        if label == "unsloth extras":
            raise _BeforeExtras
        progress(label)

    monkeypatch.setattr(stack, "_progress", stop_before_extras)
    with pytest.raises(_BeforeExtras):
        stack.install_python_stack()
    return [call for call in install.call_args_list if call.args[0].startswith("Installing MLX")]


def _repair_specs():
    path = Path(stack.SCRIPT_DIR) / "backend" / "utils" / "mlx_repair.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_MLX_INSTALL_SPECS"
            for target in node.targets
        )
    )


@pytest.mark.parametrize("platform", ["macos_arm", "macos_intel", "linux", "windows"])
@pytest.mark.parametrize("skip_base", [True, False], ids = ["fresh", "update"])
@pytest.mark.parametrize("no_torch", [False, True], ids = ["training", "gguf_only"])
def test_mlx_install_respects_platform_mode_and_pins(monkeypatch, platform, skip_base, no_torch):
    calls = _run_to_extras(monkeypatch, platform = platform, skip_base = skip_base, no_torch = no_torch)
    enabled = platform == "macos_arm" and not no_torch
    assert len(calls) == int(enabled)
    if platform.startswith("macos"):
        assert stack._TOTAL == (11 if skip_base else 12) + int(enabled)
    if enabled:
        requirements = [Requirement(arg) for arg in calls[0].args[1:] if not arg.startswith("-")]
        actual = {req.name: str(req.specifier) for req in requirements}
        expected = _repair_specs()
        expected["mlx-metal"] = expected["mlx"]
        assert actual == {
            name: str(Requirement(name + spec).specifier) for name, spec in expected.items()
        }
