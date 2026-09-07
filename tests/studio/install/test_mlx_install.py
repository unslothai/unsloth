# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""MLX installs must honor the platform, install mode, and supported versions."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
from packaging.requirements import Requirement

import install_python_stack as stack


class _BeforeExtras(Exception):
    pass


def _run_to_extras(
    monkeypatch,
    *,
    platform,
    skip_base,
    no_torch,
    shared_base = False,
    mlx_error = None,
):
    monkeypatch.setenv("SKIP_STUDIO_BASE", "1" if skip_base else "0")
    for name in ("STUDIO_LOCAL_REPO", "STUDIO_PACKAGE_NAME", "UNSLOTH_CI_SOURCE_OVERLAY"):
        monkeypatch.delenv(name, raising = False)
    for name, value in {
        "IS_WINDOWS": platform == "windows",
        "IS_LINUX": platform == "linux",
        "IS_MACOS": platform.startswith("macos"),
        "IS_MAC_ARM": platform == "macos_arm",
        "NO_TORCH": no_torch,
        "_rocm_windows_torch_installed": False,
    }.items():
        monkeypatch.setattr(stack, name, value)
    for name, value in {
        "_bootstrap_uv": True,
        "_shared_base_requirements": stack.REQ_ROOT / "base.txt" if shared_base else None,
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

    def record_install(label, *args, **kwargs):
        if label.startswith("Installing MLX") and mlx_error is not None:
            raise mlx_error

    install = Mock(side_effect = record_install)
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
@pytest.mark.parametrize("shared_base", [False, True], ids = ["empty_base", "shared_base"])
def test_mlx_install_respects_platform_mode_and_pins(
    monkeypatch, platform, skip_base, no_torch, shared_base
):
    calls = _run_to_extras(
        monkeypatch,
        platform = platform,
        skip_base = skip_base,
        no_torch = no_torch,
        shared_base = shared_base,
    )
    enabled = platform == "macos_arm" and not no_torch
    assert len(calls) == int(enabled)
    if platform.startswith("macos"):
        assert stack._TOTAL == (11 if skip_base and not shared_base else 12) + int(enabled)
    if enabled:
        requirements = [Requirement(arg) for arg in calls[0].args[1:] if not arg.startswith("-")]
        actual = {req.name: str(req.specifier) for req in requirements}
        expected = _repair_specs()
        expected["mlx-metal"] = expected["mlx"]
        assert actual == {
            name: str(Requirement(name + spec).specifier) for name, spec in expected.items()
        }


@pytest.mark.parametrize("skip_base", [True, False], ids = ["fresh", "update"])
@pytest.mark.parametrize(
    "error", [SystemExit(7), KeyboardInterrupt()], ids = ["failed", "interrupted"]
)
def test_failed_mlx_install_stops_before_extras(monkeypatch, skip_base, error):
    write_manifest = Mock()
    monkeypatch.setattr(stack.install_manifest, "write_manifest", write_manifest)
    with pytest.raises(type(error)):
        _run_to_extras(
            monkeypatch, platform = "macos_arm", skip_base = skip_base, no_torch = False, mlx_error = error
        )
    write_manifest.assert_not_called()


@pytest.mark.parametrize(
    "returncodes", [(0,), (1, 0), (1, 7)], ids = ["uv", "pip_fallback", "both_fail"]
)
def test_mlx_command_preserves_pins_and_interpreter_on_fallback(monkeypatch, returncodes):
    with monkeypatch.context() as phase:
        call = _run_to_extras(phase, platform = "macos_arm", skip_base = True, no_torch = False)[0]
    monkeypatch.setattr(stack, "USE_UV", True)
    monkeypatch.setattr(stack, "UV_NEEDS_SYSTEM", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    commands = []
    outcomes = iter(returncodes)

    def run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, next(outcomes), b"resolver failed")

    monkeypatch.setattr(stack.subprocess, "run", run)
    if returncodes[-1]:
        with pytest.raises(SystemExit) as exc:
            stack.pip_install(*call.args, **call.kwargs)
        assert exc.value.code == returncodes[-1]
    else:
        stack.pip_install(*call.args, **call.kwargs)
    assert len(commands) == len(returncodes)
    assert commands[0][:5] == ["uv", "pip", "install", "--python", sys.executable]
    if len(commands) > 1:
        assert commands[1][:4] == [sys.executable, "-m", "pip", "install"]
    for command in commands:
        assert {arg for arg in call.args[1:] if not arg.startswith("-")} <= set(command)
        assert "--upgrade" in command
        assert "-c" in command
