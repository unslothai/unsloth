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
    mlx_installable = True,
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
        # Off macOS the floor would skip every case; it has its own tests.
        "_mlx_pins_are_installable": mlx_installable,
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

    steps: list[str] = []

    def stop_before_extras(label):
        if label == "unsloth extras":
            raise _BeforeExtras
        steps.append(label)
        progress(label)

    monkeypatch.setattr(stack, "_progress", stop_before_extras)
    with pytest.raises(_BeforeExtras):
        stack.install_python_stack()
    # Labels reached before the stop, for callers asserting on the step not the install.
    _run_to_extras.steps = steps
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
        assert stack._TOTAL == (12 if skip_base and not shared_base else 13) + int(enabled)
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


@pytest.mark.parametrize(
    "python_version, macos_major, installable",
    [
        ((3, 9, 6), 15, False),  # macOS ships 3.9; no release in the pinned set has a cp39 wheel
        ((3, 10, 0), 15, True),
        ((3, 12, 0), 13, False),  # Apple Silicon on Ventura: the pins are macosx_14_0 only
        ((3, 12, 0), 14, True),  # the first macOS the pinned wheels are built for
        ((3, 12, 0), 26, True),
        # Unreadable version: skipping costs a launch, attempting costs the install.
        ((3, 12, 0), None, False),
        ((3, 9, 6), 13, False),
    ],
)
def test_mlx_pin_floor_matches_the_published_wheels(
    monkeypatch, python_version, macos_major, installable
):
    """0.32.1 ships macosx_14_0_arm64 wheels, no sdist, and the pinned set starts at cp310."""
    monkeypatch.setattr(stack.sys, "version_info", python_version)
    monkeypatch.setattr(stack, "_macos_release_major", Mock(return_value = macos_major))
    assert stack._mlx_pins_are_installable() is installable


def test_pin_floor_is_revisited_whenever_the_pins_move():
    """A pin bumped without its floor silently starts failing installs, so tie them here."""
    assert _repair_specs() == {
        "mlx": "==0.32.1",
        "mlx-lm": "==0.31.3",
        "mlx-vlm": ">=0.4.4,<0.7.0",
    }
    assert (stack._MLX_MIN_PYTHON, stack._MLX_MIN_MACOS_MAJOR) == ((3, 10), 14)


@pytest.mark.parametrize("skip_base", [True, False], ids = ["fresh", "update"])
@pytest.mark.parametrize("shared_base", [False, True], ids = ["empty_base", "shared_base"])
def test_unsupported_apple_silicon_skips_mlx_without_failing_the_install(
    monkeypatch, skip_base, shared_base
):
    """macOS 13 / Python 3.9 Apple Silicon still installs; it just stays chat-only.

    Fresh never ran this step and update was unpinned, so neither could exit here before.
    """
    calls = _run_to_extras(
        monkeypatch,
        platform = "macos_arm",
        skip_base = skip_base,
        no_torch = False,
        shared_base = shared_base,
        mlx_installable = False,
    )
    assert calls == []
    steps = _run_to_extras.steps
    assert "MLX stack (Apple Silicon)" not in steps
    assert "MLX stack (skipped, no wheel for this macOS or Python)" in steps
    # A skipped step still spends its slot.
    assert stack._TOTAL == (12 if skip_base and not shared_base else 13) + 1


def test_supported_and_unsupported_hosts_share_one_progress_budget(monkeypatch):
    """Same total either way, so the bar cannot end at 13/14 on an old Mac."""
    _run_to_extras(
        monkeypatch,
        platform = "macos_arm",
        skip_base = True,
        no_torch = False,
        mlx_installable = True,
    )
    supported = stack._TOTAL
    _run_to_extras(
        monkeypatch,
        platform = "macos_arm",
        skip_base = True,
        no_torch = False,
        mlx_installable = False,
    )
    assert stack._TOTAL == supported
