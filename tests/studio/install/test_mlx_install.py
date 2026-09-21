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
    # What the repository declares, not what the host's installed zoo narrows it to; the
    # narrowing has its own tests below.
    monkeypatch.setattr(stack, "_mlx_vlm_spec_for_installed_zoo", lambda: stack._MLX_VLM_SPEC)
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
        # An update without torch announces the no-torch runtime deps on their own slot.
        # Two mac-arm slots: the MLX step, and the re-resolve after the core phase.
        assert stack._TOTAL == (
            (12 if skip_base and not shared_base else 13)
            + 2 * int(enabled)
            + int(no_torch and not skip_base)
        )
    if enabled:
        # --upgrade-package takes a bare NAME, not a pin: skip its argument.
        args = list(calls[0].args[1:])
        pins = [
            arg
            for index, arg in enumerate(args)
            if not arg.startswith("-") and (index == 0 or args[index - 1] != "--upgrade-package")
        ]
        requirements = [Requirement(arg) for arg in pins]
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
    args = list(call.args[1:])
    upgraded = {
        arg for index, arg in enumerate(args) if index and args[index - 1] == "--upgrade-package"
    }
    pins = {
        arg for index, arg in enumerate(args) if not arg.startswith("-") and arg not in upgraded
    }
    assert upgraded == {"mlx", "mlx-metal", "mlx-lm", "mlx-vlm"}
    for command in commands:
        assert pins <= set(command)
        assert "-c" in command
    # The upgrade INTENT must survive both spellings: a dropped pip translation made the fallback a
    # silent no-op, and a bare uv --upgrade refetches ~60 MB per update.
    assert "--upgrade" not in commands[0]
    for name in upgraded:
        assert commands[0][commands[0].index("--upgrade-package") :].count(name) == 1
    if len(commands) > 1:
        assert "--upgrade" in commands[1]
        assert "--upgrade-package" not in commands[1]
        # ...and no project twice: pip refuses "mlx==0.32.2 mlx" with "Double requirement given".
        projects = [
            arg.split(";")[0].split("[")[0].split("=")[0].split("<")[0].split(">")[0].strip()
            for arg in commands[1]
            if arg and not arg.startswith("-") and arg != sys.executable
        ]
        assert len(projects) == len(set(projects)), projects


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
    """0.32.2 ships macosx_14_0_arm64 wheels, no sdist, and the pinned set starts at cp310."""
    monkeypatch.setattr(stack.sys, "version_info", python_version)
    monkeypatch.setattr(stack, "_macos_release_major", Mock(return_value = macos_major))
    assert stack._mlx_pins_are_installable() is installable


def test_pin_floor_is_revisited_whenever_the_pins_move():
    """A pin bumped without its floor silently starts failing installs, so tie them here."""
    assert _repair_specs() == {
        "mlx": "==0.32.2",
        "mlx-lm": "==0.31.3",
        "mlx-vlm": ">=0.4.4,<=0.7.1",
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
    # Two mac-arm slots even with no wheel: the re-resolve slot is spent unconditionally.
    assert stack._TOTAL == (12 if skip_base and not shared_base else 13) + 2


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


def test_mlx_vlm_spec_is_intersected_with_the_installed_zoo(monkeypatch):
    """The MLX step runs before the core phase, and SKIP_STUDIO_BASE=1 skips that phase
    altogether, so a zoo predating the gated_delta_update fix would be left beside mlx-vlm
    0.7.1 and Qwen3.5 VLM training would raise TypeError at its first step."""
    import importlib.metadata

    monkeypatch.setattr(
        importlib.metadata,
        "requires",
        lambda _name: [
            'mlx==0.32.1; sys_platform == "darwin" and platform_machine == "arm64"',
            'mlx-vlm<0.7.0,>=0.4.4; sys_platform == "darwin" and platform_machine == "arm64"',
        ],
    )
    spec = stack._mlx_vlm_spec_for_installed_zoo()
    assert spec.startswith(stack._MLX_VLM_SPEC)
    assert "<0.7.0" in spec


def test_mlx_vlm_spec_is_unchanged_without_an_installed_zoo(monkeypatch):
    import importlib.metadata

    def _raise(_name):
        raise importlib.metadata.PackageNotFoundError("unsloth_zoo")

    monkeypatch.setattr(importlib.metadata, "requires", _raise)
    assert stack._mlx_vlm_spec_for_installed_zoo() == stack._MLX_VLM_SPEC


def test_the_skip_predicate_uses_the_narrowed_range(monkeypatch):
    """0.7.1 already installed beside a zoo declaring <0.7.0 must NOT read as current, or the
    step skips the install that would put mlx-vlm back where the zoo can drive it."""
    monkeypatch.setattr(stack, "_exact_distribution_spec_is_installed", lambda _spec: True)
    monkeypatch.setattr(stack, "_installed_distribution_version", lambda _name: "0.7.1")
    monkeypatch.setattr(stack, "_mlx_closure_unmet", lambda: [])

    monkeypatch.setattr(stack, "_mlx_vlm_spec_for_installed_zoo", lambda: stack._MLX_VLM_SPEC)
    assert stack._mlx_stack_is_current() is True

    monkeypatch.setattr(
        stack, "_mlx_vlm_spec_for_installed_zoo", lambda: f"{stack._MLX_VLM_SPEC},<0.7.0"
    )
    assert stack._mlx_stack_is_current() is False


def _zoo_spec_sequence(monkeypatch, specs):
    """_mlx_vlm_spec_for_installed_zoo answering differently before and after the core phase."""
    remaining = list(specs)

    def spec():
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    monkeypatch.setattr(stack, "_mlx_vlm_spec_for_installed_zoo", spec)


def test_the_core_phase_upgrading_the_zoo_re_resolves_mlx(monkeypatch):
    """The MLX step runs before the core phase, so it honours the OLD zoo's range. Without the
    re-resolve an update that moves the zoo leaves the machine on the narrower mlx-vlm for good:
    the startup self-heal will not correct it, since 0.6.x satisfies _MLX_MIN_VERSIONS."""
    narrow = f"{stack._MLX_VLM_SPEC},<0.7.0"
    # Twice for the install step, then the widened answer after the core phase.
    _zoo_spec_sequence(monkeypatch, [narrow, narrow, stack._MLX_VLM_SPEC])
    calls = _run_to_extras(monkeypatch, platform = "macos_arm", skip_base = False, no_torch = False)
    assert len(calls) == 2
    assert "MLX stack (re-resolved for the new zoo)" in _run_to_extras.steps
    assert narrow in list(calls[0].args[1:])
    assert stack._MLX_VLM_SPEC in list(calls[1].args[1:])


def test_an_unchanged_zoo_does_not_re_resolve_mlx(monkeypatch):
    _zoo_spec_sequence(monkeypatch, [stack._MLX_VLM_SPEC])
    calls = _run_to_extras(monkeypatch, platform = "macos_arm", skip_base = False, no_torch = False)
    assert len(calls) == 1
    assert "MLX stack (zoo unchanged, skipped)" in _run_to_extras.steps
