# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth doctor` on Apple Silicon, where the MLX backend runs and torch sees no GPU.

These tests never need an Apple Silicon host: the verdict is a pure function of the
collected facts, exactly like `verdict` for the CUDA path.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli
import unsloth_cli.commands.doctor as doctor_mod


def _healthy_info(**overrides):
    info = {
        "platform": "Darwin arm64",
        "mlx": "0.32.4",
        "metal": True,
        "device": "Apple M5",
        "architecture": "applegpu_g17g",
        "memory_gib": 32.0,
        "working_set_gib": 24.96,
        "zoo": "2026.10.7",
        "zoo_mlx": True,
        "stack_blockers": [],
        "errors": [],
    }
    info.update(overrides)
    return info


def _fake_studio_gate(monkeypatch, mlx_active):
    """Install a fake studio.backend.core.training.training with the MLX gate."""
    for name in ("studio", "studio.backend", "studio.backend.core", "studio.backend.core.training"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    training = types.ModuleType("studio.backend.core.training.training")
    training.should_use_mlx_training_backend = lambda **kwargs: mlx_active
    monkeypatch.setitem(sys.modules, "studio.backend.core.training.training", training)


def test_verdict_mlx_healthy_on_apple_silicon():
    healthy, hints = doctor_mod.verdict_mlx(_healthy_info())
    assert healthy is True
    assert hints and "MLX backend" in hints[0]


def test_verdict_mlx_reports_missing_mlx():
    healthy, hints = doctor_mod.verdict_mlx(_healthy_info(mlx = None))
    assert healthy is False
    assert "mlx is not installed" in hints[0]


def test_verdict_mlx_reports_old_unsloth_zoo():
    healthy, hints = doctor_mod.verdict_mlx(_healthy_info(zoo_mlx = False))
    assert healthy is False
    assert "unsloth_zoo" in hints[0]


def test_verdict_mlx_reports_missing_metal():
    healthy, hints = doctor_mod.verdict_mlx(_healthy_info(metal = False))
    assert healthy is False
    assert "Metal is not available" in hints[0]


def test_verdict_mlx_reports_missing_device():
    healthy, hints = doctor_mod.verdict_mlx(_healthy_info(device = None))
    assert healthy is False
    assert "did not report a device" in hints[0]


def _broken_studio_gate(monkeypatch):
    _fake_studio_gate(monkeypatch, True)
    # A None entry makes the import raise ImportError, like a studio backend missing a dependency.
    monkeypatch.setitem(sys.modules, "studio.backend.core.training.training", None)


def test_verdict_mlx_reports_stack_blockers():
    blocker = "mlx_lm does not import (ModuleNotFoundError: No module named 'mlx_lm')"
    healthy, hints = doctor_mod.verdict_mlx(_healthy_info(stack_blockers = [blocker]))
    assert healthy is False
    assert blocker in hints[0]
    assert "unsloth studio update" in hints[0]


def test_verdict_mlx_fails_when_stack_check_could_not_run():
    healthy, hints = doctor_mod.verdict_mlx(_healthy_info(stack_blockers = None))
    assert healthy is False
    assert "could not run" in hints[0]


@pytest.mark.parametrize(
    "overrides",
    [
        {"mlx": None},
        {"zoo_mlx": False},
        {"stack_blockers": ["mlx-vlm 0.4.0 is older than 0.4.4"]},
        {"stack_blockers": None},
    ],
)
def test_verdict_mlx_never_suggests_a_bare_pip_install(overrides):
    # A plain pip resolve can move Transformers and leave Train/Export disabled.
    _, hints = doctor_mod.verdict_mlx(_healthy_info(**overrides))
    assert "unsloth studio update" in hints[0]
    assert "pip install" not in hints[0]


def _fake_mlx_repair(
    monkeypatch,
    blockers = None,
    broken = False,
):
    import unsloth_cli._inference as inference

    monkeypatch.setattr(inference, "ensure_studio_backend_path", lambda **kwargs: None)
    if broken:
        monkeypatch.setitem(sys.modules, "utils.mlx_repair", None)
        return
    utils = types.ModuleType("utils")
    utils.__path__ = []
    repair = types.ModuleType("utils.mlx_repair")
    repair.mlx_stack_blockers = lambda: list(blockers or [])
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.mlx_repair", repair)


def test_collect_mlx_info_reads_the_studio_mlx_gate(monkeypatch):
    _fake_mlx_repair(monkeypatch, blockers = ["mlx_vlm does not import (ImportError: boom)"])
    info = doctor_mod.collect_mlx_info()
    assert info["stack_blockers"] == ["mlx_vlm does not import (ImportError: boom)"]


def test_collect_mlx_info_notes_when_the_studio_mlx_gate_is_missing(monkeypatch):
    _fake_mlx_repair(monkeypatch, broken = True)
    info = doctor_mod.collect_mlx_info()
    assert info["stack_blockers"] is None
    assert any(e.startswith("mlx stack check:") for e in info["errors"])


def test_mlx_backend_active_honours_force_gpu_path(monkeypatch):
    _fake_studio_gate(monkeypatch, True)
    monkeypatch.setenv("UNSLOTH_FORCE_GPU_PATH", "1")
    assert doctor_mod.mlx_backend_active() is False


def test_mlx_backend_active_on_apple_silicon(monkeypatch):
    _fake_studio_gate(monkeypatch, True)
    monkeypatch.delenv("UNSLOTH_FORCE_GPU_PATH", raising = False)
    assert doctor_mod.mlx_backend_active() is True


def test_mlx_backend_active_without_mlx_installed(monkeypatch):
    # Missing mlx still routes to the MLX checks, whose verdict says how to install it.
    _fake_studio_gate(monkeypatch, True)
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.delenv("UNSLOTH_FORCE_GPU_PATH", raising = False)
    assert doctor_mod.mlx_backend_active() is True


def test_mlx_backend_active_false_off_apple_silicon(monkeypatch):
    _fake_studio_gate(monkeypatch, False)
    monkeypatch.delenv("UNSLOTH_FORCE_GPU_PATH", raising = False)
    assert doctor_mod.mlx_backend_active() is False


@pytest.mark.parametrize(
    "system, machine, expected",
    [("Darwin", "arm64", True), ("Darwin", "x86_64", False), ("Linux", "x86_64", False)],
)
def test_mlx_backend_active_falls_back_to_platform(monkeypatch, system, machine, expected):
    _broken_studio_gate(monkeypatch)
    monkeypatch.setattr(doctor_mod.platform, "system", lambda: system)
    monkeypatch.setattr(doctor_mod.platform, "machine", lambda: machine)
    monkeypatch.delenv("UNSLOTH_FORCE_GPU_PATH", raising = False)
    assert doctor_mod.mlx_backend_active() is expected


def test_collect_mlx_info_needs_every_zoo_mlx_module(monkeypatch):
    def fake_import(name):
        if name == "unsloth_zoo.mlx.trainer":
            raise ImportError("broken trainer")
        return types.ModuleType(name)

    monkeypatch.setattr(doctor_mod.importlib, "import_module", fake_import)
    info = doctor_mod.collect_mlx_info()
    assert info["zoo_mlx"] is False
    assert any("unsloth_zoo.mlx.trainer" in e for e in info["errors"])


def test_collect_mlx_info_never_raises():
    info = doctor_mod.collect_mlx_info()
    for key in ("platform", "mlx", "metal", "device", "zoo", "zoo_mlx", "errors"):
        assert key in info


def test_doctor_uses_the_mlx_path_and_exits_zero(monkeypatch):
    monkeypatch.setattr(doctor_mod, "mlx_backend_active", lambda: True)
    monkeypatch.setattr(doctor_mod, "collect_mlx_info", lambda: _healthy_info())
    result = CliRunner().invoke(unsloth_cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "backend mlx" in result.output
    assert "Apple M5" in result.output
    assert "PROBLEMS FOUND" not in result.output
    assert "nvidia-smi" not in result.output


def test_doctor_keeps_the_cuda_path_when_mlx_is_inactive(monkeypatch):
    monkeypatch.setattr(doctor_mod, "mlx_backend_active", lambda: False)

    def _fail(timeout):
        raise typer.Exit(code = 1)

    monkeypatch.setattr(doctor_mod, "_gpu_checks", _fail)
    result = CliRunner().invoke(unsloth_cli.app, ["doctor"])
    assert result.exit_code == 1


def test_doctor_without_mlx_points_at_installing_it(monkeypatch):
    monkeypatch.setattr(doctor_mod, "mlx_backend_active", lambda: True)
    info = _healthy_info(mlx = None, metal = False, device = None, zoo_mlx = False)
    monkeypatch.setattr(doctor_mod, "collect_mlx_info", lambda: info)
    result = CliRunner().invoke(unsloth_cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "mlx is not installed" in result.output
    assert "nvidia-smi" not in result.output
