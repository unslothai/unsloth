# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the hardware smoke's imports aligned with the production MLX API.

This CPU contract catches obsolete paths and symbols; the real imports and
training remain mandatory on the Apple Silicon CI runner.
"""

import importlib
from pathlib import Path
import shlex
import sys
from types import ModuleType

import yaml


REPO = Path(__file__).resolve().parents[2]
MODULES = (
    "unsloth_zoo.mlx.loader",
    "unsloth_zoo.mlx.trainer",
    "unsloth_zoo.mlx.compile",
    "unsloth_zoo.mlx.utils",
    "unsloth_zoo.mlx.cce",
    "unsloth_zoo.gated_delta_vjp",
)


def test_hardware_smoke_imports_the_current_mlx_api(monkeypatch):
    workflow = yaml.safe_load((REPO / ".github/workflows/mlx-ci.yml").read_text(encoding = "utf-8"))
    step = next(
        step
        for step in workflow["jobs"]["dispatch"]["steps"]
        if step.get("name") == "Smoke-import every MLX-only unsloth_zoo module"
    )
    command = shlex.split(step["run"])
    assert command[:2] == ["python", "-c"]
    assert len(command) == 3

    for name in ("unsloth_zoo", "unsloth_zoo.mlx", *MODULES):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["unsloth_zoo.mlx.loader"].FastMLXModel = object()
    sys.modules["unsloth_zoo.mlx.trainer"].MLXTrainer = object()
    sys.modules["unsloth_zoo.mlx.trainer"].MLXTrainingConfig = object()
    imported = []
    real_import = importlib.import_module

    def tracked_import(name):
        imported.append(name)
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", tracked_import)
    exec(compile(command[2], "mlx-ci-smoke", "exec"), {})
    assert tuple(imported) == MODULES
