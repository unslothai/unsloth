# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import importlib.util
import sys
from pathlib import Path

import pytest

_MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"


@pytest.fixture
def decision_mlx(monkeypatch):
    # By path, as unsloth/__init__.py loads it on Apple Silicon; needs neither mlx nor torch to import.
    for name in ("unsloth._decision_common", "_decision_mlx_under_test"):
        monkeypatch.delitem(sys.modules, name, raising = False)
    spec = importlib.util.spec_from_file_location(
        "_decision_mlx_under_test", _MODELS / "decision_mlx.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module._decision_zoo.cache_clear()


def test_an_install_without_torch_is_told_to_install_it(decision_mlx, monkeypatch):
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name == "torch" else find_spec(name, *args),
    )
    decision_mlx._decision_zoo.cache_clear()
    with pytest.raises(ImportError, match = "decision models on MLX need PyTorch"):
        decision_mlx._decision_zoo()
