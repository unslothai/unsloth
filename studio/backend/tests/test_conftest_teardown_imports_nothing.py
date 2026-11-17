# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""conftest's per-test resets must not import anything at teardown.

Several tests take torch out of sys.modules and block imports to exercise the no-torch paths
(test_torchao_select.py, test_mlx_stack_blockers.py, test_polled_probe_costs.py). The dynamo knob reset read
``cache_size_limit``, an alias that torch resolves with importlib.import_module, so its teardown re-imported
torch while those patches were still in place and every such test errored at teardown on a worker that had
imported torch._dynamo earlier ("function '_has_torch_function' already has a docstring").
"""

from __future__ import annotations

import importlib
import sys

import pytest

pytest.importorskip("torch._dynamo")


def test_teardown_with_torch_out_of_sys_modules_imports_nothing(monkeypatch):
    import torch._dynamo  # noqa: F401  (the knob reset snapshots dynamo's config only once it is imported)

    for name in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, name)
    real_import_module = importlib.import_module

    def no_torch(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            raise AssertionError(f"a conftest teardown imported {name}")
        return real_import_module(name, *args, **kwargs)

    # Undone after the autouse resets have torn down: any import they make hits this.
    monkeypatch.setattr(importlib, "import_module", no_torch)
