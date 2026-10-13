# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat load whose checkpoint nearly fills free VRAM asks for offload_layers = "auto" (#13388):
Gemma 4 E2B in 4-bit is 8.1 GB, 4.4 GB of it one per-layer token table, so an 8 GB card failed outright."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "core" / "inference" / "inference.py"
_GiB = 2**30


def _fit(
    free,
    size,
    cuda = True,
    devices = 1,
):
    tree = ast.parse(_SRC.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_offload_kwargs_for_fit")
        or (
            isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_OFFLOAD_FIT_FRACTION"
        )
    ]
    fake_cuda = types.SimpleNamespace(
        is_available = lambda: cuda,
        device_count = lambda: devices,
        mem_get_info = lambda i: (free, 8 * _GiB),
    )
    asked = []
    scope = {
        "torch": types.SimpleNamespace(cuda = fake_cuda),
        "logger": types.SimpleNamespace(info = lambda *a, **k: None),
        "Optional": object,
        "_loader_repo": lambda name, four: name + ("-bnb-4bit" if four else ""),
        "_checkpoint_bytes": lambda name, token: asked.append(name) or size,
    }
    exec(compile(ast.Module(nodes, []), str(_SRC), "exec"), scope)
    return scope["_offload_kwargs_for_fit"], asked


def test_checkpoint_that_fills_the_card_loads_with_auto_offload():
    fit, asked = _fit(free = int(7.3 * _GiB), size = int(8.1 * _GiB))
    assert fit("unsloth/gemma-4-E2B-it", "sequential", True) == {"offload_layers": "auto"}
    # Sized from the repo the 4-bit mapper loads, not the 16-bit name the UI sent.
    assert asked == ["unsloth/gemma-4-E2B-it-bnb-4bit"]


def test_exact_load_sizes_the_repo_it_reads():
    # use_exact_model_name: a cached 16-bit snapshot is read as-is, so the 4-bit twin says nothing.
    fit, asked = _fit(free = int(7.3 * _GiB), size = 10 * _GiB)
    assert fit("unsloth/gemma-4-E2B-it", "sequential", True, mapped = False) == {
        "offload_layers": "auto"
    }
    assert asked == ["unsloth/gemma-4-E2B-it"]


def test_checkpoint_with_room_loads_as_before():
    fit, _ = _fit(free = 7 * _GiB, size = 6 * _GiB)
    assert fit("m", "sequential", True) == {}


@pytest.mark.parametrize(
    "device_map, cuda, size",
    [
        ("unsloth_balanced", True, 9 * _GiB),  # several GPUs: the balanced planner places it
        ("sequential", False, 9 * _GiB),  # no CUDA / ROCm card
        ("sequential", True, None),  # size unknown: no guess
    ],
)
def test_nothing_changes_where_the_guess_cannot_apply(device_map, cuda, size):
    fit, _ = _fit(free = 7 * _GiB, size = size, cuda = cuda)
    assert fit("m", device_map, True) == {}
