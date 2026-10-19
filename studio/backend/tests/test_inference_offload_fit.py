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


def test_sequential_over_several_visible_cards_is_left_pooled():
    # Transformers fills every visible card in order; a one-card plan would push most of it to the host.
    fit, _ = _fit(free = 7 * _GiB, size = 13 * _GiB, devices = 2)
    assert fit("m", "sequential", True) == {}


def _local_bytes():
    tree = ast.parse(_SRC.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_complete_local_safetensors_bytes"
    )
    scope = {"json": __import__("json"), "Path": Path, "Optional": object}
    exec(compile(ast.Module([node], []), str(_SRC), "exec"), scope)
    return scope["_complete_local_safetensors_bytes"]


def test_partial_snapshot_is_not_sized_locally(tmp_path):
    # An interrupted download leaves some shards: the Hub has to size the rest.
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}',
        encoding = "utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 10)
    assert _local_bytes()(tmp_path) is None
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 5)
    assert _local_bytes()(tmp_path) == 15


def test_single_file_snapshot_is_sized_locally(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 7)
    assert _local_bytes()(tmp_path) == 7


_SPILL = "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model."


def _fallback(cuda = True, devices = 1):
    tree = ast.parse(_SRC.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_load_with_offload_fallback"
    )
    fake_cuda = types.SimpleNamespace(
        is_available = lambda: cuda,
        device_count = lambda: devices,
        empty_cache = lambda: None,
        memory_allocated = lambda: 0,
    )
    scope = {
        "torch": types.SimpleNamespace(cuda = fake_cuda),
        "logger": types.SimpleNamespace(info = lambda *a, **k: None),
    }
    exec(compile(ast.Module([node], []), str(_SRC), "exec"), scope)
    return scope["_load_with_offload_fallback"]


class _HalfBuilt:
    pass


def test_missed_guess_frees_the_failed_attempt_then_retries_with_auto():
    import weakref

    calls, seen = [], []

    def load(**kwargs):
        calls.append(kwargs)
        if "offload_layers" in kwargs:
            # The first attempt's half-built model must be gone before the retry allocates.
            seen.append(refs[0]() is None)
            return "model", "tok"
        half_built = _HalfBuilt()
        refs.append(weakref.ref(half_built))
        raise ValueError(_SPILL)

    refs = []
    assert _fallback()(load, model_name = "m", device_map = "sequential") == ("model", "tok")
    assert calls[1] == {"model_name": "m", "device_map": "sequential", "offload_layers": "auto"}
    assert seen == [True]


@pytest.mark.parametrize(
    "error, kwargs, devices",
    [
        (ValueError(_SPILL), {"offload_layers": "auto"}, 1),  # the guess already asked for it
        (ValueError("another problem"), {}, 1),
        (RuntimeError(_SPILL), {}, 1),
        (ValueError(_SPILL), {}, 2),  # several visible cards
    ],
)
def test_fallback_leaves_other_failures_alone(error, kwargs, devices):
    calls = []

    def load(**kw):
        calls.append(kw)
        raise error

    with pytest.raises(type(error)) as caught:
        _fallback(devices = devices)(load, device_map = "sequential", **kwargs)
    assert caught.value is error and len(calls) == 1
