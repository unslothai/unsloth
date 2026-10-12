# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat load the GPU cannot hold retries once with offload_layers = "auto" (#13388): Gemma 4 E2B in
4-bit is 8.1 GB, 4.4 GB of it one per-layer token table, so an 8 GB card failed the load outright."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "core" / "inference" / "inference.py"

_FIT = (
    "Unsloth: unsloth/gemma-4-E2B-it-unsloth-bnb-4bit does not fit in GPU memory (cuda:0 has 7.26 GB "
    "free), so transformers placed part of it on the CPU, which bitsandbytes cannot quantize."
)
_RAW = "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model."


def _helper(cuda = True):
    tree = ast.parse(_SRC.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_load_offloading_when_too_big"
    )
    fake_cuda = types.SimpleNamespace(is_available = lambda: cuda, empty_cache = lambda: None)
    scope = {
        "torch": types.SimpleNamespace(cuda = fake_cuda),
        "logger": types.SimpleNamespace(info = lambda *a, **k: None),
    }
    exec(compile(ast.Module([node], []), str(_SRC), "exec"), scope)
    return scope["_load_offloading_when_too_big"]


def _loader(fail_with):
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        if "offload_layers" not in kwargs and fail_with is not None:
            raise fail_with
        return "model", "tokenizer"

    return load, calls


@pytest.mark.parametrize("message", [_FIT, _RAW])
def test_too_big_single_gpu_load_retries_with_auto_offload(message):
    load, calls = _loader(ValueError(message))
    assert _helper()(load, model_name = "m", device_map = "sequential") == ("model", "tokenizer")
    assert calls == [
        {"model_name": "m", "device_map": "sequential"},
        {"model_name": "m", "device_map": "sequential", "offload_layers": "auto"},
    ]


def test_load_that_fits_is_untouched():
    load, calls = _loader(None)
    _helper()(load, model_name = "m", device_map = "sequential")
    assert calls == [{"model_name": "m", "device_map": "sequential"}]


@pytest.mark.parametrize(
    "error, device_map, cuda",
    [
        (ValueError("some other problem"), "sequential", True),
        (RuntimeError(_FIT), "sequential", True),
        (ValueError(_FIT), "unsloth_balanced", True),
        (ValueError(_FIT), "sequential", False),
    ],
)
def test_other_failures_raise_unchanged(error, device_map, cuda):
    load, calls = _loader(error)
    with pytest.raises(type(error)) as caught:
        _helper(cuda)(load, model_name = "m", device_map = device_map)
    assert caught.value is error
    assert len(calls) == 1
