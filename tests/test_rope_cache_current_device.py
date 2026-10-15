# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Unsloth's rotary caches must be built only on the device that reads them.

llama.py needs an accelerator to import, so the classes are `ast`-extracted and run against a
CPU-pinned prelude in which device index i is cpu:i.
"""

from __future__ import annotations

import ast
import math
import pathlib
import types

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LLAMA = REPO_ROOT / "unsloth" / "models" / "llama.py"
GEMMA = REPO_ROOT / "unsloth" / "models" / "gemma.py"

DEVICES = 4
DIM = 8
BASE = 10000


class _CPUTorch:
    """`torch`, with a bare-int device pinned to CPU so the test runs anywhere."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(
        self,
        spec,
        index = None,
    ):
        if isinstance(spec, int) and not isinstance(spec, bool):
            return torch.device("cpu", spec)
        return torch.device(spec) if index is None else torch.device(spec, index)

    def empty(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.empty(*args, **kwargs)


_HELPERS = ("_round_rope_size", "_PerDeviceRopeCache")


def _load(path, names, extra_ns = None):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    nodes = []
    for name in names:
        node = next(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
            ),
            None,
        )
        if node is None:
            # Helpers only exist once the cache is per device; the classes must always exist.
            if name in _HELPERS:
                continue
            pytest.fail(f"{name} is gone from {path.name}; repoint this test rather than deleting it.")
        nodes.append(node)
    ns = {
        "torch": _CPUTorch(),
        "math": math,
        "DEVICE_COUNT": DEVICES,
        "DEVICE_TYPE_TORCH": "cpu",
        "get_current_device": lambda: 0,
        "is_bfloat16_supported": lambda: False,
        "_get_rope_theta": lambda config, default = 10000: default,
        "_compute_config_rope_inv_freq": lambda config, rope_scaling: (None, 1.0),
        "_rope_scaling_as_dict": lambda value: dict(value or {}),
    }
    ns.update(extra_ns or {})
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), ns)
    return ns


def _llama_ns():
    return _load(
        LLAMA,
        list(_HELPERS) + ["LlamaRotaryEmbedding", "LongRopeRotaryEmbedding"],
    )


def _built(cache):
    return [i for i, c in enumerate(cache) if c is not None]


def _x(device_index, dtype = torch.float32):
    # extend_rope_embedding / forward only read .device and .dtype.
    return types.SimpleNamespace(device = torch.device("cpu", device_index), dtype = dtype)


def _reference(rows, dtype = torch.float32):
    inv_freq = 1.0 / (BASE ** (torch.arange(0, DIM, 2, dtype = torch.int64).float() / DIM))
    t = torch.arange(rows, dtype = torch.int64).float()
    emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim = -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _llama(max_position):
    return _llama_ns()["LlamaRotaryEmbedding"](
        dim = DIM, max_position_embeddings = max_position, base = BASE
    )


def test_init_builds_no_device():
    rope = _llama(1 << 20)
    assert _built(rope.multi_gpu_cos_cached) == []
    assert _built(rope.multi_gpu_sin_cached) == []


def test_get_cached_builds_only_the_requested_device():
    rope = _llama(1 << 20)
    cos, sin = rope.get_cached(100, 2)
    assert _built(rope.multi_gpu_cos_cached) == [2]
    ref_cos, ref_sin = _reference(cos.shape[0])
    assert cos.shape[0] == 4 * 8192
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)


def test_model_forward_extend_builds_needed_rows_on_one_device():
    # LlamaModel_fast_forward: extend to max_position_embeddings, materialise this batch's rows.
    max_position = 1 << 20
    rope = _llama(max_position)
    x = _x(1, torch.float16)
    rope.extend_rope_embedding(x, max_position, materialize_len = 40000)
    # Size and dtype move exactly as the eager extend left them.
    assert rope.current_rope_size == max_position
    assert _built(rope.multi_gpu_cos_cached) == [1]
    cos, sin = rope.get_cached(max_position, 1)
    # 40000 rounded up to the 8192 growth step, not 1M rows.
    assert cos.shape[0] == 40960
    assert cos.dtype == torch.float16
    ref_cos, ref_sin = _reference(40960, torch.float16)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)


def test_extend_grows_used_device_and_stale_peers_rebuild_lazily():
    rope = _llama(1 << 17)
    rope.get_cached(10, 0)
    rope.get_cached(10, 3)
    rope.extend_rope_embedding(_x(3, torch.float16), 50000)
    assert rope.current_rope_size == 57344
    assert rope.multi_gpu_cos_cached[3].shape[0] == 57344
    assert rope.multi_gpu_cos_cached[3].dtype == torch.float16
    # Device 0 is untouched until it is read, then matches what the eager rebuild produced.
    assert rope.multi_gpu_cos_cached[0].shape[0] == 4 * 8192
    cos, sin = rope.get_cached(50000, 0)
    ref_cos, ref_sin = _reference(57344, torch.float16)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)
    assert _built(rope.multi_gpu_cos_cached) == [0, 3]


def test_forward_builds_only_input_device():
    rope = _llama(4096)
    cos, sin = rope(_x(2), seq_len = 100)
    assert _built(rope.multi_gpu_cos_cached) == [2]
    ref_cos, ref_sin = _reference(100)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)


def test_longrope_builds_only_the_requested_device():
    rope = _llama_ns()["LongRopeRotaryEmbedding"](
        dim = DIM,
        max_position_embeddings = 128,
        original_max_position_embeddings = 16,
        short_factor = [1.0] * (DIM // 2),
        long_factor = [2.0] * (DIM // 2),
    )
    assert _built(rope.multi_gpu_short_cos_cached) == []
    rope.get_cached(8, 1)
    assert _built(rope.multi_gpu_short_cos_cached) == [1]
    assert _built(rope.multi_gpu_long_cos_cached) == []
    rope.extend_rope_embedding(_x(1, torch.float16), 100)
    assert _built(rope.multi_gpu_long_cos_cached) == [1]
    assert _built(rope.multi_gpu_short_cos_cached) == [1]


def test_gemma_builds_only_the_requested_device():
    llama_ns = _llama_ns()
    if "_PerDeviceRopeCache" not in llama_ns:
        pytest.fail("Gemma's rotary cache is built for every device")
    ns = _load(
        GEMMA,
        ["GemmaFixedRotaryEmbedding"],
        {"_PerDeviceRopeCache": llama_ns["_PerDeviceRopeCache"]},
    )
    rope = ns["GemmaFixedRotaryEmbedding"](dim = DIM, max_position_embeddings = 1 << 20, base = BASE)
    assert _built(rope.multi_gpu_cos_cached) == []
    rope.extend_rope_embedding(_x(3, torch.bfloat16), 70000)
    assert _built(rope.multi_gpu_cos_cached) == [3]
    assert rope.multi_gpu_cos_cached[3].shape[0] == 73728
    # Gemma keeps its cache in float32 whatever the input dtype.
    assert rope.multi_gpu_cos_cached[3].dtype == torch.float32
