# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""LongRope must use the short factor at exactly the pretraining length.

transformers switches to the long factor at `seq_len > original_max_position_embeddings`
(`_compute_longrope_parameters`, `_longrope_frequency_update`); `seq_len < original_max`
here switched a token early, and crashed there as well, because the long cache is built
only past `current_rope_size`, which starts at `original_max`.

llama.py needs an accelerator to import, so the class is `ast`-extracted and run against a
CPU-pinned prelude, the shape `tests/test_callback_signature_drift.py` uses.
"""

from __future__ import annotations

import ast
import pathlib
import types

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LLAMA = REPO_ROOT / "unsloth" / "models" / "llama.py"

DIM = 8
ORIGINAL_MAX = 16
MAX_POSITION = 128


class _CPUTorch:
    """`torch`, with a bare-int device pinned to CPU so the test runs anywhere."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(
        self,
        spec,
        index = None,
    ):
        # bool is an int, and torch.device(True) would be index 1, which CPU rejects.
        if isinstance(spec, int) and not isinstance(spec, bool):
            return torch.device("cpu", spec)
        return torch.device(spec) if index is None else torch.device(spec, index)

    def empty(self, *args, **kwargs):
        # get_current_device() is an int, and `device = <int>` means CUDA, so __init__'s
        # scratch buffers would allocate on a GPU the runner may not have.
        kwargs["device"] = "cpu"
        return torch.empty(*args, **kwargs)


def _load_longrope():
    tree = ast.parse(LLAMA.read_text(encoding = "utf-8"))
    # ast.walk, not tree.body: nesting the class inside an `if` would otherwise turn this
    # guard into a silent skip, which is the one failure mode a regression test must not have.
    cls = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "LongRopeRotaryEmbedding"
        ),
        None,
    )
    if cls is None:
        pytest.fail(
            "LongRopeRotaryEmbedding is gone from unsloth/models/llama.py. If it moved or was "
            "renamed, repoint this test rather than deleting it."
        )
    ns = {
        "torch": _CPUTorch(),
        "math": __import__("math"),
        "DEVICE_COUNT": 1,
        "DEVICE_TYPE_TORCH": "cpu",
        # int, matching unsloth/device_type.py: get_cached() indexes a list with it.
        "get_current_device": lambda: 0,
        "is_bfloat16_supported": lambda: False,
        # Only reached on the `config is not None` path, which these tests do not take.
        "_get_rope_theta": lambda config, default = 10000: default,
    }
    try:
        exec(compile(ast.Module(body = [cls], type_ignores = []), str(LLAMA), "exec"), ns)
    except NameError as e:
        pytest.fail(
            f"LongRopeRotaryEmbedding grew a module-level dependency this test does not stub "
            f"({e}). Add it to `ns` above."
        )
    return ns["LongRopeRotaryEmbedding"]


def _make(max_position = MAX_POSITION):
    return _load_longrope()(
        dim = DIM,
        max_position_embeddings = max_position,
        original_max_position_embeddings = ORIGINAL_MAX,
        short_factor = [1.0] * (DIM // 2),
        long_factor = [2.0] * (DIM // 2),
    )


def _fake_input():
    # forward() only reads .device and .dtype off its input.
    return types.SimpleNamespace(device = torch.device("cpu", 0), dtype = torch.float16)


@pytest.mark.parametrize("seq_len", [1, ORIGINAL_MAX - 1, ORIGINAL_MAX])
def test_short_factor_up_to_and_including_the_pretraining_length(seq_len):
    rope = _make()
    x = _fake_input()
    rope.extend_rope_embedding(x, seq_len)
    cos, sin = rope.forward(x, seq_len = seq_len)
    assert torch.equal(cos, rope.multi_gpu_short_cos_cached[0][:seq_len])
    assert torch.equal(sin, rope.multi_gpu_short_sin_cached[0][:seq_len])


def test_long_factor_past_the_pretraining_length():
    rope = _make()
    x = _fake_input()
    seq_len = ORIGINAL_MAX + 1
    rope.extend_rope_embedding(x, seq_len)
    cos, sin = rope.forward(x, seq_len = seq_len)
    assert torch.equal(cos, rope.multi_gpu_long_cos_cached[0][:seq_len])
    assert torch.equal(sin, rope.multi_gpu_long_sin_cached[0][:seq_len])


def test_get_cached_agrees_with_forward_across_the_boundary():
    # get_cached is what every attention path reads cos/sin through, so it has to pick the
    # same branch forward does.
    for seq_len in (ORIGINAL_MAX - 1, ORIGINAL_MAX, ORIGINAL_MAX + 1):
        rope = _make()
        x = _fake_input()
        rope.extend_rope_embedding(x, seq_len)
        fwd_cos, fwd_sin = rope.forward(x, seq_len = seq_len)
        cached_cos, cached_sin = rope.get_cached(seq_len = seq_len, device_index = 0)
        assert cached_cos is not None and cached_sin is not None
        assert torch.equal(fwd_cos, cached_cos[:seq_len])
        assert torch.equal(fwd_sin, cached_sin[:seq_len])


def test_the_boundary_holds_when_the_window_was_never_extended():
    # original_max == max_position (Phi-3-mini-4k's shape), so current_rope_size starts at the
    # boundary and no growth path can ever fill the long cache. Reading it is the whole bug.
    rope = _make(max_position = ORIGINAL_MAX)
    x = _fake_input()
    rope.extend_rope_embedding(x, ORIGINAL_MAX)
    cos, _ = rope.forward(x, seq_len = ORIGINAL_MAX)
    assert torch.equal(cos, rope.multi_gpu_short_cos_cached[0][:ORIGINAL_MAX])


def test_a_warm_long_cache_does_not_capture_the_boundary():
    # Once a longer sequence has run, the long cache exists, so the boundary stops crashing and
    # starts silently returning the wrong factor instead. Same branch, quieter failure.
    rope = _make()
    x = _fake_input()
    rope.extend_rope_embedding(x, ORIGINAL_MAX * 4)
    rope.forward(x, seq_len = ORIGINAL_MAX * 4)
    assert rope.multi_gpu_long_cos_cached[0] is not None
    cos, _ = rope.forward(x, seq_len = ORIGINAL_MAX)
    assert torch.equal(cos, rope.multi_gpu_short_cos_cached[0][:ORIGINAL_MAX])
    cached_cos, _ = rope.get_cached(seq_len = ORIGINAL_MAX, device_index = 0)
    assert torch.equal(cached_cos, rope.multi_gpu_short_cos_cached[0])


@pytest.mark.parametrize("seq_len", [None, 0])
def test_an_unknown_or_empty_length_reads_the_short_cache(seq_len):
    # transformers takes the long factor only on `seq_len and seq_len > original_max`, so None
    # and 0 are both short. They also have to not be None-dereferences on a cold module.
    rope = _make()
    x = _fake_input()
    cos, sin = rope.forward(x, seq_len = seq_len)
    assert cos is not None and sin is not None
    assert torch.equal(cos, rope.multi_gpu_short_cos_cached[0][:seq_len])
    cached_cos, cached_sin = rope.get_cached(seq_len = seq_len, device_index = 0)
    assert cached_cos is not None and cached_sin is not None
    assert torch.equal(cached_cos, rope.multi_gpu_short_cos_cached[0])


def test_get_cached_defaults_its_device_index():
    # device_index = None falls back to get_current_device(), which returns an int the
    # multi_gpu_* lists are indexed with.
    rope = _make()
    cos, sin = rope.get_cached(seq_len = ORIGINAL_MAX)
    assert cos is not None and sin is not None


def test_the_two_factors_really_do_differ():
    # Without this the tests above pass vacuously.
    rope = _make()
    x = _fake_input()
    rope.extend_rope_embedding(x, ORIGINAL_MAX + 1)
    rope.forward(x, seq_len = ORIGINAL_MAX + 1)
    short = rope.multi_gpu_short_cos_cached[0][:ORIGINAL_MAX]
    long = rope.multi_gpu_long_cos_cached[0][:ORIGINAL_MAX]
    assert not torch.equal(short, long)
