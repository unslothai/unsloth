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
        if isinstance(spec, int):
            return torch.device("cpu", spec)
        return torch.device(spec) if index is None else torch.device(spec, index)


def _load_longrope():
    tree = ast.parse(LLAMA.read_text(encoding = "utf-8"))
    cls = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "LongRopeRotaryEmbedding"
        ),
        None,
    )
    if cls is None:
        pytest.skip("LongRopeRotaryEmbedding not found in unsloth/models/llama.py")
    ns = {
        "torch": _CPUTorch(),
        "math": __import__("math"),
        "DEVICE_COUNT": 1,
        "DEVICE_TYPE_TORCH": "cpu",
        "get_current_device": lambda: torch.device("cpu"),
        "is_bfloat16_supported": lambda: False,
    }
    exec(compile(ast.Module(body = [cls], type_ignores = []), str(LLAMA), "exec"), ns)
    return ns["LongRopeRotaryEmbedding"]


def _make():
    return _load_longrope()(
        dim = DIM,
        max_position_embeddings = MAX_POSITION,
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


def test_the_two_factors_really_do_differ():
    # Without this the tests above pass vacuously.
    rope = _make()
    x = _fake_input()
    rope.extend_rope_embedding(x, ORIGINAL_MAX + 1)
    rope.forward(x, seq_len = ORIGINAL_MAX + 1)
    short = rope.multi_gpu_short_cos_cached[0][:ORIGINAL_MAX]
    long = rope.multi_gpu_long_cos_cached[0][:ORIGINAL_MAX]
    assert not torch.equal(short, long)
