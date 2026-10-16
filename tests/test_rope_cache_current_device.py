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


def _load(
    path,
    names,
    extra_ns = None,
):
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
            pytest.fail(
                f"{name} is gone from {path.name}; repoint this test rather than deleting it."
            )
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
    cos, sin = rope.get_cached(40000, 1)
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


def test_get_cached_materialises_the_length_it_is_asked_for():
    # Gemma2 / Cohere decode steps read get_cached(kv_seq_len) with no extend first, past the rows
    # the model forward built for max_seq_length.
    rope = _llama(1 << 17)
    rope.extend_rope_embedding(_x(0), 1 << 17, materialize_len = 2048)
    assert rope.multi_gpu_cos_cached[0].shape[0] == 4 * 8192
    cos, sin = rope.get_cached(40000, 0)
    assert cos.shape[0] >= 40000
    ref_cos, ref_sin = _reference(cos.shape[0])
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)


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


def test_granite_decode_reads_rows_past_max_seq_length():
    # GraniteModel_fast_forward_inference indexes the cache it fetches once per step at the decode
    # position, which runs past max_seq_length; the eager cache held max_position_embeddings rows.
    tree = ast.parse((REPO_ROOT / "unsloth" / "models" / "granite.py").read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "GraniteModel_fast_forward_inference"
    )
    call = next(
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "get_cached"
    )
    max_seq_length, seq_len, q_len = 32768, 32790, 1
    rows = eval(
        compile(ast.Expression(call.args[0]), "granite.py", "eval"),
        {"max": max},
        {
            "self": types.SimpleNamespace(max_seq_length = max_seq_length),
            "seq_len": seq_len,
            "q_len": q_len,
        },
    )
    rope = _llama(1 << 17)
    # Prefill of a 32760 token prompt through LlamaModel_fast_forward.
    rope.extend_rope_embedding(_x(0), 1 << 17, materialize_len = max(32760, max_seq_length))
    cos, sin = rope.get_cached(rows, 0)
    assert cos.shape[0] > seq_len
    ref_cos, _ = _reference(cos.shape[0])
    assert torch.equal(cos[seq_len], ref_cos[seq_len])


def test_covered_reads_still_rebuild_a_device_left_stale_by_growth():
    rope = _llama(1 << 17)
    rope.get_cached(10, 1)
    rope.extend_rope_embedding(_x(0, torch.float16), 50000)
    # Device 1 already holds 32768 rows covering 10, but growth changed the dtype.
    cos, _ = rope.get_cached(10, 1)
    assert cos.dtype == torch.float16 and cos.shape[0] == 57344
    ref_cos, _ = _reference(57344, torch.float16)
    assert torch.equal(cos, ref_cos)
    rope.extend_rope_embedding(_x(1, torch.float16), 11)
    assert rope.multi_gpu_cos_cached[1] is cos


def _model_forward_rope(
    rope,
    seq_length_with_past,
    position_ids,
    max_position,
    max_seq_length = 2048,
):
    # Runs the rotary block of LlamaModel_fast_forward as written in llama.py.
    tree = ast.parse(LLAMA.read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "LlamaModel_fast_forward"
    )
    block = next(
        n
        for n in fn.body
        if isinstance(n, ast.If) and "IS_ATTENTION_REFACTOR" in ast.unparse(n.test)
    )
    scope = {
        "self": types.SimpleNamespace(
            rotary_emb = rope,
            max_seq_length = max_seq_length,
            config = types.SimpleNamespace(max_position_embeddings = max_position),
        ),
        "hidden_states": _x(0, torch.float16),
        "seq_length_with_past": seq_length_with_past,
        "position_ids": position_ids,
        "IS_ATTENTION_REFACTOR": True,
        "IS_GRANITE": False,
        "torch": torch,
        # isinstance needs the class this rope was built from; other ropes never match.
        "LongRopeRotaryEmbedding": type(rope)
        if type(rope).__name__ == "LongRopeRotaryEmbedding"
        else type("LongRopeRotaryEmbedding", (), {}),
        "max": max,
        "getattr": getattr,
    }
    exec(compile(ast.Module(body = [block], type_ignores = []), str(LLAMA), "exec"), scope)
    return scope["position_embeddings"]


def test_model_forward_covers_explicit_offset_position_ids():
    # Valid positions past the batch length (offsets, generate, packing) must index real rows.
    max_position = 1 << 17
    position_ids = torch.tensor([[40000, 40001]])
    cos, sin = _model_forward_rope(_llama(max_position), 2, position_ids, max_position)
    assert cos.shape[0] >= int(position_ids.max()) + 1
    ref_cos, ref_sin = _reference(cos.shape[0], torch.float16)
    assert torch.equal(cos[position_ids], ref_cos[position_ids])
    assert torch.equal(sin[position_ids], ref_sin[position_ids])


def test_model_forward_without_position_ids_stays_short():
    # Implicit positions are 0..seq_length_with_past-1, so the short cache is enough.
    max_position = 1 << 20
    rope = _llama(max_position)
    cos, _ = _model_forward_rope(rope, 2, None, max_position)
    assert cos.shape[0] == 4 * 8192
    assert _built(rope.multi_gpu_cos_cached) == [0]


def test_starting_cache_is_computed_on_cpu():
    # The eager __init__ computed the starting cache on CPU even when loading later moved inv_freq to
    # an accelerator; computing it there instead raises the first forward's peak memory.
    rope = _llama(4096)
    rope.inv_freq = rope.inv_freq.to("meta")
    cos, sin = rope.get_cached(10, 0)
    ref_cos, ref_sin = _reference(4096)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)


def test_model_forward_position_ids_keep_longrope_short_factors():
    # Building every row for explicit positions must not switch a short input to the long factors.
    rope = _llama_ns()["LongRopeRotaryEmbedding"](
        dim = DIM,
        max_position_embeddings = 128,
        original_max_position_embeddings = 16,
        short_factor = [1.0] * (DIM // 2),
        long_factor = [2.0] * (DIM // 2),
    )
    cos, sin = _model_forward_rope(rope, 4, torch.tensor([[0, 1, 2, 3]]), 128, max_seq_length = 8)
    assert cos is rope.multi_gpu_short_cos_cached[0]
    assert sin is rope.multi_gpu_short_sin_cached[0]


def _longrope(max_position, original_max_position):
    return _llama_ns()["LongRopeRotaryEmbedding"](
        dim = DIM,
        max_position_embeddings = max_position,
        original_max_position_embeddings = original_max_position,
        short_factor = [1.0] * (DIM // 2),
        long_factor = [2.0] * (DIM // 2),
    )


def test_model_forward_longrope_offset_positions_match_transformers():
    # transformers picks LongRoPE's factors from max(position_ids) + 1, so 4 tokens at 40000 with a 4096
    # pretrained window use the long factors; the short cache has no row 40000.
    from transformers import Phi3Config
    from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

    max_position, original_max_position = 1 << 17, 4096
    position_ids = torch.tensor([[40000, 40001, 40002, 40003]])
    rope = _longrope(max_position, original_max_position)
    cos, sin = _model_forward_rope(rope, 4, position_ids, max_position)
    assert cos.shape[0] > int(position_ids.max())

    config = Phi3Config(
        hidden_size = DIM,
        num_attention_heads = 1,
        num_key_value_heads = 1,
        max_position_embeddings = max_position,
        original_max_position_embeddings = original_max_position,
        rope_scaling = {
            "type": "longrope",
            "short_factor": [1.0] * (DIM // 2),
            "long_factor": [2.0] * (DIM // 2),
        },
    )
    ref_cos, ref_sin = Phi3RotaryEmbedding(config)(torch.zeros(1, 4, DIM), position_ids)
    torch.testing.assert_close(cos[position_ids[0]].float(), ref_cos[0], atol = 2e-3, rtol = 0)
    torch.testing.assert_close(sin[position_ids[0]].float(), ref_sin[0], atol = 2e-3, rtol = 0)


def test_model_forward_longrope_offset_positions_without_sync_under_compile(monkeypatch):
    # While tracing there is no .item(): positions may span max_position_embeddings, so take the long cache.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.Tensor, "max", lambda *args, **kwargs: pytest.fail("host sync"))
    rope = _longrope(1 << 17, 4096)
    cos, _ = _model_forward_rope(rope, 4, torch.tensor([[40000, 40001, 40002, 40003]]), 1 << 17)
    assert cos is rope.multi_gpu_long_cos_cached[0]
