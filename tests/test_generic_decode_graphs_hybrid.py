# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Decode CUDA graphs for hybrid linear-attention models (Qwen3.5 / Qwen3.6), on the host stand-in backend."""

import os
import sys
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

import unsloth.models._decode_graphs as G
import unsloth.models._generic_decode_graphs as GG
from unsloth.models import vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_generic_decode_graphs import _Host  # noqa: E402
from real_accelerator import has_real_cuda  # noqa: E402

qwen3_5 = pytest.importorskip("transformers.models.qwen3_5")
from transformers import cache_utils  # noqa: E402

if getattr(cache_utils, "LinearAttentionLayer", None) is None:
    pytest.skip("transformers without LinearAttentionLayer", allow_module_level = True)
if not getattr(qwen3_5.modeling_qwen3_5.Qwen3_5ForCausalLM, "_can_compile_fullgraph", False):
    # Before 5.17 generate() never builds the static hybrid cache these tests decode over.
    pytest.skip("transformers without static-cache Qwen3.5 generate", allow_module_level = True)


# On a CUDA host the Triton GDN kernels get dispatched for these CPU tensors and fail.
host_only = pytest.mark.skipif(
    torch.cuda.is_available(), reason = "CPU stand-in: Triton GDN kernels on a CUDA host"
)


def _tiny(moe):
    torch.manual_seed(0)
    common = dict(
        vocab_size = 1000,
        hidden_size = 64,
        num_hidden_layers = 8,
        head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        linear_conv_kernel_dim = 4,
        layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 2,
        max_position_embeddings = 1024,
    )
    if moe:
        from transformers import Qwen3_5MoeForCausalLM
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

        config = Qwen3_5MoeTextConfig(
            num_attention_heads = 8,
            num_key_value_heads = 1,
            num_experts = 4,
            num_experts_per_tok = 2,
            moe_intermediate_size = 32,
            shared_expert_intermediate_size = 32,
            **common,
        )
        model = Qwen3_5MoeForCausalLM(config)
    else:
        from transformers import Qwen3_5ForCausalLM
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        config = Qwen3_5TextConfig(
            num_attention_heads = 4, num_key_value_heads = 1, intermediate_size = 64, **common
        )
        model = Qwen3_5ForCausalLM(config)
    return model.float().eval()


@pytest.fixture(scope = "module")
def dense():
    return _tiny(False)


@pytest.fixture(scope = "module")
def moe():
    return _tiny(True)


@pytest.fixture
def host(monkeypatch):
    backend = type("Host", (_Host,), {"graphs": []})
    monkeypatch.setattr(GG, "_BACKEND", backend)
    return backend


def _fresh(model):
    model.__dict__.pop(GG._ATTR, None)


def _ids(
    bsz,
    length = 7,
    pads = (),
    seed = 1,
):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 1000, (bsz, length), generator = g)
    mask = torch.ones_like(ids)
    for row, pad in enumerate(pads):
        ids[row, :pad] = 0
        mask[row, :pad] = 0
    return ids, mask


def _linear_layers(cache):
    return [layer for layer in cache.layers if type(layer) is cache_utils.LinearAttentionLayer]


def _generate(
    model,
    ids,
    mask,
    graphs,
    new,
    trace = None,
):
    """`trace` collects every linear-attention state after every decode step."""
    handle = None
    if trace is not None:

        def hook(module, args, kwargs, output):
            cache = kwargs.get("past_key_values")
            if cache is not None:
                trace.append(
                    [t.clone() for layer in _linear_layers(cache) for t in GG._linear_states(layer)]
                )

        handle = model.register_forward_hook(hook, with_kwargs = True)
    scope = vision._GraphDecodeSteps(model) if graphs else __import__("contextlib").nullcontext()
    try:
        with torch.inference_mode(), scope:
            return model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_new_tokens = new,
                min_new_tokens = new,
                do_sample = False,
                cache_implementation = "static",
                pad_token_id = 0,
                disable_compile = True,
            )
    finally:
        if handle is not None:
            handle.remove()


@host_only
@pytest.mark.parametrize("which", ["dense", "moe"])
@pytest.mark.parametrize("bsz,pads", [(1, ()), (4, (3, 0, 5, 1))])
def test_hybrid_decode_replays_with_eager_tokens_and_states(host, request, which, bsz, pads):
    model = request.getfixturevalue(which)
    _fresh(model)
    ids, mask = _ids(bsz, pads = pads)
    eager_trace, graph_trace = [], []
    new = 200
    eager = _generate(model, ids, mask, False, new, eager_trace)
    graphed = _generate(model, ids, mask, True, new, graph_trace)
    assert torch.equal(eager, graphed)
    s = GG.stats(model)
    assert s["captures"] == 1 and s["validations"] == 1 and s["replays"] == new - 2, s
    assert s["refusals"] == {}, s
    assert len(eager_trace) == len(graph_trace) == new
    for step, (a, b) in enumerate(zip(eager_trace, graph_trace)):
        assert len(a) == len(b) > 0
        for x, y in zip(a, b):
            assert torch.equal(x, y), f"state differs after step {step}"


@host_only
def test_state_comparison_catches_a_double_advance(host, dense, monkeypatch):
    # Not vacuous: a snapshot skipping linear states advances them twice.
    real = GG._snapshot

    def without_linear(cache, rows = True):
        host_part, device = real(cache, rows)
        linear = {id(t) for layer in _linear_layers(cache) for t in GG._linear_states(layer)}
        return host_part, [entry for entry in device if id(entry[0]) not in linear]

    monkeypatch.setattr(GG, "_snapshot", without_linear)
    _fresh(dense)
    ids, mask = _ids(1)
    eager_trace, graph_trace = [], []
    _generate(dense, ids, mask, False, 12, eager_trace)
    _generate(dense, ids, mask, True, 12, graph_trace)
    assert any(
        not torch.equal(x, y) for a, b in zip(eager_trace, graph_trace) for x, y in zip(a, b)
    )


@host_only
def test_linear_layer_checks(host, dense):
    from transformers import StaticCache

    cache = StaticCache(config = dense.config, max_cache_len = 32)
    linear = _linear_layers(cache)
    assert linear and GG._cache_reason(cache) == "uninitialized layer"
    ids, mask = _ids(1)
    with torch.inference_mode():
        dense(input_ids = ids, attention_mask = mask, past_key_values = cache, use_cache = True)
    assert GG._cache_reason(cache) is None
    assert GG._cache_batch(cache) == 1 and GG._cache_len(cache) == 32
    linear[0].record_past = True
    assert GG._cache_reason(cache) == "linear layer recording its past"
    linear[0].record_past = False
    linear[0].has_previous_state[0] = False
    assert GG._cache_reason(cache) == "uninitialized layer"


class _DefaultHost(_Host):
    @staticmethod
    def available():
        return True

    @staticmethod
    def free_bytes(device):
        return 1 << 40


@pytest.fixture
def default_host(monkeypatch):
    backend = type("Host", (_DefaultHost,), {"graphs": []})
    monkeypatch.setattr(GG, "_BACKEND", backend)
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    monkeypatch.setattr(G, "_STUDIO_DEFAULT_OK", True)
    for name in (
        "UNSLOTH_DECODE_CUDA_GRAPHS",
        "UNSLOTH_DECODE_GENERIC_GRAPHS",
        "UNSLOTH_DECODE_GENERIC_GRAPHS_KEEP",
    ):
        monkeypatch.delenv(name, raising = False)
    return backend


def _fast(moe):
    model = _tiny(moe)
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    return model


def _gen(
    model,
    ids,
    mask,
    new = 16,
):
    return model.generate(
        input_ids = ids,
        attention_mask = mask,
        max_new_tokens = new,
        min_new_tokens = new,
        do_sample = False,
        pad_token_id = 0,
    )


@host_only
@pytest.mark.parametrize("which", [False, True])
def test_fast_generate_graphs_hybrid_and_kept_cache_resets_state(default_host, monkeypatch, which):
    model = _fast(which)
    first_ids, first_mask = _ids(2, pads = (2, 0), seed = 1)
    other_ids, other_mask = _ids(2, pads = (0, 3), seed = 2)
    monkeypatch.setenv("UNSLOTH_DECODE_CUDA_GRAPHS", "0")
    eager_first = _gen(model, first_ids, first_mask)
    eager_other = _gen(model, other_ids, other_mask)
    monkeypatch.delenv("UNSLOTH_DECODE_CUDA_GRAPHS")
    assert torch.equal(_gen(model, first_ids, first_mask), eager_first)
    s1 = GG.stats(model)
    assert s1["captures"] == 1 and s1["replays"] >= 13 and not s1["refusals"], s1
    kept = GG.kept_cache(model)
    assert kept is not None and _linear_layers(kept)
    assert torch.equal(_gen(model, other_ids, other_mask), eager_other)
    s2 = GG.stats(model)
    assert s2["captures"] == 1 and s2["replays"] - s1["replays"] == 15, (s1, s2)
    assert GG.kept_cache(model) is kept


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda() or torch.version.hip is not None, reason = "NVIDIA CUDA")
@pytest.mark.parametrize("which", [False, True])
def test_real_capture_matches_eager_hybrid(which, monkeypatch):
    monkeypatch.setattr(G, "ENABLED", True)
    model = (
        _tiny(which)
        .to(torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16)
        .cuda()
    )
    _fresh(model)
    ids, mask = _ids(2, pads = (3, 0))
    ids, mask = ids.cuda(), mask.cuda()
    eager_trace, graph_trace = [], []
    eager = _generate(model, ids, mask, False, 48, eager_trace)
    graphed = _generate(model, ids, mask, True, 48, graph_trace)
    s = GG.stats(model)
    assert torch.equal(eager, graphed), s
    assert s["captures"] == 1 and s["replays"] == 46 and not s["refusals"], s
    assert s["check"]["mismatches"] == 0, s
    for a, b in zip(eager_trace, graph_trace):
        for x, y in zip(a, b):
            if which:
                # Graph path uses zoo's routed MoE kernels, eager uses grouped_mm: different summation order.
                assert torch.allclose(x.float(), y.float(), atol = 5e-2, rtol = 0), s
            else:
                assert torch.equal(x, y), s


def test_linear_layer_layouts():
    # LinearAttentionLayer layouts: dict per state index (5.17+), else one tensor per kind + one bool.
    states = (torch.zeros(1, 2, 3), torch.zeros(1, 2, 4, 4))
    new = types.SimpleNamespace(
        conv_states = {0: states[0]}, recurrent_states = {0: states[1]}, has_previous_state = {0: True}
    )
    old = types.SimpleNamespace(
        conv_states = states[0],
        recurrent_states = states[1],
        has_previous_state = True,
        is_conv_states_initialized = True,
        is_recurrent_states_initialized = True,
    )
    for layer in (new, old):
        assert GG._linear_states(layer) == list(states)
    old.has_previous_state = False
    assert GG._linear_reason(old) == "uninitialized layer"
    new.has_previous_state = {0: False}
    assert GG._linear_reason(new) == "uninitialized layer"
