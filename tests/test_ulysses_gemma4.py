# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Gemma 4 text models (sliding / global attention, KV sharing, PLE,
attention_k_eq_v, MoE), on CPU gloo ranks. transformers' own forward is the single-rank reference."""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad

transformers = pytest.importorskip("transformers")
gemma4 = pytest.importorskip("transformers.models.gemma4")
KINDS = ["e2b", "moe_k_eq_v"]
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]


def _config(kind):
    common = dict(
        vocab_size = 64,
        vocab_size_per_layer_input = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        head_dim = 8,
        global_head_dim = 16,
        # A window of 4 inside 13 tokens: windows cross the shard seam.
        sliding_window = 4,
        final_logit_softcapping = 30.0,
        layer_types = LAYER_TYPES,
    )
    if kind == "e2b":
        # E2B / E4B: per-layer embeddings, the last two layers reuse earlier KV, one KV head.
        common.update(
            hidden_size_per_layer_input = 8,
            num_kv_shared_layers = 2,
            num_key_value_heads = 1,
            use_double_wide_mlp = True,
        )
    else:
        # 26B-A4B / 31B: global layers share K and V and have fewer KV heads; MoE; no PLE.
        common.update(
            hidden_size_per_layer_input = 0,
            num_kv_shared_layers = 0,
            num_key_value_heads = 2,
            num_global_key_value_heads = 1,
            attention_k_eq_v = True,
            enable_moe_block = True,
            num_experts = 4,
            top_k_experts = 2,
            moe_intermediate_size = 16,
        )
    return gemma4.Gemma4TextConfig(**common)


def _model(kind):
    # Unsloth's grouped-GEMM MoE kernels need CUDA; the CPU parity check uses the torch experts.
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "native_torch")
    torch.manual_seed(0)
    model = gemma4.Gemma4ForCausalLM(_config(kind)).float().train()
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    for module in model.modules():
        # Grouped expert kernels may touch CUDA in a forked CPU worker.
        if hasattr(module, "set_experts_implementation"):
            module.set_experts_implementation("eager")
    return model


def _manager(size, rank):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(get_group = lambda: None, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.float().flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
        )
        / count
    )


def _parity_worker(rank, world, kind, control):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(kind)
    ids = torch.randint(0, 64, (2, 13), generator = torch.Generator().manual_seed(1))
    shift = F.pad(ids, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(model(input_ids = ids, use_cache = False).logits, shift, count).backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)

    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    if control:
        # Negative control: shard-local attention (the Ulysses exchange switched off).
        for module in model.modules():
            module._unsloth_cp_ulysses_attention = False
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
        )
        _loss(out.logits, inputs["shift_labels"], count).backward()
    for name, param in model.named_parameters():
        if name not in reference:
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(grad, reference[name], rtol = 1e-4, atol = 1e-5, msg = name)


def _spawn(fn, world, *args):
    # Fork: a spawned child would re-import unsloth without the conftest CUDA spoof.
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(
            _entry,
            args = (world, os.path.join(d, "init"), fn, args),
            nprocs = world,
            join = True,
            start_method = "fork",
        )


def _entry(rank, world, init_file, fn, args):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", KINDS)
def test_ulysses_gemma4_gradients_match_one_rank(kind):
    _spawn(_parity_worker, 2, kind, False)


def test_shard_local_gemma4_attention_is_wrong():
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, 2, "e2b", True)


def test_gemma4_gates():
    model = _model("e2b")
    text = model.config
    # Ulysses hands the per-layer window to the attention dispatcher; ring still cannot.
    assert cp._unsupported_attention(text, "ulysses") is None
    assert "sliding" in cp._unsupported_attention(text, "ring")
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")
    # Heterogeneous configs refuse global per-layer reads; head counts come from each layer, and
    # the global layers' own KV head count is checked too.
    assert cp._attention_head_counts(_config("moe_k_eq_v")) == {(4, 2), (4, 1)}
    # The window exemption is Gemma 4 only: another model type with a window is still refused.
    other = types.SimpleNamespace(model_type = "mistral", sliding_window = 4096, layer_types = None)
    assert cp._unsupported_attention(other, "ulysses") == "sliding window attention"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "flex sliding attention is the CUDA path")
def test_flex_sliding_attention_matches_windowed_sdpa():
    from unsloth.utils import ulysses_hybrid as uh

    torch.manual_seed(0)
    seq, heads, kv_heads, dim, window = 512, 4, 2, 64, 128
    q = torch.randn(1, heads, seq, dim, device = "cuda", requires_grad = True)
    k = torch.randn(1, kv_heads, seq, dim, device = "cuda", requires_grad = True)
    v = torch.randn(1, kv_heads, seq, dim, device = "cuda", requires_grad = True)
    idx = torch.arange(seq, device = "cuda")
    mask = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < window)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask = mask, scale = 1.0, enable_gqa = True)
    ref_grads = torch.autograd.grad(ref.pow(2).sum(), (q, k, v))
    out = uh._flex_sliding_attention(
        config = types.SimpleNamespace(n_groups = heads // kv_heads),
        context = types.SimpleNamespace(q_len = seq),
        Q = q, K = k, V = v, window = window, scale = 1.0,
    )
    torch.testing.assert_close(out, ref.transpose(1, 2), rtol = 1e-4, atol = 1e-4)
    for grad, expected in zip(torch.autograd.grad(out.pow(2).sum(), (q, k, v)), ref_grads):
        torch.testing.assert_close(grad, expected, rtol = 1e-3, atol = 1e-3)
