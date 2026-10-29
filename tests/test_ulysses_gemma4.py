# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Gemma 4 text models (sliding / global attention, KV sharing, PLE,
attention_k_eq_v, MoE), on CPU gloo ranks. transformers' own forward is the single-rank reference."""

import functools
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
            logits.to(torch.promote_types(logits.dtype, torch.float32)).flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
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


# ---------------------------------------------------------------------------------------------
# Sliding window across the shard seam, against an fp64 reference with the window written out.


def _windowed_reference(q, k, v, window, scale, lengths = None):
    """fp64 causal (optionally windowed, optionally block-diagonal per document) attention."""
    seq = q.shape[2]
    idx = torch.arange(seq)
    keep = idx[:, None] >= idx[None, :]
    if window is not None:
        keep &= (idx[:, None] - idx[None, :]) < window
    if lengths is not None:
        doc = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        doc = torch.cat([doc, torch.full((seq - len(doc),), len(lengths))])
        keep &= doc[:, None] == doc[None, :]
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~keep, float("-inf"))
    return (scores.softmax(-1) @ v).transpose(1, 2)  # [B, S, H, D]


SEQ = 16
# (window, packed lengths): smaller than a shard, the shard edge, spanning shards, covering the sequence.
WINDOW_CASES = [(1, None), (3, None), (4, None), (5, None), (8, None), (9, None), (12, None),
                (16, None), (40, None), (None, None), (3, [5, 7, 4]), (6, [5, 7, 4])]


def _attention_call(rank, world, q, k, v, *, window, scaling, lengths = None):
    from unsloth.utils import ulysses_hybrid as uh

    local = SEQ // world
    piece = lambda t: t[:, :, rank * local : (rank + 1) * local].detach().clone().requires_grad_()
    ql, kl, vl = piece(q), piece(k), piece(v)
    kwargs = {"scaling": scaling, "sliding_window": window, "dropout": 0.0}
    if lengths is not None:
        kwargs["packed_seq_lengths"] = torch.tensor(lengths)
    module = types.SimpleNamespace(training = True)
    out, _ = uh._ulysses_hf_attention(module, ql, kl, vl, None, group = None, **kwargs)
    gathered = [torch.empty_like(out) for _ in range(world)]
    dist.all_gather(gathered, out.detach().contiguous())
    return torch.cat(gathered, 1), out, (ql, kl, vl)


def _window_worker(rank, world, negative):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, SEQ, 16, generator = g, dtype = torch.float64)
    k = torch.randn(1, 2, SEQ, 16, generator = g, dtype = torch.float64)
    v = torch.randn(1, 2, SEQ, 16, generator = g, dtype = torch.float64)
    upstream = torch.randn(1, SEQ, 4, 16, generator = g, dtype = torch.float64)
    local = SEQ // world
    rows = slice(rank * local, (rank + 1) * local)
    if negative:
        # Each broken variant must disagree with the reference somewhere.
        for window, scaling, ref_window in ((None, 1.0, 3), (3, None, 3), (4, 1.0, 3), (3, 1.0, 4)):
            full, _, _ = _attention_call(rank, world, q, k, v, window = window, scaling = scaling)
            ref = _windowed_reference(q, k, v, ref_window, 1.0)
            assert not torch.allclose(full, ref, atol = 1e-6), (window, scaling, ref_window)
        return
    for window, lengths in WINDOW_CASES:
        full, out, shards = _attention_call(
            rank, world, q, k, v, window = window, scaling = 1.0, lengths = lengths
        )
        qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
        ref = _windowed_reference(qr, kr, vr, window, 1.0, lengths)
        torch.testing.assert_close(full, ref.detach(), rtol = 1e-10, atol = 1e-10, msg = str(window))
        (out * upstream[:, rows]).sum().backward()
        (ref * upstream).sum().backward()
        for mine, theirs in zip(shards, (qr, kr, vr)):
            torch.testing.assert_close(mine.grad, theirs.grad[:, :, rows], rtol = 1e-9, atol = 1e-10)

    # Structural check at the seam: zero queries attend uniformly over the kept keys, and one-hot
    # values read back exactly which keys those are (key at distance window - 1 in, window out).
    eye = torch.eye(SEQ, dtype = torch.float64)[None, None].expand(1, 2, SEQ, SEQ)
    zeros = torch.zeros(1, 4, SEQ, SEQ, dtype = torch.float64)
    for window in (2, 3, local, local + 1):
        full, _, _ = _attention_call(rank, world, zeros, eye, eye, window = window, scaling = 1.0)
        for query in range(local, min(local + window + 1, SEQ)):  # first rows past the seam
            weights = full[0, query, 0]
            assert weights[query - window + 1] > 0, (window, query)
            if query - window >= 0:
                assert weights[query - window] == 0, (window, query)


@pytest.mark.parametrize("world", [2, 4])
def test_ulysses_sliding_window_across_the_seam_matches_fp64(world):
    _spawn(_window_worker, world, False)


def test_ulysses_sliding_window_negative_controls_differ():
    _spawn(_window_worker, 2, True)


# ---------------------------------------------------------------------------------------------
# Flex sliding attention: fallback keeps scale 1.0 and the window; a failure is scoped to its
# call class; a backward-only failure is caught before loss.backward().


def _flex_inputs(dtype, requires_grad, seq = 24):
    g = torch.Generator().manual_seed(1)
    shapes = ((1, 4, seq, 16), (1, 2, seq, 16), (1, 2, seq, 16))
    return [torch.randn(*s, generator = g).to(dtype).requires_grad_(requires_grad) for s in shapes]


def _flex_attend(q, k, v, *, training, window = 5, sdpa_scale = 1.0):
    from unsloth.utils import ulysses_hybrid as uh

    seq = q.shape[2]
    config = ad.AttentionConfig(
        backend = ad.SDPA, n_kv_heads = 2, n_groups = 2,
        sdpa_kwargs = None if sdpa_scale is None else {"scale": sdpa_scale},
    )
    context = ad.AttentionContext(
        bsz = 1, q_len = seq, kv_seq_len = seq, n_heads = 4, head_dim = 16,
        requires_grad = q.requires_grad, seq_info = None, attention_mask = None, causal_mask = None,
        sliding_window = window, training = training,
    )
    out = uh._flex_or_dense(config = config, context = context, Q = q, K = k, V = v,
                            window = window, scale = 1.0)
    return out.reshape(1, seq, 4, 16)


def _flex_call(dtype, *, training, requires_grad, window = 5, sdpa_scale = 1.0):
    q, k, v = _flex_inputs(dtype, requires_grad)
    out = _flex_attend(q, k, v, training = training, window = window, sdpa_scale = sdpa_scale)
    ref_inputs = [t.detach().double().requires_grad_() for t in (q, k, v)]
    ref = _windowed_reference(*ref_inputs, window, 1.0)
    return out, ref, (q, k, v), ref_inputs


@pytest.fixture
def flex_state(monkeypatch):
    from unsloth.utils import ulysses_hybrid as uh

    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", False)
    monkeypatch.setattr(ad, "HAS_XFORMERS", False)
    monkeypatch.setattr(uh, "_FLEX_STATUS", {})
    monkeypatch.setattr(uh, "_FLEX", {})
    monkeypatch.setattr(uh, "_SLIDING_BLOCK_MASKS", {})
    return uh


def test_forced_flex_failure_falls_back_with_scale_one_and_window(flex_state, monkeypatch):
    def broken(**_):
        raise RuntimeError("injected flex compile failure")

    monkeypatch.setattr(flex_state, "_flex_sliding_attention", broken)
    out, ref, inputs, ref_inputs = _flex_call(torch.float64, training = True, requires_grad = True)
    torch.testing.assert_close(out, ref, rtol = 1e-10, atol = 1e-10)
    out.pow(2).sum().backward()
    ref.pow(2).sum().backward()
    for mine, theirs in zip(inputs, ref_inputs):
        torch.testing.assert_close(mine.grad, theirs.grad, rtol = 1e-9, atol = 1e-10)
    # Negative control: the dense path with the default scale is wrong for Gemma 4.
    out, ref, _, _ = _flex_call(torch.float64, training = True, requires_grad = True, sdpa_scale = None)
    assert not torch.allclose(out, ref, atol = 1e-6)


def test_flex_failure_on_one_class_keeps_flex_for_others(flex_state, monkeypatch):
    # Real CPU flex: float64 fails to compile, a grad-requiring call has no CPU backward, and
    # float32 inference compiles. Neither failure may switch flex off for the float32 class.
    real = flex_state._flex_sliding_attention
    ran = []

    def spy(**kwargs):
        out = real(**kwargs)
        ran.append(kwargs["Q"].dtype)
        return out

    monkeypatch.setattr(flex_state, "_flex_sliding_attention", spy)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    out, ref, _, _ = _flex_call(torch.float64, training = False, requires_grad = False)
    torch.testing.assert_close(out, ref, rtol = 1e-10, atol = 1e-10)
    out, ref, inputs, ref_inputs = _flex_call(torch.float32, training = True, requires_grad = True)
    torch.testing.assert_close(out.double(), ref, rtol = 1e-5, atol = 1e-5)
    out.pow(2).sum().backward()
    ref.pow(2).sum().backward()
    torch.testing.assert_close(inputs[0].grad.double(), ref_inputs[0].grad, rtol = 1e-4, atol = 1e-4)
    assert ran == []
    with torch.no_grad():
        out, ref, _, _ = _flex_call(torch.float32, training = False, requires_grad = False)
    torch.testing.assert_close(out.double(), ref, rtol = 1e-5, atol = 1e-5)
    assert ran == [torch.float32]
    assert sorted(flex_state._FLEX_STATUS.values()) == [False, False, True]


class _BackwardFails(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window):
        return _windowed_reference(q, k, v, window, 1.0)

    @staticmethod
    def backward(ctx, grad):
        raise RuntimeError("injected flex backward failure (out of shared memory)")


@pytest.mark.parametrize("checkpoint", [None, "reentrant", "non_reentrant"])
def test_flex_backward_only_failure_falls_back_before_backward(flex_state, monkeypatch, checkpoint):
    calls = []

    def forward_only(*, Q, K, V, window, **_):
        calls.append(torch.is_grad_enabled())
        return _BackwardFails.apply(Q, K, V, window)

    monkeypatch.setattr(flex_state, "_flex_sliding_attention", forward_only)
    inputs = _flex_inputs(torch.float64, True)
    attend = functools.partial(_flex_attend, training = True)
    if checkpoint is None:
        out = attend(*inputs)
    else:
        from torch.utils.checkpoint import checkpoint as ckpt

        out = ckpt(attend, *inputs, use_reentrant = checkpoint == "reentrant")
    ref_inputs = [t.detach().clone().requires_grad_() for t in inputs]
    ref = _windowed_reference(*ref_inputs, 5, 1.0)
    torch.testing.assert_close(out, ref, rtol = 1e-10, atol = 1e-10)
    out.pow(2).sum().backward()
    ref.pow(2).sum().backward()
    for mine, theirs in zip(inputs, ref_inputs):
        torch.testing.assert_close(mine.grad, theirs.grad, rtol = 1e-9, atol = 1e-10)
    assert calls and list(flex_state._FLEX_STATUS.values()) == [False]


def test_flex_success_probes_once_per_class(flex_state, monkeypatch):
    calls = []

    def fine(*, Q, K, V, window, **_):
        calls.append(Q.shape[2])
        return _windowed_reference(Q, K, V, window, 1.0)

    monkeypatch.setattr(flex_state, "_flex_sliding_attention", fine)
    for _ in range(2):
        out, ref, _, _ = _flex_call(torch.float64, training = True, requires_grad = True)
        torch.testing.assert_close(out, ref)
    # One probe (forward + backward) on the first call of the class, then one call per step.
    assert len(calls) == 3
    _flex_call(torch.float64, training = False, requires_grad = False)
    assert len(calls) == 4 and all(flex_state._FLEX_STATUS.values())


def _flex_gate_worker(rank, world):
    # The Ulysses wrapper hands flex the gathered sequence, the layer window and scale 1.0, and a
    # flex failure there still matches the fp64 windowed reference.
    from unsloth.utils import ulysses_hybrid as uh

    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    uh._flex_device_ok = lambda query: True
    seen = []

    def broken(*, context, window, scale, **_):
        seen.append((context.q_len, window, scale))
        raise RuntimeError("injected")

    uh._flex_sliding_attention = broken
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, SEQ, 16, generator = g, dtype = torch.float64)
    k = torch.randn(1, 2, SEQ, 16, generator = g, dtype = torch.float64)
    v = torch.randn(1, 2, SEQ, 16, generator = g, dtype = torch.float64)
    for window in (3, 9):
        full, _, _ = _attention_call(rank, world, q, k, v, window = window, scaling = 1.0)
        torch.testing.assert_close(full, _windowed_reference(q, k, v, window, 1.0), rtol = 1e-10, atol = 1e-10)
    # Covering the whole sequence (or global layers) never goes to flex.
    _attention_call(rank, world, q, k, v, window = SEQ, scaling = 1.0)
    _attention_call(rank, world, q, k, v, window = None, scaling = 1.0)
    assert seen == [(SEQ, 3, 1.0), (SEQ, 9, 1.0)], seen


def test_ulysses_flex_gate_passes_window_and_scale():
    _spawn(_flex_gate_worker, 2)


# ---------------------------------------------------------------------------------------------
# Whole model vs an fp64 single-process reference (Gemma 4's RMSNorm upcasts to float32 only, so
# tolerances are float32-sized): CP=2 / CP=4 / DP2xCP2, windows smaller than a
# shard, spanning shards and covering the sequence, right padding with unequal valid-token counts,
# a rank without labels, a label across the seam; per-parameter gradients (None / zero / value),
# one AdamW step, LoRA with nonzero B.


def _batch(mode):
    ids = torch.randint(0, 64, (2, 13), generator = torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[1, 6:] = 0  # row 1: 6 valid tokens, right padded
    labels = ids.masked_fill(mask == 0, -100)
    if mode == "prompt_masked":
        # Completion-only labels: the first shard(s) carry no labels at all.
        labels[:, :9] = -100
    return ids, mask, labels


def _grad_status(grad):
    return 0 if grad is None else (1 if not grad.any() else 2)


def _model_case(rank, world, cp, kind, window, mode, *, lora = False, control = None):
    import copy

    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    cp_rank, dp_rank = rank % cp, rank // cp
    dp = world // cp
    groups = [dist.new_group(list(range(d * cp, (d + 1) * cp))) for d in range(dp)]
    torch.manual_seed(0)
    config = _config(kind)
    config.sliding_window = window
    model = gemma4.Gemma4ForCausalLM(config).double().train()
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    if lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r = 4, lora_alpha = 8, init_lora_weights = False,  # random nonzero B
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        assert all(p.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)
    reference = copy.deepcopy(model)
    ids, mask, labels = _batch(mode)
    shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(reference(input_ids = ids, attention_mask = mask, use_cache = False).logits, shift, count).backward()

    rows = slice(dp_rank * (2 // dp), (dp_rank + 1) * (2 // dp))
    manager = _manager(cp, cp_rank)
    manager.mesh = types.SimpleNamespace(get_group = lambda: groups[dp_rank], get_local_rank = lambda: cp_rank)
    manager.attach_attention_hooks(model)
    attention = [m for n, m in model.named_modules() if n.endswith("self_attn")]
    if control == "shard_local":
        for module in attention:
            module._unsloth_cp_ulysses_attention = False
    elif control == "global_layers_windowed":
        for module in attention:
            module.sliding_window = window
    elif control == "dropped_window":
        for module in attention:
            module.sliding_window = None
    elif control == "default_scale":
        for module in attention:
            module.scaling = None
    inputs = {"input_ids": ids[rows].clone(), "attention_mask": mask[rows].clone(), "labels": labels[rows].clone()}
    with manager.apply(inputs):
        positions = inputs["position_ids"]
        if control == "position_offset":
            positions = positions - positions[:, :1]
        out = model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"],
                    position_ids = positions, use_cache = False)
        _loss(out.logits, inputs["shift_labels"], count).backward()

    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        statuses = [None] * world
        dist.all_gather_object(statuses, _grad_status(param.grad))
        expected = ref_params[name].grad
        if expected is None:
            assert statuses == [0] * world, (name, statuses)
            continue
        # Every rank holds a gradient (a rank without labels too), or the reduction desyncs.
        assert 0 not in statuses, (name, statuses)
        grad = param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(
            grad, expected, rtol = 1e-5, atol = 1e-6, msg = lambda m: f"{name}: {m}"
        )
        assert _grad_status(grad) == _grad_status(expected), name
    # One AdamW step with weight decay: None skips the parameter, zero still decays it. A large eps
    # keeps the first step Lipschitz in the gradient (Gemma 4's RMSNorm runs in float32 even under
    # float64, so gradients carry float32 rounding).
    for net in (reference, model):
        trainable = [p for p in net.parameters() if p.requires_grad]
        torch.optim.AdamW(trainable, lr = 1e-3, eps = 1e-3, weight_decay = 0.1).step()
    for (name, mine), theirs in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(
            mine, theirs, rtol = 1e-5, atol = 1e-6, msg = lambda m: f"after AdamW {name}: {m}"
        )


def _model_worker(rank, world, cp, cases):
    for case in cases:
        _model_case(rank, world, cp, *case[:3], **(case[3] if len(case) > 3 else {}))


def _controls_worker(rank, world):
    passed = []
    for control in ("shard_local", "global_layers_windowed", "dropped_window", "default_scale",
                    "position_offset"):
        try:
            _model_case(rank, world, 2, "e2b", 3, "pad", control = control)
            passed.append(control)
        except AssertionError:
            pass
    assert not passed, f"negative controls that matched the reference: {passed}"


@pytest.mark.parametrize("kind", KINDS)
def test_gemma4_cp2_windows_padding_adamw_fp64(kind):
    cases = [(kind, w, mode) for w in (3, 9, 64) for mode in ("pad", "prompt_masked")]
    _spawn(_model_worker, 2, 2, cases)


def test_gemma4_cp2_lora_nonzero_b_fp64():
    _spawn(_model_worker, 2, 2, [("e2b", 3, "pad", {"lora": True}), ("moe_k_eq_v", 9, "pad", {"lora": True})])


def test_gemma4_cp4_and_dp2_cp2_fp64():
    _spawn(_model_worker, 4, 4, [("moe_k_eq_v", 3, "pad"), ("e2b", 9, "prompt_masked")])
    _spawn(_model_worker, 4, 2, [("e2b", 5, "pad"), ("moe_k_eq_v", 3, "prompt_masked")])


def test_gemma4_negative_controls_fail():
    _spawn(_controls_worker, 2)


# ---------------------------------------------------------------------------------------------
# The lifted refusal stays narrow.


def _check(model, backend = "ulysses", allow_hybrids = True):
    try:
        cp._check_cp_model(model, 2, backend, allow_hybrids)
    except NotImplementedError as error:
        return str(error)
    return None


def test_gemma4_window_exemption_stays_narrow(monkeypatch):
    from unsloth.models.llama import LlamaAttention_fast_forward

    model = _model("e2b")
    assert _check(model) is None
    assert "sliding" in _check(model, "ring")
    assert "DPO / GRPO" in _check(model, allow_hybrids = False)
    # Llama's fast forward ignores the window: a Gemma 4 attention running it stays refused.
    attention = type(model.model.layers[0].self_attn)
    monkeypatch.setattr(attention, "forward", LlamaAttention_fast_forward)
    assert "sliding window" in _check(model)
    assert "sliding window" in _check(model, allow_hybrids = False)
    monkeypatch.undo()
    # Multimodal Gemma 4: image features would stay whole on every rank.
    vision = gemma4.Gemma4VisionConfig(
        hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1, num_attention_heads = 2,
        num_key_value_heads = 2, head_dim = 8,
    )
    multimodal = gemma4.Gemma4ForConditionalGeneration(gemma4.Gemma4Config(
        text_config = _config("e2b").to_dict(), vision_config = vision.to_dict(), audio_config = None,
    ))
    assert "multimodal" in _check(multimodal)
    # Other windowed model types, and Gemma 4 with softcapping or chunked layers, stay refused.
    for model_type in ("gemma3_text", "mistral", "qwen2"):
        other = types.SimpleNamespace(
            model_type = model_type, sliding_window = 4, layer_types = ["sliding_attention", "full_attention"]
        )
        assert cp._unsupported_attention(other, "ulysses") == "sliding_attention layers"
    text = _config("e2b")
    text.attn_logit_softcapping = 50.0
    assert cp._unsupported_attention(text, "ulysses") == "attention logit softcapping"
    text = _config("e2b")
    text.layer_types = ["chunked_attention"] * 4
    assert cp._unsupported_attention(text, "ulysses") == "chunked_attention layers"
