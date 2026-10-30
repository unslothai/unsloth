# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for DeepSeek (V2 / V3 MLA, V3.2 DSA indexer, V4 sink attention), on CPU
gloo ranks. transformers' own forwards are the single-rank reference."""

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
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
KINDS = [
    kind
    for kind, name in (
        ("v2", "DeepseekV2Config"),
        ("v3", "DeepseekV3Config"),
        ("v32", "DeepseekV32Config"),
        ("v4", "DeepseekV4Config"),
    )
    if hasattr(transformers, name)
]
if not KINDS:
    pytest.skip("transformers without DeepSeek", allow_module_level = True)
VOCAB = 128
YARN = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": 4.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 64,
}


def _mla(**extra):
    # YaRN mscale makes the softmax scale non-default; value heads (16) are narrower than q / k (24).
    return dict(
        vocab_size = VOCAB,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        kv_lora_rank = 16,
        q_lora_rank = 24,
        qk_rope_head_dim = 8,
        v_head_dim = 16,
        qk_nope_head_dim = 16,
        n_routed_experts = 4,
        n_shared_experts = 1,
        num_experts_per_tok = 2,
        first_k_dense_replace = 1,
        rope_parameters = YARN,
        **extra,
    )


def _build(kind):
    t = transformers
    if kind == "v2":
        return t.DeepseekV2ForCausalLM(t.DeepseekV2Config(**_mla()))
    if kind == "v3":
        return t.DeepseekV3ForCausalLM(t.DeepseekV3Config(**_mla(n_group = 1, topk_group = 1)))
    if kind == "v32":
        config = t.DeepseekV32Config(
            **_mla(n_group = 1, topk_group = 1, index_topk = 4, index_head_dim = 16, index_n_heads = 2)
        )
        return t.DeepseekV32ForCausalLM(config)
    config = t.DeepseekV4Config(
        vocab_size = VOCAB,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 32,
        rope_head_dim = 16,
        q_lora_rank = 32,
        o_lora_rank = 32,
        o_groups = 2,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 8,
        index_n_heads = 2,
        index_head_dim = 16,
        index_topk = 2,
        layer_types = [
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types = ["hash_moe", "moe", "moe"],
        compress_rates = {"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    )
    model = t.DeepseekV4ForCausalLM(config)
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        for module in model.modules():
            # Zero-initialised sinks / position biases would hide slicing mistakes.
            if hasattr(module, "sinks"):
                module.sinks.copy_(0.1 * torch.randn(module.sinks.shape, generator = g))
            if hasattr(module, "position_bias"):
                module.position_bias.copy_(
                    0.05 * torch.randn(module.position_bias.shape, generator = g)
                )
            if hasattr(module, "tid2eid"):
                ids = torch.arange(VOCAB)
                module.tid2eid[:, 0] = ids % 4
                module.tid2eid[:, 1] = (ids + 1) % 4
    return model


def _model(kind):
    torch.manual_seed(0)
    model = _build(kind).float().train()
    # The V3.2 / V4 Ulysses forwards reproduce transformers' eager attention, and Unsloth runs
    # DeepSeek-V4 eager only; V2 / V3 go through the attention registry, so sdpa is exercised.
    model.config._attn_implementation = "eager" if kind in ("v32", "v4") else "sdpa"
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
    # transformers' V3.2 / V4 indexers are not invariant to appending tokens (top-k over zero-score
    # ties, or over not-yet-causal compressed blocks), so the reference must see exactly the length
    # CP runs: 16 needs no padding at cp = 2. V4's rate-8 compressed layers then see two blocks.
    seq = 16 if kind in ("v32", "v4") else 15
    ids = torch.randint(0, VOCAB, (2, seq), generator = torch.Generator().manual_seed(1))
    shift = F.pad(ids, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(model(input_ids = ids, use_cache = False).logits, shift, count).backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)

    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    if control:
        for module in model.modules():
            if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
                module.forward = module._unsloth_cp_orig_forward
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
def test_ulysses_deepseek_gradients_match_one_rank(kind):
    _spawn(_parity_worker, 2, kind, False)


@pytest.mark.parametrize("kind", [k for k in KINDS if k in ("v32", "v4")])
def test_ulysses_deepseek_chunked_attention_matches(kind, monkeypatch):
    # Chunks of 4 query rows: the checkpointed chunk loop instead of one dense block.
    monkeypatch.setenv("UNSLOTH_CP_ATTN_CHUNK", "4")
    _spawn(_parity_worker, 2, kind, False)


@pytest.mark.skipif("v4" not in KINDS, reason = "transformers without DeepSeek-V4")
def test_shard_local_deepseek_v4_is_wrong():
    # Negative control: the stock V4 forward on a shard sees neither earlier keys nor full blocks.
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, 2, "v4", True)


@pytest.mark.skipif("v4" not in KINDS, reason = "transformers without DeepSeek-V4")
def test_deepseek_v4_gates():
    model = _model("v4")
    # Its own Ulysses forward builds the sliding / compressed masks; the ring backend cannot.
    assert cp._unsupported_attention(model.config, "ulysses") is None
    assert cp._unsupported_attention(model.config, "ring") is not None
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")


def test_mla_custom_scale_reaches_every_backend(monkeypatch):
    seen = {}

    def fake_ulysses(run, config, context, q, k, v, group):
        seen["config"] = config
        return torch.zeros(q.shape[0], q.shape[2], q.shape[1], q.shape[3])

    monkeypatch.setattr(uh, "ulysses_attention", fake_ulysses)
    module = types.SimpleNamespace(training = False)
    q = torch.zeros(1, 4, 3, 24)
    out, _ = uh._ulysses_hf_attention(
        module, q, q, torch.zeros(1, 4, 3, 16), None, group = None, scaling = 0.3
    )
    config = seen["config"]
    assert config.flash_dense_kwargs["softmax_scale"] == 0.3
    assert config.flash_varlen_kwargs["softmax_scale"] == 0.3
    assert config.sdpa_kwargs == {"scale": 0.3} and config.xformers_kwargs == {"scale": 0.3}
    assert out.shape[-1] == 16
    # The default scale leaves the backends on their own defaults.
    uh._ulysses_hf_attention(module, q, q, q, None, group = None, scaling = 24**-0.5)
    assert seen["config"].sdpa_kwargs is None


@pytest.mark.parametrize("wrap", ["none", "checkpoint"])
def test_chunked_attention_gradients_match_and_nest_in_checkpointing(wrap, monkeypatch):
    # The chunked V4 / V3.2 attention recomputes chunks inside one autograd op, so it must give the
    # dense gradients and also run inside activation checkpointing of the whole layer.
    from torch.utils.checkpoint import checkpoint

    from unsloth.utils import ulysses_deepseek as ud

    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 2, 12, 8, generator = g)
    kv = torch.randn(1, 1, 15, 8, generator = g)
    sinks = torch.randn(2, generator = g)
    bias = torch.zeros(1, 1, 12, 3)
    bias[..., 2] = float("-inf")
    k = torch.randn(1, 2, 12, 8, generator = g)
    v = torch.randn(1, 2, 12, 6, generator = g)
    topk = torch.randint(0, 12, (1, 12, 4), generator = g)
    topk[..., 0] = torch.arange(12)  # like the indexer, every query keeps at least one causal key

    def sink(q, kv, sinks):
        return ud._sink_attention(q, kv, bias, sinks, 5, 0.3, 0.0, True)

    def mla(q, k, v):
        return ud._sparse_mla_attention(q, k, v, topk, 0.3, 0.0, True)

    for fn, inputs in ((sink, (q, kv, sinks)), (mla, (q, k, v))):
        grads = []
        for chunk in ("100", "4"):
            monkeypatch.setenv("UNSLOTH_CP_ATTN_CHUNK", chunk)
            leaves = [x.clone().requires_grad_() for x in inputs]
            out = (
                checkpoint(fn, *leaves, use_reentrant = False)
                if wrap == "checkpoint"
                else fn(*leaves)
            )
            (out.square().sum()).backward()
            grads.append([x.grad for x in leaves])
        for dense, chunked in zip(*grads):
            torch.testing.assert_close(chunked, dense, rtol = 1e-5, atol = 1e-6)


class _PackingStubTrainer:
    def __init__(
        self,
        model = None,
        args = None,
        **kwargs,
    ):
        self.model = model
        self.args = args


class _KwargsModel(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attn = attention

    def forward(self, input_ids = None, **kwargs):
        return None


@pytest.mark.parametrize("kind", [k for k in KINDS if k in ("v32", "v4")])
def test_ulysses_sft_turns_padding_free_off_for_attention_that_refuses_packing(kind, monkeypatch):
    """V3.2 / V4 Ulysses forwards refuse packed batches, so SFT must not pick padding-free under
    Ulysses CP: the automatic choice turned it on and the first CP step raised (Colab L4, i6)."""
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    module = SimpleNamespace(SFTTrainer = type("SFTTrainer", (_PackingStubTrainer,), {}))
    trainer_module._patch_sft_trainer_auto_packing(module)

    name = {"v32": "DeepseekV32Attention", "v4": "DeepseekV4Attention"}[kind]
    refusing = _KwargsModel(type(name, (torch.nn.Module,), {})())
    plain = _KwargsModel(type("DeepseekV3Attention", (torch.nn.Module,), {})())

    def config(backend):
        return SimpleNamespace(
            packing = False,
            padding_free = None,
            max_length = 64,
            context_parallel_size = 2,
            context_parallel_backend = backend,
            eval_packing = True,
        )

    args = config("ulysses")
    module.SFTTrainer(model = refusing, args = args)
    assert args.padding_free is False and args.packing is False
    assert args.eval_packing is False
    # Ulysses attention that takes packed batches keeps the automatic padding-free choice.
    args = config("ulysses")
    module.SFTTrainer(model = plain, args = args)
    assert args.padding_free is not False
    # Without CP the refusing class trains packed as before.
    args = config("ulysses")
    args.context_parallel_size = 1
    module.SFTTrainer(model = refusing, args = args)
    assert args.padding_free is not False
