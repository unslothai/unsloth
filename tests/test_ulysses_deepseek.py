# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for DeepSeek on CPU gloo ranks: V2 / V3 MLA on main's generic route, V3.2 (DSA
indexer) and V4 (sink attention) on their own forwards. transformers' own forwards are the single-rank reference."""

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
    calls = []
    if not control:
        # The new path provably runs: once per layer, no silent fallback to the stock forward.
        owner, fn_name = uh, "_ulysses_hf_attention"
        if kind in ("v32", "v4"):
            from unsloth.utils import ulysses_deepseek as ud
            owner, fn_name = ud, f"ulysses_deepseek_{kind}_forward"
        real = getattr(owner, fn_name)

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        setattr(owner, fn_name, counted)
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
        )
        _loss(out.logits, inputs["shift_labels"], count).backward()
    if not control:
        assert len(calls) == model.config.num_hidden_layers, (kind, len(calls))
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


@pytest.mark.parametrize("kind", KINDS)
def test_ulysses_deepseek_route(kind):
    # V2 / V3 MLA (YaRN scale, value heads narrower than q / k) are plain registry attention: main's generic
    # route runs their own attention function. V3.2 / V4 must take their own forward instead: on the generic
    # route the indexer / compressor would see one shard and the rebuilt causal mask would drop the top-k.
    model = _model(kind)
    _manager(2, 0).attach_attention_hooks(model)
    layers = [m for n, m in model.named_modules() if n.endswith("self_attn")]
    assert len(layers) == model.config.num_hidden_layers
    for module in layers:
        own = getattr(module, "_unsloth_cp_orig_forward", None) is not None
        generic = getattr(module, "_unsloth_cp_ulysses_attention", False)
        assert (own, generic) == ((True, False) if kind in ("v32", "v4") else (False, True)), kind
    # transformers 4.x builds per-expert modules, which the model check refuses under CP.
    per_expert = any(
        isinstance(getattr(m, "experts", None), torch.nn.ModuleList) for m in model.modules()
    )
    assert cp._supports_context_parallel(model, "ulysses") is not per_expert


@pytest.mark.parametrize("kind", KINDS)
def test_deepseek_model_checks_as_main_does(kind):
    # SFT accepts every DeepSeek generation under Ulysses; DPO / GRPO refuse it like any generic Hugging Face
    # attention model; ring and the router auxiliary loss stay refused.
    model = _model(kind)
    if any(isinstance(getattr(m, "experts", None), torch.nn.ModuleList) for m in model.modules()):
        # transformers 4.x: per-expert modules stall DDP's reduction under CP (find_unused_parameters off).
        with pytest.raises(NotImplementedError, match = "per-expert modules"):
            cp._check_cp_model(model, 2, "ulysses", allow_hybrids = True)
        return
    cp._check_cp_model(model, 2, "ulysses", allow_hybrids = True)
    with pytest.raises(NotImplementedError, match = "verified for SFTTrainer"):
        cp._check_cp_model(model, 2, "ulysses", allow_hybrids = False)
    with pytest.raises(NotImplementedError):
        cp._check_cp_model(model, 2, "ring", allow_hybrids = True)
    model.config.output_router_logits = True
    with pytest.raises(NotImplementedError, match = "router auxiliary loss"):
        cp._check_cp_model(model, 2, "ulysses", allow_hybrids = True)


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

    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
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
    args = config("ulysses")
    args.context_parallel_size = 1
    module.SFTTrainer(model = refusing, args = args)
    assert args.padding_free is not False


@pytest.mark.parametrize("name", ["DeepseekV32Attention", "DeepseekV4Attention"])
def test_ulysses_deepseek_forward_runs_eager_inside_compiled_layers(name, monkeypatch):
    """Unsloth compiles each decoder layer as one region; traced, the V3.2 / V4 Ulysses forwards (list
    all_gather, the sequence gather, the no_grad indexer top-k) segfaulted the first CP=2 SFT step on L4
    (torch 2.14 and 2.8). They must run eagerly inside a compiled layer."""
    from unsloth.utils import ulysses_deepseek as ud

    seen = []

    def fake_forward(module, hidden_states, *args, group, **kwargs):
        seen.append(torch.compiler.is_compiling())
        return hidden_states * 2, None

    target = (
        "ulysses_deepseek_v4_forward"
        if name == "DeepseekV4Attention"
        else "ulysses_deepseek_v32_forward"
    )
    monkeypatch.setattr(ud, target, fake_forward)
    attention = type(
        name,
        (torch.nn.Module,),
        {"forward": lambda self, hidden_states, **kw: (hidden_states, None)},
    )()
    manager = types.SimpleNamespace(backend = "ulysses", group = None, packed_seq_lengths = None)
    ud.wrap_deepseek_attention_forward(attention, lambda: manager)

    def layer(x):
        return attention(x)[0] + 1

    torch._dynamo.reset()
    out = torch.compile(layer, backend = "eager")(torch.ones(2, 3))
    torch.testing.assert_close(out, torch.full((2, 3), 3.0))
    assert seen == [False], seen


def _ddp_worker(rank, world, kind):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(kind)
    seq = 16
    ids = torch.randint(0, VOCAB, (2, seq), generator = torch.Generator().manual_seed(1))
    count = F.pad(ids, (0, 1), value = -100)[:, 1:].ne(-100).sum()
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    # The CP trainer turns DDP's find_unused_parameters off (reentrant checkpointing).
    ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters = False)
    for _ in range(2):
        inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
        manager._prepare_inputs(inputs)
        with manager.apply(inputs):
            out = ddp(
                input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
            )
            _loss(out.logits, inputs["shift_labels"], count).backward()
    indexer = [n for n, p in model.named_parameters() if ".indexer." in n]
    assert indexer and all(model.get_parameter(n).grad is None for n in indexer), indexer


@pytest.mark.parametrize("kind", [k for k in KINDS if k in ("v32", "v4")])
def test_ulysses_deepseek_ddp_steps_with_trainable_indexer(kind):
    """The DSA indexer only picks keys (a discrete top-k), so its weights never get a gradient. With
    LoRA on them (V3.2 indexer.wq_b, V4 compressor.indexer.q_b_proj) and DDP's find_unused_parameters
    off under CP, the second CP=2 SFT step on L4 failed with "Expected to have finished reduction"."""
    _spawn(_ddp_worker, 2, kind)


@pytest.mark.parametrize("kind", [k for k in KINDS if k in ("v32", "v4")])
def test_ulysses_deepseek_layers_stay_out_of_block_compile(kind):
    """The eager Ulysses forward is a graph break inside each decoder layer's compiled region."""
    from unsloth.models import _block_compile as bc

    model = _model(kind)
    _manager(2, 0).attach_attention_hooks(model)
    layers = model.model.layers
    assert all(layer.__dict__.get("_unsloth_block_mode") == bc._OFF for layer in layers)


_FAILING_FAKE_LIB = []


def _failing_fake_op():
    """An op whose fake (compile-time) kernel raises, called from an autograd Function: what fp32
    experts hit on CUDA (torch._grouped_mm has no fp32 kernel for fake tensors)."""
    if not _FAILING_FAKE_LIB:
        lib = torch.library.Library("unsloth_test_cp_block", "FRAGMENT")
        lib.define("ident(Tensor x) -> Tensor")
        lib.impl("ident", lambda x: x.clone(), "CompositeExplicitAutograd")

        def fake(x):
            raise RuntimeError("no fake kernel")

        torch.library.register_fake("unsloth_test_cp_block::ident", fake, lib = lib)
        _FAILING_FAKE_LIB.append(lib)

    class Ident(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return torch.ops.unsloth_test_cp_block.ident(x)

        @staticmethod
        def backward(ctx, grad):
            return grad

    return Ident.apply


def _block_compile_checkpoint_worker(rank, world):
    from unsloth.models import _block_compile as bc

    # Other test modules set these at import (e.g. UNSLOTH_COMPILE_DISABLE=1) and spawned ranks inherit them.
    for key in ("UNSLOTH_BLOCK_COMPILE", "UNSLOTH_COMPILE_DISABLE", "UNSLOTH_FORCE_FLOAT32"):
        os.environ.pop(key, None)
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model("v32")
    # Layer 0 is dense, layer 1 MoE; only the MoE layer fails to compile after the attention break.
    ident = _failing_fake_op()
    moe = model.model.layers[1].mlp
    moe_forward = moe.forward
    moe.forward = lambda hidden_states: ident(moe_forward(hidden_states))
    available = torch.cuda.is_available
    torch.cuda.is_available = lambda: True  # Unsloth compiles decoder layers on CUDA only
    try:
        assert bc.patch_block_compile(model) == ["DeepseekV32DecoderLayer"]
    finally:
        torch.cuda.is_available = available
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    ids = torch.randint(0, VOCAB, (2, 16), generator = torch.Generator().manual_seed(1))
    count = F.pad(ids, (0, 1), value = -100)[:, 1:].ne(-100).sum()
    for _ in range(2):
        inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
        manager._prepare_inputs(inputs)
        with manager.apply(inputs):
            out = model(
                input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
            )
            _loss(out.logits, inputs["shift_labels"], count).backward()


@pytest.mark.skipif("v32" not in KINDS, reason = "transformers without DeepSeek-V3.2")
def test_ulysses_deepseek_checkpointing_with_block_compile():
    """V3.2 fp32 CP=2 SFT failed with CheckpointError (62 vs 61 saved tensors): the eager Ulysses
    forward split each compiled decoder layer, the MoE layer's post-attention part failed to compile,
    and the dense layer that ran it compiled recomputed it eagerly in backward."""
    _spawn(_block_compile_checkpoint_worker, 2)


@pytest.mark.parametrize("packing", [False, True])
@pytest.mark.parametrize("kind", [k for k in KINDS if k in ("v3", "v32", "v4")])
def test_ulysses_sft_string_model_gets_the_instance_outcome(kind, packing, monkeypatch, caplog):
    """A model ID string defers the module walk, so the Ulysses packing check must classify the
    resolved config instead: V3.2 kept packed / padding-free batches from a string and the
    attention wrapper raised NotImplementedError on the first CP step, while an instance trained."""
    import logging
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    for name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, name, lambda model, trainer: None)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    built = _build(kind)

    class _StringStub(_PackingStubTrainer):
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            super().__init__(model = built if isinstance(model, str) else model, args = args)

    module = SimpleNamespace(SFTTrainer = type("SFTTrainer", (_StringStub,), {}))
    trainer_module._patch_sft_trainer_auto_packing(module)

    def config(cp_size):
        return SimpleNamespace(
            packing = packing,
            padding_free = None,
            max_length = 64,
            context_parallel_size = cp_size,
            context_parallel_backend = "ulysses",
            eval_packing = True,
        )

    with tempfile.TemporaryDirectory() as path:
        built.config.save_pretrained(path)
        outcomes = []
        for model in (path, built):
            args = config(2)
            caplog.clear()
            with caplog.at_level(logging.WARNING):
                module.SFTTrainer(model = model, args = args)
            outcomes.append((args.packing, args.padding_free is False, args.eval_packing))
            if kind == "v32" and packing and model is path:
                assert "DeepseekV32Attention refuses packed batches" in caplog.text
        assert outcomes[0] == outcomes[1], outcomes
        if kind in ("v32", "v4"):
            assert outcomes[0] == (False, True, False)
        else:
            assert outcomes[0][1] is False
        args = config(1)
        module.SFTTrainer(model = path, args = args)
        if kind != "v4":
            assert args.padding_free is not False


def _load_ulysses_deepseek_copy(path, name):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(f"unsloth.utils.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.mark.parametrize("device", ["cuda", "xpu"])
def test_ulysses_deepseek_imports_on_torch_without_amp_custom_fwd(device, monkeypatch):
    """torch.amp.custom_fwd / custom_bwd are 2.4+, and this module is imported by plain `import
    unsloth` (models/llama.py -> context_parallel). A class-body decorator using them raised
    AttributeError at import on torch 2.1 - 2.3; the module must fall back to torch.cuda.amp."""
    import unsloth.device_type as device_type
    from unsloth.utils import ulysses_deepseek as ud

    seen = []

    def fake_amp(kind):
        def factory(
            *args,
            device_type = None,
            **kwargs,
        ):
            seen.append((kind, device_type))
            return lambda fn: fn

        return factory

    monkeypatch.setattr(device_type, "DEVICE_TYPE_TORCH", device)
    monkeypatch.setattr(torch.amp, "custom_fwd", fake_amp("fwd"), raising = False)
    monkeypatch.setattr(torch.amp, "custom_bwd", fake_amp("bwd"), raising = False)
    modern = _load_ulysses_deepseek_copy(ud.__file__, "_ud_amp_modern")
    assert seen == [("fwd", device), ("bwd", device)]

    monkeypatch.delattr(torch.amp, "custom_fwd")
    monkeypatch.delattr(torch.amp, "custom_bwd")
    # torch 2.1 - 2.3: torch.cuda.amp holds the real decorators (newer ones forward to torch.amp).
    legacy_fwd, legacy_bwd = (lambda fn: fn), (lambda fn: fn)
    monkeypatch.setattr(torch.cuda.amp, "custom_fwd", legacy_fwd)
    monkeypatch.setattr(torch.cuda.amp, "custom_bwd", legacy_bwd)
    legacy = _load_ulysses_deepseek_copy(ud.__file__, "_ud_amp_legacy")
    assert legacy._custom_fwd is legacy_fwd and legacy._custom_bwd is legacy_bwd
    for module in (modern, legacy):
        assert issubclass(module._ChunkedSinkAttention, torch.autograd.Function)
        assert issubclass(module._ChunkedSparseMLA, torch.autograd.Function)
