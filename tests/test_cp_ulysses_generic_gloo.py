# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism on any Hugging Face attention model, through the model's own attention function
(ALL_ATTENTION_FUNCTIONS, or the modeling module's eager_attention_forward): scaling, softcap, sliding window and
attention sinks survive the exchange. Two gloo CPU ranks vs one process: the loss and every parameter gradient.

Configs are the tiny test repos' (tiny_models.py: hf-internal-testing/tiny-random-Gemma3ForCausalLM,
trl-internal-testing/tiny-Qwen3ForCausalLM, trl-internal-testing/tiny-GptOssForCausalLM) with random weights, a
small vocabulary, and a window shorter than a shard so it crosses the seam. DeepSeek-V2 and MiniMax-M2 have no tiny
repo: their configs are written out (_built_config)."""

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

SEQ = 16
WINDOW = 5  # < SEQ / 2: windows cross the shard seam
REPOS = {
    "gemma3": "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
    "gemma2": "trl-internal-testing/tiny-Gemma2ForCausalLM",
    "qwen3": "trl-internal-testing/tiny-Qwen3ForCausalLM",
    "gpt_oss": "trl-internal-testing/tiny-GptOssForCausalLM",
}


def _built_config(kind):
    """Configs with no tiny hub repo: written out (random weights). DeepSeek-V2: multi-head latent attention,
    query / key head dim 24 (nope 16 + rope 8) against value head dim 16, YaRN scaling, MoE after one dense layer.
    MiniMax-M2: grouped query attention (4 heads / 2 kv heads) with q / k norms over all heads, MoE layers."""
    if kind == "deepseek_v2":
        from transformers import DeepseekV2Config
        return DeepseekV2Config(
            vocab_size = 96,
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
            rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": 4.0,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 0.707,
                "mscale_all_dim": 0.707,
                "original_max_position_embeddings": 64,
            },
            attention_dropout = 0.0,
            use_cache = False,
            pad_token_id = 0,
        )
    from transformers.models.minimax_m2 import MiniMaxM2Config

    return MiniMaxM2Config(
        vocab_size = 96,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        output_router_logits = False,
        attention_dropout = 0.0,
        use_cache = False,
        bos_token_id = 1,
        eos_token_id = 2,
        pad_token_id = 0,
    )


BUILT = ("deepseek_v2", "minimax_m2")


def _config(kind):
    from transformers import AutoConfig

    if kind in BUILT:
        try:
            return _built_config(kind)
        except ImportError as e:  # an older transformers without the architecture
            pytest.skip(f"{kind}: {e}")
    try:
        config = AutoConfig.from_pretrained(REPOS[kind])
    except Exception as e:  # offline host without the tiny repo cached
        pytest.skip(f"{REPOS[kind]}: {e}")
    config.vocab_size = 96
    config.num_hidden_layers = 2
    config.attention_dropout = 0.0
    config.use_cache = False
    config.pad_token_id = 0
    if kind in ("gemma3", "gemma2"):
        # Gemma3: 2 heads / 1 kv head, so the kv head is replicated over cp 2, and the 1 / sqrt(256) scale
        # (head_dim 8) is off the default the old path refused. Gemma2: attention logit softcapping.
        config.layer_types = ["sliding_attention", "full_attention"]
        config.sliding_window = WINDOW
        config.attn_logit_softcapping = 0.05  # small enough to bend the random tiny logits
        config.final_logit_softcapping = None
    elif kind == "gpt_oss":
        config.layer_types = ["sliding_attention", "full_attention"]
        config.sliding_window = WINDOW
        config.output_router_logits = False
    return config


def _model(kind, implementation):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_config(kind), attn_implementation = implementation)
    model.config._attn_implementation = implementation
    for name, p in model.named_parameters():
        if name.endswith("sinks"):
            with torch.no_grad():
                p.normal_(0, 1)  # zero-initialised: make every head's sink differ
    return model.float().train()


def _manager(size, rank):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.packed_seq_lengths = None
    manager.mesh = types.SimpleNamespace(get_group = lambda: None, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.float().flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
        )
        / count
    )


def _batch(packed):
    ids = torch.randint(
        0, 96, (1 if packed else 2, SEQ), generator = torch.Generator().manual_seed(1)
    )
    labels = ids.clone()
    labels[labels % 5 == 0] = -100
    return ids, labels


def _reference(model, ids, labels, lengths):
    """One process: the whole rows, or every packed document as its own row."""
    if lengths is None:
        shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
        count = shift.ne(-100).sum()
        _loss(model(input_ids = ids, use_cache = False).logits, shift, count).backward()
        return count
    pieces, start = [], 0
    for n in lengths:
        doc_ids, doc_labels = ids[:, start : start + n], labels[:, start : start + n]
        pieces.append((doc_ids, F.pad(doc_labels, (0, 1), value = -100)[:, 1:]))
        start += n
    count = sum(shift.ne(-100).sum() for _, shift in pieces)
    total = sum(
        _loss(model(input_ids = doc_ids, use_cache = False).logits, shift, count)
        for doc_ids, shift in pieces
    )
    total.backward()
    return count


def _parity_worker(
    rank,
    world,
    kind,
    implementation,
    packed,
    out_dir,
    control = None,
):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(kind, implementation)
    assert cp._generic_hf_attention(model) and cp._supports_context_parallel(model, "ulysses")
    cp._check_cp_model(model, world, "ulysses")
    ids, labels = _batch(packed)
    lengths = [6, 7, 3] if packed else None
    count = _reference(model, ids, labels, lengths)
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)

    calls, lengths_seen = [], set()
    original = uh._ulysses_hf_attention

    def counted(module, query, *args, **kwargs):
        fn = kwargs["attention_fn"]

        def seen(shard, q, *a, **k):
            lengths_seen.add(int(q.shape[2]))  # the model's own function sees the whole sequence
            return fn(shard, q, *a, **k)

        kwargs["attention_fn"] = seen
        calls.append(1)
        return original(module, query, *args, **kwargs)

    uh._ulysses_hf_attention = counted
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    if control == "window_dropped":
        uh._sliding_window = lambda module, kwargs: None
    elif control == "softcap_dropped":
        original_counted = uh._ulysses_hf_attention

        def no_softcap(*args, **kwargs):
            kwargs.pop("softcap", None)
            return original_counted(*args, **kwargs)

        uh._ulysses_hf_attention = no_softcap
    elif control == "sinks_unsliced":
        original_init = uh._HeadShard.__init__

        def full_sinks(self, module, **overrides):
            overrides.pop("sinks", None)
            original_init(self, module, **overrides)

        uh._HeadShard.__init__ = full_sinks
    inputs = {"input_ids": ids.clone(), "labels": labels.clone()}
    if packed:
        inputs["packed_seq_lengths"] = torch.tensor(lengths, dtype = torch.int32)
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"],
            position_ids = inputs["position_ids"],
            use_cache = False,
        )
        loss = _loss(out.logits, inputs["shift_labels"], count)
        loss.backward()
    dist.all_reduce(loss.detach())
    uh._ulysses_hf_attention = original
    assert len(calls) == model.config.num_hidden_layers, calls
    assert lengths_seen == {SEQ}, lengths_seen
    assert inputs["input_ids"].shape[1] == SEQ // world
    worst = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.clone()
        dist.all_reduce(grad)
        want = reference.get(name, torch.zeros_like(param))
        torch.testing.assert_close(grad, want, rtol = 1e-4, atol = 1e-6, msg = name)
        worst = max(worst, float((grad - want).abs().max()))
    if rank == 0 and out_dir:
        with open(
            os.path.join(out_dir, f"{kind}_{implementation}_{'packed' if packed else 'rows'}.txt"),
            "w",
        ) as f:
            f.write(
                f"loss {float(loss):.6f} ulysses calls {len(calls)} max grad diff {worst:.3e}\n"
            )


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


_OUT = os.environ.get("UNSLOTH_TEST_PARITY_OUT", "")


@pytest.mark.parametrize(
    "kind, implementation, packed",
    [
        ("gemma3", "eager", False),  # sliding window + non-default scale + replicated kv head
        ("gemma2", "eager", False),  # attention logit softcapping + sliding window
        ("gemma3", "sdpa", False),  # sliding window through sdpa's mask
        ("gemma3", "eager", True),  # window and packed documents in one mask
        ("qwen3", "sdpa", False),  # plain causal: sdpa's is_causal path
        ("qwen3", "sdpa", True),
        ("gpt_oss", "eager", False),  # attention sinks (local heads' slice) + sliding window
        ("deepseek_v2", "sdpa", False),  # MLA: query / key and value head dims differ, YaRN scale
        ("deepseek_v2", "eager", False),
        ("minimax_m2", "sdpa", False),  # GQA with q / k norms over all heads, MoE
    ],
)
def test_generic_ulysses_matches_one_process(kind, implementation, packed):
    _config(kind)  # skip before forking when the repo is unavailable
    if kind in BUILT and int(transformers.__version__.split(".")[0]) < 5:
        model = _model(kind, implementation)
        if not cp._supports_context_parallel(model, "ulysses"):
            # A refusal before training, not a wrong gradient: transformers 4.x builds DeepSeek-V2's experts as
            # per-expert modules, which the support check declines (_hybrid_config_blocker).
            pytest.skip(
                f"{kind}: generic Ulysses is declined on transformers {transformers.__version__}: "
                f"{cp._hybrid_config_blocker(model)}"
            )
    _spawn(_parity_worker, 2, kind, implementation, packed, _OUT)


def _fullgraph_packed_worker(rank, world, kind, implementation, out_dir):
    """fullgraph-compiled layers: a packed row with a padding tail traces with no graph break and matches one process."""
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(kind, implementation)
    for layer in model.model.layers:
        layer.compile(backend = "eager", fullgraph = True)
    ids, labels = _batch(True)
    ids, labels = ids[:, :15], labels[:, :15]  # 15 tokens: one padding token to the cp multiple
    lengths = [6, 7, 2]
    count = _reference(model, ids, labels, lengths)
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    inputs = {
        "input_ids": ids.clone(),
        "labels": labels.clone(),
        "packed_seq_lengths": torch.tensor(lengths, dtype = torch.int32),
    }
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"],
            position_ids = inputs["position_ids"],
            packed_seq_lengths = inputs["packed_seq_lengths"],
            use_cache = False,
        )
        _loss(out.logits, inputs["shift_labels"], count).backward()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = torch.zeros_like(param) if param.grad is None else param.grad.clone()
        dist.all_reduce(grad)
        want = reference.get(name, torch.zeros_like(param))
        torch.testing.assert_close(grad, want, rtol = 1e-4, atol = 1e-6, msg = name)


# Qwen3's decoder layer traces whole in one process (Gemma3's does not: its own forward breaks the graph).
@pytest.mark.parametrize("kind, implementation", [("qwen3", "sdpa"), ("qwen3", "eager")])
def test_compiled_packed_ulysses_has_no_graph_break(kind, implementation):
    _config(kind)
    try:
        _spawn(_fullgraph_packed_worker, 2, kind, implementation, _OUT)
    except mp.ProcessRaisedException as error:
        # torch 2.10's dynamo fails building a guard on transformers' sdpa path in the one-process
        # reference itself (AttributeError on transformers.quantizers.auto), before any CP code runs.
        text = str(error)
        if (
            tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 13)
            and "InternalTorchDynamoError" in text
            and "transformers.quantizers.auto" in text
        ):
            pytest.skip(f"torch {torch.__version__} dynamo guard bug in the one-process reference")
        raise


def test_whole_sequence_mask_keeps_the_padding_tail_apart():
    # Token 16 (padding past cu_seqlens[-1]) is its own segment.
    mask = uh._whole_sequence_mask(
        16,
        torch.device("cpu"),
        torch.float32,
        None,
        torch.tensor([0, 6, 13, 15], dtype = torch.int32),
    )[0, 0]
    allowed = mask == 0
    assert allowed[15].nonzero().flatten().tolist() == [15]
    assert not allowed[13:15, 15].any() and allowed[14, 13]
    assert allowed[12].nonzero().flatten().tolist() == list(range(6, 13))


def _compiled_rows_worker(rank, world, kind, implementation, out_dir):
    """Per-layer compiled model, two same-shape packed rows with different documents: each must match one process."""
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(kind, implementation)
    layers = model.model.layers
    for layer in layers:
        layer.compile(backend = "eager")  # dynamo alone: what is traced, not Inductor's numerics
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    ids, labels = _batch(True)
    for lengths in ([6, 7, 3], [3, 9, 4]):
        model.zero_grad(set_to_none = True)
        count = _reference(model, ids, labels, lengths)
        reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none = True)
        inputs = {
            "input_ids": ids.clone(),
            "labels": labels.clone(),
            "packed_seq_lengths": torch.tensor(lengths, dtype = torch.int32),
        }
        with manager.apply(inputs):
            out = model(
                input_ids = inputs["input_ids"],
                position_ids = inputs["position_ids"],
                packed_seq_lengths = inputs["packed_seq_lengths"],
                use_cache = False,
            )
            _loss(out.logits, inputs["shift_labels"], count).backward()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            grad = torch.zeros_like(param) if param.grad is None else param.grad.clone()
            dist.all_reduce(grad)
            want = reference.get(name, torch.zeros_like(param))
            torch.testing.assert_close(
                grad, want, rtol = 1e-4, atol = 1e-6, msg = f"{name} on documents {lengths}"
            )


@pytest.mark.parametrize("kind, implementation", [("qwen3", "sdpa"), ("gemma3", "eager")])
def test_compiled_layers_follow_each_packed_row(kind, implementation):
    _config(kind)
    _spawn(_compiled_rows_worker, 2, kind, implementation, _OUT)


@pytest.mark.parametrize(
    "kind, control",
    [("gemma3", "window_dropped"), ("gemma2", "softcap_dropped"), ("gpt_oss", "sinks_unsliced")],
)
def test_each_feature_is_load_bearing(kind, control):
    """Negative controls: dropping the window or the softcap after the exchange, or handing every head's sinks
    to a head shard, breaks parity, so the cases above prove the features are passed through."""
    _config(kind)
    with pytest.raises(Exception, match = "AssertionError|IndexError|RuntimeError|Mismatch"):
        _spawn(_parity_worker, 2, kind, "eager", False, "", control)


def test_windowed_hf_models_pass_the_ulysses_check_but_not_ring():
    model = _model("gemma2", "eager")
    assert cp._unsupported_attention(model.config, "ulysses", generic = True) is None
    softcapped = types.SimpleNamespace(attn_logit_softcapping = 3.0)
    assert "softcapping" in cp._unsupported_attention(softcapped, "ring")
    assert cp._unsupported_attention(softcapped, "ulysses", generic = True) is None
    assert "sliding" in cp._unsupported_attention(model.config, "ring")
    assert "sliding" in cp._unsupported_attention(model.config, "ulysses")  # Unsloth's fast forward
    with pytest.raises(NotImplementedError, match = "sliding"):
        cp._check_cp_model(model, 2, "ring")
    cp._check_cp_model(model, 2, "ulysses")
    # DPO / GRPO keep the Llama-style rule.
    with pytest.raises(NotImplementedError):
        cp._check_cp_model(model, 2, "ulysses", allow_hybrids = False)


def test_chunked_attention_stays_refused():
    model = _model("qwen3", "sdpa")
    model.config.attention_chunk_size = 8
    with pytest.raises(NotImplementedError, match = "chunked"):
        cp._check_cp_model(model, 2, "ulysses")


def test_unsloth_fast_forwards_are_not_generic():
    from unsloth.models.llama import LlamaAttention_fast_forward

    class Fast(torch.nn.Module):
        forward = LlamaAttention_fast_forward

    root = torch.nn.Module()
    root.self_attn = Fast()
    assert not uh.is_hf_registry_attention(root.self_attn)
    assert not cp._generic_hf_attention(root)
    assert cp._supports_context_parallel(root, "ulysses")  # the fast Llama path, as before


def test_ulysses_scope_restores_the_manager_when_packed_metadata_raises():
    # Malformed lengths (empty here) raise while the packed metadata is built: the manager must not stay active.
    from unsloth.utils.packing import get_packed_info_from_kwargs

    manager = _manager(1, 0)
    previous = cp._ACTIVE_MANAGER
    inputs = {
        "input_ids": torch.zeros(1, SEQ, dtype = torch.long),
        "packed_seq_lengths": torch.tensor([], dtype = torch.int32),
    }
    with pytest.raises(ValueError):
        with manager.apply(inputs):
            get_packed_info_from_kwargs(
                {"packed_seq_lengths": inputs["packed_seq_lengths"]}, torch.device("cpu")
            )
    assert cp._ACTIVE_MANAGER is previous
    assert manager.packed_seq_lengths is None
