# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 generate with partial block residency.

When the whole gather does not fit, "auto" used to stream every decoder block (one blocking all-gather
per block per decoded token). It now keeps the longest prefix of blocks that fits on every rank gathered
for the whole call and streams only the rest. Checks: the plan picks the largest prefix every rank can
hold (and the old all-or-nothing answer with UNSLOTH_FSDP2_GENERATE_RESIDENT=0); on a gloo world of 2
greedy tokens equal one process while only the streamed blocks are gathered per step. The plan comes
from one fixed-length MIN agreement in every mode, so ranks with different caps, a bad cap on one rank,
or a forced count all pick the same blocks (gloo worlds of 2 below)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

F = _load_native()

GIB = 1 << 30
MIB = 1 << 20


class _Unit(torch.nn.Module):
    pass


class _Net(torch.nn.Module):
    def __init__(self, blocks = 6):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Unit() for _ in range(blocks)])


def _plan(
    monkeypatch,
    free,
    block = 100 * MIB,
    root = 300 * MIB,
    blocks = 6,
    other_free = None,
    request_bytes = 0,
    tail = None,
    prefetch = 0,
    offload = False,
):
    """_generate_gather_plan on a fake CUDA world: units = root + `blocks` blocks of `block` bytes (+ a
    `tail` unit gathered after the blocks, e.g. a tower). other_free: a second rank's free memory; the
    agreement takes the elementwise MIN of both ranks' votes and requires equal lengths."""
    import torch.distributed as dist

    net = _Net(blocks)
    units = [net] + list(net.layers)
    sizes = {id(net): root, **{id(m): block for m in net.layers}}
    if tail is not None:
        net.tail = _Unit()
        units.append(net.tail)
        sizes[id(net.tail)] = tail
    for m in net.layers:
        m._unsloth_forward_prefetch = [None] * prefetch
    group = types.SimpleNamespace(
        mesh_info = types.SimpleNamespace(mesh = None),
        device = torch.device("cuda", 0),
        fsdp_params = [types.SimpleNamespace(offload_to_cpu = offload)],
    )
    monkeypatch.setattr(F, "_fsdp_module_class", lambda: torch.nn.Module)
    monkeypatch.setattr(F, "is_native_fsdp2", lambda model: True)
    monkeypatch.setattr(F, "_generate_gather_mode", lambda: "auto")
    monkeypatch.setattr(
        F, "_stream_block_lists", lambda model, us: [(net.layers, dict(enumerate(net.layers)))]
    )
    monkeypatch.setattr(F, "_unit_param_groups", lambda unit: [group])
    monkeypatch.setattr(F, "_unit_gathered_bytes", lambda unit: sizes[id(unit)])
    monkeypatch.setattr(F, "_synchronized_empty_cache", lambda: None)
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    model = types.SimpleNamespace(modules = lambda: iter(units))

    def run(free_bytes, agree):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (free_bytes, free_bytes))
        monkeypatch.setattr(F, "_agree_min_over_ranks", agree)
        return F._generate_gather_plan(model, request_bytes = request_bytes)

    if other_free is None:
        return run(free, lambda d, dev, values, mesh: list(values))
    other = {}

    def capture(d, dev, values, mesh):
        other["votes"] = list(values)
        return list(values)

    run(other_free, capture)
    return run(
        free,
        lambda d, dev, values, mesh: [
            min(int(v), int(o)) for v, o in zip(values, other["votes"], strict = True)
        ],
    )


def test_plan_whole_gather_when_it_fits(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # root 300 + 6 x 100 + one 300 buffer + 1 GiB scratch
    assert _plan(monkeypatch, free = 4 * GIB) == (False, 0)


def test_plan_keeps_the_prefix_that_fits(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # k resident needs 300 + 100 k + 200 (two streamed buffers) MiB + 1 GiB; 1024 + 850 fits k = 3.
    stream, resident = _plan(monkeypatch, free = GIB + 850 * MIB)
    assert stream is True and resident == 3
    # Nothing beyond the root and two buffers fits: every block streams (the old answer).
    assert _plan(monkeypatch, free = GIB + 520 * MIB) == (True, 0)


def test_plan_resident_zero_restores_all_or_nothing(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "0")
    assert _plan(monkeypatch, free = GIB + 850 * MIB) == (True, 0)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "2")
    assert _plan(monkeypatch, free = GIB + 850 * MIB) == (True, 2)  # capped by the override
    # A bad value warns and keeps nothing resident instead of raising (peers would wait).
    for bad in ("-1", "abc"):
        monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", bad)
        with pytest.warns(UserWarning):
            assert _plan(monkeypatch, free = GIB + 850 * MIB) == (True, 0)


def test_plan_without_request_estimate_keeps_nothing_resident(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    assert _plan(monkeypatch, free = GIB + 850 * MIB, request_bytes = None) == (True, 0)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "2")  # an explicit cap still applies
    assert _plan(monkeypatch, free = GIB + 850 * MIB, request_bytes = None) == (True, 2)


def test_plan_budgets_the_request_workspace(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # 3 fit with the 1 GiB floor; a 1.5 GiB KV / workspace estimate leaves room for 0.
    assert _plan(monkeypatch, free = GIB + 850 * MIB, request_bytes = GIB // 2)[1] == 3
    assert _plan(monkeypatch, free = GIB + 850 * MIB, request_bytes = GIB + GIB // 2) == (True, 0)


def test_plan_budgets_the_forward_prefetch_lookahead(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # Prefetch depth 2: k resident needs 300 + 100 k + 4 x 100 MiB + 1 GiB, so 1 fits, not 3.
    assert _plan(monkeypatch, free = GIB + 850 * MIB, prefetch = 2) == (True, 1)


def test_plan_budgets_the_upfront_gather_peak(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # A 500 MiB unit gathered after the blocks: its all-gather output next to its copied weights
    # (2 x 500) on top of root + k blocks is the peak, not decode's 300 + 500 + 100 k + 200.
    # free 1 GiB + 1650 MiB: decode alone would allow 5, the gather's 300 + 100 k + 1000 allows 3.
    assert _plan(monkeypatch, free = GIB + 1650 * MIB, tail = 500 * MIB) == (True, 3)


def test_plan_cpu_offload_streams_every_block(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    assert _plan(monkeypatch, free = GIB + 850 * MIB, offload = True) == (True, 0)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "3")
    assert _plan(monkeypatch, free = GIB + 850 * MIB, offload = True) == (True, 0)


def test_request_estimate_reads_the_generate_call():
    config = types.SimpleNamespace(
        num_hidden_layers = 4,
        hidden_size = 64,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        intermediate_size = 128,
        vocab_size = 1000,
        torch_dtype = torch.float16,
    )
    model = types.SimpleNamespace(
        config = config, generation_config = types.SimpleNamespace(max_new_tokens = None)
    )
    ids = torch.zeros(3, 10, dtype = torch.long)
    got = F._generate_request_bytes(model, (), {"input_ids": ids, "max_new_tokens": 90})
    # Decode allocator rows: 512 past the 10-token prompt + 1, rounded to 768, + one 512 increment.
    kv = 4 * 3 * (768 + 512) * (2 * 2 * 16 + 4) * 2
    act = 3 * 10 * (4 * 64 + 3 * 128) * 2
    assert got == int(1.25 * (kv + act + 2 * 3 * 1000 * 4))
    # More return sequences scale the batch; an unknown length gives no estimate.
    more = F._generate_request_bytes(
        model, (ids,), {"max_new_tokens": 90, "num_return_sequences": 2}
    )
    assert more > got
    assert F._generate_request_bytes(model, (ids,), {}) is None


def test_request_estimate_counts_retained_outputs_and_cached_history():
    """Scores / logits kept per generated token and a supplied cache's history both grow what generate
    holds; outputs that cannot be sized (attentions, hidden states) and unreadable caches give None."""
    config = types.SimpleNamespace(
        num_hidden_layers = 2, hidden_size = 64, num_attention_heads = 4, num_key_value_heads = 2,
        head_dim = 16, intermediate_size = 128, vocab_size = 50000, torch_dtype = torch.bfloat16,
    )  # fmt: skip
    model = types.SimpleNamespace(config = config, generation_config = None)
    ids = torch.zeros(2, 8, dtype = torch.long)
    base = F._generate_request_bytes(model, (ids,), {"max_new_tokens": 1000})
    kept = F._generate_request_bytes(
        model, (ids,), {"max_new_tokens": 1000, "return_dict_in_generate": True,
                        "output_scores": True, "output_logits": True},
    )  # fmt: skip
    # Two fp32 [batch, vocab] tensors per generated token, times the 1.25 margin.
    assert abs(kept - base - 1.25 * 2 * 1000 * 2 * 50000 * 4) <= 1
    # Without return_dict_in_generate nothing is retained.
    assert (
        F._generate_request_bytes(model, (ids,), {"max_new_tokens": 1000, "output_scores": True})
        == base
    )
    for name in ("output_attentions", "output_hidden_states"):
        assert (
            F._generate_request_bytes(
                model, (ids,), {"max_new_tokens": 1000, "return_dict_in_generate": True, name: True}
            )
            is None
        )

    class Cache:
        def get_seq_length(self):
            return 5000

    history = F._generate_request_bytes(
        model, (ids,), {"max_new_tokens": 10, "past_key_values": Cache()}
    )
    assert history > F._generate_request_bytes(model, (ids,), {"max_new_tokens": 10})
    legacy = ((torch.zeros(2, 2, 5000, 16), torch.zeros(2, 2, 5000, 16)),)
    assert (
        F._generate_request_bytes(model, (ids,), {"max_new_tokens": 10, "past_key_values": legacy})
        == history
    )
    assert (
        F._generate_request_bytes(
            model, (ids,), {"max_new_tokens": 10, "past_key_values": object()}
        )
        is None
    )


def test_request_estimate_max_length_ignores_cached_history():
    """generate bounds a suffix-only call by max_length - len(input_ids), so 4096 cached tokens must not
    eat the max_length budget: an 8-token suffix with max_length = 2056 decodes 2048 new tokens."""
    config = types.SimpleNamespace(
        num_hidden_layers = 2, hidden_size = 64, num_attention_heads = 4, num_key_value_heads = 2,
        head_dim = 16, intermediate_size = 128, vocab_size = 1000, torch_dtype = torch.bfloat16,
    )  # fmt: skip
    model = types.SimpleNamespace(config = config, generation_config = None)
    ids = torch.zeros(1, 8, dtype = torch.long)

    class Cache:
        def get_seq_length(self):
            return 4096

    def estimate(**kw):
        return F._generate_request_bytes(model, (ids,), {"past_key_values": Cache(), **kw})

    assert estimate(max_length = 2056) == estimate(max_new_tokens = 2048)
    assert estimate(max_length = 2056) > estimate(max_new_tokens = 0)


def _qwen_like(**extra):
    config = types.SimpleNamespace(
        num_hidden_layers = 28, hidden_size = 1536, num_attention_heads = 12, num_key_value_heads = 2,
        head_dim = 128, intermediate_size = 8960, vocab_size = 151936, torch_dtype = torch.bfloat16,
        **extra,
    )  # fmt: skip
    return types.SimpleNamespace(config = config, generation_config = None)


def test_request_estimate_counts_prefill_logits_scores_and_experts():
    """Prefill work the request itself selects: full prompt logits (explicit logits_to_keep=0, a tensor
    of positions, or a model that cannot slice logits), eager attention's score matrices (also forced by
    output_attentions) and MoE routed copies. Media inputs run a tower that is not sized: None."""
    model = _qwen_like()
    prompt = 4096
    ids = torch.zeros(1, prompt, dtype = torch.long)
    call = {"max_new_tokens": 16}
    base = F._generate_request_bytes(model, (ids,), call)
    assert F._generate_request_bytes(model, (ids,), {**call, "logits_to_keep": 1}) == base
    full = F._generate_request_bytes(model, (ids,), {**call, "logits_to_keep": 0})
    # Every prompt row of fp32 logits instead of one, times the 1.25 margin.
    assert abs(full - base - 1.25 * (prompt - 1) * 151936 * 4) <= 1
    assert (
        F._generate_request_bytes(model, (ids,), {**call, "logits_to_keep": torch.arange(prompt)})
        == full
    )
    assert F._generate_request_bytes(model, (ids,), {**call, "logits_to_keep": 64}) < full
    unsliced = types.SimpleNamespace(**vars(model), _supports_logits_to_keep = lambda: False)
    assert F._generate_request_bytes(unsliced, (ids,), call) == full

    scores = 1.25 * 12 * prompt * prompt * (2 * 2 + 4)  # bf16 scores + fp32 softmax + cast back
    eager = F._generate_request_bytes(_qwen_like(_attn_implementation = "eager"), (ids,), call)
    assert abs(eager - base - scores) <= 1
    assert F._generate_request_bytes(_qwen_like(_attn_implementation = "sdpa"), (ids,), call) == base
    assert (
        abs(F._generate_request_bytes(model, (ids,), {**call, "output_attentions": True}) - eager)
        <= 1
    )

    moe = F._generate_request_bytes(
        _qwen_like(num_experts = 128, num_experts_per_tok = 8, moe_intermediate_size = 768), (ids,), call
    )
    assert abs(moe - base - 1.25 * prompt * (8 * (2 * 1536 + 3 * 768) * 2 + 128 * 4)) <= 1

    for name in ("pixel_values", "pixel_values_videos", "input_features"):
        assert (
            F._generate_request_bytes(model, (ids,), {**call, name: torch.zeros(1, 3, 8, 8)})
            is None
        )
    assert F._generate_request_bytes(model, (ids,), {**call, "pixel_values": None}) == base


def test_plan_full_prompt_logits_keep_fewer_blocks_resident(monkeypatch):
    """A long logits_to_keep=0 request must not spend its own logits memory on resident blocks."""
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    model = _qwen_like()
    ids = torch.zeros(1, 4096, dtype = torch.long)
    call = {"max_new_tokens": 16}
    last = F._generate_request_bytes(model, (ids,), call)
    full = F._generate_request_bytes(model, (ids,), {**call, "logits_to_keep": 0})
    free = GIB + 850 * MIB
    assert _plan(monkeypatch, free = free, request_bytes = last) == (True, 3)
    assert _plan(monkeypatch, free = free, request_bytes = full) == (True, 0)


def _decode_allocated_bytes(config, batch, prompt, new_tokens):
    """Bytes alloc_decode_buffers (eager) + resize_decode_buffers really hold on CPU for one call."""
    from unsloth.models import _decode_graphs as G

    heads, kv_heads, head_dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    attn = types.SimpleNamespace(config = types.SimpleNamespace(model_type = "not_graphed"))
    K1 = torch.zeros(batch, kv_heads, prompt, head_dim, dtype = config.torch_dtype)
    G.alloc_decode_buffers(
        attn, K1, K1, batch, heads, kv_heads, head_dim, config.hidden_size,
        config.torch_dtype, torch.device("cpu"), G.KV_CACHE_INCREMENT,
    )  # fmt: skip
    for kv_seq_len in range(prompt + 1, prompt + new_tokens + 1):
        if kv_seq_len >= attn.paged_attention.shape[0]:
            G.resize_decode_buffers(attn, G.KV_CACHE_INCREMENT, batch, heads, kv_heads, head_dim)
    per_layer = sum(
        t.untyped_storage().nbytes()
        for t in (attn.paged_attention, attn.attention, attn.temp_QA, attn.temp_KV, attn.RH_Q)
    )
    return config.num_hidden_layers * per_layer


@pytest.mark.parametrize("prompt,new_tokens", [(16, 4), (40, 600), (300, 1200)])
def test_request_estimate_covers_the_decode_allocator(monkeypatch, prompt, new_tokens):
    """A short generation with a large batch: the decode allocator still holds KV_CACHE_INCREMENT rows
    past the prompt in every layer (and grows by it), far more than batch x (prompt + max_new_tokens).
    The estimate covers what alloc_decode_buffers really allocates, so the planner keeps fewer blocks
    resident than a prompt + max_new_tokens estimate would."""
    config = types.SimpleNamespace(
        num_hidden_layers = 2,
        hidden_size = 256,
        num_attention_heads = 8,
        num_key_value_heads = 8,
        head_dim = 32,
        intermediate_size = 512,
        vocab_size = 1000,
        torch_dtype = torch.bfloat16,
    )
    model = types.SimpleNamespace(config = config, generation_config = None)
    batch = 64
    ids = torch.zeros(batch, prompt, dtype = torch.long)
    got = F._generate_request_bytes(model, (ids,), {"max_new_tokens": new_tokens})
    assert got >= _decode_allocated_bytes(config, batch, prompt, new_tokens)

    if (prompt, new_tokens) != (16, 4):
        return
    # Scaled to a real model (32 layers, batch 64, 32 x 128 heads): the cache alone is GiBs, while
    # the old prompt + max_new_tokens estimate stayed near the 1 GiB floor.
    big = types.SimpleNamespace(**{**vars(config), "num_hidden_layers": 32, "num_attention_heads": 32,
                                    "num_key_value_heads": 32, "head_dim": 128, "hidden_size": 4096,
                                    "intermediate_size": 11008})  # fmt: skip
    request = F._generate_request_bytes(
        types.SimpleNamespace(config = big, generation_config = None),
        (ids,),
        {"max_new_tokens": new_tokens},
    )
    width = 2
    old = int(
        1.25
        * (
            2 * 32 * batch * (prompt + new_tokens) * 32 * 128 * width
            + batch * prompt * (4 * 4096 + 3 * 11008) * width
            + 2 * batch * 1000 * 4
        )
    )
    assert request > 4 * old
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    free = 12 * GIB
    kw = dict(block = 400 * MIB, root = 1 * GIB, blocks = 32)
    before = _plan(monkeypatch, free = free, request_bytes = old, **kw)
    after = _plan(monkeypatch, free = free, request_bytes = request, **kw)
    assert before[0] and after[0]
    assert after[1] < before[1], (before, after)


def _llama2_7b(fast_path):
    config = types.SimpleNamespace(
        num_hidden_layers = 32,
        hidden_size = 4096,
        num_attention_heads = 32,
        num_key_value_heads = 32,
        head_dim = 128,
        intermediate_size = 11008,
        vocab_size = 32000,
        torch_dtype = torch.float16,
    )
    model = types.SimpleNamespace(config = config, generation_config = None)
    if fast_path:
        model._unsloth_fast_path = True
    return model


def test_request_estimate_counts_prefill_and_paged_cache_on_the_fast_path():
    """Unsloth's fast decode path copies the prefill cache into a paged cache on the first decode step while
    generate still holds the prefill cache, so a long MHA prompt holds two full caches at once. Llama-2-7B,
    batch 1, an 8192-token prompt and 4 new tokens: ~4 GiB prefill cache next to the ~4.25 GiB paged one."""
    from unsloth.models import _decode_graphs as G

    batch, prompt, new = 1, 8192, 4
    ids = torch.zeros(batch, prompt, dtype = torch.long)
    call = {"max_new_tokens": new}
    layers, kv_heads, head_dim, heads, width = 32, 32, 128, 32, 2
    prefill_cache = layers * batch * prompt * 2 * kv_heads * head_dim * width
    rows = G.decode_cache_rows(prompt, new)
    paged = layers * batch * rows * (2 * kv_heads * head_dim + heads) * width
    assert prefill_cache == 4 * GIB
    fast = F._generate_request_bytes(_llama2_7b(True), (ids,), call)
    assert fast >= prefill_cache + paged, (fast / GIB, (prefill_cache + paged) / GIB)
    # A model off the fast path (HF decode) keeps one cache: the paged size alone covers it.
    hf = F._generate_request_bytes(_llama2_7b(False), (ids,), call)
    assert hf < prefill_cache + paged
    # A marker on an inner module (PEFT / FSDP wrapper roots) counts as well.
    inner = torch.nn.Module()
    inner._unsloth_fast_path = True
    wrapped = torch.nn.Module()
    wrapped.inner = inner
    wrapped.config = _llama2_7b(False).config
    wrapped.generation_config = None
    assert F._generate_request_bytes(wrapped, (ids,), call) == fast


def test_offloaded_stream_does_not_advertise_device_weights():
    unit = torch.nn.Module()
    group = types.SimpleNamespace(fsdp_params = [types.SimpleNamespace(offload_to_cpu = True)])
    orig = F._unit_param_groups
    F._unit_param_groups = lambda u: [group]
    try:
        assert F._GenerateBlockStream({0: unit}).keeps_weights_on_device is False
        group.fsdp_params[0].offload_to_cpu = False
        assert F._GenerateBlockStream({0: unit}).keeps_weights_on_device is True
    finally:
        F._unit_param_groups = orig


def test_plan_every_rank_picks_the_same_prefix(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    # This rank could hold 5 blocks, the other only 2: both must keep 2.
    assert _plan(monkeypatch, free = GIB + 1050 * MIB, other_free = GIB + 720 * MIB) == (True, 2)


def test_block_mode_resident_override(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(F, "_fsdp_module_class", lambda: torch.nn.Module)
    monkeypatch.setattr(F, "is_native_fsdp2", lambda model: True)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_GATHER", "block")
    monkeypatch.delenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", raising = False)
    assert F._generate_gather_plan(_Net()) == (True, 0)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "4")
    assert F._generate_gather_plan(_Net()) == (True, 4)
    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_GATHER", "all")
    assert F._generate_gather_plan(_Net()) == (False, 0)


def test_generate_keeps_resident_prefix_gathered(monkeypatch):
    """gloo world of 2, UNSLOTH_FSDP2_GENERATE_GATHER=block + 3 of 6 blocks resident: greedy tokens equal
    one process (HF generate and a decode loop that reads weights directly), the 3 resident blocks stay
    gathered (peak at least root + 3 blocks, which pure streaming never reaches) and only the other 3
    stream (peak below root + 3 + 2 blocks)."""
    pytest.importorskip("transformers")
    from test_fsdp2_native_generate_streams_blocks_gloo import _reference, _run

    monkeypatch.setenv("UNSLOTH_FSDP2_GENERATE_RESIDENT", "3")
    want = _reference()
    for rank, r in _run("block").items():
        for name in ("hf", "loop"):
            got = r[name]
            assert "error" not in got, (rank, name, got)
            assert got["tokens"] == want[name], (rank, name, got, want[name])
            assert got["sharded_after"] and got["swap_cleared"], (rank, name, got)
            assert got["peak"] >= got["root"] + 3 * got["block"], (rank, name, got)
            assert got["peak"] <= got["root"] + 5 * got["block"], (rank, name, got)
            assert got["peak"] < got["total"], (rank, name, got)


# gloo worlds of 2: real collectives, so a rank that diverges or raises shows up as a mismatch,
# an error or a hang instead of being hidden by a mock.


def _dist_env(rank, world, port, extra):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    for key, value in extra.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _plan_worker(rank, world, port, out, case):
    """_generate_gather_plan on fake units (root 300 MiB + 6 x 100 MiB blocks) under a real gloo
    world. case: {"env": [rank 0 env, rank 1 env], "device": str, "free": bytes}."""
    _dist_env(rank, world, port, case["env"][rank])
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    res = {}
    try:
        native = _load_native()
        net = _Net(6)
        units = [net] + list(net.layers)
        sizes = {id(net): 300 * MIB, **{id(m): 100 * MIB for m in net.layers}}
        group = types.SimpleNamespace(
            mesh_info = types.SimpleNamespace(mesh = None),
            device = torch.device(case["device"]),
            fsdp_params = [],
        )
        native._fsdp_module_class = lambda: torch.nn.Module
        native.is_native_fsdp2 = lambda model: True
        native._stream_block_lists = lambda model, us: [(net.layers, dict(enumerate(net.layers)))]
        native._unit_param_groups = lambda unit: [group]
        native._unit_gathered_bytes = lambda unit: sizes.get(id(unit), 0)
        native._synchronized_empty_cache = lambda: None
        free = case["free"]
        torch.cuda.mem_get_info = lambda *a, **k: (free, free)
        model = types.SimpleNamespace(modules = lambda: iter(units))
        res["plan"] = list(native._generate_gather_plan(model))
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    out.put((rank, res))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _spawn(
    worker,
    case,
    timeout = 120,
):
    import torch.multiprocessing as mp

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        worker,
        args = (2, _free_port(), out, case),
        nprocs = 2,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 2):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(f"the ranks did not finish within {timeout} s (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    print("RESIDENT_PLAN", results)
    return results


def _resident(value):
    return {"UNSLOTH_FSDP2_GENERATE_RESIDENT": value}


def test_auto_ranks_with_different_caps_agree_on_one_plan():
    """auto, caps 0 and 3 on the two ranks, both with room for 3 blocks: one fixed-length vote, so
    both keep 0 (the old per-prefix vote had 2 flags on one rank and 5 on the other)."""
    env = [
        {"UNSLOTH_FSDP2_GENERATE_GATHER": "auto", **_resident("0")},
        {"UNSLOTH_FSDP2_GENERATE_GATHER": "auto", **_resident("3")},
    ]
    results = _spawn(_plan_worker, {"env": env, "device": "cuda:0", "free": GIB + 850 * MIB})
    for rank, r in results.items():
        assert "error" not in r, (rank, r)
        assert r["plan"] == [True, 0], (rank, r)


def test_block_mode_ranks_agree_on_the_resident_count():
    """Forced block with RESIDENT=2 on one rank and 4 on the other: both keep 2 (a local count would
    send the ranks into different weight all-gathers)."""
    env = [
        {"UNSLOTH_FSDP2_GENERATE_GATHER": "block", **_resident("2")},
        {"UNSLOTH_FSDP2_GENERATE_GATHER": "block", **_resident("4")},
    ]
    results = _spawn(_plan_worker, {"env": env, "device": "cpu", "free": 0})
    for rank, r in results.items():
        assert "error" not in r, (rank, r)
        assert r["plan"] == [True, 2], (rank, r)


@pytest.mark.parametrize("mode, device", [("block", "cpu"), ("auto", "cuda:0")])
def test_bad_resident_cap_on_one_rank_does_not_strand_peers(mode, device):
    """One rank has RESIDENT=abc: no rank raises, both stream every block (the bad rank votes 0)."""
    env = [
        {"UNSLOTH_FSDP2_GENERATE_GATHER": mode, **_resident("abc")},
        {"UNSLOTH_FSDP2_GENERATE_GATHER": mode, **_resident("2")},
    ]
    results = _spawn(_plan_worker, {"env": env, "device": device, "free": GIB + 850 * MIB})
    for rank, r in results.items():
        assert "error" not in r, (rank, r)
        assert r["plan"] == [True, 0], (rank, r)


def _generate_worker(rank, world, port, out, case):
    """Real FSDP2 (CPU, gloo) _LoopNet with 6 block units, forced block + RESIDENT=case["resident"]:
    records whether generate entered the eager (no CUDA graphs) scope, and each layer's published
    execution device at generate entry (a stale cuda:0 tag is planted first)."""
    _dist_env(
        rank,
        world,
        port,
        {"UNSLOTH_FSDP2_GENERATE_GATHER": "block", **_resident(case["resident"])},
    )
    import datetime

    import torch.distributed as dist

    from test_fsdp2_native_generate_streams_blocks_gloo import _LoopNet, _loop_greedy

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    res = {}
    try:
        native = _load_native()
        eager = []
        original = native._eager_decode_scope

        def recording(stream):
            eager.append(bool(stream))
            return original(stream)

        native._eager_decode_scope = recording

        class _Net(_LoopNet):
            def generate(self, ids, new_tokens):
                self._devices_at_entry = [
                    str(layer.__dict__.get("_per_layer_device")) for layer in self.layers
                ]
                return super().generate(ids, new_tokens)

        model = _Net()
        native.shard_model(
            model,
            native.FSDP2Config(
                device = "cpu", units = list(model.layers), fit_layout = False, param_dtype = torch.float32
            ),
        )
        for layer in model.layers:
            layer.__dict__["_per_layer_device"] = torch.device("cuda", 0)
            layer.__dict__["_per_layer_device_index"] = 0
        res["tokens"] = _loop_greedy(model)
        res["eager"] = eager
        res["devices"] = model._devices_at_entry
    except Exception as exc:
        import traceback
        res["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    out.put((rank, res))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _loop_reference():
    from test_fsdp2_native_generate_streams_blocks_gloo import _LoopNet, _loop_greedy
    with torch.no_grad():
        return _loop_greedy(_LoopNet())


def test_block_mode_with_every_block_resident_keeps_cuda_graphs():
    """Forced block with all 6 blocks resident: nothing streams, so generate must not enter the eager
    scope (CUDA graphs stay on), and tokens equal one process."""
    want = _loop_reference()
    for rank, r in _spawn(_generate_worker, {"resident": "6"}).items():
        assert "error" not in r, (rank, r)
        assert r["tokens"] == want, (rank, r, want)
        assert r["eager"] == [False], (rank, r)


def test_resident_blocks_publish_their_execution_device():
    """Forced block, 2 of 6 resident: the 2 gathered up front republish their execution device over
    the stale cuda:0 tag (as a streamed block does on enter); the stream still decodes eagerly."""
    want = _loop_reference()
    for rank, r in _spawn(_generate_worker, {"resident": "2"}).items():
        assert "error" not in r, (rank, r)
        assert r["tokens"] == want, (rank, r, want)
        assert r["eager"] == [True], (rank, r)
        assert r["devices"][:2] == ["cpu", "cpu"], (rank, r)
        assert r["devices"][2:] == ["cuda:0"] * 4, (rank, r)  # streamed: published on enter
