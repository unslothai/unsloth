# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Unsloth's native DeepSeek-V4.1 (unsloth/models/deepseek_v41): registration, the reference semantics of its parts,
training forward / backward, save / reload, and the released-layout guard. CPU, tiny random configs."""

import os

import pytest

torch = pytest.importorskip("torch")

from unsloth.models.deepseek_v41 import register_deepseek_v41, transformers_has_deepseek_v41  # noqa: E402
from unsloth.models.deepseek_v41 import modeling as dsv41  # noqa: E402
from unsloth.models.deepseek_v41.configuration import DeepseekV41Config, DeepseekV41TextConfig  # noqa: E402

needs_native = pytest.mark.skipif(
    transformers_has_deepseek_v41(),
    reason = "transformers ships deepseek_v41; the native classes are not registered",
)


def _tiny_config(**over):
    text = dict(
        vocab_size = 512,
        hidden_size = 64,
        moe_intermediate_size = 32,
        num_hidden_layers = 6,
        num_attention_heads = 2,
        head_dim = 32,
        qk_rope_head_dim = 8,
        q_lora_rank = 32,
        o_groups = 2,
        o_lora_rank = 16,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 8,
        compress_ratios = [0, 0, 2, 2, 1, 1, 0],
        kv_source_layer_ids = [2, 4],
        index_source_layer_ids = [2, 3, 4, 5],
        candidate_source_layer_id = 4,
        candidate_topk_blocks = 2,
        candidate_block_size = 2,
        index_n_heads = 2,
        index_head_dim = 16,
        index_topk = 3,
        engram_layer_ids = [1, 4],
        engram_vocab_size = 101,
        engram_max_ngram_size = 3,
        engram_n_heads = 2,
        engram_head_dim = 32,
        engram_compressed_vocab_size = 97,
        num_nextn_predict_layers = 1,
        max_position_embeddings = 1024,
        rope_scaling = {
            "rope_type": "yarn",
            "factor": 4,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 64,
        },
    )
    text.update(over)
    primes, offsets = dsv41.engram_layout(
        tuple(text["engram_layer_ids"]),
        text["engram_max_ngram_size"],
        text["engram_n_heads"],
        text["engram_vocab_size"],
    )
    text["engram_num_embeddings"] = [o[-1] + p[-1][-1] for o, p in zip(offsets, primes)]
    return DeepseekV41Config(text_config = text)


def _tiny_model(seed = 0, **over):
    torch.manual_seed(seed)
    model = dsv41.DeepseekV41ForCausalLM(_tiny_config(**over)).float()
    model.bind_token_map([i % 97 for i in range(512)])
    text = model.config.text_config
    for rows, layer_id in zip(text.engram_num_embeddings, text.engram_layer_ids):
        model.model.set_engram_table(layer_id, torch.randn(rows, text.engram_head_dim) * 0.5)
    return model


def test_text_fields_are_read_through_the_composite_config():
    config = _tiny_config()
    assert isinstance(config.text_config, DeepseekV41TextConfig)
    assert config.hidden_size == 64 and config.num_experts == 4
    assert config.text_config.rope_yarn_factor == 4.0
    assert config.text_config.rope_yarn_original_max_position_embeddings == 64
    restored = DeepseekV41Config.from_dict(config.to_dict())
    assert restored.text_config.compress_ratios == config.text_config.compress_ratios
    assert restored.text_config.rope_yarn_factor == 4.0


@needs_native
def test_auto_classes_resolve_to_the_native_model():
    from transformers import AutoConfig, AutoModelForCausalLM

    assert register_deepseek_v41() is not None
    assert register_deepseek_v41() is not None  # idempotent
    config = AutoConfig.for_model("deepseek_v41", text_config = _tiny_config().text_config.to_dict())
    assert type(config) is DeepseekV41Config
    model = AutoModelForCausalLM.from_config(config)
    assert type(model).__name__ == "DeepseekV41ForCausalLM"


def test_sparse_attn_reference_matches_a_per_query_loop():
    torch.manual_seed(0)
    b, s, h, d, n, k = 2, 5, 3, 8, 7, 4
    q, kv = torch.randn(b, s, h, d), torch.randn(b, n, d)
    sink = torch.randn(h)
    idx = torch.randint(-1, n, (b, s, k), dtype = torch.int32)
    idx[0, 0] = -1  # no valid slot: zeros (kernel.py's finite running max)
    out = dsv41.sparse_attn_reference(q, kv, sink, idx, 0.3)
    for bi in range(b):
        for si in range(s):
            keep = idx[bi, si][idx[bi, si] >= 0].long()
            if keep.numel() == 0:
                assert torch.equal(out[bi, si], torch.zeros(h, d))
                continue
            scores = (q[bi, si] @ kv[bi, keep].T) * 0.3
            probs = torch.cat([scores, sink[:, None]], -1).softmax(-1)[:, :-1]
            torch.testing.assert_close(out[bi, si], probs @ kv[bi, keep], rtol = 1e-5, atol = 1e-6)


def test_window_indices_follow_the_reference_prefill_rule():
    got = dsv41._window_idxs(3, 5)
    expected = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype = torch.int32
    )
    assert torch.equal(got, expected)


def test_candidate_blocks_pin_the_newest_block_and_drop_unreachable_ones():
    inf = float("-inf")
    logits = torch.tensor(
        [[5.0, 4.0, 1.0, 0.5, 9.0, inf]]
    )  # positions 0..5, block size 2, reachable 0..4
    keep = dsv41.select_candidate_blocks(logits, torch.tensor([[5]]), topk_blocks = 2, block_size = 2)
    # block 2 holds position 4 (the newest) and is pinned; block 0 outscores block 1
    assert keep.tolist() == [[True, True, False, False, True, True]]


def test_hyper_connection_mix_is_doubly_stochastic():
    config = _tiny_config().text_config
    hc = dsv41.DeepseekV41HyperConnection(config)
    torch.nn.init.normal_(hc.fn, std = 0.5)
    pre, post, comb = hc(torch.randn(2, 3, config.hc_mult, config.hidden_size))
    # Sinkhorn ends on a column normalization (kernel.py:458-462): columns sum to 1, rows close to it after 20 steps
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 3, 4), atol = 1e-5, rtol = 0)
    torch.testing.assert_close(comb.sum(-1), torch.ones(2, 3, 4), atol = 0.1, rtol = 0)
    assert bool((pre > 0).all()) and bool((post >= 0).all()) and bool((post <= 2).all())


def test_engram_layout_hands_out_distinct_primes_with_cumulative_offsets():
    primes, offsets = dsv41.engram_layout((1, 4), 3, 2, 101)
    flat = [p for layer in primes for per in layer for p in per]
    assert len(set(flat)) == len(flat) and all(dsv41._is_prime(p) and p > 100 for p in flat)
    for layer_primes, layer_offsets in zip(primes, offsets):
        sizes = [p for per in layer_primes for p in per]
        assert list(layer_offsets) == [sum(sizes[:i]) for i in range(len(sizes))]


def test_engram_hashes_stop_at_the_row_start_and_at_padding():
    model = _tiny_model()
    hash_state = model.model.ngram_hash
    ids = torch.randint(0, 512, (1, 6))
    mask = torch.tensor([[True, True, True, False, True, True]])
    full, masked = hash_state(ids), hash_state(ids, mask)
    assert torch.equal(full[:, :3], masked[:, :3])  # before the pad: same history
    assert not torch.equal(full[:, 4], masked[:, 4])  # after it: the look-back is cut
    first = hash_state(ids[:, :1])
    assert torch.equal(first[0, 0], full[0, 0])  # position 0 hashes only padding history either way


def test_training_forward_backward_reaches_every_trainable_part():
    model = _tiny_model().train()
    ids = torch.randint(0, 512, (2, 24))
    out = model(input_ids = ids, labels = ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    graded = {
        n
        for n, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) > 0
    }
    for needle in (
        "self_attn.q_a_proj",
        "self_attn.kv_proj",
        "self_attn.o_a_proj",
        "self_attn.sinks",
        "compressor.kv_proj",
        "compressor.gate_proj",
        "engram.wkv",
        "experts.gate_up_proj",
        "shared_experts.down_proj",
        "attn_hc.fn",
        "lm_head",
    ):
        assert any(needle in n for n in graded), needle
    # selection only: the indexer and the router's correction bias get no gradient from the LM loss
    assert not any(".indexer." in n or "e_score_correction_bias" in n for n in graded)


def test_save_and_reload_keep_the_logits(tmp_path):
    from transformers import AutoModelForCausalLM

    register_deepseek_v41()
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 20))
    with torch.no_grad():
        before = model(input_ids = ids).logits
    model.save_pretrained(tmp_path)
    if dsv41.has_core_model_loading():
        # transformers 5 writes the released key names back (the renames run in reverse)
        from safetensors import safe_open
        with safe_open(os.path.join(tmp_path, "model.safetensors"), "pt") as f:
            keys = set(f.keys())
        assert not any("$" in k for k in keys)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype = torch.float32).eval()
    loaded.bind_token_map([i % 97 for i in range(512)])
    with torch.no_grad():
        after = loaded(input_ids = ids).logits
    assert torch.equal(before, after)


def test_released_layout_is_refused_without_core_model_loading(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    _tiny_config().save_pretrained(tmp_path)
    save_file({"embed.weight": torch.zeros(2, 2)}, os.path.join(tmp_path, "model.safetensors"))
    monkeypatch.setattr(dsv41, "has_core_model_loading", lambda: False)
    with pytest.raises(RuntimeError, match = "released layout"):
        dsv41.DeepseekV41ForCausalLM.from_pretrained(tmp_path)


def test_packed_position_ids_are_refused():
    model = _tiny_model()
    ids = torch.randint(0, 512, (1, 8))
    with pytest.raises(NotImplementedError, match = "packed"):
        model(input_ids = ids, position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]))


def test_experts_fallback_matches_a_per_token_fp64_oracle():
    config = _tiny_config(swiglu_limit = 0.5).text_config  # small limit: the clamps really fire
    torch.manual_seed(0)
    experts = dsv41._DeepseekV41Experts(config, 4)
    torch.nn.init.normal_(experts.gate_up_proj, std = 0.5)
    torch.nn.init.normal_(experts.down_proj, std = 0.5)
    x = torch.randn(9, config.hidden_size)
    idx = torch.randint(0, 3, (9, 2))  # expert 3 never hit: an empty group
    w = torch.rand(9, 2)
    before = dsv41.DSV41_STATS["experts_sorted_loop"]
    got = experts(x, idx, w)
    assert dsv41.DSV41_STATS["experts_sorted_loop"] == before + 1
    want = torch.zeros(9, config.hidden_size, dtype = torch.float64)
    for t in range(9):
        for k in range(2):
            e = int(idx[t, k])
            gate, up = (experts.gate_up_proj[e].double() @ x[t].double()).chunk(2)
            h = torch.nn.functional.silu(gate.clamp(max = 0.5)) * up.clamp(-0.5, 0.5)
            want[t] += w[t, k].double() * (experts.down_proj[e].double() @ h)
    torch.testing.assert_close(got.double(), want, rtol = 1e-4, atol = 1e-5)


def test_sparse_attention_engages_the_torch_reference_without_unsloth_zoo_kernel(monkeypatch):
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN", None)
    monkeypatch.setenv("UNSLOTH_DSV41_SPARSE_ATTN", "0")
    model = _tiny_model()
    before = dsv41.DSV41_STATS["sparse_attn_torch"]
    model(input_ids = torch.randint(0, 512, (1, 12)))
    assert dsv41.DSV41_STATS["sparse_attn_torch"] == before + 6  # one call per layer


@pytest.mark.parametrize("skip_shared", [True, False])
def test_shared_group_state_survives_accelerate_device_hooks(monkeypatch, skip_shared):
    """device_map splits attach accelerate hooks that rebuild dict kwargs; KV / index sources write into `shared`
    and later layers read it, so `shared` must be skipped by device placement. Flip: without the skip, readers
    see an empty copy."""
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        want = model(input_ids = ids).logits
    skip = list(dsv41.DeepseekV41PreTrainedModel._skip_keys_device_placement)
    if not skip_shared:
        skip = [k for k in skip if k != "shared"]
    for layer in model.model.layers:
        accelerate_hooks.add_hook_to_module(
            layer, accelerate_hooks.AlignDevicesHook(execution_device = "cpu", skip_keys = skip)
        )
    if skip_shared:
        with torch.no_grad():
            assert torch.equal(model(input_ids = ids).logits, want)
    else:
        with pytest.raises(KeyError):
            with torch.no_grad():
                model(input_ids = ids)


needs_frozen_tables = pytest.mark.skipif(
    not hasattr(
        __import__("unsloth_zoo.temporary_patches.deepseek_v41", fromlist = ["x"]),
        "make_deepseek_v41_engram_table",
    ),
    reason = "unsloth_zoo without the parameter-free DeepSeek-V4.1 engram table",
)


@needs_frozen_tables
def test_engram_tables_hold_no_parameters_or_buffers_and_survive_meta_init():
    model = _tiny_model()
    for table in model.model.engram_tables.values():
        assert not list(table.parameters()) and not list(table.buffers())
    assert not any("engram_tables" in n for n, _ in model.named_parameters())
    assert not any(
        "engram_tables" in k for k in model.state_dict()
    )  # DCP / FSDP2 / adapter saves skip them
    with torch.device("meta"):
        meta = dsv41.DeepseekV41ForCausalLM(_tiny_config())
    meta.to_empty(device = "cpu")  # nothing of the tables to allocate or copy
    assert all(not list(t.parameters()) for t in meta.model.engram_tables.values())


@needs_frozen_tables
def test_full_save_writes_the_tables_and_reload_rebinds_them(tmp_path):
    from transformers import AutoModelForCausalLM

    register_deepseek_v41()
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 20))
    with torch.no_grad():
        before = model(input_ids = ids).logits
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype = torch.float32).eval()
    assert all(t.is_bound for t in loaded.model.engram_tables.values())
    loaded.bind_token_map([i % 97 for i in range(512)])
    with torch.no_grad():
        assert torch.equal(loaded(input_ids = ids).logits, before)


def test_engram_hashes_of_a_sequence_shard_match_the_whole_sequence_with_left_context():
    model = _tiny_model()
    hash_state = model.model.ngram_hash
    ids = torch.randint(0, 512, (2, 30))
    whole = hash_state(ids)
    k = hash_state.max_ngram_size - 1
    for cut in (1, 2, 7, 18):
        shard = hash_state(ids[:, cut:], left_context = ids[:, max(0, cut - k) : cut])
        assert torch.equal(shard, whole[:, cut:]), cut
        # flip: without the context the shard's first n-grams differ
        assert not torch.equal(hash_state(ids[:, cut:])[:, 0], whole[:, cut])
    # -1 in the context = no token there (the sequence starts inside the window)
    assert torch.equal(
        hash_state(
            ids[:, 1:], left_context = torch.tensor([[-1, -1, 5]]).expand(2, 3).clone().fill_(-1)
        ),
        hash_state(ids[:, 1:]),
    )


def test_tp_plan_names_real_modules_and_keeps_the_kv_head_and_engram_replicated():
    import re

    plan = DeepseekV41TextConfig.base_model_tp_plan
    model = _tiny_model()
    names = {n for n, _ in model.model.named_modules()} | {
        n for n, _ in model.model.named_parameters()
    }
    for pattern in plan:
        rx = re.compile("^" + re.escape(pattern).replace(r"\*", r"\d+") + "$")
        assert any(rx.match(n) for n in names), pattern
    assert not any(
        k.endswith(("kv_proj", "compressor.kv_proj")) or "engram" in k or "indexer" in k
        for k in plan
    )
    assert (
        plan["layers.*.self_attn.q_b_proj"] == "colwise"
        and plan["layers.*.self_attn.o_b_proj"] == "rowwise"
    )


def test_attention_runs_with_a_tensor_parallel_shard_of_heads_and_groups():
    """Rank 0 of tp=2: half the q heads / sinks / o_a groups (colwise) and the matching o_b columns (rowwise); the
    partial outputs of both ranks sum to the unsharded output."""
    model = _tiny_model().eval()
    attn = model.model.layers[2].self_attn
    x = torch.randn(1, 12, model.config.text_config.hidden_size)
    pos = torch.arange(12)
    with torch.no_grad():
        full = attn(x, pos, {})
        total = 0
        for rank in range(2):
            shard = __import__("copy").deepcopy(attn)
            h, g = attn.n_heads // 2, attn.n_groups // 2
            d, r = attn.head_dim, attn.o_lora_rank
            shard.q_b_proj.weight.data = attn.q_b_proj.weight[
                rank * h * d : (rank + 1) * h * d
            ].clone()
            shard.sinks.data = attn.sinks[rank * h : (rank + 1) * h].clone()
            shard.o_a_proj.weight.data = attn.o_a_proj.weight[
                rank * g * r : (rank + 1) * g * r
            ].clone()
            shard.o_b_proj.weight.data = attn.o_b_proj.weight[
                :, rank * g * r : (rank + 1) * g * r
            ].clone()
            total = total + shard(x, pos, {})
    torch.testing.assert_close(total, full, rtol = 1e-5, atol = 1e-6)


def test_pipeline_cut_points_follow_the_group_sources():
    real = DeepseekV41TextConfig()
    assert dsv41.pipeline_safe_cut_points(real) == [0, 1, 2, 8, 14, 20]
