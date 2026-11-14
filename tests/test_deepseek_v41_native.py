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


_TO_RELEASED = [
    ("embed_tokens.", "embed."),
    (".input_layernorm.", ".attn_norm."),
    (".post_attention_layernorm.", ".ffn_norm."),
    (".attn_hc.fn", ".hc_attn_fn"),
    (".attn_hc.base", ".hc_attn_base"),
    (".attn_hc.scale", ".hc_attn_scale"),
    (".ffn_hc.fn", ".hc_ffn_fn"),
    (".ffn_hc.base", ".hc_ffn_base"),
    (".ffn_hc.scale", ".hc_ffn_scale"),
    (".self_attn.sinks", ".self_attn.attn_sink"),
    (".q_a_proj.", ".wq_a."),
    (".q_a_norm.", ".q_norm."),
    (".q_b_proj.", ".wq_b."),
    (".self_attn.kv_proj.", ".self_attn.wkv."),
    (".compressor.kv_proj.", ".compressor.wkv."),
    (".compressor.gate_proj.", ".compressor.wgate."),
    (".compressor.kv_norm.", ".compressor.norm."),
    (".indexer.k_proj.", ".indexer.wk."),
    (".o_a_proj.", ".wo_a."),
    (".o_b_proj.", ".wo_b."),
    (".gate.e_score_correction_bias_vl", ".gate.bias_vl"),
    (".gate.e_score_correction_bias", ".gate.bias"),
    (".shared_experts.gate_proj.", ".shared_experts.w1."),
    (".shared_experts.down_proj.", ".shared_experts.w2."),
    (".shared_experts.up_proj.", ".shared_experts.w3."),
    (".self_attn.", ".attn."),
    (".mlp.", ".ffn."),
]


def _released_state_dict(model):
    """The tiny model under DeepSeek's released names (written from the reference checkpoint, independently of
    convert.py's table): per-expert w1 / w3 / w2, engram rows under layers.<i>.engram.embed, mtp keys present."""
    import re

    out = {}
    for name, t in model.state_dict().items():
        t = t.detach().clone()
        if name == "lm_head.weight":
            out["head.weight"] = t
            continue
        name = name[len("model.") :]
        m = re.match(r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$", name)
        if m:
            layer = m.group(1)
            for e in range(t.shape[0]):
                p = f"layers.{layer}.ffn.experts.{e}"
                if m.group(2) == "down_proj":
                    out[f"{p}.w2.weight"] = t[e].clone()
                else:
                    w1, w3 = t[e].chunk(2, dim = 0)
                    out[f"{p}.w1.weight"], out[f"{p}.w3.weight"] = w1.clone(), w3.clone()
            continue
        for a, b in _TO_RELEASED:
            name = name.replace(a, b)
        out[name] = t
    text = model.config.text_config
    for layer_id in text.engram_layer_ids:
        out[f"layers.{layer_id}.engram.embed.weight"] = model.model.engram_tables[
            str(layer_id)
        ].frozen_table_store.weight.clone()
    out["mtp.0.norm.weight"] = torch.ones(
        text.hidden_size
    )  # DSpark / MTP head: never built, never loaded
    return out


def test_released_layout_converts_to_the_native_layout_on_any_transformers(tmp_path, monkeypatch):
    """transformers 4.57 (no core_model_loading) and FP8 quantizers without fp4 experts load the released layout
    through convert.py; forced here so every transformers version runs it."""
    from safetensors.torch import save_file
    from unsloth.models.deepseek_v41 import convert

    model = (
        _tiny_model().to(torch.bfloat16).float().eval()
    )  # bf16-exact: the converter stacks experts in bf16
    src = tmp_path / "released"
    model.config.save_pretrained(src)
    released = _released_state_dict(model)
    for name in released:
        if not name.startswith("mtp.") and ".experts." not in name and ".engram.embed." not in name:
            assert convert.native_key(name) is not None
    save_file(released, os.path.join(src, "model.safetensors"))
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", "1")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    ids = torch.randint(0, 512, (1, 20))
    with torch.no_grad():
        want = model(input_ids = ids).logits
    loaded = dsv41.DeepseekV41ForCausalLM.from_pretrained(src, dtype = torch.float32).eval()
    loaded.bind_token_map([i % 97 for i in range(512)])
    assert not any("mtp" in n for n, _ in loaded.named_parameters())
    with torch.no_grad():
        assert torch.equal(loaded(input_ids = ids).logits, want)
    # converted once, then reused
    stamp = os.listdir(tmp_path / "converted")
    dsv41.DeepseekV41ForCausalLM.from_pretrained(src, dtype = torch.float32)
    assert os.listdir(tmp_path / "converted") == stamp


def test_released_fp4_and_fp8_blocks_dequantize_like_the_reference_converter():
    """inference/convert.py: fp4 e2m1 two per byte, low nibble first; fp8 e4m3 times one scale per 32x32 block."""
    from unsloth.models.deepseek_v41 import convert

    packed = torch.full((1, 16), 0x21, dtype = torch.uint8).view(
        torch.int8
    )  # low nibble 1 (0.5), high 2 (1.0)
    scale = torch.tensor([[2.0]])
    got = convert._dequant_fp4(packed, scale)
    assert got.shape == (1, 32)
    assert torch.equal(got[0, 0::2].float(), torch.full((16,), 1.0))
    assert torch.equal(got[0, 1::2].float(), torch.full((16,), 2.0))
    if hasattr(torch, "float8_e4m3fn"):
        w = torch.randn(64, 32).to(torch.float8_e4m3fn)
        s = torch.tensor([[0.5], [4.0]])
        want = w.float() * torch.repeat_interleave(s, 32, dim = 0)
        assert torch.equal(convert._dequant_fp8(w, s).float(), want.to(torch.bfloat16).float())


def test_offset_position_ids_are_refused():
    model = _tiny_model()
    ids = torch.randint(0, 512, (1, 8))
    with pytest.raises(NotImplementedError, match = "start at 0"):
        model(input_ids = ids, position_ids = torch.arange(5, 13).unsqueeze(0))


def _alone(model, ids, width):
    """One sequence in its own row, right padded to `width` (the row length the relayout gives it): the indexer's
    top-k breaks ties among equal (relu-zero) scores by position over the whole row, so the row width must match
    for an exact comparison. Right padding never changes a real token's inputs otherwise."""
    pad = width - len(ids)
    row = torch.cat([ids, torch.zeros(pad, dtype = torch.long)]).unsqueeze(0)
    mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad)]).long().unsqueeze(0)
    with torch.no_grad():
        return model(input_ids = row, attention_mask = mask).logits[0, : len(ids)]


def test_packed_sequences_match_each_sequence_run_alone():
    """Padding-free packing (position_ids restarting at 0): compressor groups, window, top-k blocks and engram
    n-grams stay inside each sequence, so every sequence's logits equal a run of it alone."""
    model = _tiny_model().eval()
    torch.manual_seed(1)
    a, b, c = torch.randint(0, 512, (7,)), torch.randint(0, 512, (13,)), torch.randint(0, 512, (4,))
    ids = torch.cat([a, b, c]).unsqueeze(0)
    pos = torch.cat([torch.arange(7), torch.arange(13), torch.arange(4)]).unsqueeze(0)
    with torch.no_grad():
        packed = model(input_ids = ids, position_ids = pos).logits[0]
    for seq, lo in ((a, 0), (b, 7), (c, 20)):
        assert torch.allclose(
            packed[lo : lo + len(seq)], _alone(model, seq, 13), atol = 1e-5, rtol = 1e-5
        )
    # the packed row trains: gradients reach the embeddings of every sequence
    model.train()
    out = model(input_ids = ids, position_ids = pos, labels = ids)
    out.loss.backward()
    # every token but the row's last one predicts a label, so its embedding row gets a gradient
    assert torch.isfinite(out.loss)
    assert model.model.embed_tokens.weight.grad[ids[0, :-1]].abs().sum(-1).gt(0).all()


def test_left_padding_matches_the_unpadded_sequence():
    model = _tiny_model().eval()
    torch.manual_seed(2)
    a, b = torch.randint(0, 512, (11,)), torch.randint(0, 512, (16,))
    ids = torch.stack([torch.cat([torch.zeros(5, dtype = torch.long), a]), b])
    mask = torch.stack([torch.cat([torch.zeros(5), torch.ones(11)]), torch.ones(16)]).long()
    with torch.no_grad():
        logits = model(input_ids = ids, attention_mask = mask).logits
    assert torch.allclose(logits[0, 5:], _alone(model, a, 16), atol = 1e-5, rtol = 1e-5)
    assert torch.allclose(logits[1], _alone(model, b, 16), atol = 1e-5, rtol = 1e-5)
    assert torch.equal(
        logits[0, :5], torch.zeros_like(logits[0, :5])
    )  # padding columns: zero hidden states


def test_plain_and_right_padded_batches_take_no_relayout(monkeypatch):
    model = _tiny_model().eval()
    calls = []
    monkeypatch.setattr(type(model.model), "_forward_segments", lambda *a, **k: calls.append(1))
    ids = torch.randint(0, 512, (2, 12))
    mask = torch.ones(2, 12, dtype = torch.long)
    mask[1, 9:] = 0
    with torch.no_grad():
        model(input_ids = ids, attention_mask = mask, position_ids = torch.arange(12).expand(2, -1))
    assert calls == []


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
        # the group state is not a dict, so the hooks pass it through regardless; the flip needs the old dict
        monkeypatch.setattr(dsv41, "DeepseekV41GroupState", dict)
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


def test_router_sqrtsoftplus_backward_stays_finite_when_softplus_underflows():
    """model.py:819 is inference only: softplus(z) is exactly 0 in fp32 for z <= ~-104, and sqrt's backward there
    is 0 / 0 = NaN for every expert (picked or not). The clamp keeps the gradient finite."""
    config = _tiny_config().text_config
    router = dsv41.DeepseekV41TopKRouter(config, 4, 2)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[0, 0] = 1.0
    x = torch.zeros(3, config.hidden_size)
    x[:, 0] = -500.0
    x.requires_grad_(True)
    raw = torch.nn.functional.softplus(
        torch.nn.functional.linear(x, router.weight) / router.gate_temp
    )
    assert float(raw[0, 0]) == 0.0
    (g,) = torch.autograd.grad(raw.sqrt().sum(), x)
    assert not torch.isfinite(g).all()  # the trap is real for a plain sqrt
    weights, _ = router(x)
    weights.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(router.weight.grad).all()


def test_swiglu_clamp_is_asymmetric_like_the_reference():
    """model.py:845-850: gate clamped from above only, up clamped to [-limit, limit]."""
    config = _tiny_config(swiglu_limit = 10.0).text_config
    experts = dsv41.DeepseekV41Experts(config, 2)
    gate = torch.tensor([-50.0, 50.0, 3.0])
    up = torch.tensor([-50.0, 50.0, 3.0])
    got = experts._apply_gate(torch.cat([gate, up]))
    silu = torch.nn.functional.silu
    want = torch.stack(
        [
            silu(torch.tensor(-50.0)) * -10.0,
            silu(torch.tensor(10.0)) * 10.0,
            silu(torch.tensor(3.0)) * 3.0,
        ]
    )
    assert torch.equal(got, want)


def test_eps_1e20_is_added_in_fp32_so_half_precision_zero_rows_stay_finite():
    """rms_norm_eps / hc_eps / the router denominator use 1e-20, which is 0 in fp16: every one of those sums runs in
    fp32 first."""
    torch.manual_seed(0)
    config = _tiny_config(rms_norm_eps = 1e-20).text_config
    flushed = torch.zeros(16, dtype = torch.float16) * torch.rsqrt(
        torch.zeros(16, dtype = torch.float16) + torch.tensor(1e-20, dtype = torch.float16)
    )
    assert torch.isnan(flushed).all()  # what an fp16 sum would do
    norm = dsv41.DeepseekV41RMSNorm(16, 1e-20).half()
    x = torch.zeros(2, 16, dtype = torch.float16)
    assert torch.equal(norm(x), x)
    # the exact derivative at a zero row is rsqrt(1e-20) = 1e10 (the reference eps): finite in fp32 / bf16
    x32 = torch.zeros(2, 16, requires_grad = True)
    dsv41.DeepseekV41RMSNorm(16, 1e-20)(x32).sum().backward()
    assert torch.isfinite(x32.grad).all()
    hc = dsv41.DeepseekV41HyperConnection(config)
    pre, post, comb = hc(torch.zeros(1, 3, config.hc_mult, config.hidden_size, dtype = torch.float16))
    assert all(torch.isfinite(t).all() for t in (pre, post, comb))
    router = dsv41.DeepseekV41TopKRouter(config, 4, 2).half()
    weights, _ = router(torch.zeros(3, config.hidden_size, dtype = torch.float16))
    assert torch.isfinite(weights).all()


def test_loss_function_receives_num_items_in_batch():
    """Gradient accumulation passes num_items_in_batch; dropping it rescales the loss per micro-batch."""
    model = _tiny_model()
    seen = {}

    def loss_function(logits, labels, vocab_size, **kwargs):
        seen.update(kwargs)
        return logits.float().mean()

    model.loss_function = loss_function
    ids = torch.randint(0, 512, (1, 8))
    model(input_ids = ids, labels = ids, num_items_in_batch = torch.tensor(7))
    assert int(seen["num_items_in_batch"]) == 7


def test_a_callers_cache_is_left_alone():
    """No KV cache: a cache handed in is neither filled nor replaced, and none is returned."""
    cache_utils = pytest.importorskip("transformers.cache_utils")
    try:
        cache = cache_utils.DynamicCache()
    except TypeError:
        pytest.skip("DynamicCache needs a config on this transformers")
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 8))
    with torch.no_grad():
        out = model(input_ids = ids, past_key_values = cache, use_cache = True)
    assert out.past_key_values is None
    assert cache.get_seq_length() == 0


def test_each_layer_reads_only_its_own_source_compressed_kv(monkeypatch):
    """Without a cache every compressing layer attends to the window plus the seqlen // ratio latents of its group's
    source layer (model.py:740-760), never a concatenation over several source layers."""
    model = _tiny_model().eval()
    calls = []
    real = dsv41.sparse_attn

    def spy(q, kv, sink, idxs, scale):
        calls.append(kv)
        return real(q, kv, sink, idxs, scale)

    monkeypatch.setattr(dsv41, "sparse_attn", spy)
    seqlen = 12
    with torch.no_grad():
        model(input_ids = torch.randint(0, 512, (1, seqlen)), use_cache = False)
    ratios = model.config.text_config.compress_ratios[: model.config.text_config.num_hidden_layers]
    assert len(calls) == len(ratios)
    for kv, ratio in zip(calls, ratios):
        assert kv.shape[1] == seqlen + (seqlen // ratio if ratio else 0)
    # layers 2, 3 share source 2's latents; layers 4, 5 share source 4's
    assert torch.equal(calls[2][:, seqlen:], calls[3][:, seqlen:])
    assert torch.equal(calls[4][:, seqlen:], calls[5][:, seqlen:])


@pytest.mark.parametrize("needs_conversion", ["0", "1"])
def test_release_fp8_config_is_dropped_only_where_it_cannot_load(monkeypatch, needs_conversion):
    """Where transformers loads the release through convert.py (dequantized), the release's FP8 quantization_config
    must vanish from the config, else Unsloth's loader sees a pre-quantized model and skips a 4-bit request; a
    bitsandbytes config is kept."""
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", needs_conversion)
    fp8 = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [32, 32],
    }
    config = DeepseekV41Config(
        text_config = _tiny_config().text_config.to_dict(), quantization_config = fp8
    )
    assert (getattr(config, "quantization_config", None) is None) == (needs_conversion == "1")
    bnb = {"quant_method": "bitsandbytes", "load_in_4bit": True}
    assert DeepseekV41Config(quantization_config = bnb).quantization_config == bnb


@pytest.mark.parametrize(
    "targets",
    [
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "all-linear",
        None,
    ],
)
def test_default_lora_targets_map_to_the_real_projections(targets):
    """FastLanguageModel's q/k/v/o/gate/up/down defaults, "all-linear" and None reach q_a/q_b, kv, o_a/o_b, the
    compressor, the engram wkv and the shared expert; never the indexer (selection only) or a router."""
    peft = pytest.importorskip("peft")
    mapped = dsv41.DeepseekV41ForCausalLM.unsloth_lora_target_modules(targets)
    model = peft.get_peft_model(_tiny_model(), peft.LoraConfig(r = 4, target_modules = mapped))
    wrapped = {n.rsplit(".lora_A", 1)[0] for n, _ in model.named_parameters() if ".lora_A." in n}
    kinds = {".".join(n.split(".")[-2:]) for n in wrapped}
    for kind in (
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_proj",
        "self_attn.o_a_proj",
        "self_attn.o_b_proj",
        "compressor.kv_proj",
        "compressor.gate_proj",
        "engram.wkv",
        "shared_experts.gate_proj",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
    ):
        assert kind in kinds, kind
    assert not any(".indexer." in n or n.endswith(".gate") for n in wrapped)
    # the family filters and a caller's regex
    attention_only = dsv41.DeepseekV41ForCausalLM.unsloth_lora_target_modules(None, mlp = False)
    assert "down_proj" not in attention_only and "self_attn.kv_proj" in attention_only
    assert dsv41.DeepseekV41ForCausalLM.unsloth_lora_target_modules(r".*q_a_proj") == r".*q_a_proj"


def test_a_flat_text_config_loads_into_the_composite(tmp_path):
    """A model built on DeepseekV41TextConfig saves a flat config; DeepseekV41ForCausalLM.from_pretrained reads it
    through the composite config, which must take those fields as its text config (not the released defaults)."""
    text = _tiny_config().text_config
    model = dsv41.DeepseekV41ForCausalLM(text).float().eval()
    model.bind_token_map([i % 97 for i in range(512)])
    for rows, layer_id in zip(text.engram_num_embeddings, text.engram_layer_ids):
        model.model.set_engram_table(layer_id, torch.randn(rows, text.engram_head_dim) * 0.5)
    ids = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        before = model(input_ids = ids).logits
    model.save_pretrained(tmp_path)
    loaded = dsv41.DeepseekV41ForCausalLM.from_pretrained(tmp_path, dtype = torch.float32).eval()
    assert loaded.config.get_text_config().hidden_size == 64
    loaded.bind_token_map([i % 97 for i in range(512)])
    with torch.no_grad():
        assert torch.equal(loaded(input_ids = ids).logits, before)


def _fsdp2_style_input_rebuild(model):
    """What FSDP2's _cast_forward_inputs does to every fully_shard'ed layer's inputs: torch.distributed.utils.
    _apply_to_tensors rebuilds each dict (subclasses included); other objects pass through."""
    from torch.distributed.utils import _apply_to_tensors

    def hook(module, args, kwargs):
        return _apply_to_tensors(lambda t: t, (args, kwargs))

    return [layer.register_forward_pre_hook(hook, with_kwargs = True) for layer in model.model.layers]


def test_group_state_survives_fsdp2_input_casting(monkeypatch):
    pytest.importorskip("torch.distributed.utils")
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        want = model(input_ids = ids).logits
    handles = _fsdp2_style_input_rebuild(model)
    with torch.no_grad():
        assert torch.equal(model(input_ids = ids).logits, want)
    # Flip: the dict the model used to pass is rebuilt per layer, so the readers miss the sources' writes.
    monkeypatch.setattr(dsv41, "DeepseekV41GroupState", dict)
    with pytest.raises(KeyError):
        with torch.no_grad():
            model(input_ids = ids)
    for h in handles:
        h.remove()


def test_group_state_survives_fsdp2_input_casting_under_gradient_checkpointing():
    """unsloth_zoo replays each checkpointed layer with a copy of the group state; the copy is a group state too."""
    from unsloth_zoo.temporary_patches.deepseek_v41 import patch_deepseek_v41_gradient_checkpointing

    patch_deepseek_v41_gradient_checkpointing()
    ids = torch.randint(0, 512, (2, 16))

    def grads(model):
        model.train()
        out = model(input_ids = ids, labels = ids)
        out.loss.backward()
        return out.loss.detach(), {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

    loss0, g0 = grads(_tiny_model())
    model = _tiny_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    _fsdp2_style_input_rebuild(model)
    loss1, g1 = grads(model)
    torch.testing.assert_close(loss0, loss1)
    assert g0.keys() == g1.keys()
    for name in g0:
        torch.testing.assert_close(g0[name], g1[name], rtol = 1e-4, atol = 1e-6, msg = name)


def test_o_a_proj_swapped_for_a_plain_linear_stays_grouped():
    """bitsandbytes' Linear4bit swap (a load that did not skip o_a_proj) puts a plain Linear there; the diagonal
    blocks of its product are the grouped result. Plain PEFT LoRA on top matches its merged weight."""
    peft = pytest.importorskip("peft")
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        want = model(input_ids = ids).logits
    for layer in model.model.layers:
        grouped = layer.self_attn.o_a_proj
        plain = torch.nn.Linear(grouped.in_features, grouped.out_features, bias = False)
        plain.weight.data.copy_(grouped.weight.data)
        layer.self_attn.o_a_proj = plain
    before = dsv41.DSV41_STATS["o_a_ungrouped"]
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids = ids).logits, want, rtol = 1e-5, atol = 1e-5)
    assert dsv41.DSV41_STATS["o_a_ungrouped"] - before == len(model.model.layers)
    lora = peft.get_peft_model(
        model, peft.LoraConfig(r = 4, target_modules = ["self_attn.o_a_proj"], init_lora_weights = False)
    ).eval()
    with torch.no_grad():
        adapted = lora(input_ids = ids).logits
        merged = lora.merge_and_unload()(input_ids = ids).logits
    torch.testing.assert_close(adapted, merged, rtol = 1e-4, atol = 1e-4)
