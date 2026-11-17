# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Unsloth's native DeepSeek-V4.1 on CPU with tiny random configs."""

import os

import pytest

torch = pytest.importorskip("torch")

from real_accelerator import has_real_cuda  # noqa: E402
from unsloth.models.deepseek_v41 import register_deepseek_v41, transformers_has_deepseek_v41  # noqa: E402
from unsloth.models.deepseek_v41 import modeling as dsv41  # noqa: E402
from unsloth.models.deepseek_v41.configuration import DeepseekV41Config, DeepseekV41TextConfig  # noqa: E402

needs_native = pytest.mark.skipif(
    transformers_has_deepseek_v41(),
    reason = "transformers ships deepseek_v41; the native classes are not registered",
)


def _zoo_dsv41():
    try:
        import unsloth_zoo.temporary_patches.deepseek_v41 as zoo_dsv41
    except Exception:
        return None
    return zoo_dsv41


def _zoo_resolves_native():
    """unsloth_zoo's DeepSeek-V4.1 patches (checkpoint replay, grouped LoRA) find Unsloth's native module."""
    zoo_dsv41 = _zoo_dsv41()
    try:
        return zoo_dsv41 is not None and zoo_dsv41._modeling() is dsv41
    except Exception:
        return False


# Capability probes for the companion unsloth_zoo (not version strings): on an unsloth_zoo without them these tests
# skip; the model falls back (Parameter-backed engram tables, the torch sparse attention, plain-PEFT grouped
# blocks) or refuses (gradient checkpointing without the replay).
needs_frozen_tables = pytest.mark.skipif(
    not hasattr(_zoo_dsv41(), "make_deepseek_v41_engram_table"),
    reason = "unsloth_zoo without the parameter-free DeepSeek-V4.1 engram table (unsloth-zoo #2510)",
)
needs_zoo_native = pytest.mark.skipif(
    not _zoo_resolves_native(),
    reason = "unsloth_zoo's DeepSeek-V4.1 patches do not resolve Unsloth's native module (unsloth-zoo #2510)",
)


def _cpu_backend_child_env(*paths):
    """Run a child process forced onto Unsloth's CPU backend regardless of os.environ."""
    steering = (
        "UNSLOTH_DISABLE_CPU_FALLBACK",
        "UNSLOTH_FORCE_GPU_PATH",
        "UNSLOTH_ALLOW_CPU",
        "UNSLOTH_ZOO_DISABLE_GPU_INIT",
        "UNSLOTH_FORCE_CPU",
    )
    env = {k: v for k, v in os.environ.items() if k not in steering}
    env.update(CUDA_VISIBLE_DEVICES = "", UNSLOTH_FORCE_CPU = "1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [*paths, env.get("PYTHONPATH")]))
    return env


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


def test_engram_hash_compiles_fullgraph_with_the_host_helpers_disabled(monkeypatch):
    """A fullgraph hash forward must not call the compiler-disabled compute_hash_multipliers."""
    model = _tiny_model()
    hasher = model.model.ngram_hash
    ids = torch.randint(0, 512, (2, 9), generator = torch.Generator().manual_seed(0))
    want = hasher(ids)
    hasher._device_cache = {}
    monkeypatch.setattr(
        dsv41, "compute_hash_multipliers", torch.compiler.disable(dsv41.compute_hash_multipliers)
    )
    torch._dynamo.reset()
    compiled = torch.compile(type(hasher).forward, fullgraph = True, backend = "eager")
    assert torch.equal(compiled(hasher, ids), want)
    torch._dynamo.reset()


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
    """The tiny model under DeepSeek's released names, written independently of convert.py's table."""
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
        # parameter-free (frozen table store) or Parameter-backed (UNSLOTH_DSV41_FROZEN_ENGRAM=0) representation
        table = model.model.engram_tables[str(layer_id)]
        store = getattr(table, "frozen_table_store", None)
        out[f"layers.{layer_id}.engram.embed.weight"] = (
            (store.weight if store is not None else table.weight).detach().clone()
        )
    out["mtp.0.norm.weight"] = torch.ones(
        text.hidden_size
    )  # DSpark / MTP head: never built, never loaded
    return out


def test_released_layout_converts_to_the_native_layout_on_any_transformers(tmp_path, monkeypatch):
    """Load the released layout through convert.py, as transformers 4.57 and non-fp4 FP8 quantizers do."""
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
    stamp = os.listdir(tmp_path / "converted")
    dsv41.DeepseekV41ForCausalLM.from_pretrained(src, dtype = torch.float32)
    assert os.listdir(tmp_path / "converted") == stamp


def test_the_declared_released_renames_reach_every_module_name():
    """Every released name the merged export renames must land on the native name."""
    import re

    mapping = dsv41.DeepseekV41ForCausalLM._unsloth_checkpoint_conversion_mapping
    assert all("\\" not in target for target in mapping.values()), mapping
    model = _tiny_model()
    native = set(model.state_dict())
    checked = 0
    for name in _released_state_dict(model):
        if name.startswith("mtp.") or ".experts." in name and ".shared_experts." not in name:
            continue
        if ".engram.embed." in name:
            continue
        renamed = name
        for pattern, target in mapping.items():
            renamed = re.sub(pattern, target, renamed)
        if renamed != "lm_head.weight":
            renamed = "model." + renamed
        assert renamed in native, (name, renamed)
        checked += 1
    assert checked > 20


def _fp4_release(tmp_path, model):
    """The tiny model's released files with real fp4 experts (int8 e2m1 pairs + e8m0 per 32)."""
    from safetensors.torch import save_file
    from unsloth.models.deepseek_v41 import convert

    gen = torch.Generator().manual_seed(7)
    released = _released_state_dict(model)
    text = model.config.text_config
    for name in list(released):
        if ".ffn.experts." not in name:
            continue
        out_dim, in_dim = released[name].shape
        packed = torch.randint(
            0, 256, (out_dim, in_dim // 2), generator = gen, dtype = torch.uint8
        ).view(torch.int8)
        exps = torch.randint(-4, 2, (out_dim, in_dim // 32), generator = gen).float()
        scale = torch.exp2(exps)
        if hasattr(torch, "float8_e8m0fnu"):
            scale = scale.to(torch.float8_e8m0fnu)
        released[name] = packed
        released[name[: -len("weight")] + "scale"] = scale
    for layer in range(text.num_hidden_layers):
        mlp = model.model.layers[layer].mlp
        deq = lambda e, w: convert._dequant_fp4(
            released[f"layers.{layer}.ffn.experts.{e}.{w}.weight"],
            released[f"layers.{layer}.ffn.experts.{e}.{w}.scale"],
        ).float()
        n = mlp.experts.gate_up_proj.shape[0]
        with torch.no_grad():
            mlp.experts.gate_up_proj.copy_(
                torch.stack([torch.cat([deq(e, "w1"), deq(e, "w3")]) for e in range(n)])
            )
            mlp.experts.down_proj.copy_(torch.stack([deq(e, "w2") for e in range(n)]))
    src = tmp_path / "released"
    model.config.save_pretrained(src)
    save_file(released, os.path.join(src, "model.safetensors"))
    return src


def test_released_fp4_experts_stay_mxfp4_packed(tmp_path, monkeypatch, capsys):
    """Released fp4 experts load as packed MXFP4, match the bf16 conversion, and stay packed on reload."""
    pytest.importorskip("unsloth_zoo.mxfp4_dequant")
    from unsloth_zoo.mxfp4_dequant import is_mxfp4_expert_param

    model = _tiny_model().to(torch.bfloat16).float().eval()
    src = _fp4_release(tmp_path, model)
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", "1")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    ids = torch.randint(0, 512, (1, 20))
    with torch.no_grad():
        want = model(input_ids = ids).logits

    def load(path):
        m = dsv41.DeepseekV41ForCausalLM.from_pretrained(path, dtype = torch.float32).eval()
        m.bind_token_map([i % 97 for i in range(512)])
        return m

    packed = load(src)
    from unsloth.models.deepseek_v41 import convert

    if convert._transformers_major() < 5:
        assert "pip install -U transformers" in capsys.readouterr().out
    experts = packed.model.layers[0].mlp.experts
    assert is_mxfp4_expert_param(experts.gate_up_proj) and is_mxfp4_expert_param(experts.down_proj)
    assert tuple(experts.gate_up_proj._original_shape) == tuple(
        model.model.layers[0].mlp.experts.gate_up_proj.shape
    )
    bf16_bytes = model.model.layers[0].mlp.experts.gate_up_proj.numel() * 2
    assert experts.gate_up_proj.numel() * experts.gate_up_proj.element_size() * 4 == bf16_bytes
    with torch.no_grad():
        got = packed(input_ids = ids).logits
    assert torch.equal(got, want)
    # a config object read from the original repo (what Unsloth's loader passes) still builds packed experts
    via_config = dsv41.DeepseekV41ForCausalLM.from_pretrained(
        src, config = type(model.config).from_pretrained(src), dtype = torch.float32
    )
    assert is_mxfp4_expert_param(via_config.model.layers[0].mlp.experts.gate_up_proj)

    monkeypatch.setenv("UNSLOTH_DSV41_FP4_EXPERTS", "bf16")
    dense = load(src)
    assert not is_mxfp4_expert_param(dense.model.layers[0].mlp.experts.gate_up_proj)
    with torch.no_grad():
        assert torch.equal(dense(input_ids = ids).logits, want)
    assert len(os.listdir(tmp_path / "converted")) == 2  # one cached copy per expert format
    monkeypatch.delenv("UNSLOTH_DSV41_FP4_EXPERTS")

    # save / reload. Under unsloth_zoo's save patch a full save writes the experts dequantized (and a config
    # without the packed flag); without it, the blocks + scales are written and reload packed.
    from safetensors import safe_open
    from transformers import PreTrainedModel

    dense_save = getattr(PreTrainedModel.save_pretrained, "_unsloth_mxfp4_patched", False)
    packed.save_pretrained(tmp_path / "saved")
    keys = set()
    for f in os.listdir(tmp_path / "saved"):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(tmp_path / "saved", f), "pt") as h:
                keys |= set(h.keys())
    blocks_key = "model.layers.0.mlp.experts.gate_up_blocks"
    assert (blocks_key not in keys) if dense_save else (blocks_key in keys)
    assert is_mxfp4_expert_param(experts.gate_up_proj)  # restored after the save
    assert packed.config.text_config.unsloth_experts_format == "mxfp4"
    reloaded = load(tmp_path / "saved")
    assert is_mxfp4_expert_param(reloaded.model.layers[0].mlp.experts.gate_up_proj) != dense_save
    with torch.no_grad():
        assert torch.equal(reloaded(input_ids = ids).logits, want)
    # transformers' own save_pretrained (no zoo save patch): packed blocks + scales on disk, reloaded packed
    plain = PreTrainedModel.save_pretrained
    while hasattr(plain, "__wrapped__"):
        plain = plain.__wrapped__
    monkeypatch.setattr(PreTrainedModel, "save_pretrained", plain)
    packed.save_pretrained(tmp_path / "saved_packed")
    keys = set()
    for f in os.listdir(tmp_path / "saved_packed"):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(tmp_path / "saved_packed", f), "pt") as h:
                keys |= set(h.keys())
    assert blocks_key in keys and "model.layers.0.mlp.experts.gate_up_proj" not in keys
    again = load(tmp_path / "saved_packed")
    assert is_mxfp4_expert_param(again.model.layers[0].mlp.experts.gate_up_proj)
    with torch.no_grad():
        assert torch.equal(again(input_ids = ids).logits, want)
    # load_state_dict into an already packed model
    fresh = load(src)
    assert is_mxfp4_expert_param(fresh.model.layers[0].mlp.experts.gate_up_proj)
    with torch.no_grad():
        for p in fresh.model.layers[0].mlp.experts.parameters():
            p.data.zero_()
            p.mxfp4_scales.zero_()
    fresh.load_state_dict(packed.state_dict())
    with torch.no_grad():
        assert torch.equal(fresh(input_ids = ids).logits, want)


_RELEASE_FP8_LEAVES = (
    ".attn.wq_a.weight",
    ".attn.wq_b.weight",
    ".attn.wkv.weight",
    ".attn.wo_a.weight",
    ".attn.wo_b.weight",
    ".indexer.wq_b.weight",
    ".shared_experts.w1.weight",
    ".shared_experts.w2.weight",
    ".shared_experts.w3.weight",
    ".engram.wkv.weight",
)


def _e8m0_pow2(amax, qmax):
    return torch.exp2(torch.ceil(torch.log2((amax / qmax).clamp_min(2.0**-126))))


def _quantized_release(tmp_path, model):
    """Write the tiny model in the release's fp8 / fp4 storage with its quantization_config."""
    import json

    from safetensors.torch import save_file

    out = {}
    for name, t in _released_state_dict(model).items():
        t = t.float()
        if name.endswith(_RELEASE_FP8_LEAVES):
            o, i = t.shape
            wb = t.view(o // 32, 32, i // 32, 32)
            sc = _e8m0_pow2(wb.abs().amax(dim = (1, 3), keepdim = True), 448.0)
            out[name] = (wb / sc).to(torch.float8_e4m3fn).view(o, i)
            out[name[: -len("weight")] + "scale"] = sc.view(o // 32, i // 32).to(
                torch.float8_e8m0fnu
            )
        elif ".engram.embed.weight" in name:
            r, d = t.shape
            wb = t.view(r, d // 32, 32)
            sc = _e8m0_pow2(wb.abs().amax(-1, keepdim = True), 448.0)
            out[name] = (wb / sc).to(torch.float8_e4m3fn).view(r, d)
            out[name[: -len("weight")] + "scale"] = sc.view(r, d // 32).to(torch.float8_e8m0fnu)
        elif ".ffn.experts." in name:
            o, i = t.shape
            wb = t.view(o, i // 32, 32)
            sc = _e8m0_pow2(wb.abs().amax(-1, keepdim = True), 6.0)
            table = torch.tensor(
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            )
            codes = ((wb / sc).view(o, i).unsqueeze(-1) - table).abs().argmin(-1).to(torch.uint8)
            out[name] = (codes[:, 0::2] | (codes[:, 1::2] << 4)).view(torch.int8)
            out[name[: -len("weight")] + "scale"] = sc.view(o, i // 32).to(torch.float8_e8m0fnu)
        else:
            out[name] = t.to(torch.bfloat16)
    src = tmp_path / "release"
    model.config.save_pretrained(src)
    with open(src / "config.json") as f:
        cfg = json.load(f)
    cfg["dtype"] = "bfloat16"
    cfg["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [32, 32],
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4",
    }
    with open(src / "config.json", "w") as f:
        json.dump(cfg, f, indent = 1)
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(src, "model.safetensors"))
    return src


@pytest.mark.parametrize("convert", ["auto", "1"])
def test_every_call_shape_loads_the_release_without_missing_or_unexpected_keys(
    tmp_path, monkeypatch, convert
):
    """Every causal LM class, with or without an explicit config, loads every released tensor."""
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch without float8_e8m0fnu")
    from transformers import AutoConfig, AutoModelForCausalLM

    src = _quantized_release(tmp_path, _tiny_model())
    if convert == "1":
        monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", "1")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    text_dict = DeepseekV41Config.from_pretrained(src).text_config.to_dict()
    Text = dsv41.DeepseekV41TextConfig
    shapes = {
        "auto": lambda: AutoModelForCausalLM.from_pretrained(src, output_loading_info = True),
        "auto+config": lambda: AutoModelForCausalLM.from_pretrained(
            src, config = AutoConfig.from_pretrained(src), output_loading_info = True
        ),
        "causal": lambda: dsv41.DeepseekV41ForCausalLM.from_pretrained(
            src, output_loading_info = True
        ),
        "causal+config": lambda: dsv41.DeepseekV41ForCausalLM.from_pretrained(
            src, config = DeepseekV41Config.from_pretrained(src), output_loading_info = True
        ),
        "causal+text_config": lambda: dsv41.DeepseekV41ForCausalLM.from_pretrained(
            src, config = Text(**text_dict), output_loading_info = True
        ),
        "text": lambda: dsv41.DeepseekV41TextForCausalLM.from_pretrained(
            src, output_loading_info = True
        ),
        "text+text_config": lambda: dsv41.DeepseekV41TextForCausalLM.from_pretrained(
            src, config = Text(**text_dict), output_loading_info = True
        ),
    }
    ids = torch.randint(0, 512, (1, 16))
    want = None
    for name, load in shapes.items():
        model, info = load()
        assert not info["missing_keys"], (name, sorted(info["missing_keys"])[:5])
        assert not info["unexpected_keys"], (name, sorted(info["unexpected_keys"])[:5])
        model = model.float().eval()
        model.bind_token_map([i % 97 for i in range(512)])
        with torch.no_grad():
            logits = model(input_ids = ids).logits
        if want is None:
            want = logits
        torch.testing.assert_close(logits, want, rtol = 0, atol = 0, msg = name)


def _parameter_backed_engram(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DSV41_FROZEN_ENGRAM", "0")


def test_parameter_backed_engram_tables_survive_state_dict_and_reload(tmp_path, monkeypatch):
    """Parameter-backed engram tables round-trip through state_dict and save / reload."""
    _parameter_backed_engram(monkeypatch)
    model = _tiny_model().eval()
    table = model.model.engram_tables["1"]
    assert not dsv41.is_parameter_free_table(table) and isinstance(
        table, dsv41.DeepseekV41EngramEmbedding
    )
    sd = model.state_dict()
    assert "model.engram_tables.1.weight" in sd and "model.engram_tables.4.weight" in sd
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        want = model(input_ids = ids).logits
    fresh = _tiny_model(seed = 5).eval()
    fresh.load_state_dict(sd)
    with torch.no_grad():
        assert torch.equal(fresh(input_ids = ids).logits, want)
    model.save_pretrained(tmp_path / "saved")
    loaded = dsv41.DeepseekV41ForCausalLM.from_pretrained(
        tmp_path / "saved", dtype = torch.float32
    ).eval()
    loaded.bind_token_map([i % 97 for i in range(512)])
    with torch.no_grad():
        assert torch.equal(loaded(input_ids = ids).logits, want)


@needs_frozen_tables
@pytest.mark.parametrize("positional", [False, True])
def test_full_save_with_a_supplied_state_dict_still_writes_the_engram_tables(tmp_path, positional):
    """A full save writes the frozen engram tables without mutating the caller's state_dict."""
    from safetensors import safe_open

    model = _tiny_model().eval()
    assert dsv41.is_parameter_free_table(model.model.engram_tables["1"])
    sd = model.state_dict()
    assert not any(".engram_tables." in k for k in sd)
    keys_before = set(sd)
    out = tmp_path / "saved"
    if positional:
        model.save_pretrained(out, True, sd)
    else:
        model.save_pretrained(out, state_dict = sd)
    assert set(sd) == keys_before
    saved = set()
    for f in os.listdir(out):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(out, f), "pt") as h:
                saved |= set(h.keys())
    # native names, or the release's under the base-model prefix (transformers 5 writes the renames back); the
    # reload below reads them back through the anchored rule
    for layer_id in (1, 4):
        names = {
            f"model.engram_tables.{layer_id}.weight",
            f"model.layers.{layer_id}.engram.embed.weight",
        }
        assert names & saved
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        want = model(input_ids = ids).logits
    loaded = dsv41.DeepseekV41ForCausalLM.from_pretrained(out, dtype = torch.float32).eval()
    loaded.bind_token_map([i % 97 for i in range(512)])
    with torch.no_grad():
        assert torch.equal(loaded(input_ids = ids).logits, want)


@pytest.mark.parametrize("convert", ["auto", "1"])
def test_parameter_backed_fp8_engram_keeps_its_block_scale_on_load(tmp_path, monkeypatch, convert):
    """Released fp8 engram rows load into Parameter-backed tables identically to the frozen store."""
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch without float8_e8m0fnu")
    src = _quantized_release(tmp_path, _tiny_model())
    if convert == "1":
        monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", "1")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    ids = torch.randint(0, 512, (1, 16))

    def rows(model):
        model = model.float().eval()
        model.bind_token_map([i % 97 for i in range(512)])
        with torch.no_grad():
            return model.model.engram_rows(ids), model(input_ids = ids).logits

    want_rows, want_logits = rows(dsv41.DeepseekV41ForCausalLM.from_pretrained(src))
    _parameter_backed_engram(monkeypatch)
    model, info = dsv41.DeepseekV41ForCausalLM.from_pretrained(src, output_loading_info = True)
    assert not dsv41.is_parameter_free_table(model.model.engram_tables["1"])
    assert not info["missing_keys"] and not info["unexpected_keys"]
    got_rows, got_logits = rows(model)
    for layer_id in want_rows:
        torch.testing.assert_close(
            got_rows[layer_id].float(), want_rows[layer_id].float(), rtol = 0, atol = 0
        )
    torch.testing.assert_close(got_logits, want_logits, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("convert", ["auto", "1"])
def test_a_released_checkpoint_in_a_subfolder_loads_from_that_subfolder(
    tmp_path, monkeypatch, convert
):
    """from_pretrained(root, subfolder=...) discovers, converts and loads the subfolder."""
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch without float8_e8m0fnu")
    import shutil

    src = _quantized_release(tmp_path, _tiny_model())
    root = tmp_path / "repo"
    shutil.copytree(src, root / "sub")
    if convert == "1":
        monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", "1")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    ids = torch.randint(0, 512, (1, 16))
    outs = []
    for path, kw in ((root / "sub", {}), (root, {"subfolder": "sub"})):
        model, info = dsv41.DeepseekV41ForCausalLM.from_pretrained(
            path, output_loading_info = True, **kw
        )
        assert not info["missing_keys"] and not info["unexpected_keys"], kw
        model = model.float().eval()
        model.bind_token_map([i % 97 for i in range(512)])
        with torch.no_grad():
            outs.append(model(input_ids = ids).logits)
    assert torch.equal(outs[0], outs[1])


def test_tokenizer_autobinding_reads_the_checkpoint_subfolder(tmp_path, monkeypatch):
    model = _tiny_model()
    model.save_pretrained(tmp_path / "repo" / "sub")
    seen = []
    monkeypatch.setattr(
        dsv41.DeepseekV41ForCausalLM,
        "_autobind_tokenizer",
        lambda self, path = None, **hub: seen.append((str(path), hub)) or False,
    )
    dsv41.DeepseekV41ForCausalLM.from_pretrained(tmp_path / "repo", subfolder = "sub")
    assert seen and seen[0][1].get("subfolder") == "sub"


@pytest.mark.parametrize("as_path", [False, True])
def test_a_config_path_picks_up_the_converted_packed_expert_format(tmp_path, monkeypatch, as_path):
    pytest.importorskip("unsloth_zoo.mxfp4_dequant")
    from pathlib import Path

    from unsloth_zoo.mxfp4_dequant import is_mxfp4_expert_param

    src = _fp4_release(tmp_path, _tiny_model().to(torch.bfloat16).float().eval())
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", "1")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    config = Path(src) if as_path else str(src)
    model, info = dsv41.DeepseekV41ForCausalLM.from_pretrained(
        src, config = config, output_loading_info = True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert is_mxfp4_expert_param(model.model.layers[0].mlp.experts.gate_up_proj)


@pytest.mark.parametrize("convert", ["auto", "1"])
def test_loading_never_mutates_the_callers_config(tmp_path, monkeypatch, convert):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch without float8_e8m0fnu")
    src = _quantized_release(tmp_path, _tiny_model())
    if convert == "1":
        monkeypatch.setenv("UNSLOTH_DSV41_CONVERT", "1")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    for config in (
        DeepseekV41Config.from_pretrained(src),
        dsv41.DeepseekV41TextConfig(**DeepseekV41Config.from_pretrained(src).text_config.to_dict()),
    ):
        before = config.to_json_string()
        text_before = getattr(config, "text_config", None)
        text_before = text_before.to_json_string() if text_before is not None else None
        dsv41.DeepseekV41ForCausalLM.from_pretrained(src, config = config)
        assert config.to_json_string() == before
        if text_before is not None:
            assert config.text_config.to_json_string() == text_before


# One fresh CPU-backend child answers both fresh-process probes below: each child pays a full `import unsloth`
# (the slow part), and neither probe changes what the other sees. The CPU-backend registration probe runs first
# (an AutoConfig lookup touches no conversion mapping); the direct native load then drops every cached
# deepseek_v41 conversion mapping before it loads, as it did in its own process.
_FRESH_CPU_CHILD = """
import json, sys, traceback
import unsloth
from transformers import AutoConfig
try:  # a failed lookup must not take the load probe down with it
    c = AutoConfig.from_pretrained(sys.argv[1])
    print("PROBE_CPU " + json.dumps({"cpu": bool(getattr(unsloth, "_IS_CPU_BACKEND", False)), "type": type(c).__name__}))
except Exception:
    traceback.print_exc()
if len(sys.argv) > 2:
    import transformers.conversion_mapping as cm
    from unsloth.models.deepseek_v41.modeling import DeepseekV41ForCausalLM
    cache = cm._checkpoint_conversion_mapping_cache
    if cache is not None:
        for key in [k for k in list(cache) if "deepseek_v41" in k.lower()]:
            cache.pop(key)
    model, info = DeepseekV41ForCausalLM.from_pretrained(sys.argv[2], output_loading_info = True)
    print("PROBE_LOAD " + json.dumps({"missing": len(info["missing_keys"]), "unexpected": len(info["unexpected_keys"])}))
"""


def _direct_release_load_skip_reason():
    if not hasattr(torch, "float8_e8m0fnu"):
        return "torch without float8_e8m0fnu"
    from unsloth.models.deepseek_v41.convert import released_needs_conversion
    from unsloth.models.deepseek_v41.modeling import has_core_model_loading

    if not has_core_model_loading() or released_needs_conversion():
        return "transformers without the direct release load"
    return None


@pytest.fixture(scope = "module")
def fresh_cpu_child(tmp_path_factory):
    import subprocess
    import sys

    config_dir = tmp_path_factory.mktemp("dsv41_config")
    _tiny_config().save_pretrained(config_dir)
    args = [str(config_dir)]
    if _direct_release_load_skip_reason() is None:
        args.append(
            str(_quantized_release(tmp_path_factory.mktemp("dsv41_release"), _tiny_model()))
        )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = _cpu_backend_child_env(root)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env["PYTHONPATH"], *sys.path]))
    out = subprocess.run(
        [sys.executable, "-c", _FRESH_CPU_CHILD, *args],
        capture_output = True,
        text = True,
        env = env,
        timeout = 900,
    )
    return out


def _probe(out, tag):
    import json
    line = next((l for l in out.stdout.splitlines() if l.startswith(tag + " ")), None)
    return None if line is None else json.loads(line[len(tag) + 1 :])


def test_a_direct_native_class_load_registers_the_release_renames_itself(fresh_cpu_child):
    """A fresh transformers 5 process maps every released tensor without prior Auto registration."""
    reason = _direct_release_load_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    probe = _probe(fresh_cpu_child, "PROBE_LOAD")
    assert probe is not None, fresh_cpu_child.stdout[-2000:] + fresh_cpu_child.stderr[-3000:]
    assert probe == {"missing": 0, "unexpected": 0}


def test_each_release_rename_applies_once_per_model_root():
    """The nested text model does not add a second scoped copy of the conversion rules."""
    from unsloth.models.deepseek_v41.convert import released_needs_conversion
    from unsloth.models.deepseek_v41.modeling import has_core_model_loading

    if not has_core_model_loading() or released_needs_conversion():
        pytest.skip("transformers without the direct release load (convert.py loads it)")
    from transformers.conversion_mapping import get_model_conversion_mapping

    from unsloth.models import deepseek_v41 as pkg

    pkg._register_conversions()
    n_rules = len(pkg._conversion_rules())
    model = _tiny_model()
    text = dsv41.DeepseekV41TextForCausalLM(model.config.text_config)
    assert len(get_model_conversion_mapping(model, add_legacy = False)) == n_rules
    assert len(get_model_conversion_mapping(text, add_legacy = False)) == n_rules
    assert (
        len(get_model_conversion_mapping(model.model, add_legacy = False)) == n_rules - 1
    )  # no head


def test_a_config_that_does_not_fit_the_release_raises_instead_of_random_init(
    tmp_path, monkeypatch
):
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("torch without float8_e8m0fnu")
    src = _quantized_release(tmp_path, _tiny_model())
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    text = DeepseekV41Config.from_pretrained(src).text_config.to_dict()
    text["num_hidden_layers"] = 7  # one layer the release does not hold
    with pytest.raises(RuntimeError, match = "were not found in the DeepSeek-V4.1 checkpoint"):
        dsv41.DeepseekV41TextForCausalLM.from_pretrained(
            src, config = dsv41.DeepseekV41TextConfig(**text)
        )


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
    """One sequence right padded to `width`, the row length the relayout gives it."""
    pad = width - len(ids)
    row = torch.cat([ids, torch.zeros(pad, dtype = torch.long)]).unsqueeze(0)
    mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad)]).long().unsqueeze(0)
    with torch.no_grad():
        return model(input_ids = row, attention_mask = mask).logits[0, : len(ids)]


def test_packed_sequences_match_each_sequence_run_alone():
    """Padding-free packed sequences give the same logits as each sequence run alone."""
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


def test_packed_documents_at_every_boundary_match_unpacked_runs_and_never_leak():
    """Padding-free packing at the window / compressor / indexer boundaries matches per-document runs.
    16 indexer heads: with 2, relu often zeroes a whole score and top-k ties break by tensor width."""
    model = _tiny_model(index_n_heads = 16)
    text = model.config.text_config
    lengths = [1, text.sliding_window, text.sliding_window + 1, 2, 17, 5]
    assert 17 > text.index_topk * 2  # past the indexer budget at ratio 1 and 2
    gen = torch.Generator().manual_seed(3)
    docs = [torch.randint(0, 512, (n,), generator = gen) for n in lengths]
    starts = [sum(lengths[:k]) for k in range(len(lengths))]
    ids = torch.cat(docs).unsqueeze(0)
    pos = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
    labels = ids.clone()
    labels[0, starts] = -100  # no document predicts the next one's first token
    n_items = int((labels[0, 1:] != -100).sum())

    model.eval()
    with torch.no_grad():
        packed = model(input_ids = ids, position_ids = pos).logits[0]
        for doc, lo in zip(docs, starts):
            alone = model(input_ids = doc.unsqueeze(0)).logits[0]
            torch.testing.assert_close(packed[lo : lo + len(doc)], alone, rtol = 1e-5, atol = 1e-5)
        bumped = ids.clone()
        bumped[0, starts[2] : starts[2] + lengths[2]] = (
            bumped[0, starts[2] : starts[2] + lengths[2]] + 7
        ) % 512
        moved = model(input_ids = bumped, position_ids = pos).logits[0]
    for k, (lo, n) in enumerate(zip(starts, lengths)):
        diff = float((moved[lo : lo + n] - packed[lo : lo + n]).abs().max())
        assert (diff > 1e-3) if k == 2 else (diff <= 1e-6), (k, diff)

    model.train()
    model.zero_grad(set_to_none = True)
    loss = model(input_ids = ids, position_ids = pos, labels = labels, num_items_in_batch = n_items).loss
    loss.backward()
    want = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    total = 0.0
    for doc in docs:
        doc_labels = doc.clone().unsqueeze(0)
        doc_labels[0, 0] = -100
        part = model(input_ids = doc.unsqueeze(0), labels = doc_labels, num_items_in_batch = n_items).loss
        part.backward()
        total = total + part.detach()
    torch.testing.assert_close(total, loss.detach(), rtol = 1e-5, atol = 1e-6)
    got = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert got.keys() == want.keys()
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol = 1e-4, atol = 1e-6, msg = name)


@pytest.mark.parametrize("layout", ["packed", "left_padded"])
def test_packed_rows_and_left_padding_stay_separated_under_torch_compile(layout):
    """torch.compile keeps the packed-row relayout."""
    torch._dynamo.reset()
    model = _tiny_model().eval()
    torch.manual_seed(3)
    if layout == "packed":
        a, b = torch.randint(0, 512, (9,)), torch.randint(0, 512, (11,))
        kw = dict(
            input_ids = torch.cat([a, b]).unsqueeze(0),
            position_ids = torch.cat([torch.arange(9), torch.arange(11)]).unsqueeze(0),
        )
    else:
        a, b = torch.randint(0, 512, (11,)), torch.randint(0, 512, (16,))
        kw = dict(
            input_ids = torch.stack([torch.cat([torch.zeros(5, dtype = torch.long), a]), b]),
            attention_mask = torch.stack(
                [torch.cat([torch.zeros(5), torch.ones(11)]), torch.ones(16)]
            ).long(),
        )
    with torch.no_grad():
        want = model(**kw).logits
    compiled = torch.compile(model, backend = "eager")
    with torch.no_grad():
        got = compiled(**kw).logits
    torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-5)
    model.train()
    model.zero_grad(set_to_none = True)
    model(**kw, labels = kw["input_ids"]).loss.backward()
    want_g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    compiled(**kw, labels = kw["input_ids"]).loss.backward()
    for n, g in want_g.items():
        torch.testing.assert_close(
            dict(model.named_parameters())[n].grad, g, rtol = 1e-4, atol = 1e-6, msg = n
        )
    torch._dynamo.reset()


def test_embeddings_without_token_ids_are_refused_where_engram_layers_need_them():
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 12))
    embeds = model.model.embed_tokens(ids).detach()
    with pytest.raises(ValueError, match = "engram layers hash the token ids"):
        with torch.no_grad():
            model(inputs_embeds = embeds)
    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs_embeds = embeds, input_ids = ids).logits, model(input_ids = ids).logits
        )
    text = model.model
    engram = set(text.config.engram_layer_ids)
    owner = min(engram)
    with torch.no_grad():
        h, pre, state = text.pipeline_stage_forward(0, owner, input_ids = ids)
        with pytest.raises(ValueError, match = "engram layers hash the token ids"):
            text.pipeline_stage_forward(owner, owner + 1, h, pre, state)
        # a stage that owns no engram layer does not need the ids
        free = next(i for i in range(len(text.layers)) if i not in engram and i > 0)
        text.pipeline_stage_forward(free, free + 1, h, pre, dsv41.DeepseekV41GroupState())


def test_grouped_lora_is_not_marked_ready_when_nothing_was_installed(monkeypatch):
    peft = pytest.importorskip("peft")
    zoo = pytest.importorskip("unsloth_zoo.temporary_patches.deepseek_v41")
    from peft.tuners.lora.model import LoraModel

    create = LoraModel.__dict__["_create_new_module"]
    func = getattr(create, "__func__", create)
    while getattr(func, "_unsloth_dsv41_grouped", False):
        func = func.__wrapped__
    monkeypatch.setattr(LoraModel, "_create_new_module", staticmethod(func))
    monkeypatch.setattr(zoo, "patch_deepseek_v41_grouped_lora", lambda: None)
    monkeypatch.setattr(dsv41, "_GROUPED_LORA_READY", [False])
    assert dsv41.ensure_grouped_lora() is False
    assert dsv41._GROUPED_LORA_READY[0] is False


def test_gradient_checkpointing_without_the_zoo_replay_fails_closed(monkeypatch):
    """An unsloth_zoo that cannot install the decoder-layer replay must not give silently wrong gradients."""
    monkeypatch.setattr(dsv41, "ensure_checkpoint_replay", lambda: False)
    model = _tiny_model()
    with pytest.raises(RuntimeError, match = "replay"):
        model.gradient_checkpointing_enable()


def test_layer_config_ints_stay_constant_under_dynamic_compile(monkeypatch):
    """Module ints must stay constant under dynamic=True, else Inductor spends an hour in sympy."""
    from torch._dynamo.testing import EagerAndRecordGraphs

    model = _tiny_model()
    assert {
        l.self_attn.compressor.compress_ratio for l in model.model.layers if l.self_attn.compressor
    } == {1, 2}
    backend = EagerAndRecordGraphs()
    compiled = torch.compile(
        dsv41.DeepseekV41Compressor.forward, backend = backend, dynamic = True, fullgraph = True
    )
    monkeypatch.setattr(dsv41.DeepseekV41Compressor, "forward", lambda self, x: compiled(self, x))
    torch._dynamo.reset()
    before = torch._dynamo.config.allow_unspec_int_on_nn_module
    ids = torch.randint(0, 512, (2, 12), generator = torch.Generator().manual_seed(0))
    try:
        with torch._dynamo.config.patch(allow_unspec_int_on_nn_module = True), torch.no_grad():
            model(input_ids = ids)
    finally:
        torch._dynamo.reset()
    lifted = [
        node.name
        for graph in backend.graphs
        for node in graph.graph.nodes
        if node.op == "placeholder" and isinstance(node.meta.get("example_value"), torch.SymInt)
    ]
    assert backend.graphs, "the compressor was never compiled"
    attribute_ints = [n for n in lifted if "compress_ratio" in n or "head_dim" in n]
    assert not attribute_ints, (attribute_ints, lifted)
    assert (
        torch._dynamo.config.allow_unspec_int_on_nn_module is before
    )  # the forward's patch did not leak


def test_gradient_checkpointing_enable_takes_the_trainer_keywords(monkeypatch):
    """transformers 5.22+ passes every_n_layers= / offload= to gradient_checkpointing_enable."""
    import inspect

    from transformers import PreTrainedModel

    monkeypatch.setattr(dsv41, "ensure_checkpoint_replay", lambda: True)
    model = _tiny_model()
    layers = list(model.model.layers)
    assert len(layers) >= 2
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = None, every_n_layers = 1, offload = False
    )
    assert all(layer.gradient_checkpointing for layer in layers)
    model.gradient_checkpointing_disable()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs = {"use_reentrant": False}, every_n_layers = 2
    )
    flags = [bool(layer.gradient_checkpointing) for layer in layers]
    # 5.22 names every_n_layers, 5.23 takes it through **kwargs, 4.x has neither and the keyword is dropped.
    params = inspect.signature(PreTrainedModel.gradient_checkpointing_enable).parameters
    if "every_n_layers" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
        assert flags == [i % 2 == 0 for i in range(len(layers))], flags
    else:
        assert all(flags), flags


@needs_zoo_native
def test_standard_gradient_checkpointing_gives_exact_gradients_in_a_fresh_process():
    """Building the model and gradient_checkpointing_enable() install the zoo replay."""
    import json
    import subprocess
    import sys

    tests_dir = os.path.dirname(os.path.abspath(__file__))
    child = (
        "import json, sys, torch\n"
        f"sys.path.insert(0, {tests_dir!r})\n"
        "import test_deepseek_v41_native as t\n"
        "model = t._tiny_model().train()\n"
        "ids = torch.randint(0, 512, (2, 24), generator = torch.Generator().manual_seed(0))\n"
        "def grads():\n"
        "    model.zero_grad(set_to_none = True)\n"
        "    model(input_ids = ids, labels = ids).loss.backward()\n"
        "    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}\n"
        "plain = grads()\n"
        "model.gradient_checkpointing_enable()\n"
        "ckpt = grads()\n"
        "diff = max(float((plain[n] - ckpt[n]).abs().max()) for n in plain)\n"
        "print('PROBE', json.dumps({'diff': diff, 'same_keys': plain.keys() == ckpt.keys(), "
        "'checkpointing': bool(model.model.layers[0].gradient_checkpointing)}))\n"
    )
    env = _cpu_backend_child_env()
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    run = subprocess.run(
        [sys.executable, "-c", child], env = env, capture_output = True, text = True, timeout = 600
    )
    probe = [line for line in run.stdout.splitlines() if line.startswith("PROBE ")]
    assert probe, run.stdout[-2000:] + run.stderr[-3000:]
    out = json.loads(probe[0][6:])
    assert out["checkpointing"] and out["same_keys"]
    assert out["diff"] <= 1e-6, out


def test_labels_with_logits_to_keep_score_the_kept_positions_against_their_own_targets():
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (2, 10))
    labels = ids.clone()
    labels[0, 3] = -100
    with torch.no_grad():
        full = model(input_ids = ids).logits
    shifted = torch.nn.functional.pad(labels, (0, 1), value = -100)[..., 1:]

    def reference(sel):
        logits = full[:, sel].float()
        targets = shifted[:, sel]
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index = -100
        )

    with torch.no_grad():
        out = model(input_ids = ids, labels = labels, logits_to_keep = 4)
        assert out.logits.shape[1] == 4
        torch.testing.assert_close(out.loss, reference(slice(-4, None)), rtol = 1e-5, atol = 1e-6)
        positions = torch.tensor([1, 3, 6])
        out = model(input_ids = ids, labels = labels, logits_to_keep = positions)
        assert out.logits.shape[1] == 3
        torch.testing.assert_close(out.logits, full[:, positions], rtol = 1e-5, atol = 1e-6)
        torch.testing.assert_close(out.loss, reference(positions), rtol = 1e-5, atol = 1e-6)
        # without labels a tensor selector keeps only those positions
        assert model(input_ids = ids, logits_to_keep = torch.tensor([1, 3])).logits.shape[1] == 2
        # logits_to_keep = 0 (all) with labels: the usual full loss
        torch.testing.assert_close(
            model(input_ids = ids, labels = labels).loss, reference(slice(None)), rtol = 1e-5, atol = 1e-6
        )


def _zoo_ast_rewriter():
    """unsloth_zoo's fused-CE AST rewriter, loaded from its file (importing unsloth_zoo initialises the GPU)."""
    import importlib.util
    import sys

    spec = importlib.util.find_spec("unsloth_zoo")
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    path = os.path.join(locations[0], "fused_losses", "ast_rewriter.py") if locations else ""
    if not os.path.isfile(path):
        pytest.skip("unsloth_zoo's fused_losses/ast_rewriter.py is not installed")
    file_spec = importlib.util.spec_from_file_location("_dsv41_test_ast_rewriter", path)
    module = importlib.util.module_from_spec(file_spec)
    sys.modules[file_spec.name] = module
    file_spec.loader.exec_module(module)
    return module


def test_kept_position_loss_goes_through_the_fused_ce_rewrite(monkeypatch):
    """The fused-CE rewrite also handles labels with logits_to_keep."""
    import inspect
    import textwrap

    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    source = textwrap.dedent(inspect.getsource(dsv41.DeepseekV41ForCausalLM.forward))
    rewritten, _ = _zoo_ast_rewriter().rewrite_forward_source(source)
    assert rewritten is not None
    calls = []

    def fused(
        hidden,
        head,
        labels,
        vocab_size = None,
        accelerator_scaler = None,
        **kwargs,
    ):
        calls.append(hidden.shape[1])
        target = kwargs.get("shift_labels")
        if target is None:
            target = torch.nn.functional.pad(labels, (0, 1), value = -100)[..., 1:]
        logits = head(hidden).float()
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index = -100
        )

    namespace = dict(vars(dsv41))
    namespace.update(
        os = os,
        unsloth_fused_lm_head_loss = fused,
        EMPTY_LOGITS = torch.empty(0),
        _can_fuse_loss = lambda f: True,
    )
    exec(rewritten, namespace)
    forward = namespace["forward"]
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (2, 10), generator = torch.Generator().manual_seed(0))
    labels = ids.clone()
    labels[0, 3] = -100
    with torch.no_grad():
        for keep in (0, 4, torch.tensor([1, 3, 6])):
            want = model(input_ids = ids, labels = labels, logits_to_keep = keep).loss
            got = forward(model, input_ids = ids, labels = labels, logits_to_keep = keep).loss
            torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)
    assert calls == [10, 4, 3]


def test_greedy_generate_equals_full_prefix_recompute_across_the_boundaries():
    """No KV cache: greedy generate equals argmax of a fresh full-prefix forward."""
    from peft import LoraConfig, get_peft_model

    model = _tiny_model(index_n_heads = 16)
    text = model.config.text_config
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    gen = torch.Generator().manual_seed(5)
    prompts = [torch.randint(3, 512, (n,), generator = gen) for n in (3, text.sliding_window - 1)]
    width = max(len(p) for p in prompts)
    ids = torch.stack(
        [torch.cat([torch.zeros(width - len(p), dtype = torch.long), p]) for p in prompts]
    )
    mask = torch.stack(
        [torch.cat([torch.zeros(width - len(p)), torch.ones(len(p))]).long() for p in prompts]
    )
    # Both rows end past the window (8) and past index_topk (3) compressed blocks at ratio 2.
    new_tokens = text.sliding_window + 1

    def recompute(m, prompt):
        seq = prompt.unsqueeze(0)
        for _ in range(new_tokens):
            with torch.no_grad():
                nxt = m(input_ids = seq).logits[:, -1].argmax(-1, keepdim = True)
            seq = torch.cat([seq, nxt], dim = 1)
        return seq[0, len(prompt) :]

    def check(m, recompute_too = True):
        with torch.no_grad():
            out = m.generate(
                input_ids = ids, attention_mask = mask, max_new_tokens = new_tokens, do_sample = False
            )
        for k, prompt in enumerate(prompts if recompute_too else ()):
            assert torch.equal(out[k, width:], recompute(m, prompt)), k
        return out[:, width:]

    model.eval()
    plain = check(model)
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = ["q_a_proj", "kv_proj", "o_b_proj"],
            init_lora_weights = False,
        ),
    ).eval()
    with_lora = check(peft_model)
    with peft_model.disable_adapter():
        assert torch.equal(check(peft_model, recompute_too = False), plain)
    assert not torch.equal(with_lora, plain)  # the adapter changes the continuation


def test_opt_in_fast_lora_matches_peft_on_the_plain_projections(monkeypatch):
    """UNSLOTH_DSV41_FAST_LORA=1 matches PEFT's loss and gradients; a dropout adapter keeps PEFT's forward."""
    fast_lora = pytest.importorskip("unsloth.kernels.fast_lora")
    from peft import LoraConfig, get_peft_model

    targets = [
        "q_a_proj",
        "q_b_proj",
        "self_attn.kv_proj",
        "o_b_proj",
        "shared_experts.gate_proj",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
    ]

    def run(fast, dropout = 0.0):
        monkeypatch.setattr(dsv41, "_FAST_LORA", fast)
        monkeypatch.setattr(dsv41, "_LoRA_W", fast_lora.LoRA_W, raising = False)
        monkeypatch.setattr(dsv41, "_fast_lora_apply", fast_lora._apply, raising = False)
        monkeypatch.setattr(
            dsv41, "_fast_lora_extras", fast_lora._lora_dropout_and_bias, raising = False
        )
        monkeypatch.setattr(
            dsv41, "_fast_lora_parameters", fast_lora.get_lora_parameters, raising = False
        )
        for name in ("_has_multiple_active_adapters", "_has_activation_fake_quantizer"):
            monkeypatch.setattr(dsv41, name, getattr(fast_lora, name), raising = False)
        model = get_peft_model(
            _tiny_model(),
            LoraConfig(
                r = 4,
                lora_alpha = 8,
                lora_dropout = dropout,
                target_modules = targets,
                init_lora_weights = False,
            ),
        )
        model.eval()  # dropout off, so both runs see the same adapter math
        dsv41.DSV41_STATS["fast_lora"] = 0
        ids = torch.randint(0, 512, (2, 12), generator = torch.Generator().manual_seed(0))
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        return loss.detach(), grads, dsv41.DSV41_STATS["fast_lora"]

    loss0, g0, n0 = run(False)
    loss1, g1, n1 = run(True)
    layers = len(_tiny_model().model.layers)
    assert (n0, n1) == (0, 7 * layers)
    torch.testing.assert_close(loss1, loss0, rtol = 1e-6, atol = 1e-6)
    assert g1.keys() == g0.keys() and any(".lora_A." in n for n in g0)
    for name in g0:
        torch.testing.assert_close(g1[name], g0[name], rtol = 1e-5, atol = 1e-6, msg = name)
    assert run(True, dropout = 0.1)[2] == 0  # a dropout module: PEFT's forward

    # Two active adapters: PEFT's forward sums both instead of LoRA_W raising.
    run(True)
    model = get_peft_model(
        _tiny_model(),
        LoraConfig(r = 4, lora_alpha = 8, target_modules = targets, init_lora_weights = False),
    )
    model.add_adapter(
        "second", LoraConfig(r = 4, lora_alpha = 8, target_modules = targets, init_lora_weights = False)
    )
    model.base_model.set_adapter(["default", "second"])
    dsv41.DSV41_STATS["fast_lora"] = 0
    ids = torch.randint(0, 512, (2, 12), generator = torch.Generator().manual_seed(0))
    assert torch.isfinite(model(input_ids = ids, labels = ids).loss)
    assert dsv41.DSV41_STATS["fast_lora"] == 0


def test_generation_projects_only_the_last_position():
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 9))
    inputs = model.prepare_inputs_for_generation(ids, attention_mask = torch.ones_like(ids))
    assert inputs["logits_to_keep"] == 1
    with torch.no_grad():
        out = model(**inputs)
        assert out.logits.shape[1] == 1
        torch.testing.assert_close(
            out.logits[:, -1], model(input_ids = ids).logits[:, -1], rtol = 1e-5, atol = 1e-6
        )
        gen = model.generate(ids, max_new_tokens = 2, do_sample = False)
    assert gen.shape[1] == 11


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
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN_WINDOW", None)
    monkeypatch.setenv("UNSLOTH_DSV41_SPARSE_ATTN", "0")
    model = _tiny_model()
    before = dsv41.DSV41_STATS["sparse_attn_torch"]
    model(input_ids = torch.randint(0, 512, (1, 12)))
    assert dsv41.DSV41_STATS["sparse_attn_torch"] == before + 6  # one call per layer


@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA")
def test_window_only_layers_take_the_hd512_kernel_and_compressed_layers_the_window_kernel(
    monkeypatch,
):
    """head_dim 512 bf16 CUDA kernels match the concatenated path."""
    zoo = pytest.importorskip("unsloth_zoo.sparse_attention")
    if "hd512" not in getattr(zoo, "SPARSE_ATTN_STATS", {}):
        pytest.skip("this unsloth_zoo has no hd512 window dispatch")
    from unsloth_zoo.sparse_mqa_kernel import window_hd512_supported

    probe = torch.empty(1, 1, 1, 512, device = "cuda", dtype = torch.bfloat16)
    if not window_hd512_supported(probe, probe[:, :, 0]):
        pytest.skip("hd512 window kernel unsupported on this GPU")
    monkeypatch.delenv("UNSLOTH_SPARSE_ATTN_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_DSV41_SPARSE_ATTN", raising = False)
    monkeypatch.delenv("UNSLOTH_DSV41_SPARSE_ATTN_WINDOW", raising = False)
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN", None)
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN_WINDOW", None)
    model = _tiny_model(head_dim = 512).to("cuda", torch.bfloat16).eval()
    assert dsv41._SPARSE_ATTN_WINDOW is zoo.sparse_attn_window
    calls = []
    real = dsv41.sparse_attn_window

    def spy(q, window_kv, compress_kv, compress_idxs, sink, scale, window):
        before = dict(zoo.SPARSE_ATTN_STATS)
        out = real(q, window_kv, compress_kv, compress_idxs, sink, scale, window)
        used = {k for k, v in zoo.SPARSE_ATTN_STATS.items() if v != before.get(k, 0)}
        calls.append((compress_idxs.shape[-1], used))
        return out

    monkeypatch.setattr(dsv41, "sparse_attn_window", spy)
    ids = torch.randint(0, 512, (1, 24), device = "cuda")
    with torch.no_grad():
        got = model(input_ids = ids).logits.float()
    ratios = model.config.text_config.compress_ratios[: model.config.text_config.num_hidden_layers]
    assert len(calls) == len(ratios)
    for (k, used), ratio in zip(calls, ratios):
        if ratio == 0:
            assert k == 0 and used == {"hd512"}, (ratio, k, used)
        else:
            assert k > 0 and used and used <= {"gather", "triton"}, (ratio, k, used)
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN_WINDOW", None)
    with torch.no_grad():
        want = model(input_ids = ids).logits.float()
    assert ((got - want).norm() / want.norm()).item() < 2e-2


def test_an_unsloth_zoo_without_the_window_entry_keeps_the_concatenated_path(monkeypatch):
    import sys
    import types

    old = types.ModuleType("unsloth_zoo.sparse_attention")
    old.sparse_attn = lambda q, kv, sink, idxs, scale: dsv41.sparse_attn_reference(
        q, kv, sink, idxs, scale
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.sparse_attention", old)
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN", None)
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN_WINDOW", None)
    monkeypatch.delenv("UNSLOTH_DSV41_SPARSE_ATTN", raising = False)
    model = _tiny_model()
    assert dsv41._SPARSE_ATTN is old.sparse_attn and dsv41._SPARSE_ATTN_WINDOW is None
    before = dict(dsv41.DSV41_STATS)
    model(input_ids = torch.randint(0, 512, (1, 12)))
    assert dsv41.DSV41_STATS["sparse_attn_zoo"] == before["sparse_attn_zoo"] + 6
    assert dsv41.DSV41_STATS["sparse_attn_window"] == before["sparse_attn_window"]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks = pytest.mark.skipif(not has_real_cuda(), reason = "CUDA")),
    ],
)
def test_window_attention_entry_matches_the_concatenated_index_path(monkeypatch, device):
    """sparse_attn_window matches the concatenated sparse_attn path; the kill switch keeps the concatenation."""
    zoo = pytest.importorskip("unsloth_zoo.sparse_attention")
    if not hasattr(zoo, "sparse_attn_window"):
        pytest.skip("this unsloth_zoo has no sparse_attn_window")
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN", None)
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN_WINDOW", None)
    monkeypatch.delenv("UNSLOTH_DSV41_SPARSE_ATTN", raising = False)
    dsv41.ensure_sparse_attn()
    assert dsv41._SPARSE_ATTN_WINDOW is zoo.sparse_attn_window
    window_entry = dsv41._SPARSE_ATTN_WINDOW
    model = _tiny_model().to(device)
    ids = torch.randint(
        0, 512, (2, 24), device = device
    )  # longer than the window (8), several compressed groups

    def run():
        model.zero_grad(set_to_none = True)
        out = model(input_ids = ids, labels = ids)
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return out.logits.detach(), grads

    before = dsv41.DSV41_STATS["sparse_attn_window"]
    logits, grads = run()
    assert (
        dsv41.DSV41_STATS["sparse_attn_window"] == before + 6
    )  # every layer, window-only ones included
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN_WINDOW", None)
    concat_before = dsv41.DSV41_STATS["sparse_attn_zoo"]
    want_logits, want_grads = run()
    assert dsv41.DSV41_STATS["sparse_attn_zoo"] == concat_before + 6
    assert dsv41.DSV41_STATS["sparse_attn_window"] == before + 6
    torch.testing.assert_close(logits, want_logits, rtol = 1e-5, atol = 1e-5)
    assert grads.keys() == want_grads.keys() and len(grads) > 0
    for name in grads:
        torch.testing.assert_close(grads[name], want_grads[name], rtol = 1e-4, atol = 1e-5, msg = name)
    # window-only switch: the concatenation through zoo's sparse_attn
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN", None)
    monkeypatch.setenv("UNSLOTH_DSV41_SPARSE_ATTN_WINDOW", "0")
    dsv41.ensure_sparse_attn()
    assert dsv41._SPARSE_ATTN_WINDOW is None and dsv41._SPARSE_ATTN is zoo.sparse_attn
    # kill switch: neither zoo entry
    monkeypatch.setattr(dsv41, "_SPARSE_ATTN", None)
    monkeypatch.setenv("UNSLOTH_DSV41_SPARSE_ATTN", "0")
    dsv41.ensure_sparse_attn()
    assert dsv41._SPARSE_ATTN_WINDOW is None and dsv41._SPARSE_ATTN is dsv41.sparse_attn_reference
    assert window_entry is zoo.sparse_attn_window


@pytest.mark.parametrize("hooks", ["layers_skip_shared", "layers_no_skip", "every_submodule"])
def test_shared_group_state_survives_accelerate_device_hooks(hooks):
    """Accelerate hooks that rebuild Mapping arguments under device_map / offload keep the shared state intact."""
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        want = model(input_ids = ids).logits
    skip = list(dsv41.DeepseekV41PreTrainedModel._skip_keys_device_placement)
    if hooks == "layers_no_skip":
        skip = [k for k in skip if k != "shared"]
    targets = list(model.model.layers)
    if hooks == "every_submodule":
        targets = [m for layer in model.model.layers for m in layer.modules()]
    for module in targets:
        accelerate_hooks.add_hook_to_module(
            module, accelerate_hooks.AlignDevicesHook(execution_device = "cpu", skip_keys = skip)
        )
    with torch.no_grad():
        assert torch.equal(model(input_ids = ids).logits, want)


def test_shared_group_state_is_not_a_container_that_device_moves_rebuild():
    """send_to_device and FSDP2's input cast return the state object unchanged."""
    operations = pytest.importorskip("accelerate.utils.operations")
    from torch.distributed.utils import _apply_to_tensors

    state = dsv41._SharedState()
    state["index_k"] = torch.zeros(1)
    assert operations.send_to_device((state,), "cpu")[0] is state
    assert _apply_to_tensors(lambda t: t.clone(), (state,))[0] is state
    assert "index_k" in state and state.get("missing") is None


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
    """tp=2 shards sum to the unsharded attention output."""
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
    """The router clamp keeps sqrt(softplus) gradients finite where softplus underflows."""
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
    assert float(raw[0, 0].detach()) == 0.0
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
    """The 1e-20 eps sums run in fp32, since 1e-20 is 0 in fp16."""
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
    """Each compressing layer attends to its group source's latents only (model.py:740-760)."""
    model = _tiny_model().eval()
    calls = []
    real, real_window = dsv41.sparse_attn, dsv41.sparse_attn_window

    def spy(q, kv, sink, idxs, scale):
        calls.append(kv)
        return real(q, kv, sink, idxs, scale)

    def spy_window(q, window_kv, compress_kv, compress_idxs, sink, scale, window):
        # the same rows as the concatenation the index path builds
        calls.append(torch.cat([window_kv, compress_kv], dim = 1))
        return real_window(q, window_kv, compress_kv, compress_idxs, sink, scale, window)

    monkeypatch.setattr(dsv41, "sparse_attn", spy)
    monkeypatch.setattr(dsv41, "sparse_attn_window", spy_window)
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
    """A dequantized convert.py load drops the release's FP8 quantization_config so 4-bit requests still apply."""
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
    """Default, "all-linear" and None LoRA targets skip the indexer and routers."""
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
    """A flat text config saved by the text model reloads as the composite config's text config."""
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
    """Emulate FSDP2's per-layer _apply_to_tensors rebuild of dict inputs."""
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
    # Flip: a dict rebuilt per layer makes readers miss the sources' writes.
    monkeypatch.setattr(dsv41, "_SharedState", dict)
    with pytest.raises(KeyError):
        with torch.no_grad():
            model(input_ids = ids)
    for h in handles:
        h.remove()


@needs_zoo_native
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
    """A plain Linear o_a_proj takes the diagonal blocks and matches its merged LoRA weight."""
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


def test_auto_classes_reload_a_flat_text_config_checkpoint(tmp_path):
    from transformers import AutoConfig, AutoModelForCausalLM

    register_deepseek_v41()
    text = _tiny_config().text_config
    model = dsv41.DeepseekV41ForCausalLM(text).float().eval()
    model.bind_token_map([i % 97 for i in range(512)])
    for rows, layer_id in zip(text.engram_num_embeddings, text.engram_layer_ids):
        model.model.set_engram_table(layer_id, torch.randn(rows, text.engram_head_dim) * 0.5)
    ids = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        before = model(input_ids = ids).logits
    model.save_pretrained(tmp_path)
    config = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(config, DeepseekV41TextConfig) and config.hidden_size == 64
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, dtype = torch.float32).eval()
    loaded.bind_token_map([i % 97 for i in range(512)])
    with torch.no_grad():
        assert torch.equal(loaded(input_ids = ids).logits, before)


@pytest.mark.parametrize("cut", [1, 2, 3, 4, 5])
def test_pipeline_stages_with_the_group_state_handoff_match_the_whole_model(cut):
    """A two-stage pipeline split matches the unsplit model's logits and gradients."""
    model = _tiny_model().train()
    ids = torch.randint(0, 512, (2, 16))
    full = model.model(input_ids = ids).last_hidden_state
    full.float().square().mean().backward()
    want = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    text = model.config.text_config
    keys = dsv41.pipeline_handoff_keys(text, cut)
    assert (keys == []) == (cut in dsv41.pipeline_safe_cut_points(text))
    h, pre_mix, state = model.model.pipeline_stage_forward(0, cut, input_ids = ids)
    sent = state.to_tensors(keys)
    h2, _, _ = model.model.pipeline_stage_forward(
        cut,
        text.num_hidden_layers,
        hidden_states = h,
        pre_mix = pre_mix,
        group_state = sent,
        input_ids = ids,
    )
    torch.testing.assert_close(h2, full.detach(), rtol = 1e-5, atol = 1e-5)
    h2.float().square().mean().backward()
    got = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert got.keys() == want.keys()
    for name in want:
        torch.testing.assert_close(got[name], want[name], rtol = 1e-4, atol = 1e-6, msg = name)


@pytest.mark.parametrize("install", [pytest.param(True, marks = needs_zoo_native), False])
def test_building_the_model_installs_grouped_lora_for_plain_peft(monkeypatch, install):
    """o_a_proj LoRA without the zoo grouped patch matches its merged weight."""
    peft = pytest.importorskip("peft")
    pytest.importorskip("unsloth_zoo.temporary_patches.deepseek_v41")
    from peft.tuners.lora.model import LoraModel

    create = LoraModel.__dict__["_create_new_module"]
    func = getattr(create, "__func__", create)
    while getattr(func, "_unsloth_dsv41_grouped", False):
        func = func.__wrapped__  # back to PEFT's own dispatcher
    monkeypatch.setattr(LoraModel, "_create_new_module", staticmethod(func))
    monkeypatch.setattr(dsv41, "_GROUPED_LORA_READY", [False])
    if not install:
        monkeypatch.setattr(dsv41, "ensure_grouped_lora", lambda: None)
    model = _tiny_model().eval()
    ids = torch.randint(0, 512, (1, 16))
    lora = peft.get_peft_model(
        model, peft.LoraConfig(r = 4, target_modules = ["self_attn.o_a_proj"], init_lora_weights = False)
    ).eval()
    o_a = lora.base_model.model.model.layers[0].self_attn.o_a_proj
    if install:
        assert type(o_a).__name__ == "DeepseekV41GroupedLoraLinear"
    else:
        # plain PEFT LoRA (no grouped handler): grouped_projection takes the diagonal blocks of its delta too
        assert type(o_a).__name__ == "Linear" and hasattr(o_a, "lora_A")
    with torch.no_grad():
        with lora.disable_adapter():
            base = lora(input_ids = ids).logits
        adapted = lora(input_ids = ids).logits
        merged = lora.merge_and_unload()(input_ids = ids).logits
    assert not torch.allclose(adapted, base)
    torch.testing.assert_close(adapted, merged, rtol = 1e-4, atol = 1e-4)


def test_cpu_backend_import_registers_deepseek_v41(fresh_cpu_child):
    """The native registration hook is installed on the CPU backend too."""
    out = fresh_cpu_child
    probe = _probe(out, "PROBE_CPU")
    if probe is None and "CPU" in out.stderr and "fallback" in out.stderr.lower():
        pytest.skip("this unsloth_zoo has no CPU backend")
    assert probe is not None, out.stdout[-3000:] + out.stderr[-3000:]
    assert probe["cpu"] and probe["type"] == "DeepseekV41Config", probe


def test_no_module_attribute_shares_a_def_name():
    """The compiler swaps names matching a `def`, so an `import copy` beside a `.copy` method gets replaced."""
    import inspect
    import re

    source = inspect.getsource(dsv41)
    defs = set(re.findall(r"\bdef\s+(\w+)", source))
    clashes = sorted(n for n in defs if inspect.ismodule(getattr(dsv41, n, None)))
    assert not clashes, clashes


@pytest.mark.parametrize("want_info", [False, True])
def test_sharded_loading_of_a_released_checkpoint_returns_the_model(
    tmp_path, monkeypatch, want_info
):
    """Tensor parallel / FSDP2 / pipeline sharded loading of a released-layout checkpoint. DeepSeek-V4.1's
    from_pretrained asks transformers for the loading info to refuse a partial load, so the meta build returns
    (model, info): the sharded loader must fill the model, not the tuple, and hand the caller what it asked for."""
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import tp_native

    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    monkeypatch.setenv("UNSLOTH_DSV41_CONVERTED_DIR", str(tmp_path / "converted"))
    model = (
        _tiny_model().to(torch.bfloat16).float()
    )  # bf16-exact: a conversion stacks experts in bf16
    src = tmp_path / "released"
    model.config.save_pretrained(src)
    save_file(_released_state_dict(model), os.path.join(src, "model.safetensors"))
    dist.init_process_group("gloo", init_method = f"file://{tmp_path / 'pg'}", rank = 0, world_size = 1)
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names = ("tp",))["tp"]
        with tp_native.sharded_loading(mesh) as state:
            out = dsv41.DeepseekV41ForCausalLM.from_pretrained(
                src, dtype = torch.float32, output_loading_info = want_info
            )
    except ValueError as e:
        # transformers 5 merges the per-expert tensors at load (a weight conversion): refused cleanly.
        assert "weight conversion" in str(e), e
        return
    finally:
        dist.destroy_process_group()
    loaded = out
    if want_info:
        assert isinstance(out, tuple) and len(out) == 2
        loaded, info = out
        assert not info["missing_keys"] and not info["unexpected_keys"]
    assert isinstance(loaded, dsv41.DeepseekV41ForCausalLM)
    assert state["models"] == [loaded]
    assert loaded._unsloth_tp_load_stats["sharded_numel"] > 0
    got = loaded.state_dict()
    for name, value in model.state_dict().items():
        assert not got[name].is_meta, name
        assert torch.equal(got[name], value), name
