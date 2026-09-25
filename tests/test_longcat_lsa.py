# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""LongCat-Flash-Lite-Sparse loads via ``unsloth/models/longcat_lsa.py``: load, forward vs an
SGLang-transcribed reference, cache, and save back to the published layout, on a tiny checkpoint."""

import json
import math
import os
import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("transformers.models.longcat_flash")
safetensors_torch = pytest.importorskip("safetensors.torch")

import torch.nn.functional as F  # noqa: E402
from real_accelerator import has_real_cuda  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from unsloth.import_fixes import fix_transformers_longcat_lsa_config  # noqa: E402
from unsloth.models.longcat_lsa import (  # noqa: E402
    LONGCAT_LSA_MODEL_TYPE,
    is_longcat_lsa_config_dict,
    register_longcat_lsa,
)

# The published config.json, verbatim apart from the dimensions shrunk below.
REAL_CONFIG = {
    "architectures": ["LongcatCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "vocab_size": 131072,
    "hidden_size": 3072,
    "ffn_hidden_size": 6144,
    "expert_ffn_hidden_size": 1024,
    "num_layers": 14,
    "num_attention_heads": 32,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "qk_nope_head_dim": 128,
    "mla_scale_q_lora": True,
    "mla_scale_kv_lora": True,
    "routed_scaling_factor": 6.0,
    "n_routed_experts": 256,
    "max_position_embeddings": 983040,
    "rms_norm_eps": 1e-05,
    "use_cache": True,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "rope_theta": 1000000.0,
    "rope_scaling": {
        "original_max_position_embeddings": 8192,
        "rope_type": "deepseek_yarn",
        "factor": 120,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1,
        "mscale_all_dim": 1,
    },
    "attention_method": "LSA",
    "zero_expert_num": 128,
    "zero_expert_type": "identity",
    "moe_topk": 12,
    "use_mla": 1,
    "oe_vocab_size_ratio": 78,
    "oe_neighbor_num": 4,
    "oe_split_num": 4,
    "mtp_num_layers": 1,
    "index_n_heads": 16,
    "index_head_dim": 128,
    "index_topk": 2048,
    "index_k_norm_type": "rms",
    "cli_factor": 2,
    "index_local_tokens": 1024,
    "index_init_tokens": 16,
}

TINY = dict(
    vocab_size = 128,
    hidden_size = 96,
    ffn_hidden_size = 64,
    expert_ffn_hidden_size = 16,
    num_layers = 2,
    num_attention_heads = 2,
    kv_lora_rank = 16,
    q_lora_rank = 24,
    qk_rope_head_dim = 8,
    v_head_dim = 8,
    qk_nope_head_dim = 8,
    n_routed_experts = 4,
    zero_expert_num = 2,
    moe_topk = 2,
    oe_vocab_size_ratio = 3,
    index_n_heads = 2,
    index_head_dim = 8,
)


def _write_tiny(
    path,
    seed = 0,
    **overrides,
):
    cfg = dict(REAL_CONFIG, **TINY, **overrides)
    g = torch.Generator().manual_seed(seed)

    def w(*shape, scale = 0.1):
        return torch.randn(*shape, generator = g) * scale

    def norm(n):
        return 1.0 + 0.1 * torch.randn(n, generator = g)

    H, V = cfg["hidden_size"], cfg["vocab_size"]
    E, Z = cfg["n_routed_experts"], cfg["zero_expert_num"]
    I, Fh = cfg["expert_ffn_hidden_size"], cfg["ffn_hidden_size"]
    nh, ql, kl = cfg["num_attention_heads"], cfg["q_lora_rank"], cfg["kv_lora_rank"]
    rope, nope, vd = cfg["qk_rope_head_dim"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    sd = {
        "model.embed_tokens.weight": w(V, H, scale = 0.5),
        "lm_head.weight": w(V, H),
        "model.norm.weight": norm(H),
    }
    m = int(cfg["oe_vocab_size_ratio"] * V)
    tables = cfg["oe_split_num"] * (cfg["oe_neighbor_num"] - 1)
    for i in range(tables):
        sd[f"model.oe_embed_tokens{i}.weight"] = w(m + 2 * i + 1, H // tables, scale = 0.5)
        sd[f"model.oe_embed_proj{i}.weight"] = w(H, H // tables)

    def attn(prefix, indexer):
        sd[prefix + "q_a_proj.weight"] = w(ql, H)
        sd[prefix + "q_a_layernorm.weight"] = norm(ql)
        sd[prefix + "q_b_proj.weight"] = w(nh * (nope + rope), ql)
        sd[prefix + "kv_a_proj_with_mqa.weight"] = w(kl + rope, H)
        sd[prefix + "kv_a_layernorm.weight"] = norm(kl)
        sd[prefix + "kv_b_proj.weight"] = w(nh * (nope + vd), kl)
        sd[prefix + "o_proj.weight"] = w(H, nh * vd)
        if indexer:
            d, h = cfg["index_head_dim"], cfg["index_n_heads"]
            sd[prefix + "indexer.wq_b.weight"] = w(h * d, ql)
            sd[prefix + "indexer.wk.weight"] = w(d, H)
            sd[prefix + "indexer.k_norm.weight"] = norm(d)
            sd[prefix + "indexer.weights_proj.weight"] = w(h, H)

    for layer in range(cfg["num_layers"]):
        p = f"model.layers.{layer}."
        for s in range(2):
            attn(p + f"self_attn.{s}.", indexer = s == 0)
            sd[p + f"input_layernorm.{s}.weight"] = norm(H)
            sd[p + f"post_attention_layernorm.{s}.weight"] = norm(H)
            sd[p + f"mlps.{s}.gate_proj.weight"] = w(Fh, H)
            sd[p + f"mlps.{s}.up_proj.weight"] = w(Fh, H)
            sd[p + f"mlps.{s}.down_proj.weight"] = w(H, Fh)
        sd[p + "mlp.router.classifier.weight"] = w(E + Z, H, scale = 0.5)
        sd[p + "mlp.router.e_score_correction_bias"] = 0.01 * torch.randn(E + Z, generator = g)
        for e in range(E):
            sd[p + f"mlp.experts.{e}.gate_proj.weight"] = w(I, H)
            sd[p + f"mlp.experts.{e}.up_proj.weight"] = w(I, H)
            sd[p + f"mlp.experts.{e}.down_proj.weight"] = w(H, I)
    sd["model.mtp.norm.weight"] = norm(H)
    sd["model.mtp.layers.0.eh_proj.weight"] = w(H, 2 * H)
    stored = {
        k: (v if ("router" in k or "weights_proj" in k) else v.to(torch.bfloat16)).contiguous()
        for k, v in sd.items()
    }
    os.makedirs(path, exist_ok = True)
    safetensors_torch.save_file(
        stored, os.path.join(path, "model.safetensors"), metadata = {"format": "pt"}
    )
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)
    return cfg, {k: v.float() for k, v in stored.items()}


def _load(path, dtype = torch.float32):
    register_longcat_lsa()
    return AutoModelForCausalLM.from_pretrained(path, dtype = dtype)


# Reference from SGLang longcat_flash.py, deepseek_v2 MLA, deepseek_yarn, ngram_embedding.cuh.


def _ref_ngram_ids(cfg, tokens):
    V, eos = cfg["vocab_size"], cfg["eos_token_id"]
    k_, n_ = cfg["oe_split_num"], cfg["oe_neighbor_num"]
    m = int(cfg["oe_vocab_size_ratio"] * V)
    ids = torch.zeros(len(tokens), (n_ - 1) * k_, dtype = torch.long)
    for n in range(n_ - 1):
        for k in range(k_):
            idx = n * k_ + k
            mod = m + 2 * idx + 1
            for i in range(len(tokens)):
                acc = 0
                for j in range(n + 2):
                    if i - j < 0:
                        break
                    tok = int(tokens[i - j])
                    if tok == eos and j > 0:  # the context never crosses an earlier EOS
                        break
                    acc += (tok * pow(V, j, mod)) % mod
                ids[i, idx] = acc % mod
    return ids


def _ref_forward(cfg, sd, tokens):
    eps, H = cfg["rms_norm_eps"], cfg["hidden_size"]
    nh, nope, rope, vd = (
        cfg["num_attention_heads"],
        cfg["qk_nope_head_dim"],
        cfg["qk_rope_head_dim"],
        cfg["v_head_dim"],
    )
    T = len(tokens)

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + eps) * w

    def mscale(scale, m):
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    rs = cfg["rope_scaling"]
    factor, omax, base = rs["factor"], rs["original_max_position_embeddings"], cfg["rope_theta"]

    def corr(rot):
        return (rope * math.log(omax / (rot * 2 * math.pi))) / (2 * math.log(base))

    low = max(math.floor(corr(rs["beta_fast"])), 0)
    high = min(math.ceil(corr(rs["beta_slow"])), rope - 1)
    pos_freqs = base ** (torch.arange(0, rope, 2, dtype = torch.float) / rope)
    keep = 1 - ((torch.arange(rope // 2, dtype = torch.float) - low) / (high - low)).clamp(0, 1)
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - keep) + (1.0 / pos_freqs) * keep
    freqs = torch.arange(T, dtype = torch.float)[:, None] * inv_freq[None]
    cos, sin = freqs.cos()[:, None], freqs.sin()[:, None]

    def rotate(x):  # interleaved pairs (is_neox_style = False)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1).flatten(-2)

    scaling = (nope + rope) ** -0.5 * mscale(factor, rs["mscale_all_dim"]) ** 2
    q_scale, kv_scale = (H / cfg["q_lora_rank"]) ** 0.5, (H / cfg["kv_lora_rank"]) ** 0.5

    def mla(p, x):
        q = rms(x @ sd[p + "q_a_proj.weight"].T, sd[p + "q_a_layernorm.weight"] * q_scale)
        q_nope, q_pe = (q @ sd[p + "q_b_proj.weight"].T).view(T, nh, -1).split([nope, rope], -1)
        kv_a, k_pe = (x @ sd[p + "kv_a_proj_with_mqa.weight"].T).split(
            [cfg["kv_lora_rank"], rope], -1
        )
        kv = rms(kv_a, sd[p + "kv_a_layernorm.weight"] * kv_scale) @ sd[p + "kv_b_proj.weight"].T
        k_nope, v = kv.view(T, nh, -1).split([nope, vd], -1)
        q = torch.cat([q_nope, rotate(q_pe)], -1)
        k = torch.cat([k_nope, rotate(k_pe[:, None]).expand(T, nh, rope)], -1)
        s = torch.einsum("qhd,khd->hqk", q, k) * scaling
        # kv_len <= index_topk: the indexer's top-k selects every causal position.
        s = s.masked_fill(~torch.ones(T, T, dtype = torch.bool).tril(), float("-inf"))
        o = torch.einsum("hqk,khd->qhd", s.softmax(-1), v).reshape(T, -1)
        return o @ sd[p + "o_proj.weight"].T

    def mlp(p, x):
        gate = F.silu(x @ sd[p + "gate_proj.weight"].T)
        return (gate * (x @ sd[p + "up_proj.weight"].T)) @ sd[p + "down_proj.weight"].T

    def moe(p, x):
        E, rsf = cfg["n_routed_experts"], cfg["routed_scaling_factor"]
        scores = (x @ sd[p + "router.classifier.weight"].T).softmax(-1)
        choice = scores + sd[p + "router.e_score_correction_bias"][None]
        ids = torch.topk(choice, cfg["moe_topk"], dim = -1)[1]
        weights = scores.gather(1, ids) * rsf  # identity experts scaled as in transformers / vLLM
        out = torch.zeros_like(x)
        for t in range(T):
            for j in range(ids.shape[1]):
                e = int(ids[t, j])
                y = mlp(p + f"experts.{e}.", x[t : t + 1])[0] if e < E else x[t]
                out[t] += weights[t, j] * y
        return out

    ng = _ref_ngram_ids(cfg, tokens)
    parts = [sd["model.embed_tokens.weight"][tokens]]
    for i in range(ng.shape[1]):
        rows = sd[f"model.oe_embed_tokens{i}.weight"][ng[:, i]]
        parts.append(rows @ sd[f"model.oe_embed_proj{i}.weight"].T)
    h, residual = torch.stack(parts).mean(0), None

    def add_norm(h, residual, w):
        residual = h if residual is None else h + residual
        return rms(residual, w), residual

    for layer in range(cfg["num_layers"]):
        p = f"model.layers.{layer}."
        x, residual = add_norm(h, residual, sd[p + "input_layernorm.0.weight"])
        x, residual = add_norm(
            mla(p + "self_attn.0.", x), residual, sd[p + "post_attention_layernorm.0.weight"]
        )
        shortcut = moe(p + "mlp.", x)
        x, residual = add_norm(mlp(p + "mlps.0.", x), residual, sd[p + "input_layernorm.1.weight"])
        x, residual = add_norm(
            mla(p + "self_attn.1.", x), residual, sd[p + "post_attention_layernorm.1.weight"]
        )
        h = shortcut + mlp(p + "mlps.1.", x)
    x, _ = add_norm(h, residual, sd["model.norm.weight"])
    return x @ sd["lm_head.weight"].T


def _tokens(
    cfg,
    batch,
    length,
    seed = 1,
):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, cfg["vocab_size"], (batch, length), generator = g)
    eos = cfg["eos_token_id"]
    ids[0, 5] = eos
    ids[0, 6] = eos  # back-to-back EOS
    if batch > 1:
        ids[1, 11] = eos
    return ids


@pytest.fixture(scope = "module")
def tiny(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("longcat_lsa_tiny"))
    cfg, sd = _write_tiny(path)
    return path, cfg, sd


def test_recognises_only_this_config():
    assert is_longcat_lsa_config_dict(REAL_CONFIG)
    assert is_longcat_lsa_config_dict({"model_type": LONGCAT_LSA_MODEL_TYPE})
    lite = dict(REAL_CONFIG, model_type = "longcat_flash_ngram")
    assert not is_longcat_lsa_config_dict(lite)
    no_ngram = {k: v for k, v in REAL_CONFIG.items() if k != "oe_vocab_size_ratio"}
    assert not is_longcat_lsa_config_dict(no_ngram)
    assert not is_longcat_lsa_config_dict({"architectures": ["LlamaForCausalLM"]})
    assert not is_longcat_lsa_config_dict(None)


def test_autoconfig_loads_the_published_config(tmp_path):
    fix_transformers_longcat_lsa_config()
    with open(tmp_path / "config.json", "w") as f:
        json.dump(REAL_CONFIG, f)
    config = AutoConfig.from_pretrained(str(tmp_path))
    assert config.model_type == LONGCAT_LSA_MODEL_TYPE
    rope = getattr(config, "rope_parameters", None) or config.rope_scaling
    assert rope["rope_type"] == "yarn"  # SGLang's deepseek_yarn is transformers' yarn
    assert config.head_dim == config.qk_rope_head_dim == 64
    assert config.oe_vocab_size_ratio == 78 and config.index_topk == 2048

    other = tmp_path / "other"
    other.mkdir()
    with open(other / "config.json", "w") as f:
        json.dump({"architectures": ["SomethingElse"]}, f)
    with pytest.raises(ValueError):
        AutoConfig.from_pretrained(str(other))


def test_every_checkpoint_key_loads(tiny):
    path, cfg, sd = tiny
    register_longcat_lsa()
    model, info = AutoModelForCausalLM.from_pretrained(
        path, dtype = torch.float32, output_loading_info = True
    )
    assert not info["missing_keys"], info["missing_keys"]
    assert all(k.startswith("model.mtp.") for k in info["unexpected_keys"]), info
    assert not info.get("mismatched_keys"), info["mismatched_keys"]
    layer = model.model.layers[0]
    experts = layer.mlp.experts
    p = "model.layers.0.mlp.experts.3."
    if hasattr(experts, "gate_up_proj"):  # transformers 5 stacks the experts
        assert experts.gate_up_proj.shape[0] == cfg["n_routed_experts"]
        gate_up = torch.cat([sd[p + "gate_proj.weight"], sd[p + "up_proj.weight"]], 0)
        assert torch.equal(experts.gate_up_proj[3], gate_up)
        assert torch.equal(experts.down_proj[3], sd[p + "down_proj.weight"])
    else:
        assert torch.equal(experts[3].gate_proj.weight, sd[p + "gate_proj.weight"])
    ngram = model.model.ngram_embeddings
    assert torch.equal(ngram.embedders[7].weight, sd["model.oe_embed_tokens7.weight"])
    assert torch.equal(ngram.post_projs[11].weight, sd["model.oe_embed_proj11.weight"])
    indexer = layer.self_attn[0].indexer
    assert torch.equal(indexer.wq_b.weight, sd["model.layers.0.self_attn.0.indexer.wq_b.weight"])
    assert not indexer.wq_b.weight.requires_grad


def test_forward_matches_sglang_reference(tiny):
    path, cfg, sd = tiny
    model = _load(path).eval()
    ids = _tokens(cfg, 2, 19)
    with torch.no_grad():
        logits = model(input_ids = ids).logits
    for b in range(ids.shape[0]):
        ref = _ref_forward(cfg, sd, ids[b])
        torch.testing.assert_close(logits[b], ref, atol = 2e-4, rtol = 1e-4)


def test_ngram_ids_follow_the_sglang_kernel(tiny):
    path, cfg, sd = tiny
    model = _load(path).eval()
    seen = []
    handles = [
        e.register_forward_hook(lambda mod, args, out: seen.append(args[0].clone()))
        for e in model.model.ngram_embeddings.embedders
    ]
    ids = _tokens(cfg, 2, 15)
    with torch.no_grad():
        model.model.ngram_embeddings(model.model.embed_tokens(ids), ids)
    for h in handles:
        h.remove()
    ours = torch.stack(seen, -1)
    for b in range(ids.shape[0]):
        assert torch.equal(ours[b], _ref_ngram_ids(cfg, ids[b]))


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA device")
def test_split_model_keeps_the_token_table_where_accelerate_put_it(tiny):
    # Accelerate hooks move every `.to`-able arg: never pass the embed_tokens module itself.
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    path, cfg, sd = tiny
    model = _load(path).eval().to("cuda")
    ids = _tokens(cfg, 2, 15).to("cuda")
    with torch.no_grad():
        expected = model(input_ids = ids).logits.cpu()
    model.model.embed_tokens.to("cpu")
    add_hook_to_module(model.model.embed_tokens, AlignDevicesHook(execution_device = "cpu"))
    add_hook_to_module(model.model.ngram_embeddings, AlignDevicesHook(execution_device = "cuda"))
    with torch.no_grad():
        got = model(input_ids = ids).logits.cpu()
    assert model.model.embed_tokens.weight.device.type == "cpu"
    torch.testing.assert_close(got, expected, atol = 1e-4, rtol = 1e-4)


def test_cached_decode_matches_full_forward(tiny):
    path, cfg, sd = tiny
    model = _load(path).eval()
    ids = _tokens(cfg, 1, 14)
    with torch.no_grad():
        full = model(input_ids = ids).logits[0]
        out = model(input_ids = ids[:, :9], use_cache = True)
        cache, steps = out.past_key_values, [out.logits[0]]
        for t in range(9, ids.shape[1]):  # decode across the n-gram context boundary
            out = model(input_ids = ids[:, t : t + 1], past_key_values = cache, use_cache = True)
            cache = out.past_key_values
            steps.append(out.logits[0])
    torch.testing.assert_close(torch.cat(steps), full, atol = 1e-4, rtol = 1e-4)


def test_packed_training_rows_do_not_see_each_other(tiny):
    # Padding-free packing: no cache in training, so transformers' packed mask applies, and the
    # n-gram restarts with position_ids.
    path, cfg, sd = tiny
    model = _load(path).train()
    a, b = _tokens(cfg, 2, 12, seed = 3)
    b[0] = a[-1]  # b's first n-grams would read a's tail if the n-gram crossed the boundary
    packed = torch.cat([a, b])[None]
    position_ids = torch.cat([torch.arange(12), torch.arange(12)])[None]
    with torch.no_grad():
        out = model(input_ids = packed, position_ids = position_ids)
        alone = model(input_ids = b[None]).logits[0]
    assert out.past_key_values is None
    torch.testing.assert_close(out.logits[0, 12:], alone, atol = 1e-4, rtol = 1e-4)


def test_left_padded_generation_matches_unpadded(tiny):
    path, cfg, sd = tiny
    model = _load(path).eval()
    ids = _tokens(cfg, 1, 10, seed = 4)
    padded = torch.cat([torch.full((1, 3), 7), ids], dim = -1)
    mask = torch.cat([torch.zeros(1, 3, dtype = torch.long), torch.ones_like(ids)], dim = -1)
    kwargs = dict(max_new_tokens = 5, do_sample = False, pad_token_id = 0)
    with torch.no_grad():
        want = model.generate(input_ids = ids, **kwargs)[0, 10:]
        got = model.generate(input_ids = padded, attention_mask = mask, **kwargs)[0, 13:]
    assert got.tolist() == want.tolist()


def test_ngram_history_follows_beam_reorder_and_crop(tiny):
    # Beam reorder and crop must move the n-gram history with the KV cache.
    path, cfg, sd = tiny
    model = _load(path).eval()
    ids = _tokens(cfg, 2, 14)
    swapped = ids.flip(0)
    with torch.no_grad():
        full = model(input_ids = swapped).logits
        cache = model(input_ids = ids[:, :9], use_cache = True).past_key_values
        cache.reorder_cache(torch.tensor([1, 0]))
        out = model(input_ids = swapped[:, 9:10], past_key_values = cache, use_cache = True)
        torch.testing.assert_close(out.logits[:, -1], full[:, 9], atol = 1e-4, rtol = 1e-4)
        cache = out.past_key_values
        cache.crop(9)
        out = model(input_ids = swapped[:, 9:11], past_key_values = cache, use_cache = True)
        torch.testing.assert_close(out.logits[:, -1], full[:, 10], atol = 1e-4, rtol = 1e-4)


def test_cache_reset_clears_the_ngram_history(tiny):
    # A reset() static cache must not leak the previous prompt into n-gram ids.
    import transformers
    from transformers.cache_utils import StaticCache

    if int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("transformers 4's StaticCache cannot hold MLA's differing key/value head dims")
    path, cfg, sd = tiny
    model = _load(path).eval()
    first, second = _tokens(cfg, 1, 9, seed = 1), _tokens(cfg, 1, 9, seed = 2)
    try:
        cache = StaticCache(config = model.config, max_cache_len = 32)
    except TypeError:
        cache = StaticCache(config = model.config, max_batch_size = 1, max_cache_len = 32)
    with torch.no_grad():
        expected = model(input_ids = second).logits
        model(input_ids = first, past_key_values = cache, use_cache = True)
        cache.reset()
        got = model(input_ids = second, past_key_values = cache, use_cache = True).logits
    torch.testing.assert_close(got, expected, atol = 1e-4, rtol = 1e-4)


def test_inputs_embeds_is_refused(tiny):
    path, cfg, sd = tiny
    model = _load(path).eval()
    ids = _tokens(cfg, 1, 9)
    with pytest.raises(ValueError, match = "input_ids"):
        model(inputs_embeds = model.model.embed_tokens(ids))


def test_bf16_keeps_router_fp32_and_sglang_norm_eps(tiny):
    path, cfg, sd = tiny
    model = _load(path, dtype = torch.bfloat16)
    layer = model.model.layers[0]
    assert layer.mlp.router.classifier.weight.dtype == torch.float32
    assert layer.self_attn[0].indexer.weights_proj.weight.dtype == torch.float32
    assert layer.self_attn[0].q_a_proj.weight.dtype == torch.bfloat16
    for attn in layer.self_attn:
        assert attn.q_a_layernorm.variance_epsilon == cfg["rms_norm_eps"]
        assert attn.kv_a_layernorm.variance_epsilon == cfg["rms_norm_eps"]


def test_save_writes_the_published_layout(tiny, tmp_path):
    path, cfg, sd = tiny
    model = _load(path, dtype = torch.bfloat16)
    model.save_pretrained(str(tmp_path))
    saved = {}
    for name in os.listdir(tmp_path):
        if name.endswith(".safetensors"):
            saved.update(safetensors_torch.load_file(str(tmp_path / name)))
    original = safetensors_torch.load_file(os.path.join(path, "model.safetensors"))
    expected = {k for k in original if not k.startswith("model.mtp.")}
    assert set(saved) == expected
    for key in expected:
        assert saved[key].dtype == original[key].dtype, key
        assert torch.equal(saved[key], original[key]), key
    with open(tmp_path / "config.json") as f:
        assert json.load(f)["model_type"] == LONGCAT_LSA_MODEL_TYPE


def test_peft_converts_per_expert_adapters_like_longcat_flash():
    register_longcat_lsa()
    conversion = pytest.importorskip("transformers.conversion_mapping")
    table = getattr(conversion, "_MODEL_TO_CONVERSION_PATTERN", None)
    if not isinstance(table, dict) or "longcat_flash" not in table:
        pytest.skip("this transformers has no model-type conversion table")
    assert table[LONGCAT_LSA_MODEL_TYPE] == table["longcat_flash"]
    try:
        from peft.utils import transformers_weight_conversion as peft_conversion
    except Exception:
        return
    assert (
        peft_conversion._MODEL_TO_CONVERSION_PATTERN.get(LONGCAT_LSA_MODEL_TYPE)
        == (table["longcat_flash"])
    )


def test_long_sequence_warns_once(tmp_path):
    cfg, sd = _write_tiny(str(tmp_path), index_topk = 8)
    model = _load(str(tmp_path)).eval()
    ids = _tokens(cfg, 1, 12)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            model(input_ids = ids)
            model(input_ids = ids)
    messages = [str(w.message) for w in caught if "sparse attention" in str(w.message)]
    assert len(messages) == 1, messages


def test_cached_decode_past_index_topk_warns(tmp_path):
    cfg, sd = _write_tiny(str(tmp_path), index_topk = 8)
    model = _load(str(tmp_path)).eval()
    ids = _tokens(cfg, 1, 12)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            cache = model(input_ids = ids[:, :8], use_cache = True).past_key_values
            model(input_ids = ids[:, 8:9], past_key_values = cache, use_cache = True)
    messages = [str(w.message) for w in caught if "sparse attention" in str(w.message)]
    assert len(messages) == 1 and "9 token" in messages[0], messages


def test_4bit_keeps_the_mla_up_projections_in_16bit():
    # The device-map planner must size q_b_proj / kv_b_proj unquantized, as the load keeps them.
    import ast

    from unsloth.models.vision import _architecture_skip_modules

    for model_type in ("longcat_flash", "longcat_flash_lsa"):
        assert {"q_b_proj", "kv_b_proj"} <= set(_architecture_skip_modules([model_type]))
    assert _architecture_skip_modules(["llama"]) == []
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "unsloth",
        "models",
        "vision.py",
    )
    with open(path, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    planner = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "planner_quantization_kwargs"
    ]
    assert planner
    for call in planner:
        extra = next(k.value for k in call.keywords if k.arg == "extra_skip_modules")
        assert "_architecture_skip_modules" in ast.unparse(extra)
