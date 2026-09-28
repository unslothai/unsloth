# SPDX-License-Identifier: AGPL-3.0-only
"""Tiny checkpoints written under Mistral's tensor names (as in vLLM's MistralLarge3 mapper
and the real Large-3 shard headers) load through the view and match the source model.
The module is exec'd from its file so `import unsloth` (needs an accelerator) is avoided.
"""

import importlib.util
import json
import os
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Mistral4ForCausalLM"):
    pytest.skip("this transformers has no mistral4", allow_module_level = True)

from safetensors.torch import save_file  # noqa: E402
from transformers import Mistral4Config, Mistral4ForCausalLM  # noqa: E402

_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mistral_format.py")
_spec = importlib.util.spec_from_file_location("_unsloth_mistral_format", _PATH)
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)

# params.json of mistralai/Mistral-Large-3-675B-Instruct-2512 without vision_encoder.
LARGE3_PARAMS = {
    "dim": 7168,
    "head_dim": 192,
    "hidden_dim": 16384,
    "kv_lora_rank": 512,
    "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 8192},
    "max_position_embeddings": 294912,
    "moe": {
        "expert_hidden_dim": 4096,
        "expert_model_parallel": 1,
        "expert_parallel": 1,
        "first_k_dense_replace": 3,
        "num_expert_groups": 1,
        "num_expert_groups_per_tok": 1,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "num_shared_experts": 1,
        "route_every_n": 1,
        "routed_scale": 1.0,
    },
    "n_heads": 128,
    "n_kv_heads": 128,
    "n_layers": 61,
    "norm_eps": 1e-06,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "quantization_config": {
        "config_groups": {
            "FP8_BLOCK": {
                "format": "float-quantized",
                "input_activations": {
                    "dynamic": True,
                    "group_size": 128,
                    "num_bits": 8,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "float",
                },
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "block_structure": [128, 128],
                    "dynamic": False,
                    "num_bits": 8,
                    "strategy": "block",
                    "symmetric": True,
                    "type": "float",
                },
            }
        },
        "format": "float-quantized",
        "ignore": [
            "model.embed_tokens",
            "re:patch_merger.*",
            "re:vision_encoder.*",
            "re:vision_language_adapter.*",
            "re:.*kv_a_proj_with_mqa$",
            "re:.*q_a_proj$",
            "re:.*gate$",
            "lm_head",
        ],
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    },
    "rope_theta": 10000.0,
    "tied_embeddings": False,
    "v_head_dim": 128,
    "vocab_size": 131072,
    "yarn": {
        "alpha": 1,
        "apply_scale": False,
        "beta": 32,
        "factor": 36,
        "original_max_position_embeddings": 8192,
    },
}

# Text tensor names of Large-3 FP8 dense layer 0 and MoE layer 3, from the shard headers.
LARGE3_TEXT_KEYS = {
    "layers.L.attention.kv_a_norm.weight",
    "layers.L.attention.q_a_norm.weight",
    "layers.L.attention.wkv_a_with_mqa.weight",
    "layers.L.attention.wkv_b.weight",
    "layers.L.attention.wkv_b.weight_scale",
    "layers.L.attention.wo.weight",
    "layers.L.attention.wo.weight_scale",
    "layers.L.attention.wq_a.weight",
    "layers.L.attention.wq_b.weight",
    "layers.L.attention.wq_b.weight_scale",
    "layers.L.attention_norm.weight",
    "layers.L.experts.N.w1.weight",
    "layers.L.experts.N.w1.weight_scale",
    "layers.L.experts.N.w2.weight",
    "layers.L.experts.N.w2.weight_scale",
    "layers.L.experts.N.w3.weight",
    "layers.L.experts.N.w3.weight_scale",
    "layers.L.feed_forward.w1.weight",
    "layers.L.feed_forward.w1.weight_scale",
    "layers.L.feed_forward.w2.weight",
    "layers.L.feed_forward.w2.weight_scale",
    "layers.L.feed_forward.w3.weight",
    "layers.L.feed_forward.w3.weight_scale",
    "layers.L.ffn_norm.weight",
    "layers.L.gate.weight",
    "layers.L.shared_experts.w1.weight",
    "layers.L.shared_experts.w1.weight_scale",
    "layers.L.shared_experts.w2.weight",
    "layers.L.shared_experts.w2.weight_scale",
    "layers.L.shared_experts.w3.weight",
    "layers.L.shared_experts.w3.weight_scale",
    "norm.weight",
    "output.weight",
    "tok_embeddings.weight",
}

BLOCK = 128
TINY_PARAMS = {
    "dim": 256,
    "head_dim": 128,
    "hidden_dim": 256,
    "kv_lora_rank": 128,
    "llama_4_scaling": {"beta": 0.1, "original_max_position_embeddings": 16},
    "max_position_embeddings": 4096,
    "moe": {
        "expert_hidden_dim": 128,
        "expert_model_parallel": 1,
        "expert_parallel": 1,
        "first_k_dense_replace": 1,
        "num_expert_groups": 1,
        "num_expert_groups_per_tok": 1,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "route_every_n": 1,
        "routed_scale": 1.0,
    },
    "n_heads": 2,
    "n_kv_heads": 2,
    "n_layers": 3,
    "norm_eps": 1e-06,
    "q_lora_rank": 128,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 64,
    "rope_theta": 10000.0,
    "tied_embeddings": False,
    "v_head_dim": 128,
    "vocab_size": 512,
    "yarn": {
        "alpha": 1,
        "apply_scale": False,
        "beta": 32,
        "factor": 36,
        "original_max_position_embeddings": 16,
    },
}

# Mistral name -> transformers name, written out independently of the module under test.
_ATTN = {
    "attention.wq_a": "self_attn.q_a_proj",
    "attention.q_a_norm": "self_attn.q_a_layernorm",
    "attention.wq_b": "self_attn.q_b_proj",
    "attention.wkv_a_with_mqa": "self_attn.kv_a_proj_with_mqa",
    "attention.kv_a_norm": "self_attn.kv_a_layernorm",
    "attention.wkv_b": "self_attn.kv_b_proj",
    "attention.wo": "self_attn.o_proj",
    "attention_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
}
_W = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def _reference(params, seed = 0):
    torch.manual_seed(seed)
    kwargs = dict(mf.mistral_params_to_mistral4_config(params))
    kwargs.pop("architectures", None)
    kwargs.pop("quantization_config", None)
    model = Mistral4ForCausalLM(Mistral4Config(**kwargs)).float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * 0.05)
    return model


def _mistral_tensors(model, params):
    """The reference's weights under Mistral's names (per-expert w1 / w2 / w3)."""
    sd = model.state_dict()
    inter = params["moe"]["expert_hidden_dim"]
    out = {
        "tok_embeddings.weight": sd["model.embed_tokens.weight"],
        "norm.weight": sd["model.norm.weight"],
        "output.weight": sd["lm_head.weight"],
    }
    for layer in range(params["n_layers"]):
        hf = f"model.layers.{layer}"
        for mistral, name in _ATTN.items():
            out[f"layers.{layer}.{mistral}.weight"] = sd[f"{hf}.{name}.weight"]
        if layer < params["moe"]["first_k_dense_replace"]:
            for w, name in _W.items():
                out[f"layers.{layer}.feed_forward.{w}.weight"] = sd[f"{hf}.mlp.{name}.weight"]
            continue
        out[f"layers.{layer}.gate.weight"] = sd[f"{hf}.mlp.gate.weight"]
        for w, name in _W.items():
            out[f"layers.{layer}.shared_experts.{w}.weight"] = sd[
                f"{hf}.mlp.shared_experts.{name}.weight"
            ]
        gate_up, down = sd[f"{hf}.mlp.experts.gate_up_proj"], sd[f"{hf}.mlp.experts.down_proj"]
        for e in range(params["moe"]["num_experts"]):
            out[f"layers.{layer}.experts.{e}.w1.weight"] = gate_up[e, :inter]
            out[f"layers.{layer}.experts.{e}.w3.weight"] = gate_up[e, inter:]
            out[f"layers.{layer}.experts.{e}.w2.weight"] = down[e]
    return {k: v.contiguous().clone() for k, v in out.items()}


def _write(path, tensors, params):
    os.makedirs(path, exist_ok = True)
    save_file(tensors, os.path.join(path, "consolidated.safetensors"))
    with open(os.path.join(path, "params.json"), "w") as f:
        json.dump(params, f)


@pytest.fixture
def hub_cache(tmp_path, monkeypatch):
    from huggingface_hub import constants

    cache = tmp_path / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    return cache


def test_large3_params_translate_to_mistral4():
    cfg = mf.mistral_params_to_mistral4_config(LARGE3_PARAMS)
    assert cfg["model_type"] == "mistral4"
    assert cfg["architectures"] == ["Mistral4ForCausalLM"]
    expect = {
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 128,
        "num_experts_per_tok": 4,
        "moe_intermediate_size": 4096,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "intermediate_size": 16384,
        "vocab_size": 131072,
        "norm_topk_prob": True,
        "rope_interleave": True,
        "tie_word_embeddings": False,
    }
    assert {k: cfg[k] for k in expect} == expect
    rope = cfg["rope_parameters"]
    assert rope["rope_type"] == "yarn" and rope["factor"] == 36
    assert rope["original_max_position_embeddings"] == 8192
    assert rope["llama_4_scaling_beta"] == 0.1
    assert cfg["quantization_config"] == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "embed_tokens",
            "gate",
            "kv_a_proj_with_mqa",
            "lm_head",
            "q_a_proj",
        ],
    }
    # transformers accepts it and derives the MLA head sizes Mistral lists.
    config = Mistral4Config(**{k: v for k, v in cfg.items() if k != "quantization_config"})
    assert config.head_dim == LARGE3_PARAMS["head_dim"] == config.qk_head_dim


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: p.pop("q_lora_rank"),  # no MLA
        lambda p: p["moe"].update(num_shared_experts = 0),
        lambda p: p["moe"].update(route_every_n = 2),
        lambda p: p.update(sliding_window = 4096),
        lambda p: p.update(
            quantization = {"qformat_weight": "fp8_e4m3"}
        ),  # per-tensor, Small-4 style
        lambda p: p["quantization_config"]["config_groups"]["FP8_BLOCK"]["weights"].update(
            num_bits = 4
        ),
        lambda p: p["yarn"].update(apply_scale = True),
        lambda p: p["quantization_config"]["config_groups"]["FP8_BLOCK"].update(
            targets = ["re:.*experts.*"]
        ),  # vLLM scales attention differently
    ],
)
def test_unsupported_params_decline(edit):
    params = json.loads(json.dumps(LARGE3_PARAMS))
    edit(params)
    assert mf.mistral_params_to_mistral4_config(params) is None


def test_tiny_names_match_the_real_large3_names():
    names = _mistral_tensors(_reference(TINY_PARAMS), TINY_PARAMS)
    folded = {
        re.sub(r"experts\.\d+\.", "experts.N.", re.sub(r"^layers\.\d+\.", "layers.L.", k))
        for k in names
    }
    real_without_scales = {k for k in LARGE3_TEXT_KEYS if not k.endswith("_scale")}
    assert folded == real_without_scales


def test_bf16_checkpoint_loads_exactly(tmp_path, hub_cache):
    reference = _reference(TINY_PARAMS)
    with torch.no_grad():
        for p in reference.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    source = tmp_path / "large3-tiny"
    _write(
        source,
        {k: v.to(torch.bfloat16) for k, v in _mistral_tensors(reference, TINY_PARAMS).items()},
        TINY_PARAMS,
    )

    view = mf.prepare_mistral_format_checkpoint(str(source))
    assert view is not None and mf.is_mistral_format_view(view)
    assert str(hub_cache.parent) in view
    assert not any(n.endswith(".safetensors") for n in os.listdir(view))  # nothing copied
    assert mf.prepare_mistral_format_checkpoint(str(source)) == view  # reused

    with mf._mistral_format_conversions():
        model = Mistral4ForCausalLM.from_pretrained(view, dtype = torch.float32)
    mf._forget_load_conversions(model)
    ref_sd, sd = reference.state_dict(), model.state_dict()
    assert set(ref_sd) == set(sd)
    assert all(torch.equal(ref_sd[k], sd[k]) for k in ref_sd)
    ids = torch.randint(
        0, TINY_PARAMS["vocab_size"], (2, 40), generator = torch.Generator().manual_seed(1)
    )
    with torch.no_grad():
        assert torch.equal(model(input_ids = ids).logits, reference(input_ids = ids).logits)

    model.save_pretrained(tmp_path / "resaved")
    again = Mistral4ForCausalLM.from_pretrained(tmp_path / "resaved", dtype = torch.float32)
    with torch.no_grad():
        assert torch.equal(again(input_ids = ids).logits, reference(input_ids = ids).logits)


def test_view_skips_vision_tensors(tmp_path, hub_cache):
    reference = _reference(TINY_PARAMS)
    tensors = {k: v.to(torch.bfloat16) for k, v in _mistral_tensors(reference, TINY_PARAMS).items()}
    tensors["vision_encoder.transformer.layers.0.attention.wq.weight"] = torch.zeros(
        4, 4, dtype = torch.bfloat16
    )
    tensors["patch_merger.merging_layer.weight"] = torch.zeros(4, 4, dtype = torch.bfloat16)
    _write(tmp_path / "vl", tensors, TINY_PARAMS)
    view = mf.prepare_mistral_format_checkpoint(str(tmp_path / "vl"))
    with open(os.path.join(view, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    assert not any(k.startswith(("vision_encoder.", "patch_merger.")) for k in weight_map)
    assert len(weight_map) == len(tensors) - 2


def test_conversions_are_scoped_to_the_load():
    from transformers import conversion_mapping as cm

    # Class-name lookup came with USER_REGISTERED_MAPPINGS; <= 5.5 looks up model_type only.
    by_class = hasattr(cm, "USER_REGISTERED_MAPPINGS")
    key, other = (
        ("Mistral4ForCausalLM", "mistral4") if by_class else ("mistral4", "Mistral4ForCausalLM")
    )
    before = repr(cm.get_checkpoint_conversion_mapping(key))
    untouched = repr(cm.get_checkpoint_conversion_mapping(other))
    with mf._mistral_format_conversions():
        inside = repr(cm.get_checkpoint_conversion_mapping(key))
        # 5.17 also applies the model_type entry to the inner Mistral4Model: it must stay stock.
        assert repr(cm.get_checkpoint_conversion_mapping(other)) == untouched
    assert "tok_embeddings" in inside and "tok_embeddings" not in before
    assert repr(cm.get_checkpoint_conversion_mapping(key)) == before
    assert key not in getattr(cm, "USER_REGISTERED_MAPPINGS", ())


def test_view_is_published_atomically_and_names_its_source(tmp_path, hub_cache):
    tensors = {
        k: v.to(torch.bfloat16)
        for k, v in _mistral_tensors(_reference(TINY_PARAMS), TINY_PARAMS).items()
    }
    _write(tmp_path / "src", tensors, TINY_PARAMS)
    view = mf.prepare_mistral_format_checkpoint(str(tmp_path / "src"))
    assert not [n for n in os.listdir(view) if n.endswith(".tmp")]

    class _Config:
        _name_or_path = view

    class _Model:
        config = _Config()
        name_or_path = view

    model = _Model()
    mf._record_source(model)  # adapters saved later name the source, not this host's view
    assert model.name_or_path == str(tmp_path / "src")


@pytest.mark.parametrize("shard", ["../outside.safetensors", "/abs/outside.safetensors"])
def test_index_shards_outside_the_repo_are_refused(tmp_path, hub_cache, shard):
    tensors = {
        k: v.to(torch.bfloat16)
        for k, v in _mistral_tensors(_reference(TINY_PARAMS), TINY_PARAMS).items()
    }
    src = tmp_path / "src"
    _write(src, tensors, TINY_PARAMS)
    with open(src / "consolidated.safetensors.index.json", "w") as f:
        json.dump({"weight_map": {k: shard for k in tensors}}, f)
    assert mf.prepare_mistral_format_checkpoint(str(src)) is None


def test_redirect_retries_with_the_view(monkeypatch):
    calls = []

    @mf.mistral_format_redirect
    def from_pretrained(
        model_name = None,
        revision = None,
        **kwargs,
    ):
        calls.append((model_name, revision))
        if len(calls) == 1:
            raise mf.MistralFormatRedirect("/views/large3", model_name)
        return object(), "tokenizer"

    monkeypatch.setattr(mf, "_mistral_format_conversions", _null_context)
    model, tok = from_pretrained("mistralai/Mistral-Large-3", revision = "abc")
    assert calls == [("mistralai/Mistral-Large-3", "abc"), ("/views/large3", None)]
    calls.clear()
    model, tok = from_pretrained(model_name = "org/other")  # keyword form too
    assert calls == [("org/other", None), ("/views/large3", None)]


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fp8_blocks(weight):
    rows, cols = weight.shape
    blocks = weight.float().reshape(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
    scale = (blocks.abs().amax(dim = (1, 3)) / 448.0).clamp(min = 1e-12).to(torch.bfloat16)
    q = (blocks / scale.float()[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    dequant = (q.float() * scale.float()[:, None, :, None]).reshape(rows, cols)
    return q.reshape(rows, cols).contiguous(), scale.contiguous(), dequant


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "FP8 loading needs a GPU")
def test_fp8_checkpoint_dequantizes_exactly_and_loads_natively(tmp_path, hub_cache):
    from transformers import FineGrainedFP8Config

    params = dict(TINY_PARAMS, quantization_config = LARGE3_PARAMS["quantization_config"])
    reference = _reference(params)
    ignored = ("wq_a", "wkv_a_with_mqa", ".gate.weight", "tok_embeddings", "output.weight")
    tensors, dequant = {}, {}
    for name, value in _mistral_tensors(reference, params).items():
        if value.ndim == 2 and not any(s in name for s in ignored):
            tensors[name], tensors[name + "_scale"], dequant[name] = _fp8_blocks(value)
        else:
            tensors[name] = value.to(torch.bfloat16)
            dequant[name] = tensors[name].float()
    _write(tmp_path / "fp8", tensors, params)
    if not mf._fp8_dequantize_on_load_available():
        # No fp8 dequantize-on-load hook (transformers < 5.8): decline the view, never half-load it.
        assert mf.prepare_mistral_format_checkpoint(str(tmp_path / "fp8")) is None
        return

    # Reference weights: the dequantized values put back into the transformers layout.
    inter = params["moe"]["expert_hidden_dim"]
    with torch.no_grad():
        sd = reference.state_dict()
        for layer in range(params["n_layers"]):
            hf, ml = f"model.layers.{layer}", f"layers.{layer}"
            for mistral, name in _ATTN.items():
                sd[f"{hf}.{name}.weight"].copy_(dequant[f"{ml}.{mistral}.weight"])
            if layer < params["moe"]["first_k_dense_replace"]:
                for w, name in _W.items():
                    sd[f"{hf}.mlp.{name}.weight"].copy_(dequant[f"{ml}.feed_forward.{w}.weight"])
                continue
            for w, name in _W.items():
                sd[f"{hf}.mlp.shared_experts.{name}.weight"].copy_(
                    dequant[f"{ml}.shared_experts.{w}.weight"]
                )
            for e in range(params["moe"]["num_experts"]):
                sd[f"{hf}.mlp.experts.gate_up_proj"][e, :inter].copy_(
                    dequant[f"{ml}.experts.{e}.w1.weight"]
                )
                sd[f"{hf}.mlp.experts.gate_up_proj"][e, inter:].copy_(
                    dequant[f"{ml}.experts.{e}.w3.weight"]
                )
                sd[f"{hf}.mlp.experts.down_proj"][e].copy_(dequant[f"{ml}.experts.{e}.w2.weight"])
        for p in reference.parameters():
            p.data = p.data.to(torch.bfloat16)
    reference = reference.cuda()

    view = mf.prepare_mistral_format_checkpoint(str(tmp_path / "fp8"))
    with open(os.path.join(view, "config.json")) as f:
        quant = dict(json.load(f)["quantization_config"])
    quant.pop("quant_method")
    with mf._mistral_format_conversions():
        dequantized = Mistral4ForCausalLM.from_pretrained(
            view,
            dtype = torch.bfloat16,
            device_map = {"": 0},
            quantization_config = FineGrainedFP8Config(dequantize = True, **quant),
        )
    ref_sd, sd = reference.state_dict(), dequantized.state_dict()
    assert all(torch.equal(ref_sd[k], sd[k]) for k in ref_sd)
    ids = torch.randint(
        0, params["vocab_size"], (2, 40), generator = torch.Generator().manual_seed(1)
    ).cuda()
    with torch.no_grad():
        assert torch.equal(dequantized(input_ids = ids).logits, reference(input_ids = ids).logits)
    del dequantized

    with mf._mistral_format_conversions():
        native = Mistral4ForCausalLM.from_pretrained(view, dtype = torch.bfloat16, device_map = {"": 0})
    kinds = {type(m).__name__ for m in native.modules()}
    assert {"FP8Linear", "FP8Experts"} <= kinds
    sd = native.state_dict()
    layer = f"model.layers.{params['n_layers'] - 1}.mlp.experts"
    e = params["moe"]["num_experts"] - 1
    ml = f"layers.{params['n_layers'] - 1}.experts.{e}"
    assert torch.equal(sd[f"{layer}.gate_up_proj"][e, :inter].cpu(), tensors[f"{ml}.w1.weight"])
    assert torch.equal(sd[f"{layer}.gate_up_proj"][e, inter:].cpu(), tensors[f"{ml}.w3.weight"])
    assert torch.equal(
        sd[f"{layer}.down_proj_scale_inv"][e].cpu().to(torch.bfloat16),
        tensors[f"{ml}.w2.weight_scale"],
    )
    with torch.no_grad():
        logits = native(input_ids = ids).logits
    assert torch.isfinite(logits).all()


def test_redirect_without_a_view_retries_the_same_call_with_conversions():
    # Raised for a view or an adapter trained on one: same arguments, conversions on.
    calls = []

    @mf.mistral_format_redirect
    def from_pretrained(
        model_name = None,
        revision = None,
        **kwargs,
    ):
        calls.append((model_name, revision, mf.mistral_format_conversions_active()))
        if not mf.mistral_format_conversions_active():
            raise mf.MistralFormatRedirect(None, model_name)
        return object(), "tokenizer"

    from_pretrained("/adapters/large3-lora", revision = "abc")
    assert calls == [
        ("/adapters/large3-lora", "abc", False),
        ("/adapters/large3-lora", "abc", True),
    ]
    assert not mf.mistral_format_conversions_active()


def test_merged_save_from_a_view_is_refused(tmp_path, hub_cache):
    _write(tmp_path / "src", _mistral_tensors(_reference(TINY_PARAMS), TINY_PARAMS), TINY_PARAMS)
    view = mf.prepare_mistral_format_checkpoint(str(tmp_path / "src"))

    class _Model:
        config = type("C", (), {"_name_or_path": view})()

    mf.raise_if_merging_mistral_format_view(_Model(), "lora")
    for method in ("merged_16bit", "merged_4bit"):
        with pytest.raises(NotImplementedError, match = "Save the LoRA adapter"):
            mf.raise_if_merging_mistral_format_view(_Model(), method)


_ADAPTER_SCRIPT = r"""
import os, sys, torch
from unsloth import FastModel
src, adapter, stage = sys.argv[1:4]
ids = torch.arange(3, 43).view(1, -1).cuda()
if stage == "train":
    model, tok = FastModel.from_pretrained(src, max_seq_length = 64, load_in_4bit = False, dtype = torch.bfloat16)
    try:
        model.save_pretrained_gguf(os.path.join(adapter, "gguf"), tok)
        raise SystemExit("GGUF export of a view was not refused")
    except NotImplementedError:
        pass
    model = FastModel.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = ["q_b_proj", "o_proj", "gate_proj"])
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n: p.normal_(0, 0.05)
    model.save_pretrained(adapter); tok.save_pretrained(adapter)
    try:
        model.save_pretrained_merged(os.path.join(adapter, "merged"), tok, save_method = "merged_16bit")
        raise SystemExit("merged save of a view was not refused")
    except NotImplementedError:
        pass
elif stage == "reload_lm":
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(adapter, max_seq_length = 64, load_in_4bit = False, dtype = torch.bfloat16)
else:
    model, tok = FastModel.from_pretrained(adapter, max_seq_length = 64, load_in_4bit = False, dtype = torch.bfloat16)
model.eval()
with torch.no_grad():
    torch.save(model(input_ids = ids).logits.float().cpu(), os.path.join(adapter, stage + ".pt"))
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "FastModel needs a GPU")
def test_lora_adapter_trained_on_a_view_reloads_in_a_fresh_process(tmp_path):
    # The adapter's base is the view, whose index still names Mistral's tensors.
    import subprocess
    import sys

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    tok_dir = snapshot_download(
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        allow_patterns = ["tokenizer*", "vocab*", "merges*", "special*"],
    )
    params = json.loads(json.dumps(TINY_PARAMS))
    params["vocab_size"] = -(-len(AutoTokenizer.from_pretrained(tok_dir)) // 128) * 128
    src = tmp_path / "src"
    _write(
        src,
        {k: v.to(torch.bfloat16) for k, v in _mistral_tensors(_reference(params), params).items()},
        params,
    )
    AutoTokenizer.from_pretrained(tok_dir).save_pretrained(src)
    script = tmp_path / "adapter_roundtrip.py"
    script.write_text(_ADAPTER_SCRIPT)
    env = dict(
        os.environ,
        HF_HUB_CACHE = str(tmp_path / "hub"),
        UNSLOTH_COMPILE_LOCATION = str(tmp_path / "ucc"),
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(os.path.dirname(_PATH), os.pardir, os.pardir), env.get("PYTHONPATH", "")]
    )
    adapter = tmp_path / "adapter"
    for stage in ("train", "reload", "reload_lm"):
        run = subprocess.run(
            [sys.executable, str(script), str(src), str(adapter), stage],
            env = env,
            capture_output = True,
            text = True,
        )
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    with open(adapter / "adapter_config.json") as f:
        assert json.load(f)["base_model_name_or_path"] == str(src)  # portable: not the local view
    trained, reloaded = torch.load(adapter / "train.pt"), torch.load(adapter / "reload.pt")
    assert torch.equal(trained, torch.load(adapter / "reload_lm.pt"))
    assert torch.equal(trained, reloaded)


def _tiny_tekken(path):
    import base64

    words = [b"he", b"ll", b"hell", b"hello", b" w", b"or", b" wor", b"ld", b" world", b"12"]
    vocab = [bytes([b]) for b in range(256)] + words
    specials = ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", "<pad>"] + [
        f"<SPECIAL_{i}>" for i in range(6, 16)
    ]
    tekken = {
        "config": {
            "pattern": LARGE3_TEKKEN_PATTERN,
            "num_vocab_tokens": len(vocab) + len(specials),
            "default_vocab_size": len(vocab) + len(specials),
            "default_num_special_tokens": len(specials),
            "version": "v13",
        },
        "vocab": [
            {"rank": i, "token_bytes": base64.b64encode(t).decode(), "token_str": None}
            for i, t in enumerate(vocab)
        ],
        "special_tokens": [
            {"rank": i, "token_str": s, "is_control": True} for i, s in enumerate(specials)
        ],
    }
    with open(path, "w") as f:
        json.dump(tekken, f)


# config.pattern of the Large-3 tekken.json.
LARGE3_TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def test_tekken_only_checkpoint_gets_a_bos_tokenizer(tmp_path, hub_cache):
    # Large-3 BF16 ships tekken.json alone; transformers' own conversion drops BOS or shifts ids.
    from transformers import AutoTokenizer

    tensors = {
        k: v.to(torch.bfloat16)
        for k, v in _mistral_tensors(_reference(TINY_PARAMS), TINY_PARAMS).items()
    }
    _write(tmp_path / "src", tensors, TINY_PARAMS)
    _tiny_tekken(tmp_path / "src" / "tekken.json")
    view = mf.prepare_mistral_format_checkpoint(str(tmp_path / "src"))
    assert {"tokenizer.json", "tokenizer_config.json"} <= set(os.listdir(view))
    tok = AutoTokenizer.from_pretrained(view)
    text = "hello world 12 é"
    ids = tok(text).input_ids
    n_special = 16
    # BOS, then byte-level BPE over the inner vocab shifted past the special tokens.
    assert ids[:4] == [1, n_special + 256 + 3, n_special + 256 + 8, n_special + ord(" ")]
    assert tok.decode(ids, skip_special_tokens = True) == text
    assert (tok.bos_token, tok.eos_token, tok.pad_token) == ("<s>", "</s>", "<pad>")
    mistral_common = pytest.importorskip("mistral_common.tokens.tokenizers.tekken")
    reference = mistral_common.Tekkenizer.from_file(str(tmp_path / "src" / "tekken.json"))
    for sample in (text, "  hell\n\nworld12345", "wor ld 🚀"):
        assert tok(sample).input_ids == [1] + reference.encode(sample, bos = False, eos = False)
