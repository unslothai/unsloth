# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separate ComfyUI text-encoder / VAE files beside a single-file DiT (``diffusion_comfy_components.py``).

Everything runs on CPU against tiny models and tiny synthetic safetensors files written in the ComfyUI
layout: the key rules must load STRICTLY (nothing missing, nothing extra beyond the known-dead tensors),
quantized layers must dequantize (scaled fp8) or stay int8 (ConvRot), unsupported formats must be refused
by name, and the request / planner wiring must drop and re-price the replaced components.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")

import core.inference.diffusion_comfy_components as C  # noqa: E402


def _save(
    path: Path,
    tensors: dict,
    metadata: dict | None = None,
) -> str:
    path.parent.mkdir(parents = True, exist_ok = True)
    safetensors_torch.save_file(
        {k: v.detach().clone().contiguous() for k, v in tensors.items()},
        str(path),
        metadata = metadata,
    )
    return str(path)


def _quant_blob(conf: dict) -> "torch.Tensor":
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype = torch.uint8)


# ----------------------------------------------------------------------------------------------------------------------
# Spec parsing and trust


def test_parse_local_absolute_relative_and_bare(tmp_path):
    models = tmp_path / "ComfyUI" / "models"
    te = _save(models / "text_encoders" / "clip_l.safetensors", {"x": torch.zeros(1)})
    dit_dir = models / "diffusion_models"
    dit_dir.mkdir(parents = True)
    _save(dit_dir / "vae_here.safetensors", {"x": torch.zeros(1)})
    assert C.parse_component_file(te).local_path == str(Path(te).resolve())
    rel = C.parse_component_file("../text_encoders/clip_l.safetensors", model_path = str(dit_dir))
    assert rel.local_path == str(Path(te).resolve())
    bare = C.parse_component_file("vae_here.safetensors", model_path = str(dit_dir))
    assert bare.local_path.endswith("vae_here.safetensors")


def test_parse_refuses_non_safetensors_missing_and_untrusted(tmp_path):
    with pytest.raises(C.ComponentFileError, match = "not a .safetensors"):
        C.parse_component_file(str(tmp_path / "t5.gguf"))
    with pytest.raises(C.ComponentFileError, match = "does not exist"):
        C.parse_component_file(str(tmp_path / "missing.safetensors"))
    with pytest.raises(C.ComponentFileError, match = "relative path"):
        C.parse_component_file("../x.safetensors", model_path = "unsloth/FLUX.1-schnell")
    with pytest.raises(C.ComponentFileError, match = "restricted"):
        C.parse_component_file(
            "someone/repo/te.safetensors", trusted_repo = lambda r: r.startswith("unsloth/")
        )
    hub = C.parse_component_file(
        "unsloth/X-ComfyUI/split_files/te.safetensors",
        trusted_repo = lambda r: r.startswith("unsloth/"),
    )
    assert (hub.repo_id, hub.filename) == ("unsloth/X-ComfyUI", "split_files/te.safetensors")


def test_normalize_text_encoder_files():
    assert C.normalize_text_encoder_files(None) == []
    assert C.normalize_text_encoder_files(" a.safetensors ") == ["a.safetensors"]
    assert C.normalize_text_encoder_files(["a", "a", " ", "b"]) == ["a", "b"]
    with pytest.raises(C.ComponentFileError):
        C.normalize_text_encoder_files(["a", "b", "c", "d", "e"])


# ----------------------------------------------------------------------------------------------------------------------
# Classification and key rules (real ComfyUI key layouts, tiny shapes)


@pytest.mark.parametrize(
    "keys, kind",
    [
        (
            [
                "text_model.encoder.layers.0.mlp.fc1.weight",
                "text_model.embeddings.token_embedding.weight",
            ],
            "clip_l",
        ),
        (["encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight"], "t5"),
        (
            [
                "encoder.block.0.layer.0.SelfAttention.q.weight",
                "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight",
                "shared.weight",
            ],
            "umt5",
        ),
        (["model.layers.0.self_attn.q_norm.weight", "model.embed_tokens.weight"], "qwen3"),
        (
            ["model.layers.0.self_attn.q_proj.weight", "visual.blocks.0.attn.qkv.weight"],
            "qwen2_5_vl",
        ),
        (
            ["model.layers.0.self_attn.q_norm.weight", "model.visual.blocks.0.attn.qkv.weight"],
            "qwen3_vl",
        ),
        (["model.layers.0.self_attn.q_proj.weight", "tekken_model"], "mistral3"),
        (
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.pre_feedforward_layernorm.weight",
            ],
            "gemma2",
        ),
        (["model.layers.0.self_attn.q_proj.weight"], "llama"),
        (["decoder.conv_in.weight"], None),
    ],
)
def test_classify_text_encoder(keys, kind):
    assert C.classify_text_encoder(keys) == kind


def test_classify_clip_g_by_width():
    keys = [
        "text_model.encoder.layers.0.mlp.fc1.weight",
        "text_model.embeddings.token_embedding.weight",
    ]
    shapes = {"text_model.embeddings.token_embedding.weight": [49408, 1280]}
    assert C.classify_text_encoder(keys, shapes) == "clip_g"


@pytest.mark.parametrize(
    "keys, kind",
    [
        (["decoder.up.0.block.0.conv1.weight", "encoder.down.0.block.0.conv1.weight"], "ldm_kl"),
        (["decoder.upsamples.0.residual.0.gamma", "encoder.downsamples.0.residual.0.gamma"], "wan"),
        (
            [
                "decoder.upsamples.0.upsamples.0.residual.0.gamma",
                "encoder.downsamples.0.downsamples.1.resample.1.weight",
            ],
            "wan_nested",
        ),
        (["decoder.up_blocks.0.resnets.0.conv1.weight"], "diffusers"),
        (["model.layers.0.self_attn.q_proj.weight"], None),
    ],
)
def test_classify_vae(keys, kind):
    assert C.classify_vae(keys) == kind


def test_match_keys_vl_nest_strict_and_dead_lm_head():
    expected = {
        "model.language_model.embed_tokens.weight": (10, 4),
        "model.language_model.layers.0.self_attn.q_proj.weight": (4, 4),
        "model.language_model.norm.weight": (4,),
        "model.visual.blocks.0.attn.qkv.weight": (12, 4),
    }
    comfy = {
        "model.embed_tokens.weight": (10, 4),
        "model.layers.0.self_attn.q_proj.weight": (4, 4),
        "model.norm.weight": (4,),
        "model.visual.blocks.0.attn.qkv.weight": (12, 4),
        "lm_head.weight": (10, 4),  # trimmed by the family: dead
    }
    km = C.match_keys(comfy, expected)
    assert km.rule == "nest_language_model"
    assert km.dead_unexpected == ["lm_head.weight"]
    # Qwen2.5-VL keeps its vision tower at the top level in ComfyUI files.
    comfy25 = {
        ("visual." + k[len("model.visual.") :] if k.startswith("model.visual.") else k): v
        for k, v in comfy.items()
    }
    assert C.match_keys(comfy25, expected).rule == "nest_language_model"


def test_match_keys_refuses_missing_extra_and_shape():
    expected = {"layers.0.w.weight": (4, 4), "norm.weight": (4,)}
    with pytest.raises(C.ComponentFileError, match = "missing"):
        C.match_keys({"model.layers.0.w.weight": (4, 4)}, expected)
    with pytest.raises(C.ComponentFileError, match = "unexpected"):
        C.match_keys(
            {"layers.0.w.weight": (4, 4), "norm.weight": (4,), "extra.weight": (1,)}, expected
        )
    with pytest.raises(C.ComponentFileError, match = "shape"):
        C.match_keys({"layers.0.w.weight": (4, 8), "norm.weight": (4,)}, expected)


def test_match_keys_tied_lm_head_may_be_missing():
    expected = {"model.embed_tokens.weight": (10, 4), "lm_head.weight": (10, 4)}
    km = C.match_keys(
        {"model.embed_tokens.weight": (10, 4)}, expected, tied_missing = ("lm_head.weight",)
    )
    assert km.dead_missing == ["lm_head.weight"]
    with pytest.raises(C.ComponentFileError):
        C.match_keys({"model.embed_tokens.weight": (10, 4)}, expected)


# ----------------------------------------------------------------------------------------------------------------------
# Building real (tiny) transformers encoders from ComfyUI-layout files


def _tiny_clip():
    cfg = transformers.CLIPTextConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        max_position_embeddings = 16,
        projection_dim = 32,
    )
    torch.manual_seed(0)
    return cfg, transformers.CLIPTextModel(cfg).eval()


def test_clip_l_full_clip_export_loads_strictly_and_matches(tmp_path):
    cfg, ref = _tiny_clip()
    state = {k: v for k, v in ref.state_dict().items() if not k.endswith("position_ids")}
    # A full-CLIP export also carries the projection and logit scale CLIPTextModel never reads.
    state["text_projection.weight"] = torch.randn(32, 32)
    state["logit_scale"] = torch.tensor(4.6)
    path = _save(
        tmp_path / "clip_l.safetensors", {k: v.to(torch.float16) for k, v in state.items()}
    )
    enc = C.build_text_encoder(
        path, encoder_cls = transformers.CLIPTextModel, config = cfg, dtype = torch.float32
    )
    ids = torch.tensor([[1, 5, 9, 2]])
    with torch.no_grad():
        got = enc(input_ids = ids).last_hidden_state
        want = ref.to(torch.float16).float()(input_ids = ids).last_hidden_state
    assert torch.allclose(got, want, atol = 1e-5)


def _tiny_t5():
    cfg = transformers.T5Config(
        vocab_size = 64,
        d_model = 32,
        d_kv = 8,
        d_ff = 64,
        num_layers = 2,
        num_heads = 4,
        feed_forward_proj = "gated-gelu",
        is_encoder_decoder = False,
        use_cache = False,
    )
    torch.manual_seed(0)
    return cfg, transformers.T5EncoderModel(cfg).eval()


def test_t5_legacy_scaled_fp8_dequantizes_strictly(tmp_path):
    cfg, ref = _tiny_t5()
    state = dict(ref.state_dict())
    out = {}
    dequant_ref = {}
    for key, value in state.items():
        if key.endswith(".weight") and value.ndim == 2 and ".block." in key:
            scale = value.abs().max() / 448.0
            codes = (value / scale).to(torch.float8_e4m3fn)
            out[key] = codes
            out[key[: -len(".weight")] + ".scale_weight"] = scale.reshape(())
            dequant_ref[key] = codes.float() * scale
        else:
            out[key] = value
    out["scaled_fp8"] = torch.zeros(0, dtype = torch.float8_e4m3fn)
    path = _save(tmp_path / "t5xxl_fp8_e4m3fn_scaled.safetensors", out)
    enc = C.build_text_encoder(
        path, encoder_cls = transformers.T5EncoderModel, config = cfg, dtype = torch.float32
    )
    loaded = enc.state_dict()
    for key, want in dequant_ref.items():
        assert torch.equal(loaded[key], want), key
    assert enc.encoder.embed_tokens.weight.data_ptr() == enc.shared.weight.data_ptr()
    ids = torch.tensor([[3, 7, 11, 1]])
    with torch.no_grad():
        cos = torch.nn.functional.cosine_similarity(
            enc(input_ids = ids).last_hidden_state.flatten(),
            ref(input_ids = ids).last_hidden_state.flatten(),
            dim = 0,
        )
    assert cos > 0.99


def _tiny_qwen3():
    cfg = transformers.Qwen3Config(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        max_position_embeddings = 32,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    return cfg, transformers.Qwen3Model(cfg).eval()


def _convrot_int8(weight: "torch.Tensor", group: int):
    from core.inference.diffusion_convrot import build_convrot_hadamard

    h = build_convrot_hadamard(group, device = weight.device, dtype = torch.float32)
    rows, cols = weight.shape
    rotated = (weight.float().reshape(rows, cols // group, group) @ h.T).reshape(rows, cols)
    scale = rotated.abs().amax(dim = 1, keepdim = True) / 127.0
    codes = torch.round(rotated / scale).clamp(-127, 127).to(torch.int8)
    return codes, scale.reshape(rows)


def test_qwen3_comfy_layout_int8_convrot_stays_int8(tmp_path):
    cfg, ref = _tiny_qwen3()
    group = 16
    out = {}
    for key, value in ref.state_dict().items():
        comfy_key = "model." + key  # ComfyUI keeps the causal-LM naming
        if ".mlp." in key and key.endswith(".weight"):
            codes, scale = _convrot_int8(value, group)
            stem = comfy_key[: -len(".weight")]
            out[comfy_key] = codes
            out[stem + ".weight_scale"] = scale
            out[stem + ".comfy_quant"] = _quant_blob(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
            )
        else:
            out[comfy_key] = value.to(torch.bfloat16)
    path = _save(tmp_path / "qwen_3_4b_int8_convrot.safetensors", out)
    enc = C.build_text_encoder(
        path, encoder_cls = transformers.Qwen3Model, config = cfg, dtype = torch.bfloat16
    )
    assert type(enc.layers[0].mlp.gate_proj).__name__ == "Int8ConvRotLinear"
    assert enc.layers[0].mlp.gate_proj.weight.dtype == torch.int8
    assert getattr(enc, "_unsloth_te_prequant_scheme", None) == "int8"
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        got = enc(input_ids = ids).last_hidden_state.float()
        want = ref.to(torch.bfloat16)(input_ids = ids).last_hidden_state.float()
    cos = torch.nn.functional.cosine_similarity(got.flatten(), want.flatten(), dim = 0)
    assert cos > 0.98


def test_qwen3_bf16_strip_model_prefix_and_dead_lm_head(tmp_path):
    cfg, ref = _tiny_qwen3()
    out = {"model." + k: v for k, v in ref.state_dict().items()}
    out["lm_head.weight"] = torch.randn(64, 64)
    path = _save(tmp_path / "qwen_3_4b.safetensors", out)
    enc = C.build_text_encoder(
        path, encoder_cls = transformers.Qwen3Model, config = cfg, dtype = torch.float32
    )
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.allclose(
            enc(input_ids = ids).last_hidden_state, ref(input_ids = ids).last_hidden_state
        )


@pytest.mark.parametrize(
    "conf",
    [
        {"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, "convrot_groupsize": 256},
        {"format": "nvfp4"},
        {"format": "mxfp8"},
    ],
)
def test_unsupported_comfy_quant_formats_refused_by_name(tmp_path, conf):
    out = {
        "model.layers.0.mlp.down_proj.weight": torch.zeros(8, 8, dtype = torch.int8),
        "model.layers.0.mlp.down_proj.weight_scale": torch.ones(8),
        "model.layers.0.mlp.down_proj.comfy_quant": _quant_blob(conf),
        "model.layers.0.self_attn.q_norm.weight": torch.ones(8),
    }
    path = _save(tmp_path / "te_quant.safetensors", out)
    with pytest.raises(C.ComponentFileError, match = conf["format"]):
        C.quant_layers(path)
    with pytest.raises(C.ComponentFileError, match = conf["format"]):
        C.validate_component_specs([path], None, model_path = None)


def test_wrong_class_refused_with_diagnostics(tmp_path):
    cfg, ref = _tiny_clip()
    path = _save(tmp_path / "clip.safetensors", dict(ref.state_dict()))
    qcfg, _ = _tiny_qwen3()
    with pytest.raises(C.ComponentFileError, match = "does not fit"):
        C.build_text_encoder(
            path, encoder_cls = transformers.Qwen3Model, config = qcfg, dtype = torch.float32
        )


# ----------------------------------------------------------------------------------------------------------------------
# VAEs


def _ldm_vae_header(
    ch = 32,
    mult = (1, 2),
    z = 4,
    layers = 2,
) -> dict:
    """The original (``ae.safetensors``) LDM VAE naming for a tiny config."""
    t: dict = {}

    def conv(
        name,
        o,
        i,
        k = 3,
    ):
        t[name + ".weight"] = (o, i, k, k)
        t[name + ".bias"] = (o,)

    def norm(name, c):
        t[name + ".weight"] = (c,)
        t[name + ".bias"] = (c,)

    def res(name, i, o):
        norm(name + ".norm1", i)
        conv(name + ".conv1", o, i)
        norm(name + ".norm2", o)
        conv(name + ".conv2", o, o)
        if i != o:
            conv(name + ".nin_shortcut", o, i, 1)

    def attn(name, c):
        norm(name + ".norm", c)
        for p in ("q", "k", "v", "proj_out"):
            conv(f"{name}.{p}", c, c, 1)

    chans = [ch * m for m in mult]
    conv("encoder.conv_in", ch, 3)
    prev = ch
    for i, c in enumerate(chans):
        for j in range(layers):
            res(f"encoder.down.{i}.block.{j}", prev, c)
            prev = c
        if i != len(chans) - 1:
            conv(f"encoder.down.{i}.downsample.conv", c, c)
    res("encoder.mid.block_1", prev, prev)
    attn("encoder.mid.attn_1", prev)
    res("encoder.mid.block_2", prev, prev)
    norm("encoder.norm_out", prev)
    conv("encoder.conv_out", 2 * z, prev)
    conv("decoder.conv_in", chans[-1], z)
    res("decoder.mid.block_1", chans[-1], chans[-1])
    attn("decoder.mid.attn_1", chans[-1])
    res("decoder.mid.block_2", chans[-1], chans[-1])
    prev = chans[-1]
    for i in reversed(range(len(chans))):
        c = chans[i]
        for j in range(layers + 1):
            res(f"decoder.up.{i}.block.{j}", prev, c)
            prev = c
        if i != 0:
            conv(f"decoder.up.{i}.upsample.conv", c, c)
    norm("decoder.norm_out", prev)
    conv("decoder.conv_out", 3, prev)
    return t


def _tiny_kl_config():
    return {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D"] * 2,
        "up_block_types": ["UpDecoderBlock2D"] * 2,
        "block_out_channels": [32, 64],
        "layers_per_block": 2,
        "latent_channels": 4,
        "norm_num_groups": 32,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
    }


def test_ldm_ae_vae_loads_strictly_into_autoencoderkl(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    header = _ldm_vae_header()
    state = {k: torch.randn(shape) * 0.02 for k, shape in header.items()}
    path = _save(tmp_path / "ae.safetensors", state)
    vae = C.build_vae(
        path, vae_cls = diffusers.AutoencoderKL, config = _tiny_kl_config(), dtype = torch.float32
    )
    loaded = vae.state_dict()
    assert torch.equal(loaded["encoder.conv_in.weight"], state["encoder.conv_in.weight"])
    assert torch.equal(
        loaded["encoder.mid_block.attentions.0.to_q.weight"],
        state["encoder.mid.attn_1.q.weight"].squeeze(-1).squeeze(-1),
    )
    with torch.no_grad():
        assert vae.decode(torch.randn(1, 4, 8, 8)).sample.shape == (1, 3, 16, 16)


def test_vae_missing_tensor_refused(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    header = _ldm_vae_header()
    header.pop("decoder.conv_out.weight")
    path = _save(tmp_path / "ae.safetensors", {k: torch.zeros(s) for k, s in header.items()})
    with pytest.raises(C.ComponentFileError, match = "missing"):
        C.build_vae(
            path, vae_cls = diffusers.AutoencoderKL, config = _tiny_kl_config(), dtype = torch.float32
        )


def test_wan_nested_vae_names_and_time_axis_squeeze():
    state = {
        "encoder.downsamples.0.downsamples.0.residual.0.gamma": 0,
        "encoder.downsamples.0.downsamples.1.residual.2.weight": 1,
        "encoder.downsamples.0.downsamples.2.resample.1.weight": 2,
        "decoder.upsamples.1.upsamples.0.shortcut.bias": 3,
        "decoder.upsamples.1.upsamples.3.time_conv.weight": 4,
        "decoder.middle.0.residual.6.weight": 5,
        "decoder.middle.2.residual.3.gamma": 6,
        "decoder.middle.1.to_qkv.weight": 7,
        "encoder.head.2.bias": 8,
        "decoder.conv1.weight": 9,
        "conv1.weight": 10,
        "conv2.bias": 11,
    }
    got = C._convert_wan_nested_vae(state)
    assert got == {
        "encoder.down_blocks.0.resnets.0.norm1.gamma": 0,
        "encoder.down_blocks.0.resnets.1.conv1.weight": 1,
        "encoder.down_blocks.0.downsampler.resample.1.weight": 2,
        "decoder.up_blocks.1.resnets.0.conv_shortcut.bias": 3,
        "decoder.up_blocks.1.upsampler.time_conv.weight": 4,
        "decoder.mid_block.resnets.0.conv2.weight": 5,
        "decoder.mid_block.resnets.1.norm2.gamma": 6,
        "decoder.mid_block.attentions.0.to_qkv.weight": 7,
        "encoder.conv_out.bias": 8,
        "decoder.conv_in.weight": 9,
        "quant_conv.weight": 10,
        "post_quant_conv.bias": 11,
    }
    fitted = C.fit_conv_shapes(
        {"a": torch.zeros(4, 2, 1, 3, 3), "b": torch.zeros(4, 2, 1, 3, 3)}, {"a": (4, 2, 3, 3)}
    )
    assert tuple(fitted["a"].shape) == (4, 2, 3, 3) and tuple(fitted["b"].shape) == (4, 2, 1, 3, 3)


# ----------------------------------------------------------------------------------------------------------------------
# Assignment, pricing, wiring

FLUX_INDEX = {
    "text_encoder": ["transformers", "CLIPTextModel"],
    "text_encoder_2": ["transformers", "T5EncoderModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "transformer": ["diffusers", "FluxTransformer2DModel"],
}


def _hdr(*keys, shape = (4, 4)):
    return {k: {"dtype": "BF16", "shape": list(shape)} for k in keys}


def test_assign_flux_clip_t5_vae_in_any_order():
    t5 = (
        C.ComponentFileRef("t5"),
        _hdr("encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight"),
    )
    clip = (C.ComponentFileRef("clip"), _hdr("text_model.encoder.layers.0.mlp.fc1.weight"))
    ae = (C.ComponentFileRef("ae"), _hdr("decoder.up.0.block.0.conv1.weight"))
    assigned, kinds = C.assign_components(
        [t5, clip], ae, C.component_classes_from_index(FLUX_INDEX)
    )
    assert {c: r.spec for c, r in assigned.items()} == {
        "text_encoder_2": "t5",
        "text_encoder": "clip",
        "vae": "ae",
    }
    assert kinds["vae"] == "ldm_kl"


def test_assign_refuses_wrong_family_and_double_slot():
    classes = C.component_classes_from_index(FLUX_INDEX)
    qwen = (C.ComponentFileRef("qwen_3_4b"), _hdr("model.layers.0.self_attn.q_norm.weight"))
    with pytest.raises(C.ComponentFileError, match = "qwen3 text encoder"):
        C.assign_components([qwen], None, classes, family = "flux.1")
    t5 = (
        C.ComponentFileRef("t5"),
        _hdr("encoder.block.0.layer.0.SelfAttention.q.weight", "shared.weight"),
    )
    with pytest.raises(C.ComponentFileError, match = "already taken"):
        C.assign_components([t5, t5], None, classes)
    wan = (C.ComponentFileRef("wan"), _hdr("decoder.upsamples.0.residual.0.gamma"))
    with pytest.raises(C.ComponentFileError, match = "AutoencoderKL"):
        C.assign_components([], wan, classes)


def test_resident_bytes_prices_int8_at_stored_size_and_fp8_dequantized(tmp_path):
    out = {
        "a.weight": torch.zeros(64, 64, dtype = torch.int8),
        "a.weight_scale": torch.ones(64),
        "a.comfy_quant": _quant_blob(
            {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
        ),
        "b.weight": torch.zeros(64, 64, dtype = torch.float8_e4m3fn),
        "b.weight_scale": torch.ones(()),
        "b.comfy_quant": _quant_blob({"format": "float8_e4m3fn"}),
        "c.weight": torch.zeros(64, dtype = torch.bfloat16),
    }
    path = _save(tmp_path / "te.safetensors", out)
    assert C.resident_bytes(path, dtype_itemsize = 2) == (64 * 64 + 64 * 4) + 64 * 64 * 2 + 64 * 2


def test_planner_replaces_scanned_component_with_supplied_file(tmp_path):
    from core.inference.diffusion import DiffusionBackend

    base = tmp_path / "base"
    _save(
        base / "text_encoder" / "model.safetensors",
        {"w": torch.zeros(1024, 1024, dtype = torch.bfloat16)},
    )
    _save(
        base / "vae" / "diffusion_pytorch_model.safetensors",
        {"w": torch.zeros(512, 512, dtype = torch.bfloat16)},
    )
    (base / "model_index.json").write_text("{}")
    supplied = _save(
        tmp_path / "te.safetensors", {"w": torch.zeros(2048, 1024, dtype = torch.bfloat16)}
    )
    ov = C.ComponentOverrides(
        files = {"text_encoder": C.ComponentFileRef(supplied, local_path = supplied)},
        paths = {"text_encoder": supplied},
        classes = {"text_encoder": "Qwen3Model"},
    )
    token = C.set_active_component_overrides(ov)
    try:
        scanned, scanned_te, got, got_te = DiffusionBackend._supplied_component_mib(
            str(base), None, torch.bfloat16
        )
    finally:
        C.reset_active_component_overrides(token)
    assert (scanned, scanned_te, got, got_te) == (2, 2, 4, 4)
    assert DiffusionBackend._supplied_component_mib(str(base), None, torch.bfloat16) is None


def test_te_prequant_kwargs_inject_supplied_and_skip_precast(monkeypatch):
    import core.inference.diffusion_te_prequant as tp

    sentinel = object()
    ov = C.ComponentOverrides(
        files = {"text_encoder": C.ComponentFileRef("x")}, paths = {"text_encoder": "x"}, classes = {}
    )
    ov.modules["text_encoder"] = sentinel
    monkeypatch.setattr(
        tp,
        "te_prequant_sources_for_base",
        lambda *a, **k: {
            "text_encoder": types.SimpleNamespace(kind = "repo", location = "r", filename = "f")
        },
    )

    def _boom(*a, **k):
        raise AssertionError("the pre-cast encoder must not load for a supplied component")

    monkeypatch.setattr(tp, "load_prequant_text_encoder", _boom)
    token = C.set_active_component_overrides(ov)
    try:
        got = tp.te_prequant_pipe_kwargs(
            types.SimpleNamespace(name = "qwen-image-2.1"),
            "base",
            te_quant_mode = "fp8",
            target = None,
            dtype = None,
        )
    finally:
        C.reset_active_component_overrides(token)
    assert got == {"text_encoder": sentinel}
    assert tp.supplied_component_pipe_kwargs("base", dtype = None) == {}


def test_te_quant_auto_does_not_recast_supplied_encoder():
    from core.inference.diffusion import _te_quant_for_supplied_encoders

    assert _te_quant_for_supplied_encoders(None) == "none"
    assert _te_quant_for_supplied_encoders("auto") == "none"
    assert _te_quant_for_supplied_encoders("fp8") == "fp8"


def test_request_model_normalizes_and_validates():
    from models.inference import DiffusionLoadRequest

    req = DiffusionLoadRequest(model_path = "/x", text_encoder_file = " /a.safetensors ", vae_file = " ")
    assert req.text_encoder_file == ["/a.safetensors"] and req.vae_file is None
    assert req.supplied_text_encoder_files() == ["/a.safetensors"]
    with pytest.raises(Exception):
        DiffusionLoadRequest(
            model_path = "/x", text_encoder_file = ["/a.safetensors", "/a.safetensors"]
        )
    assert DiffusionLoadRequest(model_path = "/x").supplied_text_encoder_files() is None


def test_validate_load_request_refuses_pipeline_kind_and_missing_files(tmp_path):
    from core.inference.diffusion import DiffusionBackend

    fam = types.SimpleNamespace(name = "flux.1", single_file_is_pipeline = False, pipeline_only = False)
    with pytest.raises(ValueError, match = "single-file or GGUF"):
        DiffusionBackend._validate_component_files(
            fam, "pipeline", "unsloth/FLUX.1-schnell", ["/x.safetensors"], None
        )
    with pytest.raises(ValueError, match = "does not exist"):
        DiffusionBackend._validate_component_files(
            fam, "gguf", str(tmp_path), [str(tmp_path / "nope.safetensors")], None
        )
    sdxl = types.SimpleNamespace(name = "sdxl", single_file_is_pipeline = True, pipeline_only = False)
    with pytest.raises(ValueError, match = "whole-pipeline"):
        DiffusionBackend._validate_component_files(
            sdxl, "single_file", str(tmp_path), None, "/v.safetensors"
        )
    vae_as_te = _save(
        tmp_path / "ae.safetensors", {"decoder.up.0.block.0.conv1.weight": torch.zeros(2, 2)}
    )
    with pytest.raises(ValueError, match = "pass it as vae_file"):
        DiffusionBackend._validate_component_files(fam, "gguf", str(tmp_path), [vae_as_te], None)


def test_supplied_int8_encoder_reported_as_int8_without_a_request():
    from core.inference.diffusion_precision import quantize_text_encoders

    encoder = torch.nn.Linear(2, 2)
    encoder._unsloth_te_prequant_scheme = "int8"
    pipe = types.SimpleNamespace(text_encoder = encoder)
    outcome = quantize_text_encoders(pipe, types.SimpleNamespace(device = "cpu"), mode = None)
    assert outcome.mode == "int8" and "supplied file" in outcome.reason
    # A dense encoder with no request stays unreported, exactly as before.
    assert (
        quantize_text_encoders(
            types.SimpleNamespace(text_encoder = torch.nn.Linear(2, 2)), None, mode = None
        ).mode
        is None
    )
