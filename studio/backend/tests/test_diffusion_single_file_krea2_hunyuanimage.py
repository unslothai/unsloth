# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea 2 and HunyuanImage 2.1 original-layout single files (the ComfyUI ``diffusion_models`` files).

diffusers ships both transformers without a single-file converter, so every ComfyUI file of either
family failed before this with "has no single-file converter" (the ComfyUI int8 / fp8 loader) or
"FromOriginalModelMixin is currently only compatible with ..." (plain files).

The stubs below are written from the real files' headers by an inverse map kept here, independent of
the converter's own tables: a tiny diffusers model's weights go out in the original layout and must
come back bit-identical, under exactly the model's keys and shapes. The ComfyUI int8 loader is then
run on a real-format int8 file of the same tiny model, which proves the converter only moves whole
rows (that loader refuses anything else) and that the codes and scales survive exactly.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import pytest
import torch

from core.inference import diffusion as studio
from core.inference import diffusion_comfy_quant as cq
from core.inference import diffusion_single_file_converters as conv
from core.inference.diffusion_families import (
    comfy_flow_shift_for,
    detect_family,
    transformer_config_overrides_for,
    transformer_variant_differs_from_base,
)

diffusers = pytest.importorskip("diffusers")
safetensors_torch = pytest.importorskip("safetensors.torch")

KREA2_CFG = dict(
    in_channels = 8,
    num_layers = 2,
    attention_head_dim = 16,
    num_attention_heads = 2,
    num_key_value_heads = 1,
    intermediate_size = 48,
    timestep_embed_dim = 16,
    text_hidden_dim = 24,
    num_text_layers = 3,
    text_num_attention_heads = 2,
    text_num_key_value_heads = 2,
    text_intermediate_size = 40,
    num_layerwise_text_blocks = 2,
    num_refiner_text_blocks = 2,
    axes_dims_rope = (4, 6, 6),
)
HY_CFG = dict(
    in_channels = 8,
    out_channels = 8,
    num_attention_heads = 2,
    attention_head_dim = 16,
    num_layers = 2,
    num_single_layers = 2,
    num_refiner_layers = 2,
    mlp_ratio = 4.0,
    patch_size = (1, 1),
    text_embed_dim = 24,
    text_embed_2_dim = 20,
    rope_axes_dim = (8, 8),
)
HY_DISTILLED = dict(guidance_embeds = True, use_meanflow = True)


def _class(name):
    cls = getattr(diffusers, name, None)
    if cls is None:
        pytest.skip(f"this diffusers does not ship {name}")
    return cls


def _tiny(name, cfg):
    torch.manual_seed(0)
    model = _class(name)(**cfg)
    # Unique values everywhere, so a misplaced row or a swapped half cannot compare equal by accident.
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p))
    return model.eval()


# ------------------------------------------------- inverse maps: diffusers -> original, from the real headers

_KREA2_INV_BLOCK = {
    "attn.to_q.weight": "attn.wq.weight",
    "attn.to_k.weight": "attn.wk.weight",
    "attn.to_v.weight": "attn.wv.weight",
    "attn.to_out.0.weight": "attn.wo.weight",
    "attn.to_gate.weight": "attn.gate.weight",
    "attn.norm_q.weight": "attn.qknorm.qnorm.scale",
    "attn.norm_k.weight": "attn.qknorm.knorm.scale",
    "ff.gate.weight": "mlp.gate.weight",
    "ff.up.weight": "mlp.up.weight",
    "ff.down.weight": "mlp.down.weight",
    "norm1.weight": "prenorm.scale",
    "norm2.weight": "postnorm.scale",
}
_KREA2_INV_TOP = {
    "img_in.weight": "first.weight",
    "img_in.bias": "first.bias",
    "time_embed.linear_1.weight": "tmlp.0.weight",
    "time_embed.linear_1.bias": "tmlp.0.bias",
    "time_embed.linear_2.weight": "tmlp.2.weight",
    "time_embed.linear_2.bias": "tmlp.2.bias",
    "time_mod_proj.weight": "tproj.1.weight",
    "time_mod_proj.bias": "tproj.1.bias",
    "txt_in.norm.weight": "txtmlp.0.scale",
    "txt_in.linear_1.weight": "txtmlp.1.weight",
    "txt_in.linear_1.bias": "txtmlp.1.bias",
    "txt_in.linear_2.weight": "txtmlp.3.weight",
    "txt_in.linear_2.bias": "txtmlp.3.bias",
    "text_fusion.projector.weight": "txtfusion.projector.weight",
    "final_layer.linear.weight": "last.linear.weight",
    "final_layer.linear.bias": "last.linear.bias",
    "final_layer.norm.weight": "last.norm.scale",
    "final_layer.scale_shift_table": "last.modulation.lin",
}


def krea2_original(sd: dict) -> dict:
    out = {}
    for key, value in sd.items():
        if key in _KREA2_INV_TOP:
            out[_KREA2_INV_TOP[key]] = value.clone()
            continue
        m = re.match(
            r"^(transformer_blocks|text_fusion\.layerwise_blocks|text_fusion\.refiner_blocks)\.(\d+)\.(.+)$",
            key,
        )
        assert m, key
        group, index, rest = m.groups()
        stem = {
            "transformer_blocks": "blocks",
            "text_fusion.layerwise_blocks": "txtfusion.layerwise_blocks",
            "text_fusion.refiner_blocks": "txtfusion.refiner_blocks",
        }[group]
        if rest == "scale_shift_table":
            out[f"{stem}.{index}.mod.lin"] = value.reshape(
                -1
            ).clone()  # stored flat in the real files
        else:
            out[f"{stem}.{index}.{_KREA2_INV_BLOCK[rest]}"] = value.clone()
    return out


def hy_original(sd: dict, hidden: int, comfy_names: bool) -> dict:
    """Reference-repo naming (Comfy-Org's bf16 file) or ComfyUI naming under ``model.model.`` (its fp8 and
    distilled files)."""
    out: dict = {}

    def put(name, value):
        out[name] = value.clone()

    def ref_or_comfy(ref, comfy):
        return comfy if comfy_names else ref

    mlp_in, mlp_out = ref_or_comfy("fc1", "0"), ref_or_comfy("fc2", "2")
    lin = ref_or_comfy("linear", "lin")
    used = set()

    def take(key):
        used.add(key)
        return sd[key]

    def emb(dst, src):
        a, b = ref_or_comfy(("mlp.0", "mlp.2"), ("in_layer", "out_layer"))
        for p in ("weight", "bias"):
            put(f"{dst}.{a}.{p}", take(f"{src}.linear_1.{p}"))
            put(f"{dst}.{b}.{p}", take(f"{src}.linear_2.{p}"))

    for p in ("weight", "bias"):
        put(f"img_in.proj.{p}", take(f"x_embedder.proj.{p}"))
        put(f"txt_in.input_embedder.{p}", take(f"context_embedder.proj_in.{p}"))
        put(f"byt5_in.layernorm.{p}", take(f"context_embedder_2.norm.{p}"))
        for i in (1, 2, 3):
            put(f"byt5_in.fc{i}.{p}", take(f"context_embedder_2.linear_{i}.{p}"))
        put(f"final_layer.linear.{p}", take(f"proj_out.{p}"))
        scale, shift = take(f"norm_out.linear.{p}").chunk(2, dim = 0)
        put(f"final_layer.adaLN_modulation.1.{p}", torch.cat([shift, scale]))
    emb("time_in", "time_guidance_embed.timestep_embedder")
    if "time_guidance_embed.guidance_embedder.linear_1.weight" in sd:
        emb("guidance_in", "time_guidance_embed.guidance_embedder")
    if "time_guidance_embed.timestep_embedder_r.linear_1.weight" in sd:
        emb("time_r_in", "time_guidance_embed.timestep_embedder_r")
    emb("txt_in.t_embedder", "context_embedder.time_text_embed.timestep_embedder")
    # c_embedder has its own naming in the reference repo: linear_1 / linear_2
    for p in ("weight", "bias"):
        a, b = ref_or_comfy(("linear_1", "linear_2"), ("in_layer", "out_layer"))
        put(
            f"txt_in.c_embedder.{a}.{p}",
            take(f"context_embedder.time_text_embed.text_embedder.linear_1.{p}"),
        )
        put(
            f"txt_in.c_embedder.{b}.{p}",
            take(f"context_embedder.time_text_embed.text_embedder.linear_2.{p}"),
        )

    n_ref = len(
        {
            k.split(".")[3]
            for k in sd
            if k.startswith("context_embedder.token_refiner.refiner_blocks.")
        }
    )
    for i in range(n_ref):
        s, d = (
            f"context_embedder.token_refiner.refiner_blocks.{i}.",
            f"txt_in.individual_token_refiner.blocks.{i}.",
        )
        qkv, proj = ref_or_comfy(
            ("self_attn_qkv", "self_attn_proj"), ("self_attn.qkv", "self_attn.proj")
        )
        for p in ("weight", "bias"):
            put(f"{d}norm1.{p}", take(f"{s}norm1.{p}"))
            put(f"{d}norm2.{p}", take(f"{s}norm2.{p}"))
            put(f"{d}{qkv}.{p}", torch.cat([take(f"{s}attn.to_{x}.{p}") for x in "qkv"]))
            put(f"{d}{proj}.{p}", take(f"{s}attn.to_out.0.{p}"))
            put(f"{d}mlp.{mlp_in}.{p}", take(f"{s}ff.net.0.proj.{p}"))
            put(f"{d}mlp.{mlp_out}.{p}", take(f"{s}ff.net.2.{p}"))
            put(f"{d}adaLN_modulation.1.{p}", take(f"{s}norm_out.linear.{p}"))

    n_double = len({k.split(".")[1] for k in sd if k.startswith("transformer_blocks.")})
    for i in range(n_double):
        s, d = f"transformer_blocks.{i}.", f"double_blocks.{i}."
        for side, q, k, v, o, nq, nk, mod, ff in (
            ("img", "to_q", "to_k", "to_v", "to_out.0", "norm_q", "norm_k", "norm1", "ff"),
            (
                "txt",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
                "norm_added_q",
                "norm_added_k",
                "norm1_context",
                "ff_context",
            ),
        ):
            attn_qkv = ref_or_comfy(f"{side}_attn_qkv", f"{side}_attn.qkv")
            attn_proj = ref_or_comfy(f"{side}_attn_proj", f"{side}_attn.proj")
            for p in ("weight", "bias"):
                put(f"{d}{attn_qkv}.{p}", torch.cat([take(f"{s}attn.{x}.{p}") for x in (q, k, v)]))
                put(f"{d}{attn_proj}.{p}", take(f"{s}attn.{o}.{p}"))
                put(f"{d}{side}_mod.{lin}.{p}", take(f"{s}{mod}.linear.{p}"))
                put(f"{d}{side}_mlp.{mlp_in}.{p}", take(f"{s}{ff}.net.0.proj.{p}"))
                put(f"{d}{side}_mlp.{mlp_out}.{p}", take(f"{s}{ff}.net.2.{p}"))
            put(
                ref_or_comfy(
                    f"{d}{side}_attn_q_norm.weight", f"{d}{side}_attn.norm.query_norm.scale"
                ),
                take(f"{s}attn.{nq}.weight"),
            )
            put(
                ref_or_comfy(
                    f"{d}{side}_attn_k_norm.weight", f"{d}{side}_attn.norm.key_norm.scale"
                ),
                take(f"{s}attn.{nk}.weight"),
            )

    n_single = len({k.split(".")[1] for k in sd if k.startswith("single_transformer_blocks.")})
    for i in range(n_single):
        s, d = f"single_transformer_blocks.{i}.", f"single_blocks.{i}."
        for p in ("weight", "bias"):
            put(
                f"{d}linear1.{p}",
                torch.cat(
                    [take(f"{s}attn.to_{x}.{p}") for x in "qkv"] + [take(f"{s}proj_mlp.{p}")]
                ),
            )
            put(f"{d}linear2.{p}", take(f"{s}proj_out.{p}"))
            put(f"{d}modulation.{lin}.{p}", take(f"{s}norm.linear.{p}"))
        put(
            ref_or_comfy(f"{d}q_norm.weight", f"{d}norm.query_norm.scale"),
            take(f"{s}attn.norm_q.weight"),
        )
        put(
            ref_or_comfy(f"{d}k_norm.weight", f"{d}norm.key_norm.scale"),
            take(f"{s}attn.norm_k.weight"),
        )

    assert used == set(sd), sorted(set(sd) - used)[:5]  # the inverse covers the whole model
    if comfy_names:
        out = {"model.model." + k: v for k, v in out.items()}
    return out


def _assert_round_trip(model, converted):
    want = model.state_dict()
    assert set(converted) == set(want), (
        sorted(set(want) - set(converted))[:5],
        sorted(set(converted) - set(want))[:5],
    )
    for key, value in want.items():
        assert tuple(converted[key].shape) == tuple(value.shape), key
        assert torch.equal(converted[key], value), key
    # And the model takes it strictly.
    model.load_state_dict(converted, strict = True)


# ------------------------------------------------------------------------------------------ key maps, strict


def test_krea2_original_layout_round_trips_bit_identical():
    model = _tiny("Krea2Transformer2DModel", KREA2_CFG)
    original = krea2_original(model.state_dict())
    # The real files' top-level names, from Comfy-Org/Krea-2's headers.
    assert {
        "first.weight",
        "tproj.1.weight",
        "txtmlp.3.bias",
        "last.modulation.lin",
        "blocks.0.mod.lin",
    } <= set(original)
    assert original["blocks.0.mod.lin"].dim() == 1
    _assert_round_trip(model, conv.krea2_checkpoint_to_diffusers(checkpoint = original))


def test_krea2_container_prefix_is_stripped():
    model = _tiny("Krea2Transformer2DModel", KREA2_CFG)
    original = {
        "model.diffusion_model." + k: v for k, v in krea2_original(model.state_dict()).items()
    }
    _assert_round_trip(model, conv.krea2_checkpoint_to_diffusers(checkpoint = original))


@pytest.mark.parametrize("comfy_names", [False, True], ids = ["reference_names", "comfyui_names"])
@pytest.mark.parametrize("distilled", [False, True], ids = ["base", "distilled"])
def test_hunyuanimage_original_layouts_round_trip_bit_identical(comfy_names, distilled):
    cfg = {**HY_CFG, **(HY_DISTILLED if distilled else {})}
    model = _tiny("HunyuanImageTransformer2DModel", cfg)
    hidden = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    original = hy_original(model.state_dict(), hidden, comfy_names)
    _assert_round_trip(
        model, conv.hunyuanimage_checkpoint_to_diffusers(checkpoint = original, config = cfg)
    )
    # Without a config the hidden size is read off img_in.
    _assert_round_trip(model, conv.hunyuanimage_checkpoint_to_diffusers(checkpoint = original))


def test_hunyuanimage_final_modulation_halves_swap():
    """The one reorder: the original final layer holds (shift, scale), diffusers' norm_out reads (scale, shift)."""
    shift, scale = torch.full((4, 3), 1.0), torch.full((4, 3), 2.0)
    out = conv.hunyuanimage_checkpoint_to_diffusers(
        checkpoint = {
            "img_in.proj.bias": torch.zeros(8),
            "final_layer.adaLN_modulation.1.weight": torch.cat([shift, scale]),
        }
    )
    assert torch.equal(out["norm_out.linear.weight"], torch.cat([scale, shift]))


def test_unknown_keys_are_refused_not_dropped():
    with pytest.raises(ValueError, match = "not a Krea 2 transformer key"):
        conv.krea2_checkpoint_to_diffusers(
            checkpoint = {"blocks.0.attn.something.weight": torch.zeros(2, 2)}
        )
    with pytest.raises(ValueError, match = "not a HunyuanImage 2.1 transformer key"):
        conv.hunyuanimage_checkpoint_to_diffusers(
            checkpoint = {
                "img_in.proj.bias": torch.zeros(8),
                "double_blocks.0.img_attn.extra.weight": torch.zeros(2, 2),
            }
        )
    with pytest.raises(ValueError, match = "cannot be split"):
        conv.hunyuanimage_checkpoint_to_diffusers(
            checkpoint = {
                "img_in.proj.bias": torch.zeros(8),
                "double_blocks.0.img_attn.qkv.weight": torch.zeros(23, 8),
            }
        )


# --------------------------------------------------------------------------------- row-only, on every weight


@pytest.mark.parametrize("family", ["krea2", "hy_ref", "hy_comfy_distilled"])
def test_every_2d_weight_is_a_whole_row_gather(family):
    """What the ComfyUI int8 / fp8 loader needs: each output row is exactly one source row of one layer,
    every source row lands exactly once, and no column moves. Checked with that loader's own decoder."""
    if family == "krea2":
        model = _tiny("Krea2Transformer2DModel", KREA2_CFG)
        original = krea2_original(model.state_dict())
        fn = conv.krea2_checkpoint_to_diffusers
    else:
        distilled = family == "hy_comfy_distilled"
        model = _tiny(
            "HunyuanImageTransformer2DModel", {**HY_CFG, **(HY_DISTILLED if distilled else {})}
        )
        original = hy_original(model.state_dict(), 32, comfy_names = distilled)
        fn = conv.hunyuanimage_checkpoint_to_diffusers
    weights = sorted(k for k, v in original.items() if v.dim() == 2)
    sources = []
    tagged = dict(original)
    for index, key in enumerate(weights):
        rows, cols = original[key].shape
        codes = torch.zeros(rows, cols, dtype = torch.int8)
        sources.append((key, codes, None))
        tag = torch.arange(rows, dtype = torch.float64) + index * cq._TAG
        tagged[key] = tag.view(-1, 1).expand(rows, cols)
    out = fn(checkpoint = tagged)
    seen = {}
    for name, value in out.items():
        if value.dtype != torch.float64:
            continue
        for i, r, n in cq._decode_rows(name, value, sources):
            for row in range(r, r + n):
                assert (
                    i,
                    row,
                ) not in seen, f"{name}: row {row} of {weights[i]} also went to {seen[(i, row)]}"
                seen[(i, row)] = name
    assert len(seen) == sum(int(original[k].shape[0]) for k in weights)


def _int8_rows(weight, group = None):
    scale = weight.abs().amax(dim = 1, keepdim = True).clamp_min(1e-8) / 127.0
    return torch.round(weight / scale).clamp(-127, 127).to(torch.int8), scale.to(torch.float32)


def test_comfy_int8_krea2_file_loads_with_codes_exact(tmp_path, monkeypatch):
    """A Krea 2 int8_tensorwise file in the real ComfyUI layout through Studio's ComfyUI loader with the
    real converter: every int8 layer converts (the loader refuses a non-row-only converter), and each
    dequantized diffusers weight equals its own source codes times its own scales."""
    cls = _class("Krea2Transformer2DModel")
    model = _tiny("Krea2Transformer2DModel", KREA2_CFG)
    original = krea2_original(model.state_dict())
    tensors, expected = {}, {}
    inv = {v: k for k, v in _KREA2_INV_BLOCK.items()}
    for key, value in original.items():
        if re.match(r"^blocks\.\d+\.(attn\.(wq|wk|wv|wo|gate)|mlp\.(gate|up|down))\.weight$", key):
            codes, scale = _int8_rows(value)
            stem = key[: -len(".weight")]
            tensors[key] = codes
            tensors[stem + ".weight_scale"] = scale
            tensors[stem + ".comfy_quant"] = torch.tensor(
                list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype = torch.uint8
            )
            index, rest = key.split(".", 2)[1], key.split(".", 2)[2]
            expected[f"transformer_blocks.{index}.{inv[rest]}"] = codes.float() * scale
        else:
            tensors[key] = value.contiguous()
    path = tmp_path / "krea2_tiny_int8.safetensors"
    safetensors_torch.save_file(tensors, str(path))
    monkeypatch.setattr(cls, "load_config", classmethod(lambda c, *a, **k: dict(KREA2_CFG)))

    scan = cq.refuse_comfy_quant(str(path))
    loaded = cq.load_comfy_quant_transformer(
        cls,
        str(path),
        scan,
        {"torch_dtype": torch.float32, "config": "krea/Krea-2-Turbo", "subfolder": "transformer"},
        int8_backend = None,
        family = "krea-2",
    )
    assert len(expected) == 2 * 8
    sd = loaded.state_dict()
    for name, weight in expected.items():
        assert torch.equal(sd[name], weight), name
    assert loaded._unsloth_comfy_quant["dequantized"] == len(expected)


def test_plain_krea2_file_loads_like_the_base_repo(tmp_path, monkeypatch):
    """A plain bf16 ComfyUI file through Studio's loader (used while diffusers gives Krea 2 no
    from_single_file): same tensors as the diffusers weights, the _keep_in_fp32_modules norms in float32
    as from_pretrained leaves them, strict on keys."""
    cls = _class("Krea2Transformer2DModel")
    model = _tiny("Krea2Transformer2DModel", KREA2_CFG)
    path = tmp_path / "krea2_tiny_bf16.safetensors"
    safetensors_torch.save_file(
        {
            k: v.to(torch.bfloat16).contiguous()
            for k, v in krea2_original(model.state_dict()).items()
        },
        str(path),
    )
    monkeypatch.setattr(cls, "load_config", classmethod(lambda c, *a, **k: dict(KREA2_CFG)))
    loaded = conv.load_original_layout_transformer(
        cls,
        str(path),
        {"torch_dtype": torch.bfloat16, "config": "krea/Krea-2-Turbo", "subfolder": "transformer"},
    )
    keep = cls._keep_in_fp32_modules
    for key, value in model.state_dict().items():
        got = loaded.state_dict()[key]
        fp32 = any(m in key.split(".") for m in keep)
        assert got.dtype == (torch.float32 if fp32 else torch.bfloat16), key
        assert torch.equal(got, value.to(torch.bfloat16).to(got.dtype)), key

    bad = tmp_path / "krea2_extra.safetensors"
    tensors = {k: v.contiguous() for k, v in krea2_original(model.state_dict()).items()}
    del tensors["blocks.0.attn.wq.weight"]
    safetensors_torch.save_file(tensors, str(bad))
    with pytest.raises(ValueError, match = "1 missing"):
        conv.load_original_layout_transformer(cls, str(bad), {"config": "krea/Krea-2-Turbo"})


def test_the_single_file_branch_uses_studio_loader_only_without_from_single_file():
    import inspect

    # Whitespace-free, so a formatter rewrapping the condition cannot break the check.
    source = "".join(inspect.getsource(studio).split())
    gate = source.index('ifkind!="gguf"andnothasattr(transformer_cls,"from_single_file"):')
    assert (
        gate
        < source.index("transformer=load_original_layout_transformer(", gate)
        < source.index("transformer=transformer_cls.from_single_file(", gate)
    )


# ------------------------------------------------------------------------------------------- family wiring


def test_both_classes_register_and_resolve(monkeypatch):
    from diffusers.loaders import single_file_model as sfm

    monkeypatch.setattr(sfm, "SINGLE_FILE_LOADABLE_CLASSES", dict(sfm.SINGLE_FILE_LOADABLE_CLASSES))
    for name in ("Krea2Transformer2DModel", "HunyuanImageTransformer2DModel"):
        _class(name)
        sfm.SINGLE_FILE_LOADABLE_CLASSES.pop(name, None)
    added = studio._register_unregistered_single_file_classes()
    assert {"Krea2Transformer2DModel", "HunyuanImageTransformer2DModel"} <= set(added)
    assert sfm.SINGLE_FILE_LOADABLE_CLASSES["Krea2Transformer2DModel"]["checkpoint_mapping_fn"] is (
        conv.krea2_checkpoint_to_diffusers
    )
    # The ComfyUI loader's lookup, which raised "has no single-file converter" before.
    fn, _ = cq._mapping(diffusers.HunyuanImageTransformer2DModel)
    assert fn is conv.hunyuanimage_checkpoint_to_diffusers


@pytest.mark.parametrize(
    "filename, family",
    [
        ("krea2_turbo_int8_convrot.safetensors", "krea-2"),
        ("krea2_turbo_fp8_scaled.safetensors", "krea-2"),
        ("krea2_turbo_bf16.safetensors", "krea-2"),
        ("hunyuanimage2.1_bf16.safetensors", "hunyuanimage-2.1"),
        ("hunyuanimage2.1_fp8_e4m3fn.safetensors", "hunyuanimage-2.1"),
        ("hunyuanimage2.1_distilled_fp8_e4m3fn.safetensors", "hunyuanimage-2.1"),
    ],
)
def test_comfy_file_names_detect_their_family(filename, family):
    fam = detect_family(f"/models/ComfyUI/models/diffusion_models/{filename}")
    assert fam is not None and fam.name == family


def test_hunyuanimage_distilled_file_gets_its_own_config_and_shift():
    fam = detect_family("hunyuanimage-2.1")
    base = fam.base_repo
    distilled = "hunyuanimage2.1_distilled_fp8_e4m3fn.safetensors"
    plain = "hunyuanimage2.1_fp8_e4m3fn.safetensors"
    assert transformer_config_overrides_for(fam, distilled, base) == HY_DISTILLED
    assert transformer_config_overrides_for(fam, plain, base) == {}
    assert transformer_variant_differs_from_base(fam, base, distilled)
    assert not transformer_variant_differs_from_base(fam, base, plain)
    assert comfy_flow_shift_for(fam, distilled, base) == 4.0
    assert comfy_flow_shift_for(fam, plain, base) is None  # the shipped shift-5 scheduler stays


def test_distilled_single_file_drops_the_base_cfg_guiders():
    """A guidance-distilled transformer on the base repo must not inherit its two CFG guiders."""
    import inspect

    source = inspect.getsource(studio)
    gate = source.index('fam.name == "hunyuanimage-2.1" and getattr(')
    block = source[gate : gate + 800]
    assert '"guidance_embeds"' in block
    assert 'pipe_kwargs["guider"] = None' in block and 'pipe_kwargs["ocr_guider"] = None' in block
    assert gate < source.index("pipe = pipeline_cls.from_pretrained(\n", gate)


# ------------------------------------------------------------------ the real files' headers, when cached

# The suite isolates HF_HUB_CACHE, so the real files are opted into by naming a hub cache that holds them.
_REAL_HUB_ENV = "UNSLOTH_TEST_REAL_HF_HUB_CACHE"


def _real(pattern):
    import os

    root = os.environ.get(_REAL_HUB_ENV)
    hits = sorted(Path(root).glob(pattern)) if root else []
    if not hits:
        pytest.skip(f"set {_REAL_HUB_ENV} to a hub cache holding {pattern}")
    return hits[0]


def _header_stub(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F16": torch.float16,
        "F8_E4M3": torch.float8_e4m3fn,
        "I8": torch.int8,
        "U8": torch.uint8,
    }
    layers = {k[: -len(".comfy_quant")] for k in header if k.endswith(".comfy_quant")}
    stub = {}
    for key, meta in header.items():
        if key.endswith((".comfy_quant", ".weight_scale", ".input_scale")) or key == "scaled_fp8":
            continue
        dtype = torch.float64 if key[: -len(".weight")] in layers else dtypes[meta["dtype"]]
        stub[key] = torch.empty(meta["shape"], dtype = dtype, device = "meta")
    return stub


def _meta_model(cls, cfg):
    from accelerate import init_empty_weights
    with init_empty_weights():
        return cls.from_config(cfg)


@pytest.mark.parametrize(
    "pattern",
    [
        "models--Comfy-Org--Krea-2/snapshots/*/diffusion_models/krea2_turbo_int8_convrot.safetensors",
        "models--Comfy-Org--Krea-2/snapshots/*/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        "models--Comfy-Org--Krea-2/snapshots/*/diffusion_models/krea2_turbo_bf16.safetensors",
        "models--Comfy-Org--HunyuanImage_2.1_ComfyUI/snapshots/*/split_files/diffusion_models/hunyuanimage2.1_bf16.safetensors",
        "models--Comfy-Org--HunyuanImage_2.1_ComfyUI/snapshots/*/split_files/diffusion_models/hunyuanimage2.1_fp8_e4m3fn.safetensors",
        "models--Comfy-Org--HunyuanImage_2.1_ComfyUI/snapshots/*/split_files/diffusion_models/hunyuanimage2.1_distilled_fp8_e4m3fn.safetensors",
    ],
)
def test_real_comfy_headers_convert_strictly(pattern):
    path = _real(pattern)
    if "Krea-2" in pattern:
        cls, cfg, fn = _class("Krea2Transformer2DModel"), {}, conv.krea2_checkpoint_to_diffusers
    else:
        cls, fn = (
            _class("HunyuanImageTransformer2DModel"),
            conv.hunyuanimage_checkpoint_to_diffusers,
        )
        cfg = dict(
            text_embed_2_dim = 1472,
            rope_axes_dim = (64, 64),
            **(HY_DISTILLED if "distilled" in pattern else {}),
        )
    model = _meta_model(cls, cfg)
    converted = fn(checkpoint = _header_stub(path), config = dict(model.config))
    want = model.state_dict()
    assert set(converted) == set(want)
    assert all(tuple(converted[k].shape) == tuple(want[k].shape) for k in want)
