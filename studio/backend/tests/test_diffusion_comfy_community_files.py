# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI / Civitai single files of Anima, Chroma and MOSS-Image, and ComfyUI's ConvRot W4A4 int4 layout."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

import core.inference.diffusion_comfy_quant as cq  # noqa: E402
from core.inference.diffusion_convrot import build_convrot_hadamard  # noqa: E402


def _conf(**conf) -> torch.Tensor:
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype = torch.uint8)


def _save(
    path,
    tensors: dict,
    metadata: dict | None = None,
) -> str:
    safetensors_torch.save_file(
        {k: v.contiguous() for k, v in tensors.items()}, str(path), metadata = metadata
    )
    return str(path)


# ComfyUI ConvRot W4A4: rotate, symmetric per-row int4 (max 7), two codes per byte, low nibble first.
def _w4a4(weight: torch.Tensor, group: int = 256) -> tuple:
    h = build_convrot_hadamard(group)
    w = (weight.float().reshape(weight.shape[0], -1, group) @ h.T).reshape(weight.shape)
    scale = w.abs().amax(dim = 1, keepdim = True).clamp_min(1e-10) / 7.0
    q = torch.round(w / scale).clamp(-7, 7).to(torch.int32)
    packed = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).to(torch.int8)
    return packed, scale.reshape(-1), q.to(torch.int8)


def test_convrot_w4a4_loads_as_int8_convrot(tmp_path):
    weight = torch.randn(16, 512)
    packed, scale, codes = _w4a4(weight)
    path = _save(
        tmp_path / "w4.safetensors",
        {
            "a.weight": packed,
            "a.weight_scale": scale,
            "a.comfy_quant": _conf(
                format = "convrot_w4a4", convrot_groupsize = 256, linear_dtype = "int4"
            ),
        },
    )
    scan = cq.scan_comfy_quant(path)
    assert scan.problems == []
    layer = scan.layers["a"]
    assert (layer.format, layer.convrot, layer.group, layer.packed_int4) == (
        "int8_tensorwise",
        True,
        256,
        True,
    )
    unpacked = cq.unpack_int4_rows(packed)
    assert torch.equal(unpacked, codes)
    restored = cq._dequant(unpacked, scale, 256, torch.float32)
    # int4 rounding error only: at most half a step per element, in the rotated basis.
    assert (restored - weight).abs().max() <= scale.max() * 0.5 * 16**0.5 + 1e-5
    assert cq.comfy_quant_error(scan) is None


def test_convrot_w4a4_refuses_a_malformed_layer(tmp_path):
    path = _save(
        tmp_path / "bad.safetensors",
        {
            "a.weight": torch.zeros(
                8, 100, dtype = torch.int8
            ),  # K = 200: not a multiple of the group
            "a.weight_scale": torch.ones(8),
            "a.comfy_quant": _conf(format = "convrot_w4a4", convrot_groupsize = 256),
        },
    )
    with pytest.raises(ValueError, match = "not divisible by ConvRot group 256"):
        cq.refuse_comfy_quant(path)


def test_declarations_without_the_container_prefix_match_their_weights(tmp_path):
    w8 = torch.ones(8, 512, dtype = torch.int8)
    path = _save(
        tmp_path / "p.safetensors",
        {
            "model.diffusion_model.a.weight": w8,
            "model.diffusion_model.a.weight_scale": torch.ones(8, 1),
            "a.comfy_quant": _conf(format = "int8_tensorwise", convrot = True, convrot_groupsize = 256),
        },
    )
    scan = cq.scan_comfy_quant(path)
    assert scan.problems == []
    assert list(scan.layers) == ["model.diffusion_model.a"]


def _anima_state(prefix: str, blocks: int = 2) -> dict:
    t = lambda *s: torch.zeros(*s)  # noqa: E731
    state = {
        f"{prefix}x_embedder.proj.1.weight": t(4, 68),
        f"{prefix}t_embedder.1.linear_1.weight": t(4, 4),
        f"{prefix}t_embedding_norm.weight": t(4),
        f"{prefix}final_layer.linear.weight": t(4, 4),
        f"{prefix}llm_adapter.embed.weight": t(8, 4),
        f"{prefix}llm_adapter.blocks.0.self_attn.q_proj.weight": t(4, 4),
        f"{prefix}pos_embedder.dim_spatial_range": t(1),
    }
    for i in range(blocks):
        state[f"{prefix}blocks.{i}.self_attn.q_proj.weight"] = t(4, 4)
        state[f"{prefix}blocks.{i}.mlp.layer1.weight"] = t(4, 4)
    return state


@pytest.mark.parametrize("prefix", ["net.", "model.diffusion_model."])
def test_anima_file_splits_into_transformer_and_text_conditioner(prefix):
    from core.inference.diffusion_anima import _block_count, split_anima_checkpoint

    transformer, adapter = split_anima_checkpoint(_anima_state(prefix, blocks = 3))
    assert sorted(adapter) == ["blocks.0.self_attn.q_proj.weight", "embed.weight"]
    assert "transformer_blocks.2.attn1.to_q.weight" in transformer
    assert "transformer_blocks.0.ff.net.0.proj.weight" in transformer
    assert "patch_embed.proj.weight" in transformer
    assert not any("pos_embedder" in k or "llm_adapter" in k for k in transformer)
    assert _block_count(transformer) == 3


def test_anima_refuses_a_quantized_or_adapterless_file():
    from core.inference.diffusion_anima import split_anima_checkpoint

    state = _anima_state("net.")
    with pytest.raises(ValueError, match = "ComfyUI-quantised"):
        split_anima_checkpoint(
            {**state, "net.blocks.0.self_attn.q_proj.weight_scale": torch.ones(1)}
        )
    with pytest.raises(ValueError, match = "no llm_adapter"):
        split_anima_checkpoint({k: v for k, v in state.items() if "llm_adapter" not in k})


def _shapes(keys) -> dict:
    return {k: [4, 4] for k in keys}


def test_headers_name_anima_chroma_and_moss():
    from core.inference.diffusion_content import classify_tensors

    anima = classify_tensors(_shapes(_anima_state("net.")))
    assert (anima.family, anima.layout) == ("anima", "comfy")
    civitai_anima = classify_tensors(_shapes(_anima_state("model.diffusion_model.")))
    assert civitai_anima.family == "anima"
    chroma = classify_tensors(
        _shapes(
            [
                "double_blocks.0.img_attn.qkv.weight",
                "single_blocks.0.linear1.weight",
                "distilled_guidance_layer.in_proj.weight",
                "img_in.weight",
                "txt_in.weight",
                "final_layer.linear.weight",
            ]
        )
    )
    assert (chroma.family, chroma.layout) == ("chroma", "comfy")
    radiance = classify_tensors(
        _shapes(
            [
                "double_blocks.0.img_attn.qkv.weight",
                "single_blocks.0.linear1.weight",
                "distilled_guidance_layer.in_proj.weight",
                "nerf_image_embedder.embedder.0.weight",
            ]
        )
    )
    assert radiance.family is None
    moss = classify_tensors(
        _shapes(
            [
                "model.diffusion_model.layers.0.proj.qkv.weight",
                "model.diffusion_model.text_embedder.proj.weight",
                "model.diffusion_model.shared_block_modulation.weight",
                "model.diffusion_model.video_embedder.weight",
                "model.diffusion_model.video_head_modulation.weight",
            ]
        )
    )
    assert moss.family == "moss-image"


def test_moss_converter_splits_fused_qkv_and_rebuilds_rope_bands():
    from core.inference.diffusion_single_file_converters import moss_image_checkpoint_to_diffusers

    qkv = torch.arange(9 * 2, dtype = torch.float32).reshape(9, 2)
    out = moss_image_checkpoint_to_diffusers(
        checkpoint = {
            "model.diffusion_model.layers.0.proj.qkv.weight": qkv,
            "model.diffusion_model.layers.0.proj.o.weight": torch.ones(3, 3),
            "model.diffusion_model.layers.0.comfy_attention.config": torch.zeros(
                3, dtype = torch.uint8
            ),
        },
        config = {"dim": 3072, "num_heads": 24, "rope_base": 10000.0},
    )
    assert torch.equal(out["layers.0.proj.q.weight"], qkv[:3])
    assert torch.equal(out["layers.0.proj.v.weight"], qkv[6:])
    assert "layers.0.proj.o.weight" in out
    assert not any("comfy_attention" in k for k in out)
    bands = 3072 // 24 // 8
    expected = 1 / (10000.0 ** (torch.arange(bands, dtype = torch.float32) / bands))
    assert torch.allclose(out["rope.bands"], expected)


def test_family_entries():
    from core.inference.diffusion_families import default_generation_params, detect_family

    chroma = detect_family("", override = "chroma")
    assert chroma.pipeline_class == "ChromaPipeline"
    assert chroma.comfy_flow_shift == 1.0
    assert dict(chroma.scheduler_overrides) == {"use_beta_sigmas": True}
    assert "fp8" in chroma.profile.scheme_deny
    anima = detect_family("", override = "anima")
    assert anima.pipeline_class == "AnimaModularPipeline"
    assert default_generation_params("oneObsession_anima29BV2.safetensors", anima.base_repo) == (
        30,
        4.0,
    )
    # "anima" is a substring of Animagine XL, an SDXL checkpoint: its defaults stay SDXL's.
    assert default_generation_params(
        "animagineXL40_v4Opt.safetensors", "stabilityai/stable-diffusion-xl-base-1.0"
    ) == (25, 7.0)


def test_scheduler_overrides_switch_to_beta_sigmas():
    from diffusers import FlowMatchEulerDiscreteScheduler

    from core.inference.diffusion_flow_shift import (
        apply_comfy_flow_shift,
        apply_scheduler_overrides,
    )

    pipe = type("P", (), {})()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift = 3.0)
    apply_comfy_flow_shift(pipe, 1.0)
    assert apply_scheduler_overrides(pipe, (("use_beta_sigmas", True),))
    assert pipe.scheduler.config.shift == 1.0
    assert pipe.scheduler.config.use_beta_sigmas is True


def test_krea2_pipeline_takes_a_supplied_vae():
    import inspect

    from core.inference.diffusion_krea2 import load_krea2_pipeline
    assert "vae" in inspect.signature(load_krea2_pipeline).parameters
