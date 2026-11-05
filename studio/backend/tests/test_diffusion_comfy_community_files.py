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
    pytest.importorskip("diffusers")
    from core.inference.diffusion_anima import _block_count, split_anima_checkpoint

    transformer, adapter = split_anima_checkpoint(_anima_state(prefix, blocks = 3))
    assert sorted(adapter) == ["blocks.0.self_attn.q_proj.weight", "embed.weight"]
    assert "transformer_blocks.2.attn1.to_q.weight" in transformer
    assert "transformer_blocks.0.ff.net.0.proj.weight" in transformer
    assert "patch_embed.proj.weight" in transformer
    assert not any("pos_embedder" in k or "llm_adapter" in k for k in transformer)
    assert _block_count(transformer) == 3


def test_anima_refuses_a_quantized_or_adapterless_file():
    pytest.importorskip("diffusers")
    from core.inference.diffusion_anima import split_anima_checkpoint

    state = _anima_state("net.")
    with pytest.raises(ValueError, match = "ComfyUI-quantised"):
        split_anima_checkpoint(
            {**state, "net.blocks.0.self_attn.q_proj.weight_scale": torch.ones(1)}
        )
    with pytest.raises(ValueError, match = "no llm_adapter"):
        split_anima_checkpoint({k: v for k, v in state.items() if "llm_adapter" not in k})


# diffusers -> Cosmos-Predict2 names: diffusers' converter renames, undone in reverse order.
_TO_COSMOS = (
    ("proj_out", "final_layer.linear"),
    ("norm_out.linear_2", "final_layer.adaln_modulation.2"),
    ("norm_out.linear_1", "final_layer.adaln_modulation.1"),
    ("patch_embed.proj", "x_embedder.proj.1"),
    ("ff.net.2", "mlp.layer2"),
    ("ff.net.0.proj", "mlp.layer1"),
    ("norm_k", "k_norm"),
    ("norm_q", "q_norm"),
    ("to_out.0", "output_proj"),
    ("to_v", "v_proj"),
    ("to_k", "k_proj"),
    ("to_q", "q_proj"),
    ("attn2", "cross_attn"),
    ("attn1", "self_attn"),
    ("norm3.linear_2", "adaln_modulation_mlp.2"),
    ("norm3.linear_1", "adaln_modulation_mlp.1"),
    ("norm2.linear_2", "adaln_modulation_cross_attn.2"),
    ("norm2.linear_1", "adaln_modulation_cross_attn.1"),
    ("norm1.linear_2", "adaln_modulation_self_attn.2"),
    ("norm1.linear_1", "adaln_modulation_self_attn.1"),
    ("transformer_blocks", "blocks"),
    ("time_embed.norm", "t_embedding_norm"),
    ("time_embed.t_embedder", "t_embedder.1"),
)


def _tiny_anima(tmp_path):
    from diffusers import AnimaTextConditioner, CosmosTransformer3DModel

    torch.manual_seed(0)
    transformer = CosmosTransformer3DModel(
        in_channels = 4,
        out_channels = 4,
        num_attention_heads = 2,
        attention_head_dim = 16,
        num_layers = 2,
        mlp_ratio = 2.0,
        text_embed_dim = 16,
        adaln_lora_dim = 4,
        max_size = (2, 8, 8),
        patch_size = (1, 2, 2),
        rope_scale = (1.0, 1.0, 1.0),
        concat_padding_mask = True,
        extra_pos_embed_type = None,
    )
    conditioner = AnimaTextConditioner(
        source_dim = 16,
        target_dim = 16,
        model_dim = 16,
        num_layers = 1,
        num_attention_heads = 2,
        target_vocab_size = 32,
        min_sequence_length = 8,
    )
    base = tmp_path / "base"
    transformer.save_config(base / "transformer")
    conditioner.save_config(base / "text_conditioner")
    return str(base), transformer, conditioner


def _comfy_layout(transformer, conditioner) -> dict:
    state = {}
    for key, value in transformer.state_dict().items():
        for new, old in _TO_COSMOS:
            key = key.replace(new, old)
        state["net." + key] = value
    for key, value in conditioner.state_dict().items():
        state["net.llm_adapter." + key] = value
    state["net.pos_embedder.seq"] = torch.zeros(1)
    return state


def test_anima_single_file_loads_on_meta_and_matches_its_weights(tmp_path, monkeypatch):
    pytest.importorskip("diffusers")
    import core.inference.diffusion_single_file_converters as converters
    from core.inference.diffusion_anima import load_anima_single_file, split_anima_checkpoint

    base, transformer, conditioner = _tiny_anima(tmp_path)
    comfy = _comfy_layout(transformer, conditioner)
    assert set(split_anima_checkpoint(dict(comfy))[0]) == set(transformer.state_dict())
    path = _save(tmp_path / "anima.safetensors", comfy)

    # Built on meta: no fp32 model is initialised before the file's tensors arrive.
    init_devices = set()
    reset = torch.nn.Linear.reset_parameters

    def _reset(self):
        init_devices.add(self.weight.device.type)
        return reset(self)

    charged = []
    monkeypatch.setattr(torch.nn.Linear, "reset_parameters", _reset)
    monkeypatch.setattr(converters, "charge_host_ram", charged.append)
    loaded, loaded_conditioner = load_anima_single_file(
        path, base, torch.bfloat16, local_files_only = True
    )
    assert init_devices == {"meta"}
    # Each fp32 -> bf16 cast is charged to the host RAM budget before it allocates.
    assert sum(charged) == 2 * sum(v.numel() for k, v in comfy.items() if "pos_embedder" not in k)
    for model, ref in ((loaded, transformer), (loaded_conditioner, conditioner)):
        assert not model.training
        got = model.state_dict()
        assert set(got) == set(ref.state_dict())
        for key, value in ref.state_dict().items():
            assert got[key].dtype == torch.bfloat16
            assert torch.equal(got[key], value.to(torch.bfloat16)), key
    # A buffer built at init (the rotary inv_freq) stays as built, as from_pretrained leaves it.
    for (name, buf), (_, ref_buf) in zip(
        loaded_conditioner.named_buffers(), conditioner.named_buffers()
    ):
        assert buf.dtype == ref_buf.dtype and torch.equal(buf, ref_buf), name


class _AnimaScheduler:
    def step(
        self,
        noise_pred,
        t,
        latents,
        return_dict = False,
    ):
        return (latents + 1,)


class _AnimaModular:
    def __init__(self):
        self.scheduler = _AnimaScheduler()
        self.transformer = None

    def __call__(
        self,
        output = None,
        **kwargs,
    ):
        latents = 0
        for i in range(kwargs["num_inference_steps"]):
            latents = self.scheduler.step(None, 1000 - i, latents)[0]
        return [f"img:{kwargs['prompt']}"] * kwargs["num_images_per_prompt"]


def test_every_anima_render_frees_the_offload_hooks():
    # diffusers' contract for enable_model_cpu_offload: else the VAE stays on the GPU into the next image.
    from core.inference.diffusion_anima import _AnimaPipelineMixin

    class Face(_AnimaPipelineMixin):
        def __init__(self):
            self._interrupt = False
            self.mp = _AnimaModular()
            self.freed = 0

        def modular_pipeline(self):
            return self.mp

        def _set_guidance(self, mp, guidance_scale):
            pass

        def maybe_free_model_hooks(self):
            self.freed += 1

    def interrupt(pipe, i, t, kwargs):
        pipe._interrupt = True
        return kwargs

    face = Face()
    assert face(prompt = "a", num_inference_steps = 2).images == ["img:a"]
    assert face.freed == 1
    assert face(prompt = "a", num_inference_steps = 3, callback_on_step_end = interrupt).images == []
    assert face.freed == 2


def test_every_anima_image_is_its_own_generation_chunk():
    # One modular call per image (as FLUX.3), so each image is its own chunk.
    from core.inference.diffusion import _generation_chunks
    from core.inference.diffusion_families import detect_family

    jobs = [("a", 1), ("a", 2), ("b", 3)]
    assert _generation_chunks(detect_family("", override = "anima"), jobs, 1) == [[j] for j in jobs]


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
    assert {"fp8", "int8", "nvfp4", "mxfp8"} <= chroma.profile.scheme_deny
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


def test_a_local_anima_load_gets_the_anima_defaults():
    from core.inference.diffusion import _generation_defaults_for
    from core.inference.diffusion_families import detect_family

    anima = detect_family("", override = "anima")
    local = "/models/Anima-Base-v1.0-Diffusers"
    expected = {"steps": 30, "guidance": 4.0}
    # The defaults row is keyed by circlestone-labs/..., which a local folder never names.
    assert _generation_defaults_for(local, None, local, family = anima) == expected
    assert (
        _generation_defaults_for("/models/comfy", "anima-v1.safetensors", local, family = anima)
        == expected
    )
    sdxl = detect_family("", override = "sdxl")
    assert _generation_defaults_for(
        "/models/sdxl",
        "animagineXL40_v4Opt.safetensors",
        "stabilityai/stable-diffusion-xl-base-1.0",
        family = sdxl,
    ) == {"steps": 25, "guidance": 7.0}


def test_the_comfy_schedule_switches_to_beta_sigmas_unless_opted_out(monkeypatch):
    pytest.importorskip("diffusers")
    pytest.importorskip("scipy")  # diffusers' beta sigmas need it
    from diffusers import FlowMatchEulerDiscreteScheduler

    from core.inference.diffusion_flow_shift import COMFY_SIGMAS_ENV, apply_comfy_schedule

    def schedule():
        pipe = type("P", (), {})()
        pipe.scheduler = FlowMatchEulerDiscreteScheduler(shift = 3.0)
        changed = apply_comfy_schedule(pipe, 1.0, (("use_beta_sigmas", True),))
        return changed, pipe.scheduler.config.shift, pipe.scheduler.config.use_beta_sigmas

    assert schedule() == (True, 1.0, True)
    # The opt-out keeps the shipped scheduler: neither the shift nor the beta spacing.
    monkeypatch.setenv(COMFY_SIGMAS_ENV, "0")
    assert schedule() == (False, 3.0, False)


def test_krea2_pipeline_takes_a_supplied_vae():
    import inspect

    from core.inference.diffusion_krea2 import load_krea2_pipeline
    assert "vae" in inspect.signature(load_krea2_pipeline).parameters


def test_qwen3_5_text_encoder_is_classified_and_fits_moss():
    from core.inference.diffusion_comfy_components import TE_KIND_CLASSES, classify_text_encoder

    keys = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "lm_head.weight",
    ]
    assert classify_text_encoder(keys) == "qwen3_5"
    assert "Qwen3_5Model" in TE_KIND_CLASSES["qwen3_5"]
    # A Qwen2.5-VL encoder (vision tower, no linear attention, no q_norm) keeps its kind.
    assert classify_text_encoder(
        ["model.layers.0.self_attn.q_proj.weight", "visual.blocks.0.attn.qkv.weight"]
    ) == ("qwen2_5_vl")


def test_moss_pipeline_takes_a_supplied_vae():
    pytest.importorskip("diffusers")
    import inspect

    from core.inference.diffusion_moss_image import load_moss_image_pipeline

    assert "vae" in inspect.signature(load_moss_image_pipeline).parameters


def test_a_supplied_moss_vae_keeps_its_fp32_weights(tmp_path):
    pytest.importorskip("diffusers")
    from diffusers import AutoencoderKLFlux2

    from core.inference.diffusion_comfy_components import (
        ComponentFileRef,
        ComponentOverrides,
        load_override_modules,
    )

    torch.manual_seed(0)
    reference = AutoencoderKLFlux2(
        block_out_channels = (32, 32),
        down_block_types = ("DownEncoderBlock2D",) * 2,
        up_block_types = ("UpDecoderBlock2D",) * 2,
        layers_per_block = 1,
        latent_channels = 4,
        norm_num_groups = 8,
    )
    reference.save_config(tmp_path / "base" / "vae")
    path = _save(tmp_path / "vae.safetensors", reference.state_dict())

    def built(family):
        overrides = ComponentOverrides(
            files = {"vae": ComponentFileRef(spec = path, local_path = path)},
            paths = {"vae": path},
            classes = {"vae": "AutoencoderKLFlux2"},
        )
        modules = load_override_modules(
            overrides,
            base = str(tmp_path / "base"),
            dtype = torch.bfloat16,
            family = family,
            local_files_only = True,
        )
        return modules["vae"].state_dict()

    # MOSS-Image decodes in fp32, so its supplied VAE must not be rounded to the pipeline dtype on the way.
    moss = built("moss-image")
    for key, value in reference.state_dict().items():
        assert moss[key].dtype == value.dtype and torch.equal(moss[key], value), key
    other = built("krea-2")
    assert other["decoder.conv_out.weight"].dtype == torch.bfloat16


def test_anima_gguf_is_neither_offered_nor_loaded():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import detect_family, family_gguf_loadable
    from hub.utils.comfy_models import _single_file_loadable

    anima = detect_family("", override = "anima")
    assert not family_gguf_loadable(anima)
    assert not _single_file_loadable(anima, None, "anima-base-v1.0-Q8_0.gguf")
    assert _single_file_loadable(anima, None, "anima-base-v1.0.safetensors")
    with pytest.raises(ValueError, match = "GGUF files are not supported"):
        DiffusionBackend().validate_load_request(
            "someone/Anima-Base-GGUF", gguf_filename = "anima-base-v1.0-Q8_0.gguf", model_kind = "gguf"
        )
    # Chroma keeps its GGUF path (diffusers loads ChromaTransformer2DModel GGUFs).
    assert family_gguf_loadable(detect_family("", override = "chroma"))
