# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SDXL's denoiser is pipe.unet, and one single-file safetensors is the whole pipeline."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_lora
from core.inference.diffusion import (
    DiffusionBackend,
    _is_trusted_diffusion_repo,
    resolve_model_kind,
)
from core.inference.diffusion_families import detect_family, family_sd_cpp_supported


def test_sdxl_family_shape():
    fam = detect_family("stabilityai/stable-diffusion-xl-base-1.0")
    assert fam is not None and fam.name == "sdxl"
    assert fam.pipeline_class == "StableDiffusionXLPipeline"
    assert fam.denoiser_attr == "unet"
    assert fam.transformer_class == "UNet2DConditionModel"
    assert fam.single_file_is_pipeline is True
    assert fam.img2img_pipeline_class == "StableDiffusionXLImg2ImgPipeline"
    assert fam.inpaint_pipeline_class == "StableDiffusionXLInpaintPipeline"
    assert fam.controlnet_pipeline_class == "StableDiffusionXLControlNetPipeline"
    assert fam.controlnet_model_class == "ControlNetModel"
    assert fam.cfg_kwarg == "guidance_scale"


def test_sdxl_detection_by_repo_and_override():
    assert detect_family("stabilityai/sdxl-turbo").name == "sdxl"
    assert detect_family("some-org/My-Cool-SDXL-Merge").name == "sdxl"
    assert detect_family("some-org/stable-diffusion-xl-anime").name == "sdxl"
    assert detect_family("x", override = "sdxl").name == "sdxl"
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").name == "flux.1"


def test_dit_families_keep_transformer_denoiser():
    for rid in ("unsloth/FLUX.1-schnell-GGUF", "unsloth/Qwen-Image-GGUF", "unsloth/Z-Image-GGUF"):
        fam = detect_family(rid)
        assert fam.denoiser_attr == "transformer"
        assert fam.single_file_is_pipeline is False


def test_sdxl_has_no_native_sd_cpp_mapping():
    # no single-file VAE/TE mapping yet, so the no-GPU route uses diffusers, not sd-cli
    assert family_sd_cpp_supported(detect_family("stabilityai/sdxl-turbo")) is False


def test_sdxl_base_repos_are_trusted_non_gguf():
    assert _is_trusted_diffusion_repo("stabilityai/stable-diffusion-xl-base-1.0")
    assert _is_trusted_diffusion_repo("stabilityai/sdxl-turbo")
    assert _is_trusted_diffusion_repo("StabilityAI/SDXL-Turbo")
    assert not _is_trusted_diffusion_repo("randomorg/my-sdxl-merge")
    assert not _is_trusted_diffusion_repo("stabilityai/sdxl-turbo-evil")


def test_sdxl_model_kind_resolution():
    assert resolve_model_kind(None) == "pipeline"
    assert resolve_model_kind("sdxl.safetensors") == "single_file"


class _FakeVae:
    def __init__(self, dtype):
        self._dtype = dtype
        self.moved_to = None

    def parameters(self):
        yield types.SimpleNamespace(dtype = self._dtype)

    def to(self, dtype = None):
        self.moved_to = dtype
        self._dtype = dtype


def test_align_vae_dtype_uses_unet_denoiser():
    # the VAE dtype comes from a U-Net parameter, hence the _FakeVae
    import torch

    vae = _FakeVae(dtype = torch.float32)
    unet = _FakeVae(dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(unet = unet, vae = vae)
    DiffusionBackend._align_vae_dtype(pipe, "unet")
    assert vae.moved_to == torch.bfloat16


def test_align_vae_dtype_transformer_default_unchanged():
    import torch

    vae = _FakeVae(dtype = torch.float32)
    transformer = _FakeVae(dtype = torch.bfloat16)
    pipe = types.SimpleNamespace(transformer = transformer, vae = vae)
    DiffusionBackend._align_vae_dtype(pipe)
    assert vae.moved_to == torch.bfloat16
    vae2 = _FakeVae(dtype = torch.float32)
    DiffusionBackend._align_vae_dtype(types.SimpleNamespace(vae = vae2), "unet")
    assert vae2.moved_to is None


def test_align_vae_dtype_skips_gguf_packed_uint8_params():
    # GGUF leading params are packed uint8; probe the first floating dtype or .to() rejects it
    import torch

    class _GgufDenoiser:
        def parameters(self):
            yield types.SimpleNamespace(dtype = torch.uint8)
            yield types.SimpleNamespace(dtype = torch.bfloat16)

    vae = _FakeVae(dtype = torch.float32)
    pipe = types.SimpleNamespace(transformer = _GgufDenoiser(), vae = vae)
    DiffusionBackend._align_vae_dtype(pipe)
    assert vae.moved_to == torch.bfloat16

    class _AllPacked:
        def parameters(self):
            yield types.SimpleNamespace(dtype = torch.uint8)

    vae2 = _FakeVae(dtype = torch.float32)
    DiffusionBackend._align_vae_dtype(types.SimpleNamespace(transformer = _AllPacked(), vae = vae2))
    assert vae2.moved_to is None


def test_sdxl_lora_supported_on_diffusers():
    assert diffusion_lora.supports_lora(
        engine = "diffusers", family = "sdxl", model_kind = "pipeline", transformer_quant = None
    )
    assert diffusion_lora.supports_lora(
        engine = "diffusers", family = "sdxl", model_kind = "single_file", transformer_quant = None
    )


def test_pipeline_prefetch_skips_non_torch_artifacts():
    # SDXL Base ships fp16, ONNX, OpenVINO and Flax exports; the prefetch must skip them
    from core.inference.diffusion import _pipeline_file_downloaded as keep

    assert keep("model_index.json")
    assert keep("unet/diffusion_pytorch_model.safetensors")
    assert keep("text_encoder/model.safetensors")
    assert keep("scheduler/scheduler_config.json")
    assert not keep("sd_xl_base_1.0.safetensors")
    assert not keep("unet/diffusion_pytorch_model.fp16.safetensors")
    assert not keep("text_encoder/model.onnx")
    assert not keep("text_encoder/openvino_model.bin")
    assert not keep("unet/flax_model.msgpack")
    assert not keep("vae_decoder/model.onnx_data")
    assert not keep("assets/preview.png")


def test_sdxl_refiner_not_trusted():
    # the sdxl family loads every repo as base txt2img, so the img2img-only refiner is not trusted
    assert not _is_trusted_diffusion_repo("stabilityai/stable-diffusion-xl-refiner-1.0")
    assert _is_trusted_diffusion_repo("stabilityai/stable-diffusion-xl-base-1.0")
    assert _is_trusted_diffusion_repo("stabilityai/sdxl-turbo")


def test_sdxl_gguf_load_rejected_up_front():
    # SDXL's single file is the whole pipeline, so only an on-disk GGUF carrying all of it loads (#11391); a Hub
    # GGUF request still fails cheap validation before the GPU handoff.
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "also carries the text encoders and VAE"):
        backend.validate_load_request(
            "some-org/my-sdxl.gguf", gguf_filename = "my-sdxl.gguf", family_override = "sdxl"
        )


def test_base_config_filter_skips_weights():
    from core.inference.diffusion import _base_config_file_downloaded as keep

    assert keep("model_index.json")
    assert keep("text_encoder/config.json")
    assert keep("tokenizer/vocab.json")
    assert keep("scheduler/scheduler_config.json")
    assert not keep("unet/diffusion_pytorch_model.safetensors")
    assert not keep("vae/diffusion_pytorch_model.bin")
    assert not keep("text_encoder/model.onnx")
    # from_single_file resolves transformer/config.json off the Hub, so offline loads need it staged
    assert keep("transformer/config.json")
    assert not keep("transformer/diffusion_pytorch_model.safetensors")
    assert not keep("assets/x.png")
