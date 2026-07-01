# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the SDXL diffusion family.

SDXL is the one U-Net family: the denoiser is ``pipe.unet`` (not ``pipe.transformer``)
and a single-file ``.safetensors`` is the whole pipeline (not a transformer-only file).
These tests cover the pure helpers that encode those differences -- family detection,
the ``denoiser_attr`` / ``single_file_is_pipeline`` flags, the non-GGUF trust allowlist,
the VAE-dtype alignment reading the U-Net denoiser, and the LoRA-support gate -- with no
torch/diffusers/GPU needed.
"""

from __future__ import annotations

import types

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
    # The denoiser is a U-Net, addressed via pipe.unet (DiT families use pipe.transformer).
    assert fam.denoiser_attr == "unet"
    assert fam.transformer_class == "UNet2DConditionModel"
    # A single-file SDXL checkpoint is the whole pipeline, loaded via the pipeline class.
    assert fam.single_file_is_pipeline is True
    # Image-conditioned + ControlNet workflows are the standard SDXL pipelines.
    assert fam.img2img_pipeline_class == "StableDiffusionXLImg2ImgPipeline"
    assert fam.inpaint_pipeline_class == "StableDiffusionXLInpaintPipeline"
    assert fam.controlnet_pipeline_class == "StableDiffusionXLControlNetPipeline"
    assert fam.controlnet_model_class == "ControlNetModel"
    # Real CFG; SDXL uses guidance_scale, not a distilled true_cfg_scale.
    assert fam.cfg_kwarg == "guidance_scale"


def test_sdxl_detection_by_repo_and_override():
    assert detect_family("stabilityai/sdxl-turbo").name == "sdxl"
    assert detect_family("some-org/My-Cool-SDXL-Merge").name == "sdxl"
    assert detect_family("some-org/stable-diffusion-xl-anime").name == "sdxl"
    assert detect_family("x", override = "sdxl").name == "sdxl"
    # A GGUF DiT family must NOT be swallowed by the SDXL match.
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").name == "flux.1"


def test_dit_families_keep_transformer_denoiser():
    # The generalisation must not change existing DiT families: they stay on
    # pipe.transformer and their single file is transformer-only.
    for rid in ("unsloth/FLUX.1-schnell-GGUF", "unsloth/Qwen-Image-GGUF", "unsloth/Z-Image-GGUF"):
        fam = detect_family(rid)
        assert fam.denoiser_attr == "transformer"
        assert fam.single_file_is_pipeline is False


def test_sdxl_has_no_native_sd_cpp_mapping():
    # No single-file VAE/TE mapping yet, so the no-GPU route falls back to diffusers
    # rather than trying to drive sd-cli.
    assert family_sd_cpp_supported(detect_family("stabilityai/sdxl-turbo")) is False


def test_sdxl_base_repos_are_trusted_non_gguf():
    # Official safetensors-only base repos are allowlisted so their catalog entries load.
    assert _is_trusted_diffusion_repo("stabilityai/stable-diffusion-xl-base-1.0")
    assert _is_trusted_diffusion_repo("stabilityai/sdxl-turbo")
    assert _is_trusted_diffusion_repo("stabilityai/stable-diffusion-xl-refiner-1.0")
    # Case-insensitive match.
    assert _is_trusted_diffusion_repo("StabilityAI/SDXL-Turbo")
    # A random repo (even one that detects as SDXL) is NOT trusted for a non-GGUF load.
    assert not _is_trusted_diffusion_repo("randomorg/my-sdxl-merge")
    assert not _is_trusted_diffusion_repo("stabilityai/sdxl-turbo-evil")


def test_sdxl_model_kind_resolution():
    # A full-pipeline load (no single-file name) is "pipeline"; a single .safetensors
    # is "single_file" (handled by the whole-pipeline branch for SDXL).
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
    # For SDXL the denoiser lives at pipe.unet; _align_vae_dtype must read it (a pipe
    # with only .unet and no .transformer) and cast the VAE to the U-Net's dtype.
    vae = _FakeVae(dtype = "float32")
    pipe = types.SimpleNamespace(unet = types.SimpleNamespace(dtype = "bfloat16"), vae = vae)
    DiffusionBackend._align_vae_dtype(pipe, "unet")
    assert vae.moved_to == "bfloat16"


def test_align_vae_dtype_transformer_default_unchanged():
    # DiT default: reads pipe.transformer; a pipe with no transformer is a safe no-op.
    vae = _FakeVae(dtype = "float32")
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = "bfloat16"), vae = vae)
    DiffusionBackend._align_vae_dtype(pipe)
    assert vae.moved_to == "bfloat16"
    # No denoiser attribute -> no-op (does not raise, does not move the VAE).
    vae2 = _FakeVae(dtype = "float32")
    DiffusionBackend._align_vae_dtype(types.SimpleNamespace(vae = vae2), "unet")
    assert vae2.moved_to is None


def test_sdxl_lora_supported_on_diffusers():
    # SDXL is bf16/bnb-4bit on diffusers -> LoRA is allowed (unlike GGUF-via-diffusers).
    assert diffusion_lora.supports_lora(
        engine = "diffusers", family = "sdxl", model_kind = "pipeline", transformer_quant = None
    )
    assert diffusion_lora.supports_lora(
        engine = "diffusers", family = "sdxl", model_kind = "single_file", transformer_quant = None
    )


def test_pipeline_prefetch_skips_non_torch_artifacts():
    # The SDXL Base repo ships fp16 variants, ONNX, OpenVINO and Flax exports next to
    # the default safetensors; from_pretrained (no variant kwarg) loads only the
    # default torch weights, so the prefetch filter must skip everything else or a
    # catalog load pulls tens of GB of unused artifacts.
    from core.inference.diffusion import _pipeline_file_downloaded as keep

    assert keep("model_index.json")
    assert keep("unet/diffusion_pytorch_model.safetensors")
    assert keep("text_encoder/model.safetensors")
    assert keep("scheduler/scheduler_config.json")
    assert not keep("sd_xl_base_1.0.safetensors")  # top-level single-file twin
    assert not keep("unet/diffusion_pytorch_model.fp16.safetensors")
    assert not keep("text_encoder/model.onnx")
    assert not keep("text_encoder/openvino_model.bin")
    assert not keep("unet/flax_model.msgpack")
    assert not keep("vae_decoder/model.onnx_data")
    assert not keep("assets/preview.png")
