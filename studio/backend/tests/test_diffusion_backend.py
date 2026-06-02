# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the diffusion image-generation backend.

These tests cover the surface area the routes layer relies on:

* family detection from the public Unsloth GGUF naming conventions
* generation argument validation (empty prompt, bad steps, off-grid sizes)
* base64 PNG encoding round-trips
* status() shape stays compatible with the frontend status poller
* load/unload lifecycle with the heavy diffusers import monkey-patched

Real GPU loads are exercised manually via the Studio probe (see
``studio/backend/tests/test_diffusion_smoke.py``); here we keep the
suite CPU- and import-free so the consolidated CI job and the
``unslothai/unsloth`` CI fork can both run it on Ubuntu, macOS, and
Windows runners with no diffusion dependencies installed.
"""

from __future__ import annotations

import base64
import io
import logging
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

if "structlog" not in sys.modules:
    sys.modules["structlog"] = types.SimpleNamespace(
        get_logger = lambda *a, **k: logging.getLogger("structlog.stub"),
    )
if "loggers" not in sys.modules:
    sys.modules["loggers"] = types.SimpleNamespace(
        get_logger = lambda *a, **k: logging.getLogger("loggers.stub"),
    )

_OBSERVED_UNSLOTH_GGUF_QUANT_NAMES = (
    "BF16",
    "F16",
    "F32",
    "IQ1_M",
    "IQ1_S",
    "IQ2_S",
    "IQ2_XS",
    "IQ2_XXS",
    "IQ3_S",
    "IQ3_XXS",
    "IQ4_NL",
    "IQ4_XS",
    "Q2_K",
    "Q3_K",
    "Q4_0",
    "Q4_1",
    "Q4_K",
    "Q5_0",
    "Q5_1",
    "Q5_K",
    "Q6_K",
    "Q8_0",
)
_OBSERVED_NATIVE_FALLBACK_QUANT_NAMES = frozenset(
    {
        "IQ1_M",
        "IQ1_S",
        "IQ2_S",
        "IQ2_XS",
        "IQ2_XXS",
        "IQ3_S",
        "IQ3_XXS",
    }
)
_OBSERVED_DIFFUSERS_GGUF_QUANT_NAMES = frozenset(_OBSERVED_UNSLOTH_GGUF_QUANT_NAMES) - {
    "F16",
    "F32",
    *_OBSERVED_NATIVE_FALLBACK_QUANT_NAMES,
}


# ── module under test ────────────────────────────────────────────


@pytest.fixture(autouse = True)
def _reset_singleton(monkeypatch):
    """Reset the module-level singleton between tests so each test
    starts from a known state without poking globals directly."""
    import core.inference.diffusion as d

    monkeypatch.setattr(d, "_singleton", None)
    yield


# ── family detection ────────────────────────────────────────────


def test_detect_family_flux2_klein():
    from core.inference.diffusion import detect_family

    fam = detect_family("unsloth/FLUX.2-klein-4B-GGUF")
    assert fam is not None
    assert fam.name == "flux.2-klein"
    assert fam.pipeline_class == "Flux2KleinPipeline"
    assert fam.transformer_class == "Flux2Transformer2DModel"
    # Family default base must point to a real Hub repo (not the bare
    # "FLUX.2-klein" slug that does not exist). The common Unsloth 4B
    # GGUF path is distilled, so the default companion repo and default
    # steps should match the official 4-step distilled settings.
    assert fam.base_repo == "black-forest-labs/FLUX.2-klein-4B"
    assert fam.default_steps == 4
    assert fam.default_guidance_scale == 1.0


def test_detect_family_flux2_dev_is_not_klein():
    from core.inference.diffusion import detect_family

    fam = detect_family("unsloth/FLUX.2-dev-GGUF")
    assert fam is not None
    assert fam.name == "flux.2"
    # Critical: FLUX.2 dev must NOT pick up the FLUX.2 klein pipeline
    # because the transformer architectures and text encoder
    # configurations are different.
    assert fam.pipeline_class == "Flux2Pipeline"


def test_detect_family_flux1():
    from core.inference.diffusion import detect_family

    fam = detect_family("city96/FLUX.1-dev-gguf")
    assert fam is not None
    assert fam.name == "flux.1"
    assert fam.pipeline_class == "FluxPipeline"


def test_detect_family_flux1_variants():
    from core.inference.diffusion import detect_family

    cases = {
        "unsloth/FLUX.1-Kontext-dev-GGUF": ("flux.1-kontext", "FluxKontextPipeline"),
        "unsloth/FLUX.1-schnell-GGUF": ("flux.1-schnell", "FluxPipeline"),
    }
    for repo, (family, pipeline) in cases.items():
        fam = detect_family(repo)
        assert fam is not None, repo
        assert fam.name == family
        assert fam.pipeline_class == pipeline


def test_detect_family_qwen_image():
    from core.inference.diffusion import detect_family

    fam = detect_family("unsloth/Qwen-Image-GGUF")
    assert fam is not None
    assert fam.name == "qwen-image"


def test_detect_family_override_wins_over_substring():
    from core.inference.diffusion import detect_family

    fam = detect_family("unsloth/FLUX.2-dev-GGUF", override_family = "flux.1")
    assert fam is not None
    assert fam.name == "flux.1"


def test_detect_family_override_unknown_returns_none():
    from core.inference.diffusion import detect_family

    fam = detect_family("unsloth/FLUX.2-klein-4B-GGUF", override_family = "doesnotexist")
    assert fam is None


def test_detect_family_unknown_returns_none():
    from core.inference.diffusion import detect_family

    assert detect_family("random/repo") is None
    assert detect_family("") is None


def test_detect_family_sd35_is_not_sd3():
    """SD3.5 must NOT be matched as SD3 Medium. Pairing SD3.5 GGUFs
    with the Medium base produces a misleading load."""
    from core.inference.diffusion import detect_family

    assert detect_family("unsloth/SD3.5-large-GGUF") is None
    assert detect_family("unsloth/stable-diffusion-3.5-large-GGUF") is None


def test_detect_family_qwen_image_edit_is_not_qwen_image():
    """Qwen-Image-Edit must map to its own image-to-image family,
    not the base Qwen-Image text-to-image family."""
    from core.inference.diffusion import detect_family

    fam = detect_family("unsloth/Qwen-Image-Edit-GGUF")
    assert fam is not None
    assert fam.name == "qwen-image-edit"
    fam = detect_family("unsloth/Qwen-Image-Edit-2509-GGUF")
    assert fam is not None
    assert fam.name == "qwen-image-edit-2509"
    # Underscore spellings on the Hub must also be excluded; otherwise
    # qwen_image_edit-GGUF silently matches the base Qwen-Image family.
    fam = detect_family("unsloth/qwen_image_edit-GGUF")
    assert fam is not None
    assert fam.name == "qwen-image-edit"
    fam = detect_family("unsloth/QwenImageEdit-GGUF")
    assert fam is not None
    assert fam.name == "qwen-image-edit"


def test_detect_family_finds_new_image_families():
    from core.inference.diffusion import detect_family

    cases = {
        "unsloth/Qwen-Image-2512-GGUF": "qwen-image-2512",
        "unsloth/Qwen-Image-Edit-2511-GGUF": "qwen-image-edit-2511",
        "unsloth/Qwen-Image-Layered-GGUF": "qwen-image-layered",
        "unsloth/Z-Image-GGUF": "z-image",
        "unsloth/Z-Image-Turbo-GGUF": "z-image-turbo",
        "unsloth/ERNIE-Image-GGUF": "ernie-image",
        "unsloth/ERNIE-Image-Turbo-GGUF": "ernie-image-turbo",
    }
    for repo, expected in cases.items():
        fam = detect_family(repo)
        assert fam is not None, repo
        assert fam.name == expected


def test_detect_family_finds_full_repo_sdxl():
    """SDXL lives in _FULL_REPO_FAMILIES, but the auto-detector must
    still find it for ``stabilityai/stable-diffusion-xl-base-1.0`` so
    the Custom HF repo entry point does not fail with 'Could not infer
    a diffusion family' for the canonical SDXL repo."""
    from core.inference.diffusion import detect_family

    fam = detect_family("stabilityai/stable-diffusion-xl-base-1.0")
    assert fam is not None
    assert fam.name == "stable-diffusion-xl"
    fam2 = detect_family("nerijs/sdxl-lora-test")
    assert fam2 is not None
    assert fam2.name == "stable-diffusion-xl"


def test_detect_family_finds_video_full_repo_families():
    from core.inference.diffusion import detect_family

    fam = detect_family("diffusers/LTX-2.3-Distilled-Diffusers")
    assert fam is not None
    assert fam.name == "ltx2-3-distilled"
    assert fam.media_kind == "video"
    assert fam.pipeline_class == "LTX2Pipeline"
    assert fam.default_width == 768
    assert fam.default_height == 512
    assert fam.default_num_frames == 121
    assert fam.default_frame_rate == 24.0

    wan = detect_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    assert wan is not None
    assert wan.name == "wan2-2-t2v"
    assert wan.media_kind == "video"
    assert wan.pipeline_class == "WanPipeline"
    assert wan.default_width == 1280
    assert wan.default_height == 720
    assert wan.default_num_frames == 81


def test_supported_families_payload_shape():
    from core.inference.diffusion import supported_families

    payload = supported_families()
    assert isinstance(payload, list)
    assert len(payload) >= 4
    by_name = {entry["name"]: entry for entry in payload}
    assert by_name["ltx2-3-distilled"]["media_kind"] == "video"
    assert by_name["wan2-2-t2v"]["media_kind"] == "video"
    assert by_name["flux.2"]["media_kind"] == "image"
    for entry in payload:
        assert set(entry.keys()) == {
            "name",
            "pipeline_class",
            "base_repo",
            "media_kind",
            "guidance_kwarg",
            "default_steps",
            "default_guidance_scale",
            "default_width",
            "default_height",
            "default_num_frames",
            "default_frame_rate",
            "requires_image_input",
            "supports_gguf_single_file",
        }
        assert entry["media_kind"] in {"image", "video"}
        assert isinstance(entry["default_steps"], int)
        assert isinstance(entry["default_guidance_scale"], float)
        assert isinstance(entry["default_width"], int)
        assert isinstance(entry["default_height"], int)
        assert entry["default_num_frames"] is None or isinstance(
            entry["default_num_frames"], int
        )
        assert entry["default_frame_rate"] is None or isinstance(
            entry["default_frame_rate"], float
        )
        assert isinstance(entry["requires_image_input"], bool)
        assert isinstance(entry["supports_gguf_single_file"], bool)


def test_supported_optimization_options_payload_shape():
    from core.inference.diffusion import supported_optimization_options

    payload = supported_optimization_options()
    assert set(payload.keys()) == {
        "recommended_defaults",
        "offload_policies",
        "safetensors_quantizations",
        "safetensors_quantization_components",
        "compile",
    }

    recommended = payload["recommended_defaults"]
    assert recommended["gguf_image"]["offload_policy"] == "balanced"
    assert recommended["gguf_image"]["compile_dequant"] is True
    assert recommended["gguf_image"]["use_balanced_cuda_cache"] is True
    assert recommended["safetensors_image"]["safetensors_quantization"] == "none"
    assert recommended["safetensors_image"]["enable_model_cpu_offload"] is False
    assert recommended["safetensors_image"]["torch_compile"] == "regional"
    assert (
        recommended["safetensors_low_vram"]["safetensors_quantization"]
        == "bitsandbytes_4bit_nf4"
    )
    assert recommended["safetensors_low_vram"][
        "safetensors_quantization_components"
    ] == ["transformer", "unet"]
    assert recommended["safetensors_low_vram"]["torch_compile"] == "regional"
    assert "safetensors_quality_quantized" not in recommended
    assert recommended["denoiser_torch_compile"]["default_enabled"] is True
    assert recommended["denoiser_torch_compile"]["default_scope"] == "regional"
    assert recommended["denoiser_torch_compile"]["default_fullgraph"] is True
    assert recommended["denoiser_torch_compile"]["default_dynamic"] is True
    assert recommended["denoiser_torch_compile"]["default_when"] == [
        "safetensors_bf16",
        "safetensors_bitsandbytes_4bit_nf4",
    ]
    assert recommended["denoiser_torch_compile"]["default_disabled_when"] == [
        "gguf_dequant_on_the_fly",
        "safetensors_torchao",
    ]
    assert recommended["group_offload"]["image_default"] is False
    assert recommended["group_offload"]["media_kind"] == "video"

    offload_by_name = {
        entry["name"]: entry for entry in payload["offload_policies"]
    }
    assert offload_by_name["balanced"]["media_kind"] == "image"
    assert offload_by_name["balanced"]["keeps_gguf_weights_cpu_resident"] is True
    assert offload_by_name["aggressive"]["uses_diffusers_model_cpu_offload"] is True
    assert offload_by_name["hybrid"]["alias_of"] == "less_aggressive"
    assert offload_by_name["none"]["keeps_gguf_weights_cpu_resident"] is False

    quant_by_name = {
        entry["name"]: entry for entry in payload["safetensors_quantizations"]
    }
    assert quant_by_name["bitsandbytes_4bit_nf4"]["backend"] == "bitsandbytes"
    assert quant_by_name["bitsandbytes_4bit_nf4"]["requires"] == ["bitsandbytes"]
    assert quant_by_name["torchao_int8_weight_only"]["requires"] == ["torchao"]
    assert quant_by_name["torchao_int4_weight_only"]["requires"] == [
        "torchao",
        "mslk",
    ]
    assert payload["safetensors_quantization_components"] == [
        "transformer",
        "unet",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "pe",
    ]

    compile_options = payload["compile"]
    assert compile_options["gguf_balanced_dequant_compile"]["default_enabled"] is True
    cache_options = compile_options["gguf_balanced_cuda_cache"]
    assert cache_options["default_enabled"] is True
    assert cache_options["env_override"] == "UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB"
    assert cache_options["free_memory_headroom_mib"] == 8 * 1024
    assert cache_options["tiers"][-1] == {
        "min_total_vram_mib": 96 * 1024,
        "max_total_vram_mib": None,
        "cache_budget_mib": 16 * 1024,
    }
    assert cache_options["status_counters"] == {
        "modules": "gguf_prepared_module_counts.diffusion_cuda_cache_modules",
        "budget_mib": "gguf_prepared_module_counts.diffusion_cuda_cache_budget_mib",
        "candidate_mib": (
            "gguf_prepared_module_counts.diffusion_cuda_cache_candidate_mib"
        ),
        "selected_mib": (
            "gguf_prepared_module_counts.diffusion_cuda_cache_selected_mib"
        ),
    }
    assert compile_options["denoiser_torch_compile"]["default_enabled"] is True
    assert compile_options["denoiser_torch_compile"]["default_scope"] == "regional"
    assert compile_options["denoiser_torch_compile"]["default_fullgraph"] is True
    assert compile_options["denoiser_torch_compile"]["default_dynamic"] is True
    assert compile_options["denoiser_torch_compile"]["recommended_for"] == [
        "safetensors_bf16",
        "safetensors_bf16_long_session",
        "safetensors_bitsandbytes_4bit_nf4",
        "safetensors_bitsandbytes_4bit_nf4_long_session",
    ]
    assert compile_options["denoiser_torch_compile"]["not_recommended_for"] == [
        "gguf_dequant_on_the_fly",
        "safetensors_torchao",
    ]
    assert compile_options["group_offload"]["image_default"] is False
    assert compile_options["group_offload"]["media_kind"] == "video"


# ── singleton ───────────────────────────────────────────────────


def test_get_diffusion_backend_singleton():
    from core.inference.diffusion import get_diffusion_backend

    a = get_diffusion_backend()
    b = get_diffusion_backend()
    assert a is b


# ── status() shape ──────────────────────────────────────────────


def test_status_shape_unloaded():
    """Public status() (the browser-facing payload) must NOT contain
    the guard-only ``active_*`` / ``pending_*`` fields (round 16
    P1 #5)."""
    from core.inference.diffusion import get_diffusion_backend

    s = get_diffusion_backend().status()
    expected_keys = {
        "is_loaded",
        "is_loading",
        "repo_id",
        "pipeline_repo",
        "family",
        "pipeline_class",
        "media_kind",
        "base_repo",
        "base_repo_source",
        "base_repo_confidence",
        "base_repo_variant",
        "base_repo_warning",
        "sampling_contract",
        "gguf_filename",
        "transformer_gguf_repo",
        "transformer_gguf_filename",
        "text_encoder_gguf_repo",
        "text_encoder_gguf_filename",
        "prompt_enhancer_gguf_repo",
        "prompt_enhancer_gguf_filename",
        "lora",
        "component_sources",
        "gguf_quantized_cpu_resident",
        "gguf_pin_cpu_resident",
        "offload_policy",
        "gguf_execution_backend",
        "gguf_prepared_module_counts",
        "safetensors_quantization",
        "safetensors_quantization_components",
        "load_timings",
        "device",
        "dtype",
        "loaded_at",
        "last_error",
        "supported_families",
        "optimization_options",
    }
    assert expected_keys.issubset(s.keys())
    # Guard-facing fields are gated behind include_internal=True.
    for guard_key in (
        "active_repo_id",
        "active_diffusion_gguf_repo",
        "active_base_repo",
        "active_base_repo_source",
        "active_base_repo_confidence",
        "active_base_repo_variant",
        "active_base_repo_warning",
        "active_gguf_filename",
        "active_text_encoder_gguf_repo",
        "active_text_encoder_gguf_filename",
        "active_prompt_enhancer_gguf_repo",
        "active_prompt_enhancer_gguf_filename",
        "active_lora_repo",
        "active_lora_weight_name",
        "pending_repo_id",
        "pending_diffusion_gguf_repo",
        "pending_base_repo",
        "pending_base_repo_source",
        "pending_base_repo_confidence",
        "pending_base_repo_variant",
        "pending_base_repo_warning",
        "pending_gguf_filename",
        "pending_text_encoder_gguf_repo",
        "pending_text_encoder_gguf_filename",
        "pending_prompt_enhancer_gguf_repo",
        "pending_prompt_enhancer_gguf_filename",
        "pending_lora_repo",
        "pending_lora_weight_name",
    ):
        assert guard_key not in s, f"public status() must not expose {guard_key}"
    assert s["is_loaded"] is False
    assert s["repo_id"] is None
    assert s["media_kind"] is None

    # Internal status() exposes the guard fields for delete/route use.
    s_internal = get_diffusion_backend().status(include_internal = True)
    assert s_internal["active_diffusion_gguf_repo"] is None
    assert s_internal["pending_diffusion_gguf_repo"] is None
    assert s_internal["active_gguf_filename"] is None
    assert s_internal["pending_gguf_filename"] is None
    assert s_internal["active_text_encoder_gguf_filename"] is None
    assert s_internal["pending_text_encoder_gguf_filename"] is None
    assert s_internal["active_prompt_enhancer_gguf_filename"] is None
    assert s_internal["pending_prompt_enhancer_gguf_filename"] is None
    assert s_internal["active_lora_repo"] is None
    assert s_internal["pending_lora_repo"] is None


# ── encode_png_base64 ───────────────────────────────────────────


def test_encode_png_base64_round_trip():
    from PIL import Image

    from core.inference.diffusion import encode_png_base64

    img = Image.new("RGB", (16, 16), color = (255, 0, 0))
    b64 = encode_png_base64(img)
    raw = base64.b64decode(b64)
    decoded = Image.open(io.BytesIO(raw))
    assert decoded.format == "PNG"
    assert decoded.size == (16, 16)


# ── generation validation (no real pipeline) ────────────────────


def _stub_pipeline(monkeypatch, *, returns = None, raises = None):
    """Mount a fake torch pipeline on the singleton so generate_image's
    argument validation runs without diffusers / torch being involved."""
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()

    class _StubPipe:
        def __call__(self, **kwargs):
            if raises is not None:
                raise raises

            class _Out:
                pass

            o = _Out()
            o.images = [
                returns
                or Image.new(
                    "RGB", (kwargs["width"], kwargs["height"]), color = (0, 255, 0)
                )
            ]
            return o

    backend._pipe = _StubPipe()
    backend._device = "cpu"
    backend._family = d._FAMILIES[0]
    backend._repo_id = "stub/stub"
    return backend


def test_generate_image_rejects_empty_prompt(monkeypatch):
    backend = _stub_pipeline(monkeypatch)
    with pytest.raises(ValueError, match = "prompt is empty"):
        backend.generate_image(prompt = "   ")


def test_generate_image_rejects_bad_steps(monkeypatch):
    backend = _stub_pipeline(monkeypatch)
    with pytest.raises(ValueError, match = "num_inference_steps"):
        backend.generate_image(prompt = "cat", num_inference_steps = 0)
    with pytest.raises(ValueError, match = "num_inference_steps"):
        backend.generate_image(prompt = "cat", num_inference_steps = 999)


def test_generate_image_rejects_off_grid_size(monkeypatch):
    backend = _stub_pipeline(monkeypatch)
    with pytest.raises(ValueError, match = "multiples of 8"):
        backend.generate_image(prompt = "cat", width = 513, height = 512)


def test_generate_image_rejects_oversized(monkeypatch):
    backend = _stub_pipeline(monkeypatch)
    with pytest.raises(ValueError, match = "width and height"):
        backend.generate_image(prompt = "cat", width = 4096, height = 512)


def test_generate_image_calls_pipeline_with_kwargs(monkeypatch):
    backend = _stub_pipeline(monkeypatch)
    img = backend.generate_image(
        prompt = "a red sphere",
        negative_prompt = "blue",
        num_inference_steps = 4,
        guidance_scale = 1.0,
        width = 256,
        height = 256,
        seed = 42,
    )
    assert img.size == (256, 256)


def test_generate_image_low_vram_gguf_drains_cuda_cache(monkeypatch):
    import core.inference.diffusion as d

    backend = _stub_pipeline(monkeypatch)
    backend._device = "cuda"
    backend._gguf_quantized_cpu_resident = True
    calls: list[str] = []
    monkeypatch.setattr(d, "_drain_cuda_cache", lambda: calls.append("drain"))

    backend.generate_image(prompt = "a red sphere", width = 256, height = 256)

    assert calls == ["drain"]


def test_generate_image_device_resident_gguf_keeps_cuda_cache(monkeypatch):
    import core.inference.diffusion as d

    backend = _stub_pipeline(monkeypatch)
    backend._device = "cuda"
    backend._gguf_quantized_cpu_resident = False
    calls: list[str] = []
    monkeypatch.setattr(d, "_drain_cuda_cache", lambda: calls.append("drain"))

    backend.generate_image(prompt = "a red sphere", width = 256, height = 256)

    assert calls == []


def test_generate_image_reuses_prompt_embedding_cache(monkeypatch):
    import torch
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    fam = next(f for f in d._FAMILIES if f.name == "z-image")

    class _PromptCachePipe:
        _execution_device = "cpu"

        def __init__(self):
            self.encode_calls = 0
            self.calls: list[dict[str, Any]] = []

        def encode_prompt(
            self,
            *,
            prompt,
            negative_prompt = None,
            do_classifier_free_guidance = True,
            device = None,
            max_sequence_length = 512,
        ):
            self.encode_calls += 1
            assert prompt == "same prompt"
            assert negative_prompt is None
            assert do_classifier_free_guidance is True
            assert max_sequence_length == 512
            return [torch.ones(1, 2, device = device)], [
                torch.zeros(1, 2, device = device)
            ]

        def __call__(
            self,
            *,
            prompt = None,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            num_inference_steps,
            width,
            height,
            guidance_scale = None,
            generator = None,
        ):
            self.calls.append(
                {
                    "prompt": prompt,
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "generator": generator,
                    "guidance_scale": guidance_scale,
                }
            )

            class _Out:
                pass

            out = _Out()
            out.images = [Image.new("RGB", (width, height), color = (0, 255, 0))]
            return out

    pipe = _PromptCachePipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = fam
    backend._repo_id = "stub/z-image"

    backend.generate_image(
        prompt = "same prompt",
        width = 256,
        height = 256,
        guidance_scale = 4,
        seed = 1,
    )
    backend.generate_image(
        prompt = "same prompt",
        width = 256,
        height = 256,
        guidance_scale = 4,
        seed = 2,
    )

    assert pipe.encode_calls == 1
    assert len(pipe.calls) == 2
    assert all(call["prompt"] is None for call in pipe.calls)
    assert all(call["prompt_embeds"][0].device.type == "cpu" for call in pipe.calls)
    assert all(
        call["negative_prompt_embeds"][0].device.type == "cpu"
        for call in pipe.calls
    )
    assert backend._prompt_embedding_cache_value is not None
    assert backend._prompt_embedding_cache_value[0][0].device.type == "cpu"


def test_prompt_embedding_cache_unwraps_flux2_klein_auxiliary_text_ids(monkeypatch):
    import torch
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    fam = next(f for f in d._FAMILIES if f.name == "flux.2-klein")

    class _Flux2KleinPromptCachePipe:
        _execution_device = "cpu"

        def __init__(self):
            self.encode_calls = 0
            self.prompt_embeds_seen = []

        def encode_prompt(
            self,
            prompt,
            device = None,
            num_images_per_prompt = 1,
            prompt_embeds = None,
            max_sequence_length = 512,
            text_encoder_out_layers = (9, 18, 27),
        ):
            self.encode_calls += 1
            embeds = torch.ones(1, 2, 3, device = device)
            text_ids = torch.zeros(1, 2, 4, device = device)
            return embeds, text_ids

        def __call__(
            self,
            *,
            prompt = None,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            num_inference_steps,
            width,
            height,
            guidance_scale = None,
            generator = None,
        ):
            assert prompt is None
            assert hasattr(prompt_embeds, "shape")
            assert negative_prompt_embeds is None
            self.prompt_embeds_seen.append(prompt_embeds)

            class _Out:
                pass

            out = _Out()
            out.images = [Image.new("RGB", (width, height), color = (0, 255, 0))]
            return out

    pipe = _Flux2KleinPromptCachePipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = fam
    backend._repo_id = "stub/flux2-klein"

    backend.generate_image(
        prompt = "same prompt",
        width = 256,
        height = 256,
        guidance_scale = 1,
        seed = 1,
    )
    backend.generate_image(
        prompt = "same prompt",
        width = 256,
        height = 256,
        guidance_scale = 1,
        seed = 2,
    )

    assert pipe.encode_calls == 1
    assert len(pipe.prompt_embeds_seen) == 2
    assert backend._prompt_embedding_cache_value is not None
    assert hasattr(backend._prompt_embedding_cache_value[0], "shape")


def test_prompt_embedding_cache_preserves_qwen_masks(monkeypatch):
    import torch
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    fam = next(f for f in d._FAMILIES if f.name == "qwen-image")

    class _QwenPromptCachePipe:
        _execution_device = "cpu"

        def __init__(self):
            self.encode_calls = 0
            self.calls: list[dict[str, Any]] = []

        def encode_prompt(
            self,
            prompt,
            device = None,
            num_images_per_prompt = 1,
        ):
            self.encode_calls += 1
            batch = len(prompt) if isinstance(prompt, list) else 1
            embeds = torch.ones(batch, 4, 3, device = device) * self.encode_calls
            mask = torch.tensor([[1, 1, 0, 0]], device = device).repeat(batch, 1)
            return embeds, mask

        def __call__(
            self,
            *,
            prompt = None,
            prompt_embeds = None,
            prompt_embeds_mask = None,
            negative_prompt_embeds = None,
            negative_prompt_embeds_mask = None,
            num_inference_steps,
            width,
            height,
            true_cfg_scale = None,
            generator = None,
        ):
            self.calls.append(
                {
                    "prompt": prompt,
                    "prompt_embeds": prompt_embeds,
                    "prompt_embeds_mask": prompt_embeds_mask,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
                    "true_cfg_scale": true_cfg_scale,
                }
            )

            class _Out:
                pass

            out = _Out()
            out.images = [Image.new("RGB", (width, height), color = (0, 255, 0))]
            return out

    pipe = _QwenPromptCachePipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = fam
    backend._repo_id = "stub/qwen-image"

    backend.generate_image(
        prompt = "same prompt",
        negative_prompt = " ",
        width = 256,
        height = 256,
        guidance_scale = 4,
        seed = 1,
    )
    backend.generate_image(
        prompt = "same prompt",
        negative_prompt = " ",
        width = 256,
        height = 256,
        guidance_scale = 4,
        seed = 2,
    )

    assert pipe.encode_calls == 2
    assert len(pipe.calls) == 2
    assert all(call["prompt"] is None for call in pipe.calls)
    assert all(call["prompt_embeds_mask"] is not None for call in pipe.calls)
    assert all(call["negative_prompt_embeds_mask"] is not None for call in pipe.calls)
    assert torch.equal(
        pipe.calls[0]["prompt_embeds_mask"],
        pipe.calls[1]["prompt_embeds_mask"],
    )
    assert torch.equal(
        pipe.calls[0]["negative_prompt_embeds_mask"],
        pipe.calls[1]["negative_prompt_embeds_mask"],
    )
    assert backend._prompt_embedding_cache_value is not None
    assert backend._prompt_embedding_cache_value[1].device.type == "cpu"
    assert backend._prompt_embedding_cache_value[3].device.type == "cpu"


def test_prompt_embedding_cache_synthesizes_qwen_all_valid_masks(monkeypatch):
    import torch
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    fam = next(f for f in d._FAMILIES if f.name == "qwen-image")

    class _QwenPromptCachePipe:
        _execution_device = "cpu"

        def __init__(self):
            self.calls: list[dict[str, Any]] = []

        def encode_prompt(
            self,
            prompt,
            device = None,
            num_images_per_prompt = 1,
        ):
            batch = len(prompt) if isinstance(prompt, list) else 1
            return torch.ones(batch, 4, 3, device = device), None

        def __call__(
            self,
            *,
            prompt = None,
            prompt_embeds = None,
            prompt_embeds_mask = None,
            negative_prompt_embeds = None,
            negative_prompt_embeds_mask = None,
            num_inference_steps,
            width,
            height,
            true_cfg_scale = None,
            generator = None,
        ):
            self.calls.append(
                {
                    "prompt_embeds_mask": prompt_embeds_mask,
                    "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
                }
            )

            class _Out:
                pass

            out = _Out()
            out.images = [Image.new("RGB", (width, height), color = (0, 255, 0))]
            return out

    pipe = _QwenPromptCachePipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = fam
    backend._repo_id = "stub/qwen-image"

    backend.generate_image(
        prompt = "same prompt",
        negative_prompt = " ",
        width = 256,
        height = 256,
        guidance_scale = 4,
        seed = 1,
    )

    assert len(pipe.calls) == 1
    assert torch.equal(
        pipe.calls[0]["prompt_embeds_mask"],
        torch.ones(1, 4, dtype = torch.long),
    )
    assert torch.equal(
        pipe.calls[0]["negative_prompt_embeds_mask"],
        torch.ones(1, 4, dtype = torch.long),
    )
    assert backend._prompt_embedding_cache_value is not None
    assert torch.equal(
        backend._prompt_embedding_cache_value[1],
        torch.ones(1, 4, dtype = torch.long),
    )
    assert torch.equal(
        backend._prompt_embedding_cache_value[3],
        torch.ones(1, 4, dtype = torch.long),
    )


def test_flux2_klein_embedded_guidance_patch_disables_cfg_and_forwards_guidance(monkeypatch):
    import torch
    from core.inference import diffusion as d

    fam = next(f for f in d._FAMILIES if f.name == "flux.2-klein")
    captured: dict[str, torch.Tensor | None] = {}

    class _Transformer:
        def forward(self, *, hidden_states, guidance = None, **_kwargs):
            captured["guidance"] = guidance
            return ("ok",)

    class _Pipe:
        def __init__(self):
            self.transformer = _Transformer()
            self._guidance_scale = 4.0
            self.config = SimpleNamespace(is_distilled = False)

        def register_to_config(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self.config, key, value)

        def check_inputs(self):
            assert self.config.is_distilled is False

    pipe = _Pipe()
    monkeypatch.setenv("UNSLOTH_STUDIO_FLUX2_KLEIN_SINGLE_PASS_GUIDANCE", "1")

    assert d._enable_flux2_klein_embedded_guidance(pipe, fam) is True
    assert pipe.config.is_distilled is False
    pipe.check_inputs()
    assert pipe.config.is_distilled is True

    hidden_states = torch.zeros((2, 4, 8))
    assert pipe.transformer.forward(hidden_states = hidden_states) == ("ok",)

    guidance = captured["guidance"]
    assert guidance is not None
    assert guidance.device == hidden_states.device
    assert guidance.dtype is torch.float32
    assert guidance.tolist() == [4.0, 4.0]


def test_flux2_klein_embedded_guidance_patch_is_opt_in(monkeypatch):
    from core.inference import diffusion as d

    fam = next(f for f in d._FAMILIES if f.name == "flux.2-klein")
    pipe = SimpleNamespace(transformer = SimpleNamespace(forward = lambda **_: None))

    monkeypatch.delenv("UNSLOTH_STUDIO_FLUX2_KLEIN_SINGLE_PASS_GUIDANCE", raising = False)

    assert d._enable_flux2_klein_embedded_guidance(pipe, fam) is False


def test_flux2_klein_batched_cfg_batches_cond_and_uncond(monkeypatch):
    import torch
    from core.inference import diffusion as d

    fam = next(f for f in d._FAMILIES if f.name == "flux.2-klein")
    calls: list[dict[str, torch.Tensor]] = []

    class _Transformer:
        def forward(self, **kwargs):
            calls.append(kwargs)
            return (kwargs["hidden_states"] + kwargs["encoder_hidden_states"],)

    pipe = SimpleNamespace(
        transformer = _Transformer(),
        do_classifier_free_guidance = True,
    )
    monkeypatch.delenv("UNSLOTH_STUDIO_FLUX2_KLEIN_BATCHED_CFG", raising = False)

    assert d._enable_flux2_klein_batched_cfg(pipe, fam) is True

    common = {
        "timestep": torch.ones((1,), dtype = torch.float32),
        "txt_ids": torch.zeros((1, 2, 4), dtype = torch.float32),
        "img_ids": torch.zeros((1, 2, 4), dtype = torch.float32),
        "return_dict": False,
    }
    cond = pipe.transformer.forward(
        hidden_states = torch.ones((1, 2, 3), dtype = torch.float32),
        encoder_hidden_states = torch.full((1, 2, 3), 10.0),
        guidance = None,
        **common,
    )[0]
    cond_view = cond[:, :2]
    uncond = pipe.transformer.forward(
        hidden_states = torch.full((1, 2, 3), 2.0),
        encoder_hidden_states = torch.full((1, 2, 3), 20.0),
        guidance = None,
        **common,
    )[0]

    assert len(calls) == 1
    assert calls[0]["hidden_states"].shape[0] == 2
    assert calls[0]["encoder_hidden_states"].shape[0] == 2
    torch.testing.assert_close(cond_view, torch.full((1, 2, 3), 11.0))
    torch.testing.assert_close(uncond, torch.full((1, 2, 3), 22.0))
    guided = uncond + 4.0 * (cond_view - uncond)
    torch.testing.assert_close(guided, torch.full((1, 2, 3), -22.0))


def test_flux2_klein_batched_cfg_can_be_disabled(monkeypatch):
    from core.inference import diffusion as d

    fam = next(f for f in d._FAMILIES if f.name == "flux.2-klein")
    pipe = SimpleNamespace(transformer = SimpleNamespace(forward = lambda **_: None))
    monkeypatch.setenv("UNSLOTH_STUDIO_FLUX2_KLEIN_BATCHED_CFG", "0")

    assert d._enable_flux2_klein_batched_cfg(pipe, fam) is False


def test_aggressive_memory_policy_enables_vae_slicing_and_tiling():
    from core.inference import diffusion as d

    calls: list[str] = []

    class _VAE:
        def enable_slicing(self):
            calls.append("slicing")

        def enable_tiling(self):
            calls.append("tiling")

    pipe = SimpleNamespace(vae = _VAE())

    d._apply_diffusion_memory_policy(pipe, d.DIFFUSION_OFFLOAD_POLICY_AGGRESSIVE)

    assert calls == ["slicing", "tiling"]


def test_balanced_memory_policy_leaves_vae_decode_untouched():
    from core.inference import diffusion as d

    calls: list[str] = []

    class _VAE:
        def enable_slicing(self):
            calls.append("slicing")

        def enable_tiling(self):
            calls.append("tiling")

    pipe = SimpleNamespace(vae = _VAE())

    d._apply_diffusion_memory_policy(pipe, d.DIFFUSION_OFFLOAD_POLICY_BALANCED)

    assert calls == []


def test_generate_image_forwards_family_default_call_kwargs(monkeypatch):
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    seen: dict[str, Any] = {}

    class _ErnieStubPipe:
        def __call__(
            self,
            *,
            prompt,
            num_inference_steps,
            width,
            height,
            guidance_scale,
            use_pe = False,
            **_kwargs,
        ):
            seen.update(
                {
                    "prompt": prompt,
                    "num_inference_steps": num_inference_steps,
                    "width": width,
                    "height": height,
                    "guidance_scale": guidance_scale,
                    "use_pe": use_pe,
                }
            )

            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (width, height), color = (0, 255, 0))]
            return o

    backend._pipe = _ErnieStubPipe()
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/ERNIE-Image-Turbo-GGUF")
    backend._repo_id = "stub/ernie"

    img = backend.generate_image(prompt = "poster")

    assert img.size == (1024, 1024)
    assert seen["num_inference_steps"] == 8
    assert seen["guidance_scale"] == 1.0
    assert seen["use_pe"] is True


def test_generate_image_unloaded_raises(monkeypatch):
    import core.inference.diffusion as d

    backend = d.get_diffusion_backend()
    backend._pipe = None
    with pytest.raises(RuntimeError, match = "No diffusion model"):
        backend.generate_image(prompt = "x")


def test_unload_clears_state(monkeypatch):
    backend = _stub_pipeline(monkeypatch)
    assert backend.is_loaded
    backend.unload_model()
    assert not backend.is_loaded
    s = backend.status()
    assert s["repo_id"] is None
    assert s["family"] is None


def test_guard_diffusers_optional_bitsandbytes_stubs_failed_import(monkeypatch):
    import builtins
    import sys
    import core.inference.diffusion as d

    original_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "diffusers.quantizers.bitsandbytes", raising = False)
    monkeypatch.delitem(sys.modules, "bitsandbytes", raising = False)

    def _fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "bitsandbytes":
            raise RuntimeError("broken optional bnb")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    d._guard_diffusers_optional_bitsandbytes()

    stub = sys.modules["diffusers.quantizers.bitsandbytes"]
    assert hasattr(stub, "BnB4BitDiffusersQuantizer")
    assert "diffusers.quantizers.bitsandbytes.utils" in sys.modules
    assert sys.modules[
        "diffusers.quantizers.bitsandbytes.utils"
    ]._check_bnb_status(None) == (False, False, False)
    with pytest.raises(RuntimeError, match = "bitsandbytes failed to import"):
        stub.BnB4BitDiffusersQuantizer()
    with pytest.raises(RuntimeError, match = "bitsandbytes failed to import"):
        sys.modules[
            "diffusers.quantizers.bitsandbytes.utils"
        ].replace_with_bnb_linear()


def test_guard_peft_optional_bitsandbytes_marks_bnb_unavailable(monkeypatch):
    import builtins
    import sys
    import core.inference.diffusion as d

    original_import = builtins.__import__
    fake_peft = types.ModuleType("peft")
    fake_peft.__path__ = []
    fake_import_utils = types.ModuleType("peft.import_utils")
    fake_import_utils.is_bnb_available = lambda: True
    fake_import_utils.is_bnb_4bit_available = lambda: True
    fake_lora_model = types.ModuleType("peft.tuners.lora.model")
    fake_lora_model.is_bnb_available = lambda: True
    fake_lora_model.is_bnb_4bit_available = lambda: True

    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "peft.import_utils", fake_import_utils)
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.model", fake_lora_model)
    monkeypatch.delitem(sys.modules, "bitsandbytes", raising = False)

    def _fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "bitsandbytes":
            raise RuntimeError("broken optional bnb")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    d._guard_peft_optional_bitsandbytes()

    assert fake_import_utils.is_bnb_available() is False
    assert fake_import_utils.is_bnb_4bit_available() is False
    assert fake_lora_model.is_bnb_available() is False
    assert fake_lora_model.is_bnb_4bit_available() is False


# ── load_model (with monkey-patched diffusers) ──────────────────


def _install_fake_diffusers(monkeypatch, *, raise_on_pipeline = False):
    """Build a tiny ``diffusers`` shim so we can exercise load_model
    without dragging the real 1+ GB diffusers / torch import in."""
    from PIL import Image

    fake = types.ModuleType("diffusers")
    fake.__version__ = "fake"

    class _FakeQuantConfig:
        def __init__(self, compute_dtype = None):
            self.compute_dtype = compute_dtype

    class _FakePipelineQuantizationConfig:
        def __init__(
            self,
            quant_backend = None,
            quant_kwargs = None,
            components_to_quantize = None,
            quant_mapping = None,
        ):
            self.quant_backend = quant_backend
            self.quant_kwargs = quant_kwargs or {}
            self.components_to_quantize = components_to_quantize
            self.quant_mapping = quant_mapping

    class _FakeTransformer:
        def __init__(self):
            self.compile_calls = []
            self.compile_repeated_blocks_calls = []
            self._repeated_blocks = ("fake_block",)

        @classmethod
        def from_single_file(cls, path, **kw):
            inst = cls()
            inst.path = path
            inst.qc = kw.get("quantization_config")
            inst.dtype = kw.get("torch_dtype")
            inst.config = kw.get("config")
            inst.subfolder = kw.get("subfolder")
            inst.token = kw.get("token")
            return inst

        def compile(self, **kwargs):
            self.compile_calls.append(kwargs)

        def compile_repeated_blocks(self, **kwargs):
            self.compile_repeated_blocks_calls.append(kwargs)

    class _FakeWanVAE:
        @classmethod
        def from_pretrained(cls, base_repo, **kwargs):
            inst = cls()
            inst.base_repo = base_repo
            inst.kwargs = kwargs
            return inst

    class _FakeSchedulerConfig:
        pass

    class _FakeScheduler:
        def __init__(self):
            self.config = _FakeSchedulerConfig()

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, base_repo, **kwargs):
            if raise_on_pipeline:
                raise RuntimeError("simulated load failure")
            inst = cls()
            inst.base_repo = base_repo
            inst.kwargs = kwargs
            inst.scheduler = _FakeScheduler()
            inst.lora_loads = []
            inst.adapter_calls = []
            inst.fuse_calls = []
            inst.transformer = kwargs.get("transformer") or _FakeTransformer()
            inst.unet = kwargs.get("unet")
            inst.vae = kwargs.get("vae")
            inst.text_encoder = kwargs.get("text_encoder")
            inst.text_encoder_2 = kwargs.get("text_encoder_2")
            inst.text_encoder_3 = kwargs.get("text_encoder_3")
            inst.pe = kwargs.get("pe")
            inst.config = SimpleNamespace(
                is_distilled = (
                    isinstance(base_repo, str)
                    and "FLUX.2-klein-" in base_repo
                    and "base" not in base_repo.lower()
                )
            )
            return inst

        def __call__(self, **kwargs):
            class _Out:
                pass

            o = _Out()
            o.images = [
                Image.new("RGB", (kwargs["width"], kwargs["height"]), color = (0, 0, 255))
            ]
            return o

        def enable_model_cpu_offload(self):
            self.cpu_offload = True

        def to(self, device):
            self.device = device
            for component in self.kwargs.values():
                to = getattr(component, "to", None)
                if callable(to):
                    to(device)
            return self

        def load_lora_weights(self, repo, **kwargs):
            self.lora_loads.append({"repo": repo, **kwargs})

        def set_adapters(self, adapter_names, adapter_weights = None):
            self.adapter_calls.append(
                {
                    "adapter_names": adapter_names,
                    "adapter_weights": adapter_weights,
                }
            )

        def fuse_lora(self, **kwargs):
            self.fuse_calls.append(kwargs)

    fake.GGUFQuantizationConfig = _FakeQuantConfig
    fake.PipelineQuantizationConfig = _FakePipelineQuantizationConfig
    fake.DiffusionPipeline = _FakePipeline
    fake.Flux2KleinPipeline = _FakePipeline
    fake.Flux2Transformer2DModel = _FakeTransformer
    fake.Flux2Pipeline = _FakePipeline
    fake.FluxPipeline = _FakePipeline
    fake.FluxKontextPipeline = _FakePipeline
    fake.FluxTransformer2DModel = _FakeTransformer
    fake.QwenImagePipeline = _FakePipeline
    fake.QwenImageTransformer2DModel = _FakeTransformer
    fake.QwenImageEditPipeline = _FakePipeline
    fake.QwenImageEditPlusPipeline = _FakePipeline
    fake.QwenImageLayeredPipeline = _FakePipeline
    fake.ZImagePipeline = _FakePipeline
    fake.ZImageTransformer2DModel = _FakeTransformer
    fake.ErnieImagePipeline = _FakePipeline
    fake.ErnieImageTransformer2DModel = _FakeTransformer
    fake.LTX2Pipeline = _FakePipeline
    fake.LTX2VideoTransformer3DModel = _FakeTransformer
    fake.WanPipeline = _FakePipeline
    fake.WanTransformer3DModel = _FakeTransformer
    fake.AutoencoderKLWan = _FakeWanVAE
    fake.SD3Transformer2DModel = _FakeTransformer
    fake.StableDiffusion3Pipeline = _FakePipeline
    fake.StableDiffusionXLPipeline = _FakePipeline

    monkeypatch.setitem(sys.modules, "diffusers", fake)

    # Pretend HF Hub gave us a local file without actually fetching.
    # Round 21: accept arbitrary kwargs (round 20 preflight adds
    # ``filename="model_index.json"`` and round 21 preflight adds
    # ``subfolder="transformer"``) so existing tests that exercise
    # the GGUF path do not hit a TypeError from the fake signature.
    fake_hub = types.ModuleType("huggingface_hub")

    def _fake_download(repo_id, filename, token = None, subfolder = None, **_kwargs):
        sub = f"{subfolder}/" if subfolder else ""
        return f"/fake/{repo_id}/{sub}{filename}"

    fake_hub.hf_hub_download = _fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    # Force CPU dtype so the test does not need CUDA.
    import core.inference.diffusion as d

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cpu", "fake_dtype"),
    )

    # Round 16 reordered _release_other_gpu_owners_for_diffusion to
    # run BEFORE the chat unload. That helper imports core.training /
    # core.export and raises on active or unverifiable status. Stub
    # both modules with idle backends so the load_model fast path
    # works in CI environments where neither module is fully wired
    # (Windows runners without the training/export deps).
    fake_training_mod = types.ModuleType("core.training")
    fake_training_mod.get_training_backend = lambda: SimpleNamespace(
        is_training_active = lambda: False,
    )
    monkeypatch.setitem(sys.modules, "core.training", fake_training_mod)

    fake_export_mod = types.ModuleType("core.export")
    fake_export_mod.get_export_backend = lambda: SimpleNamespace(
        is_export_active = lambda: False,
        current_checkpoint = None,
    )
    monkeypatch.setitem(sys.modules, "core.export", fake_export_mod)

    return fake


def test_load_model_full_repo_uses_safetensors_quantization(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/my-flux-diffusers",
        family_override = "flux.1",
        safetensors_quantization = "bitsandbytes_4bit_nf4",
        safetensors_quantization_components = ["transformer", "text_encoder_2"],
        enable_model_cpu_offload = False,
    )

    pipe = backend._pipe
    quant_config = pipe.kwargs["quantization_config"]
    assert status["is_loaded"] is True
    assert status["safetensors_quantization"] == "bitsandbytes_4bit_nf4"
    assert status["safetensors_quantization_components"] == [
        "transformer",
        "text_encoder_2",
    ]
    assert quant_config.quant_backend == "bitsandbytes_4bit"
    assert quant_config.quant_kwargs["load_in_4bit"] is True
    assert quant_config.quant_kwargs["bnb_4bit_quant_type"] == "nf4"
    assert quant_config.components_to_quantize == [
        "transformer",
        "text_encoder_2",
    ]
    assert pipe.transformer.compile_repeated_blocks_calls == [
        {"fullgraph": True, "dynamic": True}
    ]
    assert status["torch_compile_config"] == {
        "scope": "regional",
        "source": "default",
        "mode": None,
        "fullgraph": True,
        "dynamic": True,
        "options": {},
    }
    assert status["torch_compile_stats"]["compiled_components"][0]["component"] == "transformer"


def test_load_model_full_repo_defaults_to_regional_torch_compile(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/my-flux-diffusers",
        family_override = "flux.1",
        enable_model_cpu_offload = False,
    )

    pipe = backend._pipe
    assert pipe.transformer.compile_repeated_blocks_calls == [
        {"fullgraph": True, "dynamic": True}
    ]
    assert pipe.transformer.compile_calls == []
    assert status["torch_compile_config"]["scope"] == "regional"
    assert status["torch_compile_config"]["source"] == "default"
    assert status["torch_compile_stats"]["compiled_components"][0]["method"] == (
        "compile_repeated_blocks"
    )


def test_load_model_cpu_offload_defaults_to_regional_torch_compile(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/my-flux-diffusers",
        family_override = "flux.1",
        enable_model_cpu_offload = True,
    )

    pipe = backend._pipe
    assert backend._cpu_offload_enabled is True
    assert pipe.cpu_offload is True
    assert pipe.transformer.compile_repeated_blocks_calls == [
        {"fullgraph": True, "dynamic": True}
    ]
    assert status["torch_compile_config"]["scope"] == "regional"
    assert status["torch_compile_config"]["source"] == "default"


def test_load_model_full_repo_respects_explicit_torch_compile_off(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/my-flux-diffusers",
        family_override = "flux.1",
        enable_model_cpu_offload = False,
        torch_compile = "none",
    )

    pipe = backend._pipe
    assert pipe.transformer.compile_repeated_blocks_calls == []
    assert pipe.transformer.compile_calls == []
    assert status["torch_compile_config"] is None
    assert status["torch_compile_stats"] is None


def test_default_torch_compile_policy_skips_gguf_torchao_and_video():
    import core.inference.diffusion as d

    assert (
        d._default_diffusion_torch_compile_scope(
            requested = None,
            has_gguf_components = False,
            safetensors_quantization = None,
            media_kind = "image",
        )
        == "regional"
    )
    assert (
        d._default_diffusion_torch_compile_scope(
            requested = None,
            has_gguf_components = False,
            safetensors_quantization = "bitsandbytes_4bit_nf4",
            media_kind = "image",
        )
        == "regional"
    )
    assert (
        d._default_diffusion_torch_compile_scope(
            requested = None,
            has_gguf_components = True,
            safetensors_quantization = None,
            media_kind = "image",
        )
        == "none"
    )
    assert (
        d._default_diffusion_torch_compile_scope(
            requested = None,
            has_gguf_components = False,
            safetensors_quantization = "torchao_int8_weight_only",
            media_kind = "image",
        )
        == "none"
    )
    assert (
        d._default_diffusion_torch_compile_scope(
            requested = None,
            has_gguf_components = False,
            safetensors_quantization = None,
            media_kind = "video",
        )
        == "none"
    )
    assert (
        d._default_diffusion_torch_compile_scope(
            requested = "transformer",
            has_gguf_components = True,
            safetensors_quantization = "torchao_int8_weight_only",
            media_kind = "video",
        )
        == "transformer"
    )


def test_load_model_rejects_safetensors_quantization_with_gguf(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(ValueError, match = "safetensors_quantization"):
        backend.load_model(
            "unsloth/FLUX.2-klein-4B-GGUF",
            gguf_filename = "flux-2-klein-4b-Q4_K_M.gguf",
            base_repo = "black-forest-labs/FLUX.2-klein-4B",
            family_override = "flux.2-klein",
            safetensors_quantization = "bitsandbytes_4bit_nf4",
        )


def test_build_safetensors_quantization_preflights_torchao_int4_mslk(
    monkeypatch,
):
    import core.inference.diffusion as d

    fake_diffusers = types.SimpleNamespace(
        PipelineQuantizationConfig = lambda **kwargs: kwargs,
        TorchAoConfig = lambda config: ("diffusers_torchao", config),
    )
    real_find_spec = d.importlib.util.find_spec
    monkeypatch.setattr(
        d.importlib.util,
        "find_spec",
        lambda name: None if name == "mslk" else real_find_spec(name),
    )

    with pytest.raises(RuntimeError, match = "mslk"):
        d._build_safetensors_pipeline_quantization_config(
            fake_diffusers,
            "torchao_int4_weight_only",
            ["transformer"],
            "fake_dtype",
        )


def test_load_model_ernie_gguf_uses_state_dict_fallback(monkeypatch):
    fake = _install_fake_diffusers(monkeypatch)
    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    class _ErnieTransformerWithoutSingleFile:
        @classmethod
        def load_config(cls, *_args, **_kwargs):
            return {"ok": True}

        @classmethod
        def from_config(cls, _config):
            return cls()

    fake.ErnieImageTransformer2DModel = _ErnieTransformerWithoutSingleFile
    calls: list[dict[str, Any]] = []

    def _fake_state_dict_loader(transformer_cls, path, **kwargs):
        calls.append(
            {
                "transformer_cls": transformer_cls,
                "path": path,
                **kwargs,
            }
        )
        return SimpleNamespace(source = "state-dict-fallback")

    monkeypatch.setattr(d, "_load_transformer_gguf_from_state_dict", _fake_state_dict_loader)

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/ERNIE-Image-Turbo-GGUF",
        gguf_filename = "ernie-image-turbo-UD-Q4_K_M.gguf",
        base_repo = "baidu/ERNIE-Image-Turbo",
        family_override = "ernie-image-turbo",
    )

    assert status["is_loaded"] is True
    assert status["family"] == "ernie-image-turbo"
    assert calls
    assert calls[0]["transformer_cls"] is _ErnieTransformerWithoutSingleFile
    assert calls[0]["base_repo"] == "baidu/ERNIE-Image-Turbo"
    assert calls[0]["path"].endswith("ernie-image-turbo-UD-Q4_K_M.gguf")


@pytest.mark.parametrize(
    ("repo_id", "filename", "family", "base_repo", "variant"),
    [
        (
            "unsloth/FLUX.1-Kontext-dev-GGUF",
            "flux1-kontext-dev-Q4_K_M.gguf",
            "flux.1-kontext",
            "black-forest-labs/FLUX.1-Kontext-dev",
            None,
        ),
        (
            "unsloth/FLUX.1-dev-GGUF",
            "flux1-dev-Q4_K_M.gguf",
            "flux.1",
            "black-forest-labs/FLUX.1-dev",
            None,
        ),
        (
            "unsloth/FLUX.1-schnell-GGUF",
            "flux1-schnell-Q4_K_M.gguf",
            "flux.1-schnell",
            "black-forest-labs/FLUX.1-schnell",
            None,
        ),
        (
            "unsloth/FLUX.2-dev-GGUF",
            "flux2-dev-Q4_K_M.gguf",
            "flux.2",
            "black-forest-labs/FLUX.2-dev",
            None,
        ),
        (
            "unsloth/FLUX.2-klein-4B-GGUF",
            "flux-2-klein-4b-Q4_K_M.gguf",
            "flux.2-klein",
            "black-forest-labs/FLUX.2-klein-4B",
            "distilled-4b",
        ),
        (
            "unsloth/FLUX.2-klein-9B-GGUF",
            "flux-2-klein-9b-Q4_K_M.gguf",
            "flux.2-klein",
            "black-forest-labs/FLUX.2-klein-9B",
            "distilled-9b",
        ),
        (
            "unsloth/FLUX.2-klein-base-4B-GGUF",
            "flux-2-klein-base-4b-Q4_K_M.gguf",
            "flux.2-klein",
            "black-forest-labs/FLUX.2-klein-base-4B",
            "base-4b",
        ),
        (
            "unsloth/FLUX.2-klein-base-9B-GGUF",
            "flux-2-klein-base-9b-Q4_K_M.gguf",
            "flux.2-klein",
            "black-forest-labs/FLUX.2-klein-base-9B",
            "base-9b",
        ),
        (
            "unsloth/Z-Image-GGUF",
            "z-image-Q4_K_M.gguf",
            "z-image",
            "Tongyi-MAI/Z-Image",
            None,
        ),
        (
            "unsloth/Z-Image-Turbo-GGUF",
            "z-image-turbo-Q4_K_M.gguf",
            "z-image-turbo",
            "Tongyi-MAI/Z-Image-Turbo",
            None,
        ),
        (
            "unsloth/ERNIE-Image-Turbo-GGUF",
            "ernie-image-turbo-UD-Q4_K_M.gguf",
            "ernie-image-turbo",
            "baidu/ERNIE-Image-Turbo",
            None,
        ),
        (
            "unsloth/ERNIE-Image-GGUF",
            "ernie-image-UD-Q4_K_M.gguf",
            "ernie-image",
            "baidu/ERNIE-Image",
            None,
        ),
        (
            "unsloth/Qwen-Image-GGUF",
            "qwen-image-Q4_K_M.gguf",
            "qwen-image",
            "Qwen/Qwen-Image",
            None,
        ),
        (
            "unsloth/Qwen-Image-Edit-GGUF",
            "qwen-image-edit-Q4_K_M.gguf",
            "qwen-image-edit",
            "Qwen/Qwen-Image-Edit",
            None,
        ),
        (
            "unsloth/Qwen-Image-Edit-2509-GGUF",
            "qwen-image-edit-2509-Q4_K_M.gguf",
            "qwen-image-edit-2509",
            "Qwen/Qwen-Image-Edit-2509",
            None,
        ),
        (
            "unsloth/Qwen-Image-2512-GGUF",
            "qwen-image-2512-Q4_K_M.gguf",
            "qwen-image-2512",
            "Qwen/Qwen-Image-2512",
            None,
        ),
        (
            "unsloth/Qwen-Image-Edit-2511-GGUF",
            "qwen-image-edit-2511-Q4_K_M.gguf",
            "qwen-image-edit-2511",
            "Qwen/Qwen-Image-Edit-2511",
            None,
        ),
        (
            "unsloth/Qwen-Image-Layered-GGUF",
            "qwen-image-layered-Q4_K_M.gguf",
            "qwen-image-layered",
            "Qwen/Qwen-Image-Layered",
            None,
        ),
    ],
)
def test_load_model_curated_unsloth_diffusion_gguf_manifest(
    monkeypatch,
    repo_id,
    filename,
    family,
    base_repo,
    variant,
):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(repo_id, gguf_filename = filename)

    assert status["is_loaded"] is True
    assert status["family"] == family
    assert status["base_repo"] == base_repo
    assert status["base_repo_variant"] == variant
    assert status["gguf_filename"] == filename
    assert status["sampling_contract"]["gguf"] is True
    assert backend._pipe.base_repo == base_repo
    assert backend._pipe.kwargs["transformer"].path.endswith(filename)


def test_curated_unsloth_diffusion_gguf_manifest_covers_all_quant_filenames():
    from core.inference.diffusion import (
        _CURATED_UNSLOTH_DIFFUSION_GGUFS,
        _filename_matches_curated_diffusion_gguf,
    )

    common_suffixes = (
        "BF16.gguf",
        "F16.gguf",
        "Q2_K.gguf",
        "Q3_K_M.gguf",
        "Q3_K_S.gguf",
        "Q4_0.gguf",
        "Q4_1.gguf",
        "Q4_K_M.gguf",
        "Q4_K_S.gguf",
        "Q5_0.gguf",
        "Q5_1.gguf",
        "Q5_K_M.gguf",
        "Q5_K_S.gguf",
        "Q6_K.gguf",
        "Q8_0.gguf",
    )
    extra_suffixes_by_repo = {
        "unsloth/FLUX.2-dev-GGUF": ("Q3_K_L.gguf",),
        "unsloth/Qwen-Image-GGUF": ("Q3_K_L.gguf",),
        "unsloth/Qwen-Image-Edit-GGUF": ("Q3_K_L.gguf",),
        "unsloth/Qwen-Image-Edit-2509-GGUF": ("Q3_K_L.gguf",),
        "unsloth/Qwen-Image-Edit-2511-GGUF": ("Q3_K_L.gguf",),
        "unsloth/Qwen-Image-Layered-GGUF": ("Q3_K_L.gguf",),
        "unsloth/ERNIE-Image-Turbo-GGUF": (
            "UD-Q2_K.gguf",
            "UD-Q3_K_M.gguf",
            "UD-Q4_K_M.gguf",
            "UD-Q5_K_M.gguf",
        ),
        "unsloth/ERNIE-Image-GGUF": (
            "UD-Q2_K.gguf",
            "UD-Q3_K_M.gguf",
            "UD-Q4_K_M.gguf",
            "UD-Q5_K_M.gguf",
        ),
        "unsloth/Z-Image-GGUF": ("Q3_K_L.gguf",),
    }
    expected_repos = {
        "unsloth/FLUX.1-Kontext-dev-GGUF",
        "unsloth/FLUX.1-dev-GGUF",
        "unsloth/FLUX.1-schnell-GGUF",
        "unsloth/FLUX.2-dev-GGUF",
        "unsloth/FLUX.2-klein-4B-GGUF",
        "unsloth/FLUX.2-klein-9B-GGUF",
        "unsloth/FLUX.2-klein-base-4B-GGUF",
        "unsloth/FLUX.2-klein-base-9B-GGUF",
        "unsloth/Z-Image-GGUF",
        "unsloth/Z-Image-Turbo-GGUF",
        "unsloth/ERNIE-Image-Turbo-GGUF",
        "unsloth/ERNIE-Image-GGUF",
        "unsloth/Qwen-Image-GGUF",
        "unsloth/Qwen-Image-Edit-GGUF",
        "unsloth/Qwen-Image-Edit-2509-GGUF",
        "unsloth/Qwen-Image-2512-GGUF",
        "unsloth/Qwen-Image-Edit-2511-GGUF",
        "unsloth/Qwen-Image-Layered-GGUF",
    }

    assert {
        spec.repo_id
        for spec in _CURATED_UNSLOTH_DIFFUSION_GGUFS
    } == expected_repos

    checked = 0
    for spec in _CURATED_UNSLOTH_DIFFUSION_GGUFS:
        for prefix in spec.filename_prefixes:
            suffixes = common_suffixes + extra_suffixes_by_repo.get(spec.repo_id, ())
            for suffix in suffixes:
                checked += 1
                assert _filename_matches_curated_diffusion_gguf(spec, prefix + suffix)

    assert checked == 285


def test_curated_diffusion_presets_cover_manifest():
    from core.inference.diffusion import (
        _CURATED_UNSLOTH_DIFFUSION_GGUFS,
        curated_diffusion_presets,
    )

    presets = curated_diffusion_presets()
    assert len(presets) == len(_CURATED_UNSLOTH_DIFFUSION_GGUFS)
    by_repo = {entry["transformer_gguf_repo"]: entry for entry in presets}
    assert set(by_repo) == {
        spec.repo_id
        for spec in _CURATED_UNSLOTH_DIFFUSION_GGUFS
    }
    flux2 = by_repo["unsloth/FLUX.2-dev-GGUF"]
    assert flux2["id"] == "flux.2-dev"
    assert flux2["pipeline_repo"] == "black-forest-labs/FLUX.2-dev"
    assert flux2["default_steps"] == 50
    assert flux2["default_guidance_scale"] == 4.0
    assert flux2["default_text_encoder_gguf_repo"] == (
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
    )
    assert flux2["recommended_offload_policy"] == "less_aggressive"
    assert by_repo["unsloth/Z-Image-GGUF"]["recommended_offload_policy"] == (
        "less_aggressive"
    )
    assert by_repo["unsloth/Z-Image-Turbo-GGUF"]["recommended_offload_policy"] == (
        "less_aggressive"
    )
    assert by_repo["unsloth/ERNIE-Image-GGUF"]["recommended_offload_policy"] == (
        "less_aggressive"
    )
    assert by_repo["unsloth/ERNIE-Image-Turbo-GGUF"]["recommended_offload_policy"] == (
        "less_aggressive"
    )


def test_resolve_diffusion_load_plan_expands_preset_component_swap():
    from core.inference.diffusion import resolve_diffusion_load_plan

    plan = resolve_diffusion_load_plan(
        preset_id = "flux.2-klein-base-4b",
        transformer_quant = "Q4_K_M",
        text_encoder_gguf_filename = "qwen3-4b-BF16.gguf",
        require_loadable = True,
    )

    assert plan["ready_to_load"] is True
    assert plan["load_kwargs"]["repo_id"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert plan["load_kwargs"]["gguf_filename"] is None
    assert plan["load_kwargs"]["transformer_gguf_repo"] == (
        "unsloth/FLUX.2-klein-base-4B-GGUF"
    )
    assert plan["load_kwargs"]["transformer_gguf_filename"] == (
        "flux-2-klein-base-4b-Q4_K_M.gguf"
    )
    assert plan["load_kwargs"]["text_encoder_gguf_repo"] == (
        "unsloth/Qwen3-4B-GGUF"
    )
    assert plan["load_kwargs"]["family_override"] == "flux.2-klein"
    assert plan["load_kwargs"]["offload_policy"] == "balanced"
    assert plan["sampling_defaults"]["num_inference_steps"] == 50
    assert plan["sampling_defaults"]["guidance_scale"] == 4.0
    assert plan["component_sources"]["transformer"] == {
        "source": "gguf",
        "repo": "unsloth/FLUX.2-klein-base-4B-GGUF",
        "filename": "flux-2-klein-base-4b-Q4_K_M.gguf",
    }


def test_resolve_diffusion_load_plan_requires_quant_for_preset_load():
    from core.inference.diffusion import resolve_diffusion_load_plan

    with pytest.raises(ValueError, match = "require transformer_gguf_filename"):
        resolve_diffusion_load_plan(
            preset_id = "flux.2-dev",
            require_loadable = True,
        )


def test_resolve_diffusion_load_plan_uses_flux2_dev_speed_policy():
    from core.inference.diffusion import resolve_diffusion_load_plan

    plan = resolve_diffusion_load_plan(
        preset_id = "flux.2-dev",
        transformer_quant = "Q4_K_M",
    )

    assert plan["load_kwargs"]["offload_policy"] == "less_aggressive"


def test_resolve_diffusion_load_plan_uses_fast_image_gguf_policies():
    from core.inference.diffusion import resolve_diffusion_load_plan

    for preset_id, quant in (
        ("z-image", "Q4_K_M"),
        ("z-image-turbo", "Q4_K_M"),
        ("ernie-image", "UD-Q4_K_M"),
        ("ernie-image-turbo", "UD-Q4_K_M"),
    ):
        plan = resolve_diffusion_load_plan(
            preset_id = preset_id,
            transformer_quant = quant,
        )

        assert plan["load_kwargs"]["offload_policy"] == "less_aggressive"


def test_resolve_diffusion_load_plan_keeps_qwen_balanced_by_default():
    import core.inference.diffusion as d

    for preset_id in (
        "qwen-image",
        "qwen-image-2512",
        "qwen-image-edit",
        "qwen-image-edit-2509",
        "qwen-image-edit-2511",
        "qwen-image-layered",
    ):
        plan = d.resolve_diffusion_load_plan(
            preset_id = preset_id,
            transformer_quant = "Q4_K_M",
        )

        assert plan["load_kwargs"]["offload_policy"] == "balanced"


def test_resolve_diffusion_load_plan_keeps_qwen_balanced_for_low_memory():
    import core.inference.diffusion as d

    for preset_id in (
        "qwen-image",
        "qwen-image-2512",
        "qwen-image-edit",
        "qwen-image-edit-2509",
        "qwen-image-edit-2511",
        "qwen-image-layered",
    ):
        plan = d.resolve_diffusion_load_plan(
            preset_id = preset_id,
            transformer_quant = "Q4_K_M",
        )

        assert plan["load_kwargs"]["offload_policy"] == "balanced"


def test_curated_gguf_recommended_offload_policy_for_direct_loads():
    import core.inference.diffusion as d

    assert d._curated_gguf_recommended_offload_policy(
        repo_id = "unsloth/Qwen-Image-GGUF",
        gguf_filename = "qwen-image-Q4_K_M.gguf",
        device = "cpu",
    ) == "balanced"
    assert d._curated_gguf_recommended_offload_policy(
        repo_id = "unsloth/Qwen-Image-GGUF",
        gguf_filename = "qwen-image-Q4_K_M.gguf",
        free_bytes = 64 * 1024**3,
        total_bytes = 80 * 1024**3,
    ) == "balanced"
    assert d._curated_gguf_recommended_offload_policy(
        repo_id = "unsloth/Qwen-Image-GGUF",
        gguf_filename = "qwen-image-Q4_K_M.gguf",
        free_bytes = 30 * 1024**3,
        total_bytes = 48 * 1024**3,
    ) == "balanced"
    for repo_id, filename in (
        ("unsloth/Qwen-Image-Edit-GGUF", "qwen-image-edit-Q4_K_M.gguf"),
        ("unsloth/Qwen-Image-Edit-2509-GGUF", "qwen-image-edit-2509-Q4_K_M.gguf"),
        ("unsloth/Qwen-Image-Edit-2511-GGUF", "qwen-image-edit-2511-Q4_K_M.gguf"),
        ("unsloth/Qwen-Image-Layered-GGUF", "qwen-image-layered-Q4_K_M.gguf"),
    ):
        assert d._curated_gguf_recommended_offload_policy(
            repo_id = repo_id,
            gguf_filename = filename,
            free_bytes = 64 * 1024**3,
            total_bytes = 80 * 1024**3,
        ) == "balanced"
        assert d._curated_gguf_recommended_offload_policy(
            repo_id = repo_id,
            gguf_filename = filename,
            free_bytes = 30 * 1024**3,
            total_bytes = 48 * 1024**3,
        ) == "balanced"
    assert d._curated_gguf_recommended_offload_policy(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
    ) == "less_aggressive"
    assert d._curated_gguf_recommended_offload_policy(
        repo_id = "Qwen/Qwen-Image",
        transformer_gguf_repo = "unsloth/Qwen-Image-GGUF",
        transformer_gguf_filename = "qwen-image-Q4_K_M.gguf",
        free_bytes = 64 * 1024**3,
        total_bytes = 80 * 1024**3,
    ) == "balanced"
    assert d._curated_gguf_recommended_offload_policy(
        repo_id = "Qwen/Qwen-Image",
        transformer_gguf_repo = "unsloth/Qwen-Image-GGUF",
        transformer_gguf_filename = "qwen-image-Q4_K_M.gguf",
        free_bytes = 30 * 1024**3,
        total_bytes = 48 * 1024**3,
    ) == "balanced"
    assert d._curated_gguf_recommended_offload_policy(
        repo_id = "Qwen/Qwen-Image",
    ) is None


def test_load_model_ernie_can_replace_text_encoder_and_prompt_enhancer_ggufs(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")

    class _FakeMistral3TextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return SimpleNamespace(kind = "text", path = path)

    class _FakePromptEnhancer:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            subfolder = "pe",
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "subfolder": subfolder,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return SimpleNamespace(kind = "pe", path = path)

    fake_text_mod.LazyMistral3TextEncoder = _FakeMistral3TextEncoder
    fake_text_mod.LazyMinistral3PromptEnhancer = _FakePromptEnhancer
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/ERNIE-Image-Turbo-GGUF",
        gguf_filename = "ernie-image-turbo-UD-Q4_K_M.gguf",
        base_repo = "baidu/ERNIE-Image-Turbo",
        family_override = "ernie-image-turbo",
        text_encoder_gguf_filename = "Ministral-3-3B-Instruct-2512-UD-Q4_K_XL.gguf",
        prompt_enhancer_gguf_filename = "Ernie-Image-Prompt-Enhancer-Ministral-3.8B-Q4_K_M.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["family"] == "ernie-image-turbo"
    assert status["text_encoder_gguf_repo"] == "unsloth/Ministral-3-3B-Instruct-2512-GGUF"
    assert status["prompt_enhancer_gguf_repo"] == (
        "Green-Sky/Ernie-Image-Prompt-Enhancer-Ministral-3B-GGUF"
    )
    assert backend._pipe.kwargs["text_encoder"].kind == "text"
    assert backend._pipe.kwargs["pe"].kind == "pe"
    assert _FakeMistral3TextEncoder.calls == [
        {
            "path": "/fake/unsloth/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-UD-Q4_K_XL.gguf",
            "base_repo_or_path": "baidu/ERNIE-Image-Turbo",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert _FakePromptEnhancer.calls == [
        {
            "path": "/fake/Green-Sky/Ernie-Image-Prompt-Enhancer-Ministral-3B-GGUF/Ernie-Image-Prompt-Enhancer-Ministral-3.8B-Q4_K_M.gguf",
            "base_repo_or_path": "baidu/ERNIE-Image-Turbo",
            "subfolder": "pe",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]


def test_load_model_unknown_gguf_family_still_requires_family(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "Could not infer"):
        backend.load_model("private/random-repo", gguf_filename = "model.gguf")


def test_load_model_gguf_path_happy(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
    )
    assert status["is_loaded"] is True
    assert status["family"] == "flux.2-klein"
    assert status["pipeline_class"] == "Flux2KleinPipeline"
    # _smart_base_repo picks the distilled 4B (not the Base) for the
    # "FLUX.2-klein-4B-GGUF" repo name. The Base variant kicks in only
    # when "base" is part of the repo id.
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-4B"
    assert status["base_repo_source"] == "name_heuristic"
    assert status["base_repo_confidence"] == "heuristic"
    assert status["base_repo_variant"] == "distilled-4b"
    assert status["gguf_filename"] == "flux-2-klein-4b-Q4_K_S.gguf"
    contract = status["sampling_contract"]
    assert contract["family"] == "flux.2-klein"
    assert contract["gguf"] is True
    assert contract["base_repo_variant"] == "distilled-4b"
    assert contract["pipeline_is_distilled"] is True
    assert contract["guidance_semantics"] == "distilled_single_pass"
    assert contract["default_steps"] == 4
    assert contract["default_guidance_scale"] == 1.0
    assert contract["scheduler_class"] == "_FakeScheduler"


def test_load_model_pipeline_repo_with_transformer_gguf_component(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "black-forest-labs/FLUX.2-klein-base-4B",
        transformer_gguf_repo = "unsloth/FLUX.2-klein-base-4B-GGUF",
        transformer_gguf_filename = "flux-2-klein-base-4b-Q4_K_M.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["repo_id"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert status["pipeline_repo"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert status["base_repo_source"] == "full_repo"
    assert status["base_repo_variant"] == "base-4b"
    assert status["gguf_filename"] == "flux-2-klein-base-4b-Q4_K_M.gguf"
    assert status["transformer_gguf_repo"] == "unsloth/FLUX.2-klein-base-4B-GGUF"
    assert status["transformer_gguf_filename"] == "flux-2-klein-base-4b-Q4_K_M.gguf"
    assert status["sampling_contract"]["default_steps"] == 50
    assert status["sampling_contract"]["default_guidance_scale"] == 4.0
    assert status["component_sources"]["pipeline"]["repo"] == (
        "black-forest-labs/FLUX.2-klein-base-4B"
    )
    assert status["component_sources"]["transformer"] == {
        "source": "gguf",
        "repo": "unsloth/FLUX.2-klein-base-4B-GGUF",
        "filename": "flux-2-klein-base-4b-Q4_K_M.gguf",
    }
    internal = backend.status(include_internal = True)
    assert internal["active_repo_id"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert internal["active_diffusion_gguf_repo"] == (
        "unsloth/FLUX.2-klein-base-4B-GGUF"
    )


def test_load_model_unknown_full_diffusers_repo_uses_generic_pipeline(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/new-diffusion-model",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["family"] == "diffusers"
    assert status["pipeline_class"] == "DiffusionPipeline"
    assert status["pipeline_repo"] == "owner/new-diffusion-model"
    assert status["base_repo"] == "owner/new-diffusion-model"
    assert status["component_sources"]["transformer"] == {
        "source": "pipeline_repo",
        "repo": "owner/new-diffusion-model",
    }


def test_load_model_flux2_dev_text_encoder_gguf(monkeypatch):
    """FLUX.2 dev can pair a transformer GGUF with a lazy GGUF
    Mistral text encoder so diffusers does not materialize the 24B
    text model in bf16 resident VRAM."""
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    patch_calls: list[tuple[Any, str, bool | None]] = []

    class _FakeLazyTextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.base_repo_or_path = base_repo_or_path
            inst.compute_dtype = compute_dtype
            inst.resident_device = resident_device
            inst.token = token
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    fake_text_mod.LazyFlux2MistralTextEncoder = _FakeLazyTextEncoder
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: patch_calls.append((root, resident_device, pin_memory)) or 7
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-dev-GGUF",
        gguf_filename = "flux2-dev-Q4_K_M.gguf",
        text_encoder_gguf_filename = "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["family"] == "flux.2"
    assert status["text_encoder_gguf_repo"] == "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
    assert (
        status["text_encoder_gguf_filename"]
        == "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
    )
    assert _FakeLazyTextEncoder.calls == [
        {
            "path": "/fake/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
            "base_repo_or_path": "black-forest-labs/FLUX.2-dev",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert backend._pipe.kwargs["text_encoder"].path == _FakeLazyTextEncoder.calls[0]["path"]

    internal = backend.status(include_internal = True)
    assert internal["active_text_encoder_gguf_repo"] == "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
    assert (
        internal["active_text_encoder_gguf_filename"]
        == "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
    )


def test_load_model_flux2_dev_text_encoder_gguf_cpu_resident_when_offloading(monkeypatch):
    """When model CPU offload is enabled on CUDA, lazy text-encoder
    GGUF weights should stay CPU-resident and copy only the active
    quantized tensor/chunk during forward."""
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    patch_calls: list[tuple[Any, str, bool | None]] = []

    class _FakeLazyTextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.resident_device = resident_device
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    fake_text_mod.LazyFlux2MistralTextEncoder = _FakeLazyTextEncoder
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: patch_calls.append((root, resident_device, pin_memory)) or 7
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-dev-GGUF",
        gguf_filename = "flux2-dev-Q4_K_M.gguf",
        text_encoder_gguf_filename = "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
        enable_model_cpu_offload = True,
        offload_policy = "aggressive",
    )

    assert status["is_loaded"] is True
    assert _FakeLazyTextEncoder.calls[-1]["resident_device"] == "cpu"
    assert backend._pipe.kwargs["text_encoder"].resident_device == "cpu"
    assert patch_calls == [
        (backend._pipe.kwargs["transformer"], "cpu", True),
        (backend._pipe.kwargs["text_encoder"], "cpu", True),
    ]


def test_load_model_text_encoder_gguf_cpu_resident_without_full_cpu_offload(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    patch_calls: list[tuple[Any, str, bool | None]] = []

    class _FakeLazyTextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.resident_device = resident_device
            inst.to_calls = []
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

        def to(self, device):
            self.to_calls.append(device)
            return self

    fake_text_mod.LazyFlux2MistralTextEncoder = _FakeLazyTextEncoder
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: patch_calls.append((root, resident_device, pin_memory)) or 7
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-dev-GGUF",
        gguf_filename = "flux2-dev-Q4_K_M.gguf",
        text_encoder_gguf_filename = "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf",
        enable_model_cpu_offload = False,
        gguf_quantized_cpu_resident = True,
        gguf_pin_cpu_resident = True,
    )

    assert status["is_loaded"] is True
    assert status["gguf_quantized_cpu_resident"] is True
    assert status["gguf_pin_cpu_resident"] is True
    assert backend._cpu_offload_enabled is False
    assert getattr(backend._pipe, "cpu_offload", False) is False
    assert backend._pipe.device == "cuda"
    assert _FakeLazyTextEncoder.calls[-1]["resident_device"] == "cpu"
    assert backend._pipe.kwargs["text_encoder"].resident_device == "cpu"
    assert backend._pipe.kwargs["text_encoder"].to_calls == ["cuda"]
    assert patch_calls == [
        (backend._pipe.kwargs["transformer"], "cpu", True),
        (backend._pipe.kwargs["text_encoder"], "cpu", True),
    ]


def test_load_model_diffusion_gguf_cpu_resident_when_offloading(monkeypatch):
    """Diffusion GGUF weights should follow the same CPU-resident
    quantized-weight contract as the lazy text encoder when model CPU
    offload is active on CUDA."""
    _install_fake_diffusers(monkeypatch)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 123

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )
    monkeypatch.setattr(d, "_patch_gguf_modules_for_resident_device", _fake_patch)

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        enable_model_cpu_offload = True,
    )

    assert status["is_loaded"] is True
    transformer = backend._pipe.kwargs["transformer"]
    assert calls == [(transformer, "cpu", True)]


def test_load_model_diffusion_gguf_cpu_resident_without_full_cpu_offload(monkeypatch):
    """Hybrid mode keeps packed GGUF weights CPU-resident without
    installing Diffusers' full model CPU offload hooks."""
    _install_fake_diffusers(monkeypatch)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 123

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )
    monkeypatch.setattr(d, "_patch_gguf_modules_for_resident_device", _fake_patch)

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        enable_model_cpu_offload = False,
        gguf_quantized_cpu_resident = True,
        gguf_pin_cpu_resident = True,
    )

    assert status["is_loaded"] is True
    assert status["gguf_quantized_cpu_resident"] is True
    assert status["gguf_pin_cpu_resident"] is True
    assert backend._cpu_offload_enabled is False
    assert getattr(backend._pipe, "cpu_offload", False) is False
    assert backend._pipe.device == "cuda"
    transformer = backend._pipe.kwargs["transformer"]
    assert calls == [(transformer, "cpu", True)]


def test_resolve_diffusion_offload_policy_modes(monkeypatch):
    import core.inference.diffusion as d

    monkeypatch.delenv("UNSLOTH_STUDIO_GGUF_PIN_CPU_RESIDENT", raising = False)

    assert d._resolve_diffusion_offload_policy(
        offload_policy = "aggressive",
        enable_model_cpu_offload = False,
        gguf_quantized_cpu_resident = False,
        gguf_pin_cpu_resident = False,
    ) == ("aggressive", True, True, True)
    assert d._resolve_diffusion_offload_policy(
        offload_policy = "less-aggressive",
        enable_model_cpu_offload = True,
        gguf_quantized_cpu_resident = False,
        gguf_pin_cpu_resident = False,
    ) == ("less_aggressive", False, False, True)
    assert d._resolve_diffusion_offload_policy(
        offload_policy = "none",
        enable_model_cpu_offload = True,
        gguf_quantized_cpu_resident = True,
        gguf_pin_cpu_resident = True,
    ) == ("none", False, False, False)
    assert d._resolve_diffusion_offload_policy(
        offload_policy = None,
        enable_model_cpu_offload = True,
        gguf_quantized_cpu_resident = None,
        gguf_pin_cpu_resident = None,
    ) == (None, True, True, True)
    assert d._resolve_diffusion_offload_policy(
        offload_policy = None,
        enable_model_cpu_offload = False,
        gguf_quantized_cpu_resident = True,
        gguf_pin_cpu_resident = None,
    ) == (None, False, True, False)


def test_balanced_gguf_cuda_cache_budget_is_automatic_and_headroom_aware(monkeypatch):
    import core.inference.diffusion as d

    monkeypatch.delenv("UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB", raising = False)
    gib = 1024**3

    assert d._balanced_gguf_cuda_cache_bytes(
        device = "cpu",
        free_bytes = 64 * gib,
        total_bytes = 64 * gib,
    ) == 0
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 16 * gib,
        total_bytes = 16 * gib,
    ) == 0
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 30 * gib,
        total_bytes = 24 * gib,
    ) == 2 * gib
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 64 * gib,
        total_bytes = 64 * gib,
    ) == 8 * gib
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 64 * gib,
        total_bytes = 48 * gib,
    ) == 4 * gib
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 10 * gib,
        total_bytes = 64 * gib,
    ) == 2 * gib
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 128 * gib,
        total_bytes = 96 * gib,
    ) == 16 * gib


def test_balanced_gguf_cuda_cache_budget_env_override(monkeypatch):
    import core.inference.diffusion as d

    gib = 1024**3
    monkeypatch.setenv("UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB", "0")
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 64 * gib,
        total_bytes = 64 * gib,
    ) == 0

    monkeypatch.setenv("UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB", "123")
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 16 * gib,
        total_bytes = 16 * gib,
    ) == 123 * 1024 * 1024

    monkeypatch.setenv("UNSLOTH_STUDIO_GGUF_CUDA_CACHE_MIB", "bad")
    assert d._balanced_gguf_cuda_cache_bytes(
        free_bytes = 64 * gib,
        total_bytes = 64 * gib,
    ) == 0


def test_load_model_offload_policy_aggressive(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 123

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )
    monkeypatch.setattr(d, "_patch_gguf_modules_for_resident_device", _fake_patch)

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        enable_model_cpu_offload = False,
        gguf_quantized_cpu_resident = False,
        gguf_pin_cpu_resident = False,
        offload_policy = "aggressive",
    )

    assert status["offload_policy"] == "aggressive"
    assert status["gguf_quantized_cpu_resident"] is True
    assert status["gguf_pin_cpu_resident"] is True
    assert backend._cpu_offload_enabled is True
    assert getattr(backend._pipe, "cpu_offload", False) is True
    assert not hasattr(backend._pipe, "device")
    transformer = backend._pipe.kwargs["transformer"]
    assert calls == [(transformer, "cpu", True)]


def test_load_model_auto_uses_curated_gguf_policy(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 123

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )
    monkeypatch.setattr(d, "_patch_gguf_modules_for_resident_device", _fake_patch)

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        enable_model_cpu_offload = True,
        offload_policy = None,
    )

    assert status["offload_policy"] == "less_aggressive"
    assert status["gguf_quantized_cpu_resident"] is False
    assert status["gguf_pin_cpu_resident"] is False
    assert backend._cpu_offload_enabled is False
    assert getattr(backend._pipe, "cpu_offload", False) is False
    assert backend._pipe.device == "cuda"
    assert calls == []


def test_load_model_auto_keeps_qwen_balanced(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 123

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )
    monkeypatch.setattr(d, "_patch_gguf_modules_for_resident_device", _fake_patch)
    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/Qwen-Image-GGUF",
        gguf_filename = "qwen-image-Q4_K_M.gguf",
        enable_model_cpu_offload = True,
        offload_policy = None,
    )

    assert status["offload_policy"] == "balanced"
    assert status["gguf_quantized_cpu_resident"] is True
    assert status["gguf_pin_cpu_resident"] is True
    assert backend._cpu_offload_enabled is False
    assert getattr(backend._pipe, "cpu_offload", False) is False
    assert backend._pipe.device == "cuda"
    transformer = backend._pipe.kwargs["transformer"]
    assert calls == [(transformer, "cpu", True)]


def test_load_model_offload_policy_balanced(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 123

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )
    monkeypatch.setattr(d, "_patch_gguf_modules_for_resident_device", _fake_patch)

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        enable_model_cpu_offload = True,
        gguf_quantized_cpu_resident = False,
        gguf_pin_cpu_resident = False,
        offload_policy = "balanced",
    )

    assert status["offload_policy"] == "balanced"
    assert status["gguf_quantized_cpu_resident"] is True
    assert status["gguf_pin_cpu_resident"] is True
    assert backend._cpu_offload_enabled is False
    assert getattr(backend._pipe, "cpu_offload", False) is False
    assert backend._pipe.device == "cuda"
    transformer = backend._pipe.kwargs["transformer"]
    assert calls == [(transformer, "cpu", True)]


def test_load_model_offload_policy_none(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    import core.inference.diffusion as d
    from core.inference.diffusion import get_diffusion_backend

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 123

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cuda", "fake_dtype"),
    )
    monkeypatch.setattr(d, "_patch_gguf_modules_for_resident_device", _fake_patch)

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        enable_model_cpu_offload = True,
        gguf_quantized_cpu_resident = True,
        gguf_pin_cpu_resident = True,
        offload_policy = "none",
    )

    assert status["offload_policy"] == "none"
    assert status["gguf_quantized_cpu_resident"] is False
    assert status["gguf_pin_cpu_resident"] is False
    assert backend._cpu_offload_enabled is False
    assert getattr(backend._pipe, "cpu_offload", False) is False
    assert backend._pipe.device == "cuda"
    assert calls == []


def test_load_model_torch_compile_regional(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        torch_compile = "regional",
        torch_compile_mode = "default",
        torch_compile_dynamic = True,
        torch_compile_fullgraph = False,
    )

    transformer = backend._pipe.transformer
    assert transformer.compile_repeated_blocks_calls == [
        {"mode": "default", "fullgraph": False, "dynamic": True}
    ]
    assert transformer.compile_calls == []
    assert status["torch_compile_config"]["scope"] == "regional"
    assert status["torch_compile_stats"]["compiled_components"][0]["component"] == "transformer"
    assert status["load_timings"]["torch_compile"] >= 0


def test_load_model_torch_compile_transformer(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        torch_compile = "transformer",
        torch_compile_mode = "reduce-overhead",
    )

    transformer = backend._pipe.transformer
    assert transformer.compile_calls == [{"mode": "reduce-overhead"}]
    assert transformer.compile_repeated_blocks_calls == []
    assert status["torch_compile_config"]["scope"] == "transformer"
    assert status["torch_compile_stats"]["compiled_components"][0]["method"] == "module.compile"


def test_load_model_torch_compile_pipeline(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        torch_compile = "pipeline",
        torch_compile_mode = "default",
        torch_compile_options = {"triton.cudagraphs": False},
    )

    transformer = backend._pipe.transformer
    assert transformer.compile_calls == [
        {
            "options": {"triton.cudagraphs": False},
        }
    ]
    assert status["torch_compile_config"]["scope"] == "pipeline"
    assert status["torch_compile_stats"]["compiled_components"][0]["component"] == "transformer"


def test_load_model_torch_compile_invalid_scope(monkeypatch):
    _install_fake_diffusers(monkeypatch)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(ValueError, match = "torch_compile must be one of"):
        backend.load_model(
            "unsloth/FLUX.2-klein-4B-GGUF",
            gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
            torch_compile = "everything",
        )


def test_patch_gguf_modules_for_resident_device_keeps_weight_resident():
    import torch

    from core.inference.diffusion import _patch_gguf_modules_for_resident_device

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            return self

    class _FakeGGUFLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = _FakeGGUFParameter(
                torch.ones(2, 2, dtype = torch.float32),
                quant_type = "Q4_K_M",
            )
            self.forward_devices: list[torch.device] = []

        def forward_native(self, inputs):
            self.forward_devices.append(self.weight.device)
            return torch.nn.functional.linear(inputs, self.weight)

        def forward(self, inputs):
            return self.forward_native(inputs)

    root = torch.nn.Sequential(_FakeGGUFLinear())
    layer = root[0]

    assert _patch_gguf_modules_for_resident_device(root, "cpu") == 1
    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == torch.float32
    assert layer.weight.quant_type == "Q4_K_M"

    root.to(dtype = torch.float64)

    assert layer.weight.device.type == "cpu"
    assert layer.weight.dtype == torch.float32
    assert layer.weight.quant_type == "Q4_K_M"
    out = root(torch.ones(1, 2, dtype = torch.float32))
    assert out.shape == (1, 2)
    assert layer.forward_devices == [torch.device("cpu")]


def test_patch_gguf_modules_for_resident_device_uses_shared_helper(monkeypatch):
    import core.inference.diffusion as d
    import core.inference.gguf_text_encoder as g

    calls: list[tuple[Any, str, bool | None]] = []

    def _fake_shared_patch(root, resident_device, *, pin_memory = None):
        calls.append((root, resident_device, pin_memory))
        return 42

    monkeypatch.setattr(g, "patch_gguf_text_encoder_for_resident_device", _fake_shared_patch)
    root = object()

    assert d._patch_gguf_modules_for_resident_device(root, "cpu", pin_memory = True) == 42
    assert calls == [(root, "cpu", True)]


def test_replace_diffusers_gguf_linear_with_studio_lazy_module(monkeypatch):
    import torch
    pytest.importorskip("diffusers")
    from diffusers.quantizers.gguf.utils import GGUFLinear

    import core.inference.diffusion as d
    import core.inference.gguf_text_encoder as g

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0, dtype = torch.uint8) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            self.quant_shape = tuple(data.shape)
            return self

    root = torch.nn.Sequential(GGUFLinear(2, 3, bias = True, compute_dtype = torch.float32))
    root[0].weight = _FakeGGUFParameter(
        torch.arange(6, dtype = torch.uint8).reshape(3, 2),
        quant_type = "fake",
    )
    root[0].bias = torch.nn.Parameter(torch.ones(3), requires_grad = False)

    monkeypatch.setattr(d, "_diffusers_gguf_fused_cuda_available", lambda: False)

    assert d._replace_diffusers_gguf_linear_parameters(root, torch.float32, resident_device = "cpu") == 1
    assert isinstance(root[0], g.LazyGGUFLinear)
    assert root[0]._resident_device == torch.device("cpu")
    assert root[0].qweight.device.type == "cpu"
    assert root[0].bias is not None


def test_replace_diffusers_gguf_linear_keeps_fused_diffusers_module(monkeypatch):
    import torch
    pytest.importorskip("diffusers")
    from diffusers.quantizers.gguf.utils import GGUFLinear

    import core.inference.diffusion as d

    root = torch.nn.Sequential(GGUFLinear(2, 3, bias = False, compute_dtype = torch.float32))
    root[0].weight.quant_type = "fake"

    monkeypatch.setattr(d, "_diffusers_gguf_fused_cuda_available", lambda: True)

    assert d._replace_diffusers_gguf_linear_parameters(root, torch.float32, resident_device = "cpu") == 0
    assert isinstance(root[0], GGUFLinear)


def test_materialize_gguf_embedding_parameters_dequantizes_logical_shape(monkeypatch):
    import torch

    import core.inference.diffusion as d

    calls: list[Any] = []

    def _fake_dequantize_gguf_parameter(weight, dtype = None):
        calls.append(weight)
        assert dtype is torch.bfloat16
        return torch.arange(6, dtype = torch.bfloat16).reshape(2, 3)

    monkeypatch.setattr(d, "_dequantize_diffusers_gguf_parameter", _fake_dequantize_gguf_parameter)

    root = torch.nn.Sequential(torch.nn.Embedding(2, 6))
    raw_weight = torch.nn.Parameter(torch.zeros(2, 6, dtype = torch.uint8), requires_grad = False)
    raw_weight.quant_type = "BF16"
    root[0].weight = raw_weight

    assert d._materialize_gguf_embedding_parameters(root, torch.bfloat16) == 1
    assert calls == [raw_weight]
    assert root[0].weight.shape == (2, 3)
    assert root[0].weight.dtype == torch.bfloat16
    assert root[0].weight.requires_grad is False
    assert not hasattr(root[0].weight, "quant_type")
    out = root(torch.tensor([0, 1]))
    assert out.shape == (2, 3)


def test_replace_gguf_conv2d_parameters_wraps_lazy_module(monkeypatch):
    import torch

    import core.inference.diffusion as d
    import core.inference.gguf_text_encoder as g

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        assert quant_type == "Q4"
        assert logical_shape == (1, 1, 1, 1)
        return torch.ones(logical_shape, dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    conv = torch.nn.Conv2d(1, 1, kernel_size = 1, bias = True)
    conv.bias.data.zero_()
    raw_weight = torch.nn.Parameter(
        torch.ones(1, 1, 1, 2, dtype = torch.uint8),
        requires_grad = False,
    )
    raw_weight.quant_type = "Q4"
    raw_weight.quant_shape = (1, 1, 1, 1)
    conv.weight = raw_weight
    root = torch.nn.Sequential(conv)

    assert d._replace_gguf_conv2d_parameters(
        root,
        torch.float32,
        resident_device = "cpu",
    ) == 1

    assert isinstance(root[0], g.LazyGGUFConv2d)
    assert root[0].qweight.device == torch.device("cpu")
    out = root(torch.full((1, 1, 2, 2), 4.0))
    torch.testing.assert_close(out, torch.full((1, 1, 2, 2), 4.0))


def test_replace_gguf_conv2d_parameters_keeps_quantized_bias_lazy(monkeypatch):
    import torch

    import core.inference.diffusion as d
    import core.inference.gguf_text_encoder as g

    def fake_dequant(qbuffer, quant_type, *, dtype = None, logical_shape = None):
        return qbuffer.to(dtype = dtype).reshape(logical_shape)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    conv = torch.nn.Conv2d(1, 1, kernel_size = 1, bias = True)
    raw_weight = torch.nn.Parameter(
        torch.ones(1, 1, 1, 1, dtype = torch.uint8),
        requires_grad = False,
    )
    raw_weight.quant_type = "Q4"
    raw_weight.quant_shape = (1, 1, 1, 1)
    raw_bias = torch.nn.Parameter(torch.tensor([2], dtype = torch.uint8), requires_grad = False)
    raw_bias.quant_type = "Q4"
    raw_bias.quant_shape = (1,)
    conv.weight = raw_weight
    conv.bias = raw_bias
    root = torch.nn.Sequential(conv)

    assert d._replace_gguf_conv2d_parameters(root, torch.float32, resident_device = "cpu") == 1

    assert isinstance(root[0], g.LazyGGUFConv2d)
    assert root[0].qweight.device == torch.device("cpu")
    assert root[0].qbias.device == torch.device("cpu")
    out = root(torch.full((1, 1, 1, 1), 3.0))
    torch.testing.assert_close(out, torch.tensor([[[[5.0]]]]))


def test_replace_gguf_embedding_parameters_wraps_lazy_module(monkeypatch):
    import torch

    import core.inference.diffusion as d
    import core.inference.gguf_text_encoder as g

    seen_rows: list[torch.Tensor] = []

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        seen_rows.append(qweight.clone())
        assert quant_type == "Q4"
        assert logical_shape is None
        return qweight.to(dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    embedding = torch.nn.Embedding(4, 2)
    raw_weight = torch.nn.Parameter(
        torch.tensor(
            [
                [10, 11],
                [20, 21],
                [30, 31],
                [40, 41],
            ],
            dtype = torch.uint8,
        ),
        requires_grad = False,
    )
    raw_weight.quant_type = "Q4"
    raw_weight.quant_shape = (4, 2)
    embedding.weight = raw_weight
    root = torch.nn.Sequential(embedding)

    assert d._replace_gguf_embedding_parameters(
        root,
        torch.float32,
        resident_device = "cpu",
    ) == 1

    assert isinstance(root[0], g.LazyGGUFEmbedding)
    assert root[0].qweight.device == torch.device("cpu")
    out = root(torch.tensor([[3, 1, 3]]))
    torch.testing.assert_close(
        seen_rows[0],
        torch.tensor([[20, 21], [40, 41]], dtype = torch.uint8),
    )
    torch.testing.assert_close(
        out,
        torch.tensor([[[40.0, 41.0], [20.0, 21.0], [40.0, 41.0]]]),
    )


def test_replace_gguf_embedding_parameters_materializes_unknown_shape_with_shared_dequant(
    monkeypatch,
):
    import torch

    import core.inference.diffusion as d
    import core.inference.gguf_text_encoder as g

    calls: list[dict[str, Any]] = []

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        calls.append(
            {
                "qweight": qweight.clone(),
                "quant_type": quant_type,
                "dtype": dtype,
                "logical_shape": logical_shape,
            }
        )
        return torch.arange(6, dtype = dtype).reshape(2, 3)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    embedding = torch.nn.Embedding(2, 6)
    raw_weight = torch.nn.Parameter(
        torch.arange(6, dtype = torch.uint8).reshape(2, 3),
        requires_grad = False,
    )
    raw_weight.quant_type = "IQ1_M"
    embedding.weight = raw_weight
    root = torch.nn.Sequential(embedding)

    assert d._replace_gguf_embedding_parameters(root, torch.float32, resident_device = "cpu") == 1

    assert len(calls) == 1
    torch.testing.assert_close(
        calls[0]["qweight"],
        torch.arange(6, dtype = torch.uint8).reshape(2, 3),
    )
    assert calls[0]["quant_type"] == "IQ1_M"
    assert calls[0]["dtype"] is torch.float32
    assert calls[0]["logical_shape"] is None
    assert root[0].weight.shape == (2, 3)
    assert root[0].weight.dtype is torch.float32
    assert root[0].weight.requires_grad is False


def test_replace_gguf_norm_parameters_wraps_lazy_layer_norm(monkeypatch):
    import torch

    import core.inference.diffusion as d
    import core.inference.gguf_text_encoder as g

    calls: list[torch.Tensor] = []

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        calls.append(qweight.clone())
        assert quant_type == "Q4"
        return qweight.to(dtype = dtype).reshape(logical_shape)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)

    root = torch.nn.Sequential(torch.nn.LayerNorm(3))
    raw_weight = torch.nn.Parameter(
        torch.tensor([1, 1, 1], dtype = torch.uint8),
        requires_grad = False,
    )
    raw_weight.quant_type = "Q4"
    raw_weight.quant_shape = (3,)
    root[0].weight = raw_weight

    assert d._replace_gguf_norm_parameters(root, torch.float32, resident_device = "cpu") == 1
    assert isinstance(root[0], g.LazyGGUFLayerNorm)
    assert root[0].qweight.device == torch.device("cpu")
    assert root(torch.ones(2, 3)).shape == (2, 3)
    torch.testing.assert_close(calls[0], torch.tensor([1, 1, 1], dtype = torch.uint8))


def test_patch_diffusers_gguf_checkpoint_loader_no_copy_is_scoped(monkeypatch):
    import core.inference.diffusion as d

    fake_diffusers = types.ModuleType("diffusers")
    fake_models = types.ModuleType("diffusers.models")
    fake_model_loading_utils = types.ModuleType("diffusers.models.model_loading_utils")

    def _original_loader(path):
        return {"original": path}

    fake_model_loading_utils.load_gguf_checkpoint = _original_loader
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", fake_models)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.models.model_loading_utils",
        fake_model_loading_utils,
    )

    with d._patch_diffusers_gguf_checkpoint_loader_no_copy():
        assert (
            fake_model_loading_utils.load_gguf_checkpoint
            is d._load_gguf_checkpoint_no_copy
        )

    assert fake_model_loading_utils.load_gguf_checkpoint is _original_loader


def test_load_gguf_checkpoint_no_copy_preserves_numpy_storage(monkeypatch):
    import numpy as np
    import torch

    import core.inference.diffusion as d

    class _QType:
        F32 = object()
        F16 = object()
        BF16 = object()
        IQ1_M = object()
        UNKNOWN = object()

    tensor_data = np.arange(12, dtype = np.uint8).reshape(3, 4)
    dense_data = np.arange(6, dtype = np.float32)
    f16_data = np.arange(6, dtype = np.float16)
    native_iq_data = np.arange(56, dtype = np.uint8).reshape(1, 56)

    class _FakeField:
        def __init__(self, values):
            self.parts = [np.array([value], dtype = np.int32) for value in values]
            self.data = list(range(len(values)))

    class _FakeTensor:
        def __init__(self, name, tensor_type, data, shape):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = shape

    class _FakeReader:
        def __init__(self, path):
            self.path = path
            self.tensors = [
                _FakeTensor("embed.weight", _QType.BF16, tensor_data, (6, 2)),
                _FakeTensor("dense.weight", _QType.F32, dense_data, (6,)),
                _FakeTensor("dense_f16.weight", _QType.F16, f16_data, (6,)),
                _FakeTensor("native_iq.weight", _QType.IQ1_M, native_iq_data, (256,)),
            ]

        def get_field(self, name):
            if name == "comfy.gguf.orig_shape.embed.weight":
                return _FakeField([2, 6])
            if name == "comfy.gguf.orig_shape.dense.weight":
                return _FakeField([2, 3])
            if name == "comfy.gguf.orig_shape.dense_f16.weight":
                return _FakeField([2, 3])
            if name == "comfy.gguf.orig_shape.native_iq.weight":
                return _FakeField([256])
            return None

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            return self

    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = _FakeReader
    fake_gguf.GGMLQuantizationType = _QType
    fake_gguf.quants = SimpleNamespace(_type_traits = {_QType.IQ1_M: object()})

    fake_utils = types.ModuleType("diffusers.quantizers.gguf.utils")
    fake_utils.GGUFParameter = _FakeGGUFParameter
    fake_utils.SUPPORTED_GGUF_QUANT_TYPES = {_QType.BF16}

    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", types.ModuleType("diffusers.quantizers"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.gguf", types.ModuleType("diffusers.quantizers.gguf"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.gguf.utils", fake_utils)

    parsed = d._load_gguf_checkpoint_no_copy("/tmp/fake.gguf")
    weight = parsed["embed.weight"]
    dense = parsed["dense.weight"]
    dense_f16 = parsed["dense_f16.weight"]
    native_iq = parsed["native_iq.weight"]

    assert isinstance(weight, _FakeGGUFParameter)
    assert weight.quant_type is _QType.BF16
    assert weight.quant_shape == (2, 6)
    assert weight.data_ptr() == torch.from_numpy(tensor_data).data_ptr()
    assert getattr(weight, "_unsloth_gguf_reader").path == "/tmp/fake.gguf"
    assert dense.shape == (2, 3)
    assert dense.data_ptr() == torch.from_numpy(dense_data).data_ptr()
    assert not isinstance(dense_f16, _FakeGGUFParameter)
    assert dense_f16.shape == (2, 3)
    assert dense_f16.dtype is torch.float16
    assert dense_f16.data_ptr() == torch.from_numpy(f16_data).data_ptr()
    assert isinstance(native_iq, _FakeGGUFParameter)
    assert native_iq.quant_type is _QType.IQ1_M
    assert native_iq.quant_shape == (256,)
    assert native_iq.data_ptr() == torch.from_numpy(native_iq_data).data_ptr()


def test_load_gguf_checkpoint_no_copy_accepts_observed_unsloth_quant_types(monkeypatch):
    import numpy as np
    import torch

    import core.inference.diffusion as d

    _QType = type(
        "_QType",
        (),
        {quant_name: object() for quant_name in _OBSERVED_UNSLOTH_GGUF_QUANT_NAMES},
    )
    quant_by_name = {
        quant_name: getattr(_QType, quant_name)
        for quant_name in _OBSERVED_UNSLOTH_GGUF_QUANT_NAMES
    }

    class _FakeField:
        def __init__(self, values):
            self.parts = [np.array([value], dtype = np.int32) for value in values]
            self.data = list(range(len(values)))

    class _FakeTensor:
        def __init__(self, name, tensor_type, data, shape):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = shape

    class _FakeReader:
        def __init__(self, path):
            self.path = path
            self.tensors = []
            for quant_name, quant_type in quant_by_name.items():
                if quant_name == "F32":
                    data = np.array([1.0], dtype = np.float32)
                elif quant_name == "F16":
                    data = np.array([1.0], dtype = np.float16)
                else:
                    data = np.zeros((1, 8), dtype = np.uint8)
                self.tensors.append(
                    _FakeTensor(
                        f"{quant_name}.weight",
                        quant_type,
                        data,
                        data.shape,
                    )
                )

        def get_field(self, name):
            if not name.startswith("comfy.gguf.orig_shape."):
                return None
            return _FakeField([1, 1])

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            return self

    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = _FakeReader
    fake_gguf.GGMLQuantizationType = _QType
    fake_gguf.quants = SimpleNamespace(
        _type_traits = {
            quant_by_name[quant_name]: object()
            for quant_name in _OBSERVED_NATIVE_FALLBACK_QUANT_NAMES
        }
    )

    fake_utils = types.ModuleType("diffusers.quantizers.gguf.utils")
    fake_utils.GGUFParameter = _FakeGGUFParameter
    fake_utils.SUPPORTED_GGUF_QUANT_TYPES = {
        quant_by_name[quant_name]
        for quant_name in _OBSERVED_DIFFUSERS_GGUF_QUANT_NAMES
    }

    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", types.ModuleType("diffusers.quantizers"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.gguf", types.ModuleType("diffusers.quantizers.gguf"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.gguf.utils", fake_utils)

    parsed = d._load_gguf_checkpoint_no_copy("/tmp/fake.gguf")

    assert set(parsed) == {
        f"{quant_name}.weight"
        for quant_name in _OBSERVED_UNSLOTH_GGUF_QUANT_NAMES
    }
    for quant_name, quant_type in quant_by_name.items():
        weight = parsed[f"{quant_name}.weight"]
        if quant_name in {"F16", "F32"}:
            assert not isinstance(weight, _FakeGGUFParameter)
            continue
        assert isinstance(weight, _FakeGGUFParameter)
        assert weight.quant_type is quant_type
        assert weight.quant_shape == (1, 1)


def test_load_gguf_checkpoint_no_copy_rejects_unknown_native_quant(monkeypatch):
    import numpy as np
    import torch

    import core.inference.diffusion as d

    class _QType:
        F32 = object()
        F16 = object()
        BF16 = object()
        UNKNOWN = object()

    class _FakeTensor:
        def __init__(self, name, tensor_type, data, shape):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = shape

    class _FakeReader:
        def __init__(self, path):
            self.tensors = [
                _FakeTensor(
                    "unsupported.weight",
                    _QType.UNKNOWN,
                    np.arange(8, dtype = np.uint8),
                    (256,),
                )
            ]

        def get_field(self, name):
            return None

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            return self

    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = _FakeReader
    fake_gguf.GGMLQuantizationType = _QType
    fake_gguf.quants = SimpleNamespace(_type_traits = {})

    fake_utils = types.ModuleType("diffusers.quantizers.gguf.utils")
    fake_utils.GGUFParameter = _FakeGGUFParameter
    fake_utils.SUPPORTED_GGUF_QUANT_TYPES = {_QType.BF16}

    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", types.ModuleType("diffusers.quantizers"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.gguf", types.ModuleType("diffusers.quantizers.gguf"))
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.gguf.utils", fake_utils)

    with pytest.raises(ValueError, match = "unsupported.weight"):
        d._load_gguf_checkpoint_no_copy("/tmp/fake.gguf")


def test_load_model_text_encoder_gguf_rejects_unsupported_family_without_component(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "text_encoder_gguf_component"):
        backend.load_model(
            "city96/FLUX.1-dev-gguf",
            gguf_filename = "flux1-dev-Q4_K_S.gguf",
            text_encoder_gguf_filename = "text.gguf",
        )


def test_load_model_text_encoder_gguf_rejects_wrong_builtin_family_with_detected_arch(
    monkeypatch,
    tmp_path,
):
    import core.inference.gguf_text_encoder as g

    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    text_repo = tmp_path / "text"
    text_repo.mkdir()
    (text_repo / "text.gguf").touch()
    monkeypatch.setattr(
        g,
        "inspect_text_encoder_gguf",
        lambda path: SimpleNamespace(architecture = "qwen2vl"),
    )

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "Detected text GGUF architecture: qwen2vl"):
        backend.load_model(
            "city96/FLUX.1-dev-gguf",
            gguf_filename = "flux1-dev-Q4_K_S.gguf",
            text_encoder_gguf_repo = str(text_repo),
            text_encoder_gguf_filename = "text.gguf",
        )


def test_load_model_qwen_image_text_encoder_gguf_uses_qwen2vl_loader(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    text_repo = tmp_path / "text"
    text_repo.mkdir()
    text_path = text_repo / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    mmproj_path = text_repo / "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf"
    text_path.touch()
    mmproj_path.touch()

    class _FakeQwenTextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            mmproj_gguf_path = None,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.mmproj_gguf_path = mmproj_gguf_path
            inst.resident_device = resident_device
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "mmproj_gguf_path": mmproj_gguf_path,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    fake_text_mod.LazyQwen2VLTextEncoder = _FakeQwenTextEncoder
    fake_text_mod.LazyFlux2MistralTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.inspect_text_encoder_gguf = lambda path: SimpleNamespace(
        architecture = "qwen2vl",
        mmproj_path = mmproj_path,
    )
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/Qwen-Image-GGUF",
        gguf_filename = "qwen-image-Q4_K_M.gguf",
        text_encoder_gguf_repo = str(text_repo),
        text_encoder_gguf_filename = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["family"] == "qwen-image"
    assert status["sampling_contract"]["guidance_kwarg"] == "true_cfg_scale"
    assert (
        status["sampling_contract"]["guidance_semantics"]
        == "true_classifier_free_guidance"
    )
    assert status["sampling_contract"]["has_default_negative_prompt"] is True
    assert _FakeQwenTextEncoder.calls == [
        {
            "path": str(text_path),
            "base_repo_or_path": "Qwen/Qwen-Image",
            "mmproj_gguf_path": mmproj_path,
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert backend._pipe.kwargs["text_encoder"].path == str(text_path)
    assert backend._pipe.kwargs["text_encoder"].mmproj_gguf_path == mmproj_path


def test_load_model_qwen_image_text_encoder_gguf_downloads_remote_mmproj(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)

    fake_hub = types.ModuleType("huggingface_hub")
    text_path = tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    mmproj_path = tmp_path / "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf"
    text_path.touch()
    mmproj_path.touch()
    downloads: list[tuple[str, str]] = []

    def fake_download(repo_id, filename, token = None, subfolder = None, **_kwargs):
        path = f"{subfolder}/{filename}" if subfolder else filename
        downloads.append((repo_id, path))
        if repo_id == "unsloth/Qwen2.5-VL-7B-Instruct-GGUF":
            if filename == "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf":
                return str(text_path)
            if filename == "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf":
                return str(mmproj_path)
            raise FileNotFoundError(filename)
        return f"/fake/{repo_id}/{path}"

    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")

    class _FakeQwenTextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            mmproj_gguf_path = None,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.mmproj_gguf_path = mmproj_gguf_path
            cls.calls.append(
                {
                    "path": path,
                    "mmproj_gguf_path": mmproj_gguf_path,
                }
            )
            return inst

    fake_text_mod.LazyQwen2VLTextEncoder = _FakeQwenTextEncoder
    fake_text_mod.inspect_text_encoder_gguf = lambda path: SimpleNamespace(
        architecture = "qwen2vl",
        mmproj_path = None,
    )
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/Qwen-Image-GGUF",
        gguf_filename = "qwen-image-Q4_K_M.gguf",
        text_encoder_gguf_repo = "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        text_encoder_gguf_filename = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert (
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf",
    ) in downloads
    assert _FakeQwenTextEncoder.calls[-1] == {
        "path": str(text_path),
        "mmproj_gguf_path": mmproj_path,
    }


def test_load_model_z_image_text_encoder_gguf_uses_qwen3_loader(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    text_repo = tmp_path / "text"
    text_repo.mkdir()
    text_path = text_repo / "Qwen3-4B-UD-Q4_K_XL.gguf"
    text_path.touch()

    class _FakeQwen3TextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.resident_device = resident_device
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    fake_text_mod.LazyQwen3TextEncoder = _FakeQwen3TextEncoder
    fake_text_mod.LazyQwen2VLTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.LazyFlux2MistralTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.inspect_text_encoder_gguf = lambda path: SimpleNamespace(
        architecture = "qwen3",
        mmproj_path = None,
    )
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/Z-Image-Turbo-GGUF",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        text_encoder_gguf_repo = str(text_repo),
        text_encoder_gguf_filename = "Qwen3-4B-UD-Q4_K_XL.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["family"] == "z-image-turbo"
    assert _FakeQwen3TextEncoder.calls == [
        {
            "path": str(text_path),
            "base_repo_or_path": "Tongyi-MAI/Z-Image-Turbo",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert backend._pipe.kwargs["text_encoder"].path == str(text_path)
    assert backend._pipe.kwargs["text_encoder"].resident_device is None


def test_load_model_flux1_text_encoder_gguf_uses_t5_loader_as_text_encoder_2(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    text_repo = tmp_path / "text"
    text_repo.mkdir()
    text_path = text_repo / "t5-v1_1-xxl-encoder-Q3_K_S.gguf"
    text_path.touch()

    class _FakeT5TextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            subfolder,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.subfolder = subfolder
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "subfolder": subfolder,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    fake_text_mod.LazyT5TextEncoder = _FakeT5TextEncoder
    fake_text_mod.LazyQwen3TextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.LazyQwen2VLTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.LazyFlux2MistralTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.inspect_text_encoder_gguf = lambda path: SimpleNamespace(
        architecture = "t5encoder",
        mmproj_path = None,
    )
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.1-dev-GGUF",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        text_encoder_gguf_repo = str(text_repo),
        text_encoder_gguf_filename = "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["family"] == "flux.1"
    assert _FakeT5TextEncoder.calls == [
        {
            "path": str(text_path),
            "base_repo_or_path": "black-forest-labs/FLUX.1-dev",
            "subfolder": "text_encoder_2",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert "text_encoder" not in backend._pipe.kwargs
    assert backend._pipe.kwargs["text_encoder_2"].path == str(text_path)


def test_load_model_sd3_text_encoder_gguf_uses_t5_loader_as_text_encoder_3(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    text_repo = tmp_path / "text"
    text_repo.mkdir()
    text_path = text_repo / "t5-v1_1-xxl-encoder-Q3_K_S.gguf"
    text_path.touch()

    class _FakeT5TextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            subfolder,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.subfolder = subfolder
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "subfolder": subfolder,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    fake_text_mod.LazyT5TextEncoder = _FakeT5TextEncoder
    fake_text_mod.LazyQwen3TextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.LazyQwen2VLTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.LazyFlux2MistralTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.inspect_text_encoder_gguf = lambda path: SimpleNamespace(
        architecture = "t5encoder",
        mmproj_path = None,
    )
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/stable-diffusion-3-medium-GGUF",
        gguf_filename = "sd3-medium-Q4_K_M.gguf",
        text_encoder_gguf_repo = str(text_repo),
        text_encoder_gguf_filename = "t5-v1_1-xxl-encoder-Q3_K_S.gguf",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert status["family"] == "stable-diffusion-3"
    assert _FakeT5TextEncoder.calls == [
        {
            "path": str(text_path),
            "base_repo_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
            "subfolder": "text_encoder_3",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert "text_encoder" not in backend._pipe.kwargs
    assert backend._pipe.kwargs["text_encoder_3"].path == str(text_path)


@pytest.mark.parametrize(
    ("architecture", "loader_name", "filename"),
    [
        ("llama", "LazyLlamaTextEncoder", "llama-text-Q4_K_M.gguf"),
        ("qwen3vl", "LazyQwen3VLTextEncoder", "qwen3vl-text-Q4_K_M.gguf"),
        ("gemma3", "LazyGemma3TextEncoder", "gemma3-text-Q4_K_M.gguf"),
    ],
)
def test_load_model_explicit_text_encoder_component_routes_generic_architectures(
    monkeypatch,
    tmp_path,
    architecture,
    loader_name,
    filename,
):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    text_repo = tmp_path / "text"
    text_repo.mkdir()
    text_path = text_repo / filename
    text_path.touch()

    class _FakeGenericTextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            subfolder,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            inst.subfolder = subfolder
            inst.resident_device = resident_device
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "subfolder": subfolder,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    setattr(fake_text_mod, loader_name, _FakeGenericTextEncoder)
    fake_text_mod.inspect_text_encoder_gguf = lambda path: SimpleNamespace(
        architecture = architecture,
        mmproj_path = None,
    )
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "city96/FLUX.1-dev-gguf",
        gguf_filename = "flux1-dev-Q4_K_M.gguf",
        text_encoder_gguf_repo = str(text_repo),
        text_encoder_gguf_filename = filename,
        text_encoder_gguf_component = "text_encoder",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert _FakeGenericTextEncoder.calls == [
        {
            "path": str(text_path),
            "base_repo_or_path": "black-forest-labs/FLUX.1-dev",
            "subfolder": "text_encoder",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert backend._pipe.kwargs["text_encoder"].path == str(text_path)


def test_load_model_flux2_mistral_text_encoder_overrides_generic_llama_architecture(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)

    fake_text_mod = types.ModuleType("core.inference.gguf_text_encoder")
    text_repo = tmp_path / "text"
    text_repo.mkdir()
    text_path = text_repo / "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
    text_path.touch()

    class _FakeFlux2TextEncoder:
        calls: list[dict[str, Any]] = []

        @classmethod
        def from_gguf(
            cls,
            path,
            *,
            base_repo_or_path,
            compute_dtype,
            resident_device = None,
            token = None,
        ):
            inst = cls()
            inst.path = path
            cls.calls.append(
                {
                    "path": path,
                    "base_repo_or_path": base_repo_or_path,
                    "compute_dtype": compute_dtype,
                    "resident_device": resident_device,
                    "token": token,
                }
            )
            return inst

    fake_text_mod.LazyFlux2MistralTextEncoder = _FakeFlux2TextEncoder
    fake_text_mod.LazyLlamaTextEncoder = SimpleNamespace(
        from_gguf = lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader"))
    )
    fake_text_mod.inspect_text_encoder_gguf = lambda path: SimpleNamespace(
        architecture = "llama",
        mmproj_path = None,
    )
    fake_text_mod.patch_gguf_text_encoder_for_resident_device = (
        lambda root, resident_device, pin_memory = None: 0
    )
    monkeypatch.setitem(sys.modules, "core.inference.gguf_text_encoder", fake_text_mod)
    import core.inference as inference_pkg

    monkeypatch.setattr(inference_pkg, "gguf_text_encoder", fake_text_mod, raising = False)

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-dev-GGUF",
        gguf_filename = "flux2-dev-Q4_K_M.gguf",
        text_encoder_gguf_repo = str(text_repo),
        text_encoder_gguf_filename = text_path.name,
        text_encoder_gguf_component = "text_encoder",
        enable_model_cpu_offload = False,
    )

    assert status["is_loaded"] is True
    assert _FakeFlux2TextEncoder.calls == [
        {
            "path": str(text_path),
            "base_repo_or_path": "black-forest-labs/FLUX.2-dev",
            "compute_dtype": "fake_dtype",
            "resident_device": None,
            "token": None,
        }
    ]
    assert backend._pipe.kwargs["text_encoder"].path == str(text_path)


def test_load_model_rejects_unknown_text_encoder_component(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(ValueError, match = "text_encoder_gguf_component"):
        backend.load_model(
            "city96/FLUX.1-dev-gguf",
            gguf_filename = "flux1-dev-Q4_K_M.gguf",
            text_encoder_gguf_filename = "llama-text-Q4_K_M.gguf",
            text_encoder_gguf_component = "tokenizer",
        )


def test_load_model_recovers_after_failure(monkeypatch):
    _install_fake_diffusers(monkeypatch, raise_on_pipeline = True)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "Failed to load diffusion model"):
        backend.load_model(
            "unsloth/FLUX.2-klein-4B-GGUF",
            gguf_filename = "x.gguf",
        )
    # Failed load must leave the singleton unloaded but with last_error set.
    s = backend.status()
    assert s["is_loaded"] is False
    assert s["last_error"] and "simulated load failure" in s["last_error"]


def test_failed_swap_clears_previous_metadata(monkeypatch):
    """After a successful load, a subsequent failing load must NOT
    leave status() reporting the OLD repo/family/base_repo on top of
    is_loaded=false. The clear must be atomic with the pipe drop."""
    import sys

    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    # First load succeeds.
    backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
    )
    s_before = backend.status()
    assert s_before["is_loaded"] is True
    assert s_before["repo_id"] == "unsloth/FLUX.2-klein-4B-GGUF"

    # Replace from_pretrained on the SAME fake module with a raising one
    # without re-installing the rest of the fakes.
    fake = sys.modules["diffusers"]

    def _boom(cls, *a, **kw):
        raise RuntimeError("simulated swap failure")

    fake.Flux2KleinPipeline.from_pretrained = classmethod(_boom)

    with pytest.raises(RuntimeError, match = "Failed to load diffusion model"):
        backend.load_model(
            "unsloth/FLUX.2-dev-GGUF",
            gguf_filename = "flux2-dev-Q4_K_S.gguf",
        )

    s_after = backend.status()
    assert s_after["is_loaded"] is False
    # Critically: stale metadata from the previous successful load
    # must be cleared, not just the pipe.
    assert s_after["repo_id"] is None
    assert s_after["family"] is None
    assert s_after["base_repo"] is None
    assert s_after["gguf_filename"] is None
    assert s_after["last_error"] and "simulated swap failure" in s_after["last_error"]


def test_load_model_swap_drops_previous(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
    )
    first_pipe = backend._pipe
    backend.load_model(
        "unsloth/FLUX.2-dev-GGUF",
        gguf_filename = "flux2-dev-Q4_K_S.gguf",
    )
    assert backend._pipe is not first_pipe
    assert backend.status()["family"] == "flux.2"


def test_load_model_base_repo_override(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-9B-GGUF",
        gguf_filename = "flux-2-klein-9b-Q4_K_S.gguf",
        base_repo = "black-forest-labs/FLUX.2-klein-base-9B",
    )
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-9B"
    assert status["base_repo_source"] == "explicit"
    assert status["base_repo_confidence"] == "explicit"


def test_load_model_ambiguous_flux2_klein_gguf_requires_base_repo(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "base_repo candidates") as exc_info:
        backend.load_model(
            "owner/my-flux2-klein-finetune-GGUF",
            gguf_filename = "model-Q4_K_M.gguf",
        )
    message = str(exc_info.value)
    assert "black-forest-labs/FLUX.2-klein-4B" in message
    assert "black-forest-labs/FLUX.2-klein-base-9B" in message
    assert "original base repo" in message


def test_load_model_flux2_klein_uses_filename_variant_hint(monkeypatch):
    """Third-party finetune repos often have a generic repo name while
    the quant filename carries the useful variant marker. Accept that
    as a heuristic instead of forcing base_repo for every custom GGUF."""
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/my-flux2-klein-finetune-GGUF",
        gguf_filename = "my-flux2-klein-base-9b-UD-Q4_K_XL.gguf",
    )

    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-9B"
    assert status["base_repo_source"] == "filename_heuristic"
    assert status["base_repo_confidence"] == "heuristic"
    assert status["base_repo_variant"] == "base-9b"


def test_load_model_flux2_klein_rejects_conflicting_variant_hints(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "Conflicting FLUX.2 Klein"):
        backend.load_model(
            "owner/my-flux2-klein-base-4b-GGUF",
            gguf_filename = "my-flux2-klein-9b-UD-Q4_K_XL.gguf",
        )


def test_diffusion_gguf_inspection_detects_flux_comfy_signature():
    from core.inference.diffusion import _inspect_diffusion_gguf_tensor_names

    inspection = _inspect_diffusion_gguf_tensor_names(
        {"double_blocks.0.img_attn.proj.weight", "other.weight"},
        metadata = {"general.architecture": "flux"},
    )

    assert inspection.architecture == "flux"
    assert inspection.layout == "comfy"
    assert "flux_comfy" in inspection.matched_signatures
    assert "flux.2-klein" in inspection.family_hints
    assert inspection.warnings == ()


def test_diffusion_gguf_inspection_detects_layout_conflict():
    from core.inference.diffusion import _inspect_diffusion_gguf_tensor_names

    inspection = _inspect_diffusion_gguf_tensor_names(
        {
            "double_blocks.0.img_attn.proj.weight",
            "transformer_blocks.0.attn.norm_added_k.weight",
        },
        metadata = {"general.architecture": "flux"},
    )

    assert inspection.architecture == "flux"
    assert inspection.layout is None
    assert "flux_comfy" in inspection.matched_signatures
    assert "flux_diffusers" in inspection.matched_signatures
    assert any(warning.startswith("layout_conflict") for warning in inspection.warnings)


def test_diffusion_gguf_inspection_ernie_signature_overrides_wan_metadata():
    from core.inference.diffusion import _inspect_diffusion_gguf_tensor_names

    inspection = _inspect_diffusion_gguf_tensor_names(
        {
            "adaLN_modulation.1.weight",
            "layers.0.self_attention.to_q.weight",
            "layers.0.self_attention.to_out.0.weight",
        },
        metadata = {"general.architecture": "wan"},
    )

    assert inspection.architecture == "ernie_image"
    assert inspection.layout == "diffusers"
    assert "ernie_image" in inspection.matched_signatures
    assert "ernie-image" in inspection.family_hints
    assert "ernie-image-turbo" in inspection.family_hints
    assert "wan2-2-t2v" not in inspection.family_hints
    assert (
        "architecture_conflict:metadata=wan,signature=ernie_image"
        in inspection.warnings
    )


def test_resolve_flux2_klein_uses_gguf_metadata_variant_hint():
    from core.inference.diffusion import (
        DiffusionGGUFInspection,
        _resolve_diffusion_base_repo,
        detect_family,
    )

    fam = detect_family("owner/my-flux2-klein-finetune-GGUF")
    assert fam is not None
    resolution = _resolve_diffusion_base_repo(
        fam = fam,
        repo_id = "owner/my-flux2-klein-finetune-GGUF",
        gguf_filename = "model-Q4_K_M.gguf",
        base_repo = None,
        gguf_inspection = DiffusionGGUFInspection(
            architecture = "flux",
            layout = "comfy",
            family_hints = ("flux.2-klein",),
            variant_hints = (("metadata:general.name", "base-9b"),),
        ),
    )

    assert resolution.base_repo == "black-forest-labs/FLUX.2-klein-base-9B"
    assert resolution.source == "gguf_metadata"
    assert resolution.variant == "base-9b"


def test_variant_resolver_exposes_flux2_klein_candidates():
    from core.inference.diffusion import (
        _candidate_base_repo_message,
        _variant_from_text_for_family,
    )

    assert _variant_from_text_for_family(
        "flux.2-klein",
        "my-flux2-klein-base-9b-UD-Q4_K_XL.gguf",
    ) == "base-9b"
    assert _variant_from_text_for_family(
        "z-image",
        "z-image-turbo-Q4_K_M.gguf",
    ) is None

    candidates = _candidate_base_repo_message("flux.2-klein")
    assert "black-forest-labs/FLUX.2-klein-4B (distilled 4B)" in candidates
    assert "black-forest-labs/FLUX.2-klein-base-9B (base 9B)" in candidates


def test_load_model_flux2_klein_uses_gguf_metadata_hint_after_local_inspection(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)
    import core.inference.diffusion as d
    from core.inference.diffusion import DiffusionGGUFInspection, get_diffusion_backend

    repo_dir = tmp_path / "my-flux2-klein-finetune-GGUF"
    repo_dir.mkdir()
    (repo_dir / "model-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr(
        d,
        "_inspect_diffusion_gguf_file",
        lambda _path: DiffusionGGUFInspection(
            architecture = "flux",
            layout = "comfy",
            family_hints = ("flux.2-klein",),
            variant_hints = (("metadata:general.name", "distilled-9b"),),
        ),
    )

    backend = get_diffusion_backend()
    status = backend.load_model(
        str(repo_dir),
        gguf_filename = "model-Q4_K_M.gguf",
        family_override = "flux.2-klein",
    )

    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-9B"
    assert status["base_repo_source"] == "gguf_metadata"
    assert status["base_repo_variant"] == "distilled-9b"


def test_load_model_rejects_gguf_family_mismatch_after_local_inspection(
    monkeypatch,
    tmp_path,
):
    _install_fake_diffusers(monkeypatch)
    import core.inference.diffusion as d
    from core.inference.diffusion import DiffusionGGUFInspection, get_diffusion_backend

    repo_dir = tmp_path / "my-flux2-klein-finetune-GGUF"
    repo_dir.mkdir()
    (repo_dir / "model-Q4_K_M.gguf").write_bytes(b"fake")
    monkeypatch.setattr(
        d,
        "_inspect_diffusion_gguf_file",
        lambda _path: DiffusionGGUFInspection(
            architecture = "z_image",
            layout = "comfy",
            family_hints = ("z-image", "z-image-turbo"),
        ),
    )

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "does not match the GGUF architecture"):
        backend.load_model(
            str(repo_dir),
            gguf_filename = "model-Q4_K_M.gguf",
            family_override = "flux.2-klein",
        )


def test_load_model_ambiguous_flux2_klein_gguf_accepts_explicit_base_repo(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/my-flux2-klein-finetune-GGUF",
        gguf_filename = "model-Q4_K_M.gguf",
        base_repo = "black-forest-labs/FLUX.2-klein-base-4B",
    )

    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert status["base_repo_source"] == "explicit"
    assert status["base_repo_confidence"] == "explicit"
    assert status["sampling_contract"]["guidance_semantics"] == "classifier_free_guidance"


def test_load_model_gguf_only_repo_without_filename_errors(monkeypatch):
    """When the caller points at a -GGUF repo but forgets the filename,
    surface a clear error instead of calling from_pretrained on the
    GGUF-only repo (which 500s deep in diffusers)."""
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "looks like a GGUF-only repo"):
        backend.load_model("unsloth/FLUX.2-klein-4B-GGUF")


def test_smart_base_repo_picks_9b(monkeypatch):
    """For unsloth/FLUX.2-klein-9B-GGUF without an explicit base_repo,
    the backend must fall through to FLUX.2-klein-9B, not the 4B base."""
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-9B-GGUF",
        gguf_filename = "flux-2-klein-9b-Q4_K_S.gguf",
    )
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-9B"
    assert status["base_repo_source"] == "name_heuristic"
    assert status["base_repo_confidence"] == "heuristic"
    assert status["base_repo_variant"] == "distilled-9b"


def test_smart_base_repo_picks_base_9b(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-base-9B-GGUF",
        gguf_filename = "flux-2-klein-base-9b-Q4_K_S.gguf",
    )
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-9B"
    assert status["base_repo_variant"] == "base-9b"


def test_smart_base_repo_picks_base_4b(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-base-4B-GGUF",
        gguf_filename = "flux-2-klein-base-4b-Q4_K_S.gguf",
    )
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert status["base_repo_variant"] == "base-4b"
    assert status["sampling_contract"]["pipeline_is_distilled"] is False
    assert status["sampling_contract"]["guidance_semantics"] == "classifier_free_guidance"
    assert status["sampling_contract"]["default_steps"] == 50
    assert status["sampling_contract"]["default_guidance_scale"] == 4.0


def test_gguf_transformer_load_passes_config_subfolder_token(monkeypatch):
    """Diffusers-format GGUFs require config=<base_repo>+subfolder=
    transformer at from_single_file time; gated GGUFs also need the
    token. Verify all three kwargs are forwarded."""
    fake = _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    captured: dict = {}
    original = fake.Flux2Transformer2DModel.from_single_file.__func__

    def _capture(cls, path, **kw):
        captured.update(kw)
        return original(cls, path, **kw)

    fake.Flux2Transformer2DModel.from_single_file = classmethod(_capture)

    backend = get_diffusion_backend()
    backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        hf_token = "hf_test_token",
    )
    assert captured.get("config") == "black-forest-labs/FLUX.2-klein-4B"
    assert captured.get("subfolder") == "transformer"
    assert captured.get("token") == "hf_test_token"


def test_release_chat_backend_calls_unload_with_model_name(monkeypatch):
    """The safetensors backend unload helper must call unload_model
    with the active model name (the orchestrator's signature requires
    it). The previous behaviour swallowed TypeError and left the chat
    model resident, defeating the lifecycle handoff."""
    import sys
    import types

    fake_pkg = types.ModuleType("core.inference")
    calls: list = []

    class _Stub:
        active_model_name = "owner/some-model"

        def unload_model(self, name):
            calls.append(name)
            self.active_model_name = None
            return True

    stub = _Stub()
    fake_pkg.get_inference_backend = lambda: stub
    monkeypatch.setitem(sys.modules, "core.inference", fake_pkg)

    # Skip the llama-server branch by also stubbing routes.inference.
    fake_routes = types.ModuleType("routes.inference")
    fake_routes.get_llama_cpp_backend = lambda: types.SimpleNamespace(is_loaded = False)
    monkeypatch.setitem(sys.modules, "routes.inference", fake_routes)

    from core.inference.diffusion import _release_chat_backend_for_diffusion

    _release_chat_backend_for_diffusion()
    assert calls == ["owner/some-model"], calls
    assert stub.active_model_name is None


def test_load_model_uses_safetensors_flag(monkeypatch):
    """The pipeline.from_pretrained call must pass use_safetensors=True
    so pickle-backed .bin weights are refused at load time."""
    fake = _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    captured: dict = {}

    original = fake.Flux2KleinPipeline.from_pretrained.__func__

    def _capture(cls, base_repo, **kw):
        captured.update(kw)
        return original(cls, base_repo, **kw)

    fake.Flux2KleinPipeline.from_pretrained = classmethod(_capture)

    backend = get_diffusion_backend()
    backend.load_model(
        "unsloth/FLUX.2-klein-base-4B-GGUF",
        gguf_filename = "flux-2-klein-base-4b-Q4_K_S.gguf",
    )
    assert captured.get("use_safetensors") is True


def test_load_model_full_repo_does_not_substitute(monkeypatch):
    """A full diffusers repo (no gguf_filename) must call from_pretrained
    with the user-supplied repo, not the family default. This was the
    silent-substitution bug surfaced by review."""
    fake = _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/FLUX.1-finetune-diffusers",
        family_override = "flux.1",
    )
    # base_repo must echo the user repo, not the family default.
    assert status["base_repo"] == "owner/FLUX.1-finetune-diffusers"
    assert status["base_repo_source"] == "full_repo"
    assert status["base_repo_confidence"] == "explicit"
    assert status["repo_id"] == "owner/FLUX.1-finetune-diffusers"
    assert status["sampling_contract"]["gguf"] is False
    assert status["sampling_contract"]["base_repo_source"] == "full_repo"
    # And the fake pipeline records what we called from_pretrained with.
    assert backend._pipe.base_repo == "owner/FLUX.1-finetune-diffusers"


@pytest.mark.parametrize(
    ("repo_id", "family"),
    [
        ("black-forest-labs/FLUX.2-dev", "flux.2"),
        ("black-forest-labs/FLUX.2-klein-4B", "flux.2-klein"),
        ("black-forest-labs/FLUX.2-klein-base-4B", "flux.2-klein"),
        ("Tongyi-MAI/Z-Image", "z-image"),
        ("Tongyi-MAI/Z-Image-Turbo", "z-image-turbo"),
        ("baidu/ERNIE-Image-Turbo", "ernie-image-turbo"),
        ("Qwen/Qwen-Image-2512", "qwen-image-2512"),
        ("Qwen/Qwen-Image-Edit-2511", "qwen-image-edit-2511"),
    ],
)
def test_load_model_full_diffusers_transformer_pipeline_smoke(
    monkeypatch,
    repo_id,
    family,
):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(repo_id, family_override = family)

    assert status["is_loaded"] is True
    assert status["family"] == family
    assert status["base_repo"] == repo_id
    assert status["base_repo_source"] == "full_repo"
    assert status["sampling_contract"]["gguf"] is False
    assert backend._pipe.base_repo == repo_id


def test_load_model_full_repo_applies_unfused_lora(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/FLUX.1-finetune-diffusers",
        family_override = "flux.1",
        lora_repo = "owner/my-flux-lora",
        lora_weight_name = "pytorch_lora_weights.safetensors",
        lora_adapter_name = "studio-style",
        lora_scale = 0.75,
    )

    assert status["is_loaded"] is True
    assert status["lora"] == {
        "repo": "owner/my-flux-lora",
        "weight_name": "pytorch_lora_weights.safetensors",
        "adapter_name": "studio-style",
        "scale": 0.75,
        "fused": False,
    }
    assert backend._pipe.lora_loads == [
        {
            "repo": "owner/my-flux-lora",
            "adapter_name": "studio-style",
            "use_safetensors": True,
            "weight_name": "pytorch_lora_weights.safetensors",
        }
    ]
    assert backend._pipe.adapter_calls == [
        {
            "adapter_names": "studio-style",
            "adapter_weights": 0.75,
        }
    ]
    assert backend._pipe.fuse_calls == []


def test_load_model_full_repo_can_fuse_lora(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "owner/FLUX.1-finetune-diffusers",
        family_override = "flux.1",
        lora_repo = "owner/my-flux-lora",
        lora_adapter_name = "default",
        lora_scale = 0.5,
        lora_fuse = True,
    )

    assert status["lora"]["fused"] is True
    assert backend._pipe.lora_loads == [
        {
            "repo": "owner/my-flux-lora",
            "adapter_name": "default",
            "use_safetensors": True,
        }
    ]
    assert backend._pipe.fuse_calls == [
        {
            "lora_scale": 0.5,
            "adapter_names": ["default"],
        }
    ]


def test_load_model_gguf_applies_unfused_lora(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
        lora_repo = "owner/my-klein-lora",
    )

    assert status["is_loaded"] is True
    assert status["lora"] == {
        "repo": "owner/my-klein-lora",
        "weight_name": None,
        "adapter_name": "default",
        "scale": 1.0,
        "fused": False,
    }
    assert backend._pipe.lora_loads == [
        {
            "repo": "owner/my-klein-lora",
            "adapter_name": "default",
            "use_safetensors": True,
        }
    ]


def test_load_model_rejects_non_safetensors_lora_weight(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "only accepts safetensors"):
        backend.load_model(
            "owner/FLUX.1-finetune-diffusers",
            family_override = "flux.1",
            lora_repo = "owner/my-flux-lora",
            lora_weight_name = "pytorch_lora_weights.bin",
        )


def test_load_model_gguf_rejects_lora_fusion(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "Fusing LoRA into a GGUF"):
        backend.load_model(
            "unsloth/FLUX.2-klein-4B-GGUF",
            gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
            lora_repo = "owner/my-klein-lora",
            lora_fuse = True,
        )
    status = backend.status(include_internal = True)
    assert status["is_loaded"] is False
    assert status["active_lora_repo"] is None
    assert status["pending_lora_repo"] is None


def test_apply_lora_rejects_studio_lazy_gguf_modules():
    from core.inference.diffusion import _apply_diffusion_lora

    pipe = SimpleNamespace(load_lora_weights = lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match = "lazy quantized modules"):
        _apply_diffusion_lora(
            pipe,
            lora_repo = "owner/my-klein-lora",
            lora_weight_name = None,
            lora_adapter_name = None,
            lora_scale = None,
            lora_fuse = False,
            hf_token = None,
            gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
            uses_studio_lazy_gguf_modules = True,
        )


def test_load_model_wan_full_repo_uses_fp32_vae(monkeypatch):
    """Wan follows the official Diffusers recipe: VAE in FP32, while
    the rest of the pipeline uses the backend-selected dtype."""
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend
    import torch

    backend = get_diffusion_backend()
    status = backend.load_model(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        family_override = "wan2-2-t2v",
    )

    assert status["is_loaded"] is True
    assert status["family"] == "wan2-2-t2v"
    assert status["media_kind"] == "video"
    assert backend._pipe.base_repo == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert backend._pipe.kwargs["torch_dtype"] == "fake_dtype"
    assert backend._pipe.kwargs["use_safetensors"] is True
    vae = backend._pipe.kwargs["vae"]
    assert vae.base_repo == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert vae.kwargs["subfolder"] == "vae"
    assert vae.kwargs["torch_dtype"] is torch.float32


def test_load_model_concurrent_serialises(monkeypatch):
    """Two concurrent load_model() calls must NOT both reach
    pipeline_cls.from_pretrained at the same time (race fix)."""
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend
    import threading
    import time as _t

    backend = get_diffusion_backend()
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    import sys as _sys

    fake_pipeline_cls = _sys.modules["diffusers"].Flux2KleinPipeline
    original_from_pretrained = fake_pipeline_cls.from_pretrained.__func__

    def _instrumented_from_pretrained(cls, base_repo, **kwargs):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        try:
            _t.sleep(0.1)
            return original_from_pretrained(cls, base_repo, **kwargs)
        finally:
            with lock:
                active["n"] -= 1

    fake_pipeline_cls.from_pretrained = classmethod(_instrumented_from_pretrained)

    errors: list = []

    def _do_load():
        try:
            backend.load_model(
                "unsloth/FLUX.2-klein-base-4B-GGUF",
                gguf_filename = "flux-2-klein-base-4b-Q4_K_S.gguf",
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target = _do_load) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert (
        active["max"] == 1
    ), f"Expected concurrent loads to serialise; max_active={active['max']}"


def test_pipe_accepts_kwarg_filter():
    """The negative_prompt filter must drop the kwarg on classes that
    do not accept it (FLUX.2 / FLUX.2 klein) and keep it on the rest."""
    from core.inference.diffusion import _pipe_accepts_kwarg

    class _NoNeg:
        def __call__(
            self, *, prompt, num_inference_steps, guidance_scale, width, height
        ):
            pass

    class _Neg:
        def __call__(
            self,
            *,
            prompt,
            negative_prompt = None,
            num_inference_steps,
            guidance_scale,
            width,
            height,
        ):
            pass

    class _VarKw:
        def __call__(self, **kw):
            pass

    assert _pipe_accepts_kwarg(_NoNeg(), "negative_prompt") is False
    assert _pipe_accepts_kwarg(_Neg(), "negative_prompt") is True
    # Anything with **kwargs is assumed to accept the kwarg (the
    # alternative is to silently drop legitimate params).
    assert _pipe_accepts_kwarg(_VarKw(), "negative_prompt") is True


def test_generate_image_strips_negative_prompt_on_flux2(monkeypatch):
    """generate_image must drop negative_prompt when the loaded pipeline
    does not accept it; otherwise FLUX.2 would 500 on a user-visible
    field."""
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()

    received: dict = {}

    class _Flux2LikePipe:
        # Signature mirrors Flux2Pipeline.__call__: NO negative_prompt.
        # No **kw either, since the real FLUX.2 pipeline does not accept
        # arbitrary kwargs (passing negative_prompt to it raises TypeError).
        def __call__(
            self,
            *,
            prompt,
            num_inference_steps,
            guidance_scale,
            width,
            height,
            generator = None,
        ):
            received["prompt"] = prompt

            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (width, height), (1, 2, 3))]
            return o

    backend._pipe = _Flux2LikePipe()
    backend._device = "cpu"
    backend._family = d._FAMILIES[0]
    backend._repo_id = "stub/stub"

    # If generate_image forwarded negative_prompt, the pipeline call
    # would raise TypeError. The PR's filter drops it, so the call
    # succeeds and we observe the prompt was still delivered.
    backend.generate_image(
        prompt = "a sloth",
        negative_prompt = "blurry, low quality",
        num_inference_steps = 4,
        guidance_scale = 1.0,
        width = 256,
        height = 256,
    )
    assert received["prompt"] == "a sloth"


def test_generate_image_keeps_negative_prompt_on_supporting_pipe(monkeypatch):
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    captured: dict = {}

    class _NegOK:
        def __call__(
            self,
            *,
            prompt,
            negative_prompt = None,
            num_inference_steps,
            guidance_scale,
            width,
            height,
            **kw,
        ):
            captured["negative_prompt"] = negative_prompt

            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (width, height), (4, 5, 6))]
            return o

    backend._pipe = _NegOK()
    backend._device = "cpu"
    backend._family = next(f for f in d._FAMILIES if f.name == "flux.1")
    backend._repo_id = "stub/stub"

    backend.generate_image(
        prompt = "a sloth",
        negative_prompt = "blurry",
        num_inference_steps = 4,
        guidance_scale = 1.0,
        width = 256,
        height = 256,
    )
    assert captured["negative_prompt"] == "blurry"


def test_generate_image_forwards_true_cfg_scale_when_supported(monkeypatch):
    """When a pipeline accepts both negative_prompt and true_cfg_scale
    (QwenImagePipeline, FluxPipeline) the user's guidance_scale must be
    forwarded as true_cfg_scale as well, otherwise the negative prompt
    is silently ignored (Qwen leaves the default true_cfg_scale=4.0
    while the user value lands on guidance_scale)."""
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    captured: dict = {}

    class _QwenLikePipe:
        def __call__(
            self,
            *,
            prompt,
            negative_prompt = None,
            num_inference_steps,
            guidance_scale,
            true_cfg_scale = 4.0,
            width,
            height,
            **kw,
        ):
            captured["guidance_scale"] = guidance_scale
            captured["true_cfg_scale"] = true_cfg_scale
            captured["negative_prompt"] = negative_prompt

            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (width, height), (7, 8, 9))]
            return o

    backend._pipe = _QwenLikePipe()
    backend._device = "cpu"
    backend._family = next(f for f in d._FAMILIES if f.name == "flux.1")
    backend._repo_id = "stub/stub"

    backend.generate_image(
        prompt = "a sloth",
        negative_prompt = "blurry",
        num_inference_steps = 4,
        guidance_scale = 7.5,
        width = 256,
        height = 256,
    )
    assert captured["negative_prompt"] == "blurry"
    assert captured["guidance_scale"] == 7.5
    assert captured["true_cfg_scale"] == 7.5


def test_generate_image_skips_true_cfg_scale_without_negative_prompt(monkeypatch):
    """Pipelines that accept true_cfg_scale must NOT have it forwarded
    when no negative_prompt is given; otherwise distilled CFG models
    would unintentionally switch into real-CFG mode and degrade
    quality / double inference cost."""
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    captured: dict = {}

    class _QwenLikePipe:
        def __call__(
            self,
            *,
            prompt,
            negative_prompt = None,
            num_inference_steps,
            guidance_scale,
            true_cfg_scale = 4.0,
            width,
            height,
            **kw,
        ):
            captured["guidance_scale"] = guidance_scale
            captured["true_cfg_scale"] = true_cfg_scale

            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (width, height), (1, 1, 1))]
            return o

    backend._pipe = _QwenLikePipe()
    backend._device = "cpu"
    backend._family = next(f for f in d._FAMILIES if f.name == "flux.1")
    backend._repo_id = "stub/stub"

    backend.generate_image(
        prompt = "a sloth",
        negative_prompt = None,
        num_inference_steps = 4,
        guidance_scale = 7.5,
        width = 256,
        height = 256,
    )
    assert captured["guidance_scale"] == 7.5
    # Default left untouched: real CFG only activates with neg prompt.
    assert captured["true_cfg_scale"] == 4.0


def test_generate_image_does_not_block_status(monkeypatch):
    """status() must return promptly while a generation is in flight;
    holding _lock for the whole forward froze the Images UI on the
    polling endpoint for the entire (minutes long) generation."""
    import threading
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    pipe_started = threading.Event()
    pipe_release = threading.Event()

    class _SlowPipe:
        def __call__(self, **kw):
            pipe_started.set()
            # Wait until the test releases us; status() should return
            # before this lock is released.
            pipe_release.wait(timeout = 5)

            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (kw["width"], kw["height"]), (1, 2, 3))]
            return o

    backend._pipe = _SlowPipe()
    backend._device = "cpu"
    backend._family = d._FAMILIES[0]
    backend._repo_id = "stub/stub"

    t = threading.Thread(
        target = backend.generate_image,
        kwargs = dict(
            prompt = "a sloth",
            num_inference_steps = 1,
            guidance_scale = 1.0,
            width = 64,
            height = 64,
        ),
    )
    t.start()
    try:
        assert pipe_started.wait(timeout = 5)
        # Forward is in progress; status() must not block on _lock.
        completed = [False]

        def call_status():
            backend.status()
            completed[0] = True

        s = threading.Thread(target = call_status)
        s.start()
        s.join(timeout = 2)
        assert completed[0], "status() blocked on generate_image"
    finally:
        pipe_release.set()
        t.join(timeout = 5)


def test_load_publishes_pending_target_during_loading():
    """status() must expose the pending repo_id / base_repo / gguf
    file while is_loading=True so cache- and finetuned-delete guards
    can refuse to rmtree the repo being downloaded right now.

    The pending exposure is purely a state-shape contract: load_model
    sets _loading + _pending_* under _lock at the start, and status()
    snapshots them under _lock. Test the contract directly instead of
    racing a fake pipeline through a background thread, which was
    flaky on the Windows runner (the chat-release helpers' transitive
    imports of core.training.resume failed there and the load thread
    exited cleanly before the main thread observed the pending state).
    """
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()
    # Simulate the state load_model publishes at the top of its
    # critical section, before from_pretrained runs.
    with backend._lock:
        backend._loading = True
        backend._pending_repo_id = "unsloth/FLUX.2-klein-4B-GGUF"
        backend._pending_base_repo = "black-forest-labs/FLUX.2-klein-4B"
        backend._pending_base_repo_source = "name_heuristic"
        backend._pending_base_repo_confidence = "heuristic"
        backend._pending_base_repo_variant = "distilled-4b"
        backend._pending_gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf"
        backend._pending_text_encoder_gguf_repo = "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
        backend._pending_text_encoder_gguf_filename = "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
        backend._pending_lora_repo = "owner/my-klein-lora"
        backend._pending_lora_weight_name = "adapter.safetensors"

    public = backend.status()
    assert public["is_loading"] is True
    assert public["repo_id"] == "unsloth/FLUX.2-klein-4B-GGUF"
    assert public["base_repo"] == "black-forest-labs/FLUX.2-klein-4B"
    assert public["base_repo_source"] == "name_heuristic"
    assert public["base_repo_confidence"] == "heuristic"
    assert public["base_repo_variant"] == "distilled-4b"
    assert public["text_encoder_gguf_repo"] == "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
    assert public["text_encoder_gguf_filename"] == "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
    assert public["lora"] == {
        "repo": "owner/my-klein-lora",
        "weight_name": "adapter.safetensors",
        "adapter_name": None,
        "scale": None,
        "fused": None,
    }
    # Guard-facing internal payload also reports the pending fields
    # under their dedicated keys.
    internal = backend.status(include_internal = True)
    assert internal["pending_repo_id"] == "unsloth/FLUX.2-klein-4B-GGUF"
    assert internal["pending_base_repo"] == "black-forest-labs/FLUX.2-klein-4B"
    assert internal["pending_base_repo_source"] == "name_heuristic"
    assert internal["pending_base_repo_confidence"] == "heuristic"
    assert internal["pending_base_repo_variant"] == "distilled-4b"
    assert internal["pending_gguf_filename"] == "flux-2-klein-4b-Q4_K_S.gguf"
    assert internal["pending_text_encoder_gguf_repo"] == "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"
    assert (
        internal["pending_text_encoder_gguf_filename"]
        == "Mistral-Small-3.2-24B-Instruct-2506-UD-Q4_K_XL.gguf"
    )
    assert internal["pending_lora_repo"] == "owner/my-klein-lora"
    assert internal["pending_lora_weight_name"] == "adapter.safetensors"


def test_unload_waits_for_in_flight_generation(monkeypatch):
    """unload_model() must not return is_loaded=False while a
    generate_image forward is still iterating; otherwise routes/...
    callers see the pipe as freed while it still owns GPU memory and
    can race a subsequent load."""
    import threading
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.get_diffusion_backend()
    started = threading.Event()
    release = threading.Event()
    generation_finished = threading.Event()

    class _SlowPipe:
        def __call__(self, **kw):
            started.set()
            release.wait(timeout = 5)

            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (kw["width"], kw["height"]))]
            return o

    backend._pipe = _SlowPipe()
    backend._device = "cpu"
    backend._family = d._FAMILIES[0]
    backend._repo_id = "stub/stub"

    def do_generate():
        try:
            backend.generate_image(
                prompt = "x",
                num_inference_steps = 1,
                guidance_scale = 1.0,
                width = 64,
                height = 64,
            )
        finally:
            generation_finished.set()

    gen_thread = threading.Thread(target = do_generate)
    gen_thread.start()
    try:
        assert started.wait(timeout = 5)
        unload_returned = threading.Event()

        def do_unload():
            backend.unload_model()
            unload_returned.set()

        unload_thread = threading.Thread(target = do_unload)
        unload_thread.start()
        # unload should block until release sets, NOT return early.
        unload_thread.join(timeout = 0.5)
        assert (
            not unload_returned.is_set()
        ), "unload_model returned while generation was still running"
        release.set()
        unload_thread.join(timeout = 5)
        assert unload_returned.is_set()
        assert generation_finished.is_set()
    finally:
        release.set()
        gen_thread.join(timeout = 5)


def test_bf16_falls_back_to_fp16_on_old_cuda(monkeypatch):
    """CUDA availability does not imply BF16 support; old GPUs report
    is_available()=True and is_bf16_supported()=False. The backend
    must fall back to FP16 rather than picking BF16 and failing
    deep inside from_pretrained."""
    import core.inference.diffusion as d

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported():
            return False

    class _FakeBackends:
        class mps:
            @staticmethod
            def is_available():
                return False

    class _FakeTorch:
        cuda = _FakeCuda
        backends = _FakeBackends
        # Sentinel objects so the dtype identity comparison works.
        bfloat16 = object()
        float16 = object()
        float32 = object()

    fake_torch = _FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = d.DiffusionBackend()
    device, dtype = backend._pick_device_and_dtype()
    assert device == "cuda"
    assert dtype is fake_torch.float16


# ── round 13 regressions ──────────────────────────────────────────


def test_smart_base_repo_uses_windows_leaf_only():
    """Round 13 P2 #13: a Windows path whose PARENT directory contains
    'base' must not be misclassified as the Klein Base 4B variant."""
    from core.inference.diffusion import _smart_base_repo, detect_family

    repo = r"C:\Users\me\base\FLUX.2-klein-4B-GGUF"
    fam = detect_family(repo)
    assert fam is not None and fam.name == "flux.2-klein"
    assert _smart_base_repo(fam, repo) == "black-forest-labs/FLUX.2-klein-4B"


def test_resolve_local_gguf_child_rejects_traversal(tmp_path):
    """Round 13 P1 #2: gguf_filename must not escape the repo root."""
    from core.inference.diffusion import _resolve_local_gguf_child

    repo_root = tmp_path / "my-flux"
    repo_root.mkdir()
    (repo_root / "model.gguf").write_bytes(b"x")
    sibling = tmp_path / "other.gguf"
    sibling.write_bytes(b"y")

    assert _resolve_local_gguf_child(repo_root, "model.gguf").name == "model.gguf"

    # ``./model.gguf`` is normalised by PurePosixPath to ``model.gguf``
    # and stays inside the repo, so it is intentionally accepted.
    for bad in ("../other.gguf", "", "sub/../model.gguf"):
        with pytest.raises(RuntimeError):
            _resolve_local_gguf_child(repo_root, bad)
    with pytest.raises(RuntimeError):
        _resolve_local_gguf_child(repo_root, "/etc/passwd")


def test_resolve_local_gguf_child_rejects_backslash(tmp_path):
    """Round 13 P1 #2: a Windows-style separator inside gguf_filename
    must be rejected even on POSIX so it never becomes a literal name."""
    from core.inference.diffusion import _resolve_local_gguf_child

    repo_root = tmp_path / "my-flux"
    repo_root.mkdir()
    (repo_root / "model.gguf").write_bytes(b"x")

    with pytest.raises(RuntimeError):
        _resolve_local_gguf_child(repo_root, r"..\\other.gguf")


def test_load_model_accepts_relative_local_dir(monkeypatch, tmp_path):
    """Round 13 P1 #2: relative directory paths (Studio exports) must
    NOT be routed through hf_hub_download."""
    import core.inference.diffusion as d

    repo_root = tmp_path / "exports" / "my-flux"
    repo_root.mkdir(parents = True)
    gguf_file = repo_root / "model.gguf"
    gguf_file.write_bytes(b"x")

    # cwd so the relative path resolves to repo_root
    monkeypatch.chdir(tmp_path)

    fake_transformer = object()
    fake_pipe = SimpleNamespace(
        to = lambda *a, **kw: None,
        enable_model_cpu_offload = lambda: None,
    )

    class _FakeQuantConfig:
        def __init__(self, **_):
            pass

    class _FakeTransformerCls:
        from_single_file_calls: list[tuple[str, dict]] = []

        @classmethod
        def from_single_file(cls, path, **kwargs):
            cls.from_single_file_calls.append((path, kwargs))
            return fake_transformer

    class _FakePipeCls:
        @classmethod
        def from_pretrained(cls, base, **kwargs):
            return fake_pipe

    fake_diffusers = SimpleNamespace(
        __version__ = "0.99",
        GGUFQuantizationConfig = _FakeQuantConfig,
        Flux2Transformer2DModel = _FakeTransformerCls,
        Flux2KleinPipeline = _FakePipeCls,
    )

    fake_torch = SimpleNamespace(
        cuda = SimpleNamespace(
            is_available = lambda: False,
            is_bf16_supported = lambda: False,
            empty_cache = lambda: None,
        ),
        bfloat16 = "bf16",
        float16 = "fp16",
        float32 = "fp32",
        backends = SimpleNamespace(
            mps = SimpleNamespace(is_available = lambda: False),
        ),
    )

    def _boom(**kwargs):
        # Round 20 P1 #1 added a base-repo preflight that downloads
        # the diffusers ``model_index.json`` of the auto-picked
        # companion repo BEFORE the chat unload. Round 21 P2 #6
        # added a second preflight for ``transformer/config.json``
        # on that same companion. Allow both preflight kinds through
        # but still reject any attempt to download the GGUF itself,
        # which is what this test guards.
        if kwargs.get("filename") in ("model_index.json", "config.json"):
            return "/tmp/preflight"
        raise AssertionError("hf_hub_download must not run for a local dir")

    fake_hub = SimpleNamespace(hf_hub_download = _boom)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = d.DiffusionBackend()
    backend.load_model(
        repo_id = "exports/my-flux",
        gguf_filename = "model.gguf",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        family_override = "flux.2-klein",
        enable_model_cpu_offload = False,
    )

    assert _FakeTransformerCls.from_single_file_calls
    resolved_path = _FakeTransformerCls.from_single_file_calls[0][0]
    assert str(gguf_file.resolve()) == resolved_path


def test_generate_image_with_metadata_returns_active_pipeline(monkeypatch):
    """Round 13 P2 #9: meta returns the resident pipeline's identity."""
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()
    fake_fam = d.DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2KleinTransformer3DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        aliases = (),
    )

    def _fake_unlocked(**kwargs):
        from PIL import Image as _Image

        return _Image.new("RGB", (8, 8))

    backend._pipe = object()
    backend._repo_id = "unsloth/FLUX.2-klein-4B-GGUF"
    backend._family = fake_fam
    monkeypatch.setattr(backend, "_generate_image_unlocked", _fake_unlocked)

    _, meta = backend.generate_image_with_metadata(prompt = "x")
    assert meta == {
        "model": "unsloth/FLUX.2-klein-4B-GGUF",
        "family": "flux.2-klein",
    }


@pytest.mark.parametrize(
    "repo_id",
    [
        "unsloth/Qwen_Image-Edit-GGUF",
        "unsloth/Qwen-Image_Edit-GGUF",
        "unsloth/Qwen-ImageEdit-GGUF",
        "unsloth/qwen-image_edit-2509-GGUF",
        "unsloth/Qwen.Image.Edit-GGUF",
    ],
)
def test_detect_family_qwen_image_edit_mixed_separators(repo_id):
    """Round 14 P2 #8: every spelling of Qwen-Image-Edit must NOT
    match the base Qwen-Image text-to-image family."""
    from core.inference.diffusion import detect_family

    fam = detect_family(repo_id)
    if fam is not None:
        assert fam.name != "qwen-image"


def test_redact_hf_tokens_removes_url_embedded_token():
    """Round 14 P2 #9: tokens embedded in user-supplied paths /
    URLs must be scrubbed before logging."""
    from core.inference.diffusion import _redact_hf_tokens

    leaky = (
        "https://hf_abcdefghij0123456789@huggingface.co/unsloth/FLUX.2-klein-4B-GGUF"
    )
    redacted = _redact_hf_tokens(leaky)
    assert "hf_" not in redacted
    assert "<redacted>" in redacted
    # Non-strings pass through unchanged so the helper is safe in
    # logger argument lists where families / dtypes mix in.
    assert _redact_hf_tokens(None) is None
    assert _redact_hf_tokens(42) == 42


def test_status_preserves_active_gguf_subdir(monkeypatch):
    """Round 14 P1 #4: status() must surface the original caller-
    supplied gguf_filename (``BF16/model.gguf``) instead of the
    collapsed basename."""
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()
    backend._pipe = object()
    backend._repo_id = "unsloth/FLUX.2-klein-4B-GGUF"
    backend._gguf_path = "/cache/models/unsloth/FLUX.2-klein-4B-GGUF/BF16/model.gguf"
    backend._gguf_filename = "BF16/model.gguf"
    backend._family = d.DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        aliases = (),
    )

    s = backend.status(include_internal = True)
    assert s["active_gguf_filename"] == "BF16/model.gguf"
    # UI-facing field still collapses to the basename.
    assert s["gguf_filename"] == "model.gguf"


def test_generator_uses_cpu_when_cpu_offload_enabled(monkeypatch):
    """Round 14 P1 #6: seeded CUDA generation must NOT create a
    CUDA torch.Generator when the pipeline was loaded with CPU
    offload enabled, otherwise it crashes mid-forward."""
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()

    class _FakePipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            from PIL import Image

            return SimpleNamespace(images = [Image.new("RGB", (8, 8))])

    fake_pipe = _FakePipe()
    backend._pipe = fake_pipe
    backend._device = "cuda"
    backend._cpu_offload_enabled = True

    captured_devices: list[str] = []

    class _FakeGenerator:
        def __init__(self, device):
            captured_devices.append(device)

        def manual_seed(self, seed):
            return self

    class _FakeTorchCuda:
        @staticmethod
        def is_available():
            return True

    fake_torch = SimpleNamespace(Generator = _FakeGenerator, cuda = _FakeTorchCuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend._generate_image_unlocked(prompt = "x", seed = 7, width = 8, height = 8)
    assert captured_devices == ["cpu"]


def test_generate_image_uses_family_guidance_kwarg_and_default_negative_prompt():
    """Qwen-Image uses true_cfg_scale for real CFG; guidance_scale is
    reserved for guidance-distilled variants and is ineffective there."""
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()

    class _FakeQwenPipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(
            self,
            *,
            prompt,
            num_inference_steps,
            true_cfg_scale,
            negative_prompt,
            width,
            height,
            generator = None,
        ):
            self.last_kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "generator": generator,
            }
            from PIL import Image

            return SimpleNamespace(images = [Image.new("RGB", (width, height))])

    pipe = _FakeQwenPipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/Qwen-Image-2512-GGUF")

    backend._generate_image_unlocked(
        prompt = "x",
        width = 16,
        height = 16,
        guidance_scale = 3.5,
    )

    assert pipe.last_kwargs["true_cfg_scale"] == 3.5
    assert pipe.last_kwargs["negative_prompt"] == backend._family.default_negative_prompt
    assert "guidance_scale" not in pipe.last_kwargs


def test_generate_image_uses_family_sampling_defaults_when_omitted():
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()

    class _FakeZImageTurboPipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(
            self,
            *,
            prompt,
            num_inference_steps,
            guidance_scale,
            width,
            height,
            generator = None,
        ):
            self.last_kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "generator": generator,
            }
            from PIL import Image

            return SimpleNamespace(images = [Image.new("RGB", (width, height))])

    pipe = _FakeZImageTurboPipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/Z-Image-Turbo-GGUF")

    backend._generate_image_unlocked(prompt = "x")

    assert pipe.last_kwargs["num_inference_steps"] == 9
    assert pipe.last_kwargs["guidance_scale"] == 0.0
    assert pipe.last_kwargs["width"] == 1024
    assert pipe.last_kwargs["height"] == 1024


def test_generate_image_uses_flux2_klein_variant_sampling_defaults():
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()

    class _FakeFlux2KleinPipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(
            self,
            *,
            prompt,
            num_inference_steps,
            guidance_scale,
            width,
            height,
        ):
            self.last_kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
            }
            from PIL import Image

            return SimpleNamespace(images = [Image.new("RGB", (width, height))])

    pipe = _FakeFlux2KleinPipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/FLUX.2-klein-base-4B-GGUF")
    backend._base_repo_variant = "base-4b"

    backend._generate_image_unlocked(prompt = "x")

    assert pipe.last_kwargs["num_inference_steps"] == 50
    assert pipe.last_kwargs["guidance_scale"] == 4.0


def test_generate_image_uses_qwen_edit_plus_repo_defaults():
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.DiffusionBackend()

    class _FakeQwenEditPlusPipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(
            self,
            *,
            image,
            prompt,
            negative_prompt,
            true_cfg_scale,
            guidance_scale,
            num_inference_steps,
            width,
            height,
        ):
            self.last_kwargs = {
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "true_cfg_scale": true_cfg_scale,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
            }
            return SimpleNamespace(images = [Image.new("RGB", (width, height))])

    pipe = _FakeQwenEditPlusPipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/Qwen-Image-Edit-2509-GGUF")

    backend._generate_image_unlocked(
        prompt = "change the sign",
        input_images = [Image.new("RGB", (16, 16))],
        width = 64,
        height = 64,
    )

    assert pipe.last_kwargs["num_inference_steps"] == 40
    assert pipe.last_kwargs["true_cfg_scale"] == 4.0
    assert pipe.last_kwargs["guidance_scale"] == 1.0


def test_generate_image_rejects_image_required_family():
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()
    backend._pipe = object()
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/Qwen-Image-Layered-GGUF")

    with pytest.raises(RuntimeError, match = "requires image input"):
        backend._generate_image_unlocked(prompt = "x", width = 16, height = 16)


def test_generate_image_required_family_forwards_input_image():
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.DiffusionBackend()

    class _FakeQwenEditPipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(
            self,
            *,
            image,
            prompt,
            negative_prompt,
            true_cfg_scale,
            num_inference_steps,
            width,
            height,
        ):
            self.last_kwargs = {
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "true_cfg_scale": true_cfg_scale,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
            }
            return SimpleNamespace(images = [Image.new("RGB", (width, height))])

    source = Image.new("RGB", (32, 32))
    pipe = _FakeQwenEditPipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/Qwen-Image-Edit-2511-GGUF")

    image = backend._generate_image_unlocked(
        prompt = "change the background",
        input_images = [source],
        width = 64,
        height = 64,
    )

    assert image.size == (64, 64)
    assert pipe.last_kwargs["image"] is source
    assert pipe.last_kwargs["negative_prompt"] == backend._family.default_negative_prompt
    assert pipe.last_kwargs["true_cfg_scale"] == 4.0
    assert pipe.last_kwargs["num_inference_steps"] == 40


def test_generate_image_rejects_video_family_on_image_route():
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()
    backend._pipe = object()
    backend._device = "cpu"
    backend._family = d.detect_family("diffusers/LTX-2.3-Distilled-Diffusers")

    with pytest.raises(RuntimeError, match = "video generation family"):
        backend._generate_image_unlocked(prompt = "x", width = 768, height = 512)


def test_generate_video_with_metadata_uses_family_defaults():
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()

    class _FakeWanPipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return SimpleNamespace(frames = [["frame0", "frame1"]])

    pipe = _FakeWanPipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._repo_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    backend._family = d.detect_family("Wan-AI/Wan2.2-T2V-A14B-Diffusers")

    video, meta = backend.generate_video_with_metadata(prompt = "a crane shot")

    assert video == ["frame0", "frame1"]
    assert pipe.last_kwargs["num_inference_steps"] == 40
    assert pipe.last_kwargs["guidance_scale"] == 4.0
    assert pipe.last_kwargs["guidance_scale_2"] == 3.0
    assert pipe.last_kwargs["width"] == 1280
    assert pipe.last_kwargs["height"] == 720
    assert pipe.last_kwargs["num_frames"] == 81
    assert pipe.last_kwargs["output_type"] == "np"
    assert meta["family"] == "wan2-2-t2v"
    assert meta["num_frames"] == 81
    assert meta["frame_rate"] == 16.0


def test_generate_video_rejects_image_family():
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()
    backend._pipe = object()
    backend._device = "cpu"
    backend._family = d.detect_family("unsloth/Z-Image-GGUF")

    with pytest.raises(RuntimeError, match = "not a video generation family"):
        backend.generate_video_with_metadata(prompt = "x")


def test_generate_images_with_metadata_flattens_layered_outputs():
    import core.inference.diffusion as d
    from PIL import Image

    backend = d.DiffusionBackend()

    class _FakeLayeredPipe:
        def __init__(self):
            self.last_kwargs = None

        def __call__(
            self,
            *,
            image,
            prompt,
            negative_prompt,
            true_cfg_scale,
            num_inference_steps,
            resolution,
            layers,
            cfg_normalize,
            use_en_prompt,
        ):
            self.last_kwargs = {
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "true_cfg_scale": true_cfg_scale,
                "num_inference_steps": num_inference_steps,
                "resolution": resolution,
                "layers": layers,
                "cfg_normalize": cfg_normalize,
                "use_en_prompt": use_en_prompt,
            }
            return SimpleNamespace(
                images = [[Image.new("RGBA", (resolution, resolution)) for _ in range(3)]]
            )

    source = Image.new("RGB", (32, 32))
    pipe = _FakeLayeredPipe()
    backend._pipe = pipe
    backend._device = "cpu"
    backend._repo_id = "unsloth/Qwen-Image-Layered-GGUF"
    backend._family = d.detect_family("unsloth/Qwen-Image-Layered-GGUF")

    images, meta = backend.generate_images_with_metadata(
        prompt = "separate foreground",
        input_images = [source],
        width = 640,
        height = 640,
    )

    assert len(images) == 3
    assert meta["output_count"] == 3
    assert pipe.last_kwargs["image"].mode == "RGBA"
    assert pipe.last_kwargs["resolution"] == 640
    assert pipe.last_kwargs["layers"] == 4
    assert pipe.last_kwargs["cfg_normalize"] is True
    assert pipe.last_kwargs["use_en_prompt"] is True


def test_smart_base_repo_uses_windows_leaf_only_already_set_separator_round14():
    """Sanity: relative paths still work after the Windows fix."""
    from core.inference.diffusion import _smart_base_repo, detect_family

    repo = "owner/FLUX.2-klein-9B-GGUF"
    fam = detect_family(repo)
    assert fam is not None
    assert _smart_base_repo(fam, repo) == "black-forest-labs/FLUX.2-klein-9B"


def test_display_repo_id_collapses_absolute_path(tmp_path):
    """Round 15 P2 #6: absolute local paths must NOT leak through
    status(). Hub-style repo ids pass through unchanged. Uses
    ``tmp_path`` so the absolute path is platform-correct (POSIX
    ``/`` paths read as drive-relative on Windows)."""
    from core.inference.diffusion import _display_repo_id

    # Hub id passes through.
    assert (
        _display_repo_id("black-forest-labs/FLUX.2-klein-4B")
        == "black-forest-labs/FLUX.2-klein-4B"
    )
    # Absolute local path collapses to leaf. ``tmp_path`` is absolute
    # on every OS pytest supports.
    absolute_local = tmp_path / "private-flux"
    absolute_local.mkdir()
    assert _display_repo_id(str(absolute_local)) == "private-flux"
    # HF tokens are scrubbed defensively.
    leaky = "https://hf_abcdefghij0123456789@huggingface.co/owner/repo"
    out = _display_repo_id(leaky)
    assert "hf_" not in out


def test_detect_family_rejects_substring_collisions():
    """Round 15 P2 #8: ``flux.20-model`` must NOT match ``flux.2``."""
    from core.inference.diffusion import detect_family

    # ``flux.20`` is a different number and must not collide with ``flux.2``.
    assert detect_family("owner/flux.20-model") is None
    # ``stable-diffusion-30`` must not match ``stable-diffusion-3``.
    assert detect_family("foo/stable-diffusion-30") is None
    # Legitimate ``flux.2`` still matches.
    fam = detect_family("black-forest-labs/FLUX.2-dev")
    assert fam is not None and fam.name == "flux.2"


def test_detect_family_compact_aliases_with_owner_prefix():
    """Round 16 P2 #9: compact aliases must match when the repo has
    an owner prefix. ``unsloth/Flux2Klein-GGUF`` -> flux.2-klein
    via the ``flux2-klein`` alias's compact form. Embedded compact
    matches (e.g. ``flux2`` inside ``flux20``) must NOT match."""
    from core.inference.diffusion import detect_family

    fam = detect_family("unsloth/Flux2Klein-GGUF")
    assert fam is not None and fam.name == "flux.2-klein"
    # 20 is a different number; must not collide with flux.2.
    assert detect_family("unsloth/Flux20-GGUF") is None


def test_public_status_does_not_leak_local_path_via_active_fields(
    monkeypatch, tmp_path
):
    """Round 16 P1 #5: even the guard-facing active_*/pending_* keys
    must be absent from the public status payload. Uses ``tmp_path``
    so the absolute path is correct on every OS."""
    import core.inference.diffusion as d

    absolute_repo = tmp_path / "private-flux"
    absolute_repo.mkdir()
    absolute_base = tmp_path / "base-private"
    absolute_base.mkdir()

    backend = d.DiffusionBackend()
    backend._pipe = object()
    backend._repo_id = str(absolute_repo)
    backend._base_repo = str(absolute_base)
    backend._family = d.DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        aliases = (),
    )

    public = backend.status()
    # UI-facing fields collapse to leaf and the guard-only fields are absent.
    assert public["repo_id"] == "private-flux"
    assert public["base_repo"] == "base-private"
    for key in (
        "active_repo_id",
        "active_base_repo",
        "active_gguf_filename",
        "pending_repo_id",
        "pending_base_repo",
        "pending_gguf_filename",
    ):
        assert key not in public

    internal = backend.status(include_internal = True)
    assert internal["active_repo_id"] == str(absolute_repo)
    assert internal["active_base_repo"] == str(absolute_base)


def test_generate_image_with_metadata_redacts_local_path(monkeypatch, tmp_path):
    """Round 16 P1 #6: the generation response must not echo a raw
    absolute path back to the browser."""
    import core.inference.diffusion as d

    absolute_repo = tmp_path / "secret-flux"
    absolute_repo.mkdir()

    backend = d.DiffusionBackend()
    backend._pipe = object()
    backend._repo_id = str(absolute_repo)
    backend._family = d.DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        aliases = (),
    )

    def _fake_unlocked(**kwargs):
        from PIL import Image as _Image

        return _Image.new("RGB", (8, 8))

    monkeypatch.setattr(backend, "_generate_image_unlocked", _fake_unlocked)
    _, meta = backend.generate_image_with_metadata(prompt = "x")
    assert meta["model"] == "secret-flux"
    assert str(tmp_path) not in meta["model"]


def test_release_other_gpu_owners_raises_on_active_training(monkeypatch):
    """Round 15 P1 #3: direct backend callers must not bypass the
    route layer's training-active 409 guard."""
    import core.inference.diffusion as d

    fake_training_mod = types.ModuleType("core.training")
    fake_training_mod.get_training_backend = lambda: SimpleNamespace(
        is_training_active = lambda: True
    )
    monkeypatch.setitem(sys.modules, "core.training", fake_training_mod)

    # Ensure export module import does not fail the test before the
    # training raise lands.
    fake_export_mod = types.ModuleType("core.export")
    fake_export_mod.get_export_backend = lambda: SimpleNamespace(
        is_export_active = lambda: False,
        current_checkpoint = None,
    )
    monkeypatch.setitem(sys.modules, "core.export", fake_export_mod)

    with pytest.raises(RuntimeError) as exc_info:
        d._release_other_gpu_owners_for_diffusion()
    assert "Training is currently active" in str(exc_info.value)


def test_generate_image_with_metadata_blocks_concurrent_unload(monkeypatch):
    """Round 13 P2 #9: _generate_lock serialises the forward AND the
    meta snapshot, so a queued unload cannot wipe state in between."""
    import threading
    import core.inference.diffusion as d

    backend = d.DiffusionBackend()
    fake_fam = d.DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2KleinTransformer3DModel",
        base_repo = "black-forest-labs/FLUX.2-klein-4B",
        aliases = (),
    )

    started = threading.Event()
    finish = threading.Event()

    def _fake_unlocked(**kwargs):
        from PIL import Image as _Image

        started.set()
        # Hold long enough for the unload thread to race the metadata
        # snapshot if the lock were released too early.
        finish.wait(timeout = 2.0)
        return _Image.new("RGB", (8, 8))

    backend._pipe = object()
    backend._repo_id = "unsloth/FLUX.2-klein-4B-GGUF"
    backend._family = fake_fam
    monkeypatch.setattr(backend, "_generate_image_unlocked", _fake_unlocked)

    result: list = []

    def _gen():
        result.append(backend.generate_image_with_metadata(prompt = "x"))

    gen_thread = threading.Thread(target = _gen)
    gen_thread.start()
    assert started.wait(timeout = 2.0)

    def _unload():
        backend.unload_model()

    un_thread = threading.Thread(target = _unload)
    un_thread.start()
    # The unload must NOT have completed yet; it queues behind the
    # generation's _generate_lock.
    un_thread.join(timeout = 0.2)
    assert un_thread.is_alive()
    finish.set()
    gen_thread.join(timeout = 5.0)
    un_thread.join(timeout = 5.0)

    assert result
    _, meta = result[0]
    assert meta["model"] == "unsloth/FLUX.2-klein-4B-GGUF"
    assert meta["family"] == "flux.2-klein"
