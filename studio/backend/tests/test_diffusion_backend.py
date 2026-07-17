# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion backend.

The family helpers are pure functions, tested directly. The backend lifecycle is
exercised with ``torch`` / ``diffusers`` stubbed via ``sys.modules`` so no real
GPU, weights, or network access is needed (sub-second, CI-friendly).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import types

import pytest

from core.inference.diffusion import (
    DiffusionBackend,
    _LoadState,
    _base_file_downloaded,
    _clamp_max_side,
    _resolve_base_repo,
    _resolve_diffusion_compute_dtype,
)

# diffusion.py imports the compile/arch patch modules LAZILY (they pull torch at module level,
# and diffusion.py must stay importable on a torchless native install). Import them here at
# collection time -- under the real torch -- so they're cached in sys.modules before the
# fake-torch fixtures swap it out; else the lazy import would build them against the stub torch.
import core.inference.diffusion_eager_patches  # noqa: E402,F401
import core.inference.diffusion_arch_patches  # noqa: E402,F401
from core.inference.diffusion_families import (
    detect_family,
    resolve_base_repo,
    resolve_local_gguf_child,
    supported_family_names,
)


# Pure family helpers


def test_clamp_max_side_bounds_oversized_init():
    # img2img / inpaint derive OUTPUT size from the uploaded image; an oversized upload (up to
    # the 4096/side decode cap = 4x the txt2img 2048 ceiling) drives an OOM-scale latent.
    # _clamp_max_side bounds the longest side to 2048, preserving aspect ratio.
    from PIL import Image

    # A 12MP-shaped landscape photo -> longest side clamped to 2048, 4:3 aspect preserved.
    out = _clamp_max_side(Image.new("RGB", (4096, 3072)), 2048)
    assert out.size == (2048, 1536)
    # A portrait upload clamps on its longest (height) side.
    assert _clamp_max_side(Image.new("RGB", (1000, 4000)), 2048).size == (512, 2048)
    # An image already within bound is returned unchanged (no needless resample).
    small = Image.new("RGB", (768, 512))
    assert _clamp_max_side(small, 2048) is small


def test_detect_family_from_repo_id():
    # Detection is by architecture; Turbo/full and schnell/dev map to one family.
    assert detect_family("unsloth/Z-Image-Turbo-GGUF").name == "z-image"
    assert detect_family("unsloth/Z-Image-GGUF").name == "z-image"
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").name == "qwen-image"
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").name == "flux.1"
    # FLUX.2-klein is its own pipeline (Qwen3 TE), distinct from FLUX.1.
    klein = detect_family("unsloth/FLUX.2-klein-4B-GGUF")
    assert klein.name == "flux.2-klein"
    assert klein.pipeline_class == "Flux2KleinPipeline"
    assert klein.cfg_kwarg == "guidance_scale"
    # Both klein sizes share the one family (base repo resolved per-variant).
    assert detect_family("unsloth/FLUX.2-klein-9B-GGUF").name == "flux.2-klein"
    # FLUX.2-dev is the Mistral-based Flux2Pipeline, a distinct family from klein; its
    # gated base repo is reachable with an HF token. It must not collide with klein.
    dev = detect_family("unsloth/FLUX.2-dev-GGUF")
    assert dev.name == "flux.2-dev"
    assert dev.pipeline_class == "Flux2Pipeline"
    assert dev.base_repo == "black-forest-labs/FLUX.2-dev"
    assert detect_family("black-forest-labs/FLUX.2-dev").name == "flux.2-dev"
    # Qwen-Image guides via true_cfg_scale, not guidance_scale.
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").cfg_kwarg == "true_cfg_scale"
    assert detect_family("unsloth/Z-Image-GGUF").cfg_kwarg == "guidance_scale"
    # Qwen-Image-Edit is a SUPPORTED instruction-editing family (its own edit pipeline);
    # the most-specific match wins so it doesn't fall back to the generic qwen-image.
    edit = detect_family("unsloth/Qwen-Image-Edit-2511-GGUF")
    assert edit.name == "qwen-image-edit"
    assert edit.pipeline_class == "QwenImageEditPlusPipeline"
    assert edit.edit is True
    assert detect_family("unsloth/Qwen-Image-Edit-2509-GGUF").name == "qwen-image-edit"
    # FLUX Kontext is a SUPPORTED editing family (FluxKontextPipeline); the "kontext"
    # keyword is un-rejected for it, and it must win over the generic "flux.1" match.
    kontext = detect_family("unsloth/FLUX.1-Kontext-dev-GGUF")
    assert kontext.name == "flux.1-kontext"
    assert kontext.pipeline_class == "FluxKontextPipeline"
    assert kontext.edit is True
    assert kontext.cfg_kwarg == "guidance_scale"
    # A plain FLUX.1 checkpoint must still resolve to the base flux.1 family, not kontext.
    assert detect_family("unsloth/FLUX.1-dev-GGUF").name == "flux.1"
    # A plain Qwen-Image checkpoint must still resolve to the base family, not edit.
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").name == "qwen-image"
    # Krea 2 (diffusers >= 0.39): bf16-only single-stream DiT, no GGUF/sd.cpp mapping.
    krea2 = detect_family("krea/Krea-2-Turbo")
    assert krea2.name == "krea-2"
    assert krea2.pipeline_class == "Krea2Pipeline"
    assert krea2.transformer_class == "Krea2Transformer2DModel"
    assert krea2.cfg_kwarg == "guidance_scale"
    assert krea2.fp16_incompatible is True
    assert krea2.sd_cpp_text_encoders == ()
    assert detect_family("meta-llama/Llama-3-8B") is None


def test_detect_family_matches_reject_and_alias_by_segment():
    # Reject keywords and short aliases must match whole path/name segments, not raw substrings,
    # so an unrelated word that CONTAINS one doesn't misroute a valid base model (regression:
    # substring matching broke these).
    assert detect_family("/models/edited/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    assert detect_family("unsloth/Z-Image-Edition-GGUF").name == "z-image"
    assert detect_family("/models/kontextual/z-image-turbo-Q4_K_M.gguf").name == "z-image"
    # Supported edit families still resolve (edit / kontext are whole tokens there).
    assert detect_family("unsloth/Qwen-Image-Edit-2511-GGUF").name == "qwen-image-edit"
    assert detect_family("unsloth/FLUX.1-Kontext-dev-GGUF").name == "flux.1-kontext"
    # Unsupported variants sharing only a base arch keyword are still rejected.
    assert detect_family("unsloth/Qwen-Image-Layered-GGUF") is None
    assert detect_family("unsloth/Qwen-Image-2512-Inpaint") is None


def test_detect_family_edit_keyword_scoped_to_basename():
    from core.inference.diffusion_families import detect_family_for_pick

    # A parent directory named `edit`/`inpaint` must NOT poison a valid pick: only the model id
    # / filename basename is scanned for reject keywords. A direct local pick arrives as
    # (parent_dir, filename).
    assert detect_family("/models/edit") is None  # the dir alone is ambiguous
    assert detect_family_for_pick("/models/edit", "Z-Image-Turbo-Q4.gguf").name == "z-image"
    assert detect_family_for_pick("/models/inpaint", "qwen-image-2512-Q4.gguf").name == "qwen-image"
    # A genuinely unsupported variant keyword in the FILENAME still rejects.
    assert detect_family_for_pick("/models/misc", "Qwen-Image-Layered-Q4.gguf") is None


def test_detect_family_override():
    assert detect_family("local/path", override = "z-image").name == "z-image"
    assert detect_family("local/path", override = "zimage").name == "z-image"
    assert detect_family("local/path", override = "not-a-family") is None


def test_supported_family_names():
    names = supported_family_names()
    # The unknown-model error lists these, so the key families must be present.
    for expected in ("flux.1", "flux.2-klein", "flux.2-dev", "qwen-image", "z-image", "krea-2"):
        assert expected in names
    # Every listed name is a valid family_override (round-trips through detect_family).
    for name in names:
        assert detect_family("some/unknown-repo", override = name) is not None


def test_resolve_base_repo():
    fam = detect_family("x", override = "z-image")
    assert resolve_base_repo(fam, None) == fam.base_repo
    assert resolve_base_repo(fam, "   ") == fam.base_repo
    assert resolve_base_repo(fam, "custom/base") == "custom/base"


def test_resolve_local_gguf_child(tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    assert resolve_local_gguf_child(tmp_path, "model.gguf") == (tmp_path / "model.gguf").resolve()
    with pytest.raises(ValueError):
        resolve_local_gguf_child(tmp_path, "/etc/passwd")
    with pytest.raises(ValueError):
        resolve_local_gguf_child(tmp_path, "../secret.gguf")
    with pytest.raises(ValueError):
        resolve_local_gguf_child(tmp_path, "..\\secret.gguf")
    with pytest.raises(FileNotFoundError):
        resolve_local_gguf_child(tmp_path, "missing.gguf")


def test_resolve_local_gguf_child_blocks_symlink_escape(tmp_path):
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"secret")
    repo = tmp_path / "repo"
    repo.mkdir()
    try:
        (repo / "model.gguf").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(ValueError):
        resolve_local_gguf_child(repo, "model.gguf")


# Stubbed runtime for backend lifecycle


class _FakeDtype:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"torch.{self._name}"

    __str__ = __repr__


class _FakeGenerator:
    def __init__(self, device = None) -> None:
        self.device = device
        self.manual = None

    def seed(self) -> int:
        return 4242

    def manual_seed(self, value: int):
        self.manual = value
        return self


class _FakeImage:
    """Stand-in for a generated PIL image (the route persists it; here we only
    count how many come back)."""


class _FakePipe:
    def __init__(self) -> None:
        self.moved_to = None
        self.offloaded = False
        self.sequential_offloaded = False
        self.vae_tiled = False
        self.vae_sliced = False
        self.last_kwargs = None

    def to(self, device):
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self, device = None) -> None:
        self.offloaded = True
        self.offload_device = device

    def enable_sequential_cpu_offload(self, device = None) -> None:
        self.sequential_offloaded = True
        self.offload_device = device

    def enable_vae_tiling(self) -> None:
        self.vae_tiled = True

    def enable_vae_slicing(self) -> None:
        self.vae_sliced = True

    # Explicit signature (not just **kwargs) so generate()'s signature-gated guards for
    # negative_prompt / callback_on_step_end take effect -- a **kwargs-only fake would make
    # `"negative_prompt" in signature` always False.
    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        **kwargs,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "callback_on_step_end": callback_on_step_end,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
            **kwargs,
        }
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakePipeline:
    last: dict = {}
    last_single_file: dict = {}

    @classmethod
    def from_pretrained(cls, base, **kwargs):
        _FakePipeline.last = {"base": base, **kwargs}
        return _FakePipe()

    @classmethod
    def from_single_file(cls, path, **kwargs):
        # SDXL-style single-file: the WHOLE pipeline comes from one .safetensors file.
        _FakePipeline.last_single_file = {"path": path, **kwargs}
        return _FakePipe()


class _FakeTransformer:
    last: dict = {}

    @classmethod
    def from_single_file(cls, path, **kwargs):
        _FakeTransformer.last = {"path": path, **kwargs}
        return object()


class _FakeImg2ImgPipe:
    """An img2img pipeline call: records the image-conditioned kwargs. Its signature
    declares image/strength but NOT width/height, mirroring real img2img pipelines
    (which derive the output size from the input image)."""

    last_kwargs: dict = {}

    def __call__(
        self,
        *,
        prompt = None,
        image = None,
        strength = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        **kwargs,
    ):
        _FakeImg2ImgPipe.last_kwargs = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            **kwargs,
        }
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakeImg2ImgPipeline:
    built_from: object = None
    from_pipe_kwargs: dict = {}

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeImg2ImgPipeline.built_from = base_pipe
        _FakeImg2ImgPipeline.from_pipe_kwargs = kwargs
        return _FakeImg2ImgPipe()


class _FakeInpaintPipe:
    """An inpaint pipeline call: records image + mask_image + strength. Real inpaint
    pipelines take both an init image and a grayscale mask and derive output size from
    the input, so width/height are not in its signature."""

    last_kwargs: dict = {}

    def __call__(
        self,
        *,
        prompt = None,
        image = None,
        mask_image = None,
        strength = None,
        negative_prompt = None,
        callback_on_step_end = None,
        guidance_scale = None,
        true_cfg_scale = None,
        **kwargs,
    ):
        _FakeInpaintPipe.last_kwargs = {
            "prompt": prompt,
            "image": image,
            "mask_image": mask_image,
            "strength": strength,
            **kwargs,
        }
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakeInpaintPipeline:
    built_from: object = None

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeInpaintPipeline.built_from = base_pipe
        return _FakeInpaintPipe()


@pytest.fixture
def fake_runtime(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = _FakeDtype("bfloat16")
    torch.float16 = _FakeDtype("float16")
    torch.float32 = _FakeDtype("float32")
    torch.Generator = _FakeGenerator
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    torch.backends = types.SimpleNamespace(mps = None)
    # generate() wraps the pipe call in torch.inference_mode(); a no-op CM here.
    torch.inference_mode = lambda: contextlib.nullcontext()

    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)
    diffusers.ZImagePipeline = _FakePipeline
    diffusers.ZImageTransformer2DModel = _FakeTransformer
    diffusers.ZImageImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.ZImageInpaintPipeline = _FakeInpaintPipeline
    # Qwen-Image too, so the true_cfg_scale cfg-kwarg path is exercisable.
    diffusers.QwenImagePipeline = _FakePipeline
    diffusers.QwenImageTransformer2DModel = _FakeTransformer
    diffusers.QwenImageImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.QwenImageInpaintPipeline = _FakeInpaintPipeline
    # Instruction-editing pipeline (Qwen-Image-Edit): its own pipeline IS the loaded one.
    diffusers.QwenImageEditPlusPipeline = _FakePipeline
    # Ideogram 4, so its guidance_scale/guidance_schedule pairing is exercisable. It loads only
    # as a full pipeline (two DiTs), assembled per-component by load_ideogram4_pipeline -- stub
    # that to a fake pipe so the guidance path is reachable without real weights.
    diffusers.Ideogram4Pipeline = _FakePipeline
    diffusers.Ideogram4Transformer2DModel = _FakeTransformer
    # SDXL: a U-Net family. Its single-file checkpoint is the whole pipeline, so the pipeline
    # class carries from_single_file; UNet2DConditionModel is the denoiser class (fetched but
    # unused on the pipeline/single-file-pipeline paths).
    diffusers.StableDiffusionXLPipeline = _FakePipeline
    diffusers.UNet2DConditionModel = _FakeTransformer
    diffusers.StableDiffusionXLImg2ImgPipeline = _FakeImg2ImgPipeline
    diffusers.StableDiffusionXLInpaintPipeline = _FakeInpaintPipeline

    monkeypatch.setattr(
        "core.inference.diffusion.load_ideogram4_pipeline",
        lambda repo_id, dtype, hf_token = None: _FakePipe(),
    )

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    # The backend imports clear_gpu_cache by reference; no-op it so unload doesn't
    # run real hardware detection against the stubbed torch.
    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", lambda: None)
    _FakePipeline.last = {}
    _FakePipeline.last_single_file = {}
    _FakeTransformer.last = {}
    _FakeImg2ImgPipeline.built_from = None
    _FakeImg2ImgPipe.last_kwargs = {}
    _FakeInpaintPipeline.built_from = None
    _FakeInpaintPipe.last_kwargs = {}
    yield


def test_load_generate_unload_gguf(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
        hf_token = "hf_secret",
    )
    assert status["loaded"] is True
    assert status["family"] == "z-image"
    assert status["base_repo"] == "base/repo"
    assert status["device"] == "cpu"
    assert status["dtype"] == "float32"
    assert status["cpu_offload"] is False
    # Transformer built from the local GGUF, pipeline assembled from the base repo.
    assert _FakeTransformer.last["path"] == str((tmp_path / "model.gguf").resolve())
    assert _FakeTransformer.last["subfolder"] == "transformer"
    # The token reaches the (possibly gated) base config fetch and the pipeline.
    assert _FakeTransformer.last["token"] == "hf_secret"
    assert _FakePipeline.last["base"] == "base/repo"
    assert "transformer" in _FakePipeline.last

    gen = backend.generate(
        prompt = "a sloth", negative_prompt = "blurry", width = 512, height = 512, steps = 4, guidance = 3.0
    )
    assert gen["seed"] == 4242  # random seed reported back
    assert gen["repo_id"] == str(tmp_path)  # echoed so the route can record the model
    assert len(gen["images"]) == 1  # PIL images handed to the route for persistence
    # z-image guides via guidance_scale (not true_cfg_scale); the signature-gated
    # negative_prompt and per-step callback both reach the pipeline call.
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 3.0 and call["true_cfg_scale"] is None
    assert call["negative_prompt"] == "blurry"
    assert callable(call["callback_on_step_end"])

    gen2 = backend.generate(prompt = "again", seed = 99)
    assert gen2["seed"] == 99

    # batch_size produces that many images in one call, all sharing the seed.
    batch = backend.generate(prompt = "batch", seed = 7, batch_size = 3)
    assert len(batch["images"]) == 3 and batch["seed"] == 7

    assert backend.unload()["loaded"] is False
    assert backend.is_loaded is False


def test_generate_progress_active_during_setup(fake_runtime, tmp_path, monkeypatch):
    # A generation must report active from the moment it holds the lock, before the slow pre-denoise
    # setup. _apply_loras runs inside that window, so probe generate_progress() from there.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
        hf_token = "hf_secret",
    )

    seen = {}

    def fake_apply(self, state, loras, cancel):
        seen["progress"] = self.generate_progress()

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", fake_apply)

    # Idle before the run.
    assert backend.generate_progress()["active"] is False

    gen = backend.generate(prompt = "a sloth", steps = 4)
    assert len(gen["images"]) == 1

    # Active was published during setup, with the requested step total and step 0.
    assert seen["progress"]["active"] is True
    assert seen["progress"]["total_steps"] == 4
    assert seen["progress"]["step"] == 0

    # And it is cleared once the generation returns.
    assert backend.generate_progress()["active"] is False


def test_generate_progress_cleared_on_setup_error(fake_runtime, tmp_path, monkeypatch):
    # A setup-time failure skips the inner finally that nulls _gen, so the outer finally must
    # clear the published progress; otherwise a crashed generation leaves the UI stuck "active".
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
        hf_token = "hf_secret",
    )

    def boom(self, state, loras, cancel):
        raise RuntimeError("setup failed")

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", boom)

    with pytest.raises(RuntimeError, match = "setup failed"):
        backend.generate(prompt = "a sloth", steps = 4)

    assert backend.generate_progress()["active"] is False


def test_dense_speed_auto_defers_compile_to_third_generation(fake_runtime, tmp_path, monkeypatch):
    # Dense models with speed unset stay bit-identical eager for the first two generations; the
    # 3rd engages the `default` profile mid-session (repeated use amortises the one-time compile),
    # upgrading attention alongside it.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {"compiled": k.get("speed_mode") == "default"},
    )
    monkeypatch.setattr(dmod, "apply_attention_backend", lambda pipe, backend, logger = None: backend)
    monkeypatch.setattr(
        dmod,
        "select_attention_backend",
        lambda target, requested, speed_active = False: ("_native_cudnn" if speed_active else None),
    )
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
    assert status["speed_mode"] == "off"
    assert status["resolved"]["speed_mode"]["value"] == "deferred"
    assert status["resolved"]["speed_mode"]["source"] == "auto"

    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    assert backend.status()["speed_mode"] == "off"  # first two stay exact eager
    backend.generate(prompt = "three")
    status3 = backend.status()
    assert status3["speed_mode"] == "default"
    assert "compiled" in status3["speed_optims"]
    assert status3["attention_backend"] == "_native_cudnn"
    assert status3["resolved"]["speed_mode"]["value"] == "default"

    # An explicit "off" is pinned: no deferral, still eager after 3 generations.
    backend.unload()
    status_off = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
        speed_mode = "off",
    )
    assert status_off["resolved"]["speed_mode"]["value"] == "off"
    for p in ("a", "b", "c"):
        backend.generate(prompt = p)
    assert backend.status()["speed_mode"] == "off"
    backend.unload()


def test_deferred_speed_skips_when_lora_requested(fake_runtime, tmp_path, monkeypatch):
    # A compiled transformer rejects LoRA (supports_lora False once compiled), and _apply_loras
    # raises before its unchanged-selection no-op, so engaging the deferred compile on a LoRA
    # generation would permanently break every LoRA generation on this load. The deferral must
    # skip while a LoRA is requested and engage only on a later LoRA-free generation.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    engaged: list = []

    def fake_engage(self, state):
        engaged.append(state.generation_count)
        state.speed_deferred = False  # mirror the real helper: engage once, then clear

    monkeypatch.setattr(DiffusionBackend, "_engage_deferred_speed", fake_engage)
    # LoRA loading is covered elsewhere; stub it so this test needs no adapter file.
    monkeypatch.setattr(DiffusionBackend, "_apply_loras", lambda self, state, loras, cancel: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    # 3rd generation requests a LoRA: the deferral must be skipped (pipe stays eager, LoRA-capable).
    backend.generate(prompt = "three", loras = [("adapter", 1.0)])
    assert engaged == []
    # 4th generation without a LoRA: the deferral now engages (the guard is LoRA-specific, not off).
    backend.generate(prompt = "four")
    assert len(engaged) == 1


def test_deferred_speed_skips_while_adapter_attached(fake_runtime, tmp_path, monkeypatch):
    # Even a NO-LoRA generation must defer the compile while an adapter from a PRIOR generation is
    # still attached: _apply_loras runs AFTER the engage, so compiling here would bake the resident
    # adapter into the graph and the later unload (swallowed on a compiled pipe) would leave it
    # active forever -- silent wrong output. Defer until _apply_loras clears it.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    engaged: list = []

    def fake_engage(self, state):
        engaged.append(state.generation_count)
        state.speed_deferred = False

    monkeypatch.setattr(DiffusionBackend, "_engage_deferred_speed", fake_engage)

    # Track the attached set on the pipe, mirroring the real _apply_loras marker (_unsloth_loras).
    def fake_apply(self, state, loras, cancel):
        specs = [(i, w) for (i, w) in (loras or []) if w != 0]
        state.pipe._unsloth_loras = tuple(specs)

    monkeypatch.setattr(DiffusionBackend, "_apply_loras", fake_apply)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
    # Gens 1-2 attach an adapter, so it is still resident going into gen 3.
    backend.generate(prompt = "one", loras = [("adapter", 1.0)])
    backend.generate(prompt = "two", loras = [("adapter", 1.0)])
    # Gen 3 requests NO LoRA but the adapter is still attached -> defer (no compile-with-adapter).
    backend.generate(prompt = "three")
    assert engaged == []
    # Gen 3's _apply_loras([]) cleared the adapter; gen 4 is genuinely LoRA-free -> engage.
    backend.generate(prompt = "four")
    assert len(engaged) == 1


def test_deferred_speed_preserves_explicit_attention(fake_runtime, tmp_path, monkeypatch):
    # A dense model loaded with Speed on Auto but Attention explicitly pinned (e.g. "native" to
    # avoid cuDNN) must KEEP that choice when the 3rd generation engages the deferred `default`
    # profile. The auto cuDNN upgrade applies only when attention was left on auto, never when the
    # caller pinned a backend.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(dmod, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        dmod,
        "apply_speed_optims",
        lambda pipe, target, **k: {"compiled": k.get("speed_mode") == "default"},
    )
    monkeypatch.setattr(dmod, "apply_attention_backend", lambda pipe, backend, logger = None: backend)

    # A select mock that -- unlike a bare "auto -> cuDNN" stub -- HONORS an explicit request:
    # "native" stays on the default (None) even under a speed profile, and only a left-unset
    # ("auto"/None) request upgrades to cuDNN when speed is active.
    def fake_select(
        target,
        requested,
        speed_active = False,
    ):
        if requested in (None, "", "auto"):
            return "_native_cudnn" if speed_active else None
        if str(requested).lower() in ("native", "sdpa"):
            return None
        return requested

    monkeypatch.setattr(dmod, "select_attention_backend", fake_select)
    monkeypatch.setattr(dmod.compile_cache, "begin", lambda **k: None)

    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
        attention_backend = "native",
    )
    backend.generate(prompt = "one")
    backend.generate(prompt = "two")
    backend.generate(prompt = "three")  # deferred profile engages here
    status = backend.status()
    assert status["speed_mode"] == "default"  # the compile profile still engaged
    assert "compiled" in status["speed_optims"]
    # The pinned "native" survived: NOT silently upgraded to cuDNN.
    assert status["attention_backend"] is None
    assert status["resolved"]["attention_backend"]["value"] == "native"
    assert status["resolved"]["attention_backend"]["source"] == "explicit"

    # Control: with attention left on auto, the same 3rd-generation deferral DOES upgrade
    # to cuDNN -- so the assertion above is not vacuously passing.
    backend.unload()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
    for p in ("a", "b", "c"):
        backend.generate(prompt = p)
    assert backend.status()["attention_backend"] == "_native_cudnn"
    backend.unload()


def _tiny_png_b64() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 30, 30)).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_generate_img2img_uses_from_pipe(fake_runtime, tmp_path):
    """An init_image routes generate() through the family's img2img pipeline, built via
    Pipeline.from_pipe around the loaded pipe (no reload), with image + strength passed
    and width/height dropped (the img2img pipe derives size from the input image)."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    # The loaded family advertises the image-conditioned workflows for UI gating
    # (upscale rides the img2img pipeline, so it appears whenever img2img does).
    assert backend.status()["workflows"] == ["txt2img", "img2img", "upscale", "inpaint", "outpaint"]

    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a car at sunset",
        steps = 4,
        guidance = 0.0,
        seed = 3,
        init_image = _tiny_png_b64(),
        strength = 0.5,
    )
    assert len(out["images"]) == 1
    # from_pipe was handed the loaded text-to-image pipe (component reuse, no reload).
    assert _FakeImg2ImgPipeline.built_from is loaded_pipe
    # ...and with torch_dtype=None so from_pipe SKIPS its default float32 recast, which
    # both upcasts the reused bf16 modules and crashes on torchao-quantized weights.
    assert _FakeImg2ImgPipeline.from_pipe_kwargs.get("torch_dtype", "MISSING") is None
    call = _FakeImg2ImgPipe.last_kwargs
    assert call["image"] is not None  # decoded source image passed through
    assert call["strength"] == 0.5
    assert "width" not in call and "height" not in call  # img2img derives size from image

    # A txt2img call after it still uses the base pipe (no image kwarg).
    backend.generate(prompt = "plain", steps = 4, seed = 1)
    assert backend._state.pipe.last_kwargs.get("image") is None


def test_generate_img2img_unsupported_family_raises(fake_runtime, tmp_path, monkeypatch):
    """A family with no image-conditioning at all (no img2img/inpaint/edit/reference) rejects
    an init_image with a clear error rather than failing deep in the pipeline."""
    from core.inference.diffusion_families import DiffusionFamily

    # A synthetic txt2img-only family: no img2img/inpaint pipeline, not edit, not reference.
    # (Every shipped family now supports some image workflow, so build one for this case.)
    plain = DiffusionFamily(
        name = "plain-test",
        pipeline_class = "ZImagePipeline",
        transformer_class = "ZImageTransformer2DModel",
        base_repo = "base/repo",
    )
    monkeypatch.setattr(
        "core.inference.diffusion.detect_family_for_pick",
        lambda repo_id, gguf_filename = None, override = None: plain,
    )
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo")
    assert backend.status()["workflows"] == ["txt2img"]
    with pytest.raises(ValueError, match = "img2img"):
        backend.generate(prompt = "x", steps = 4, init_image = _tiny_png_b64())


def test_generate_rejects_conditioning_without_init_image(fake_runtime, tmp_path):
    """mask / upscale / reference all need an input image; without one they must raise a
    clear ValueError rather than silently degrading to txt2img."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    with pytest.raises(ValueError, match = "mask_image requires"):
        backend.generate(prompt = "x", steps = 4, mask_image = _mask_b64(64))
    with pytest.raises(ValueError, match = "upscale requires"):
        backend.generate(prompt = "x", steps = 4, upscale = 2.0)
    with pytest.raises(ValueError, match = "reference_images require"):
        backend.generate(prompt = "x", steps = 4, reference_images = [_tiny_png_b64()])


def test_generate_rejects_reference_on_unsupported_family(fake_runtime, tmp_path):
    """A non-reference family rejects reference_images instead of silently dropping them."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    with pytest.raises(ValueError, match = "Reference images are not supported"):
        backend.generate(
            prompt = "x",
            steps = 4,
            init_image = _tiny_png_b64(),
            reference_images = [_tiny_png_b64()],
        )


def test_generate_upscale_enlarges_and_low_strength(fake_runtime, tmp_path):
    """An init_image + upscale factor routes generate() through the family's img2img
    pipeline (hires fix): the source is enlarged to size*factor (rounded to /16) before the
    denoise, the strength defaults low, and the factor is capped so a huge value can't OOM."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    # Upscale rides the img2img pipeline, so it is advertised alongside img2img.
    assert "upscale" in backend.status()["workflows"]

    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a crisp photo",
        steps = 4,
        guidance = 0.0,
        seed = 3,
        init_image = _tiny_png_b64(),
        upscale = 2.0,  # 64 -> 128, no explicit strength
    )
    assert len(out["images"]) == 1
    # Reuses the resident modules via from_pipe (no reload, no extra VRAM).
    assert _FakeImg2ImgPipeline.built_from is loaded_pipe
    call = _FakeImg2ImgPipe.last_kwargs
    # The image handed to the pipe is the ENLARGED source (64 * 2 = 128, already /16).
    assert call["image"].size == (128, 128)
    # Strength defaults to the hires-fix value when the caller sends none.
    assert call["strength"] == 0.35

    # The factor is capped at 4x so a large request can't blow up the VAE/transformer.
    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _tiny_png_b64(),
        upscale = 99.0,
    )
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (256, 256)  # 64 * 4 (capped)

    # An explicit strength overrides the hires-fix default.
    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _tiny_png_b64(),
        upscale = 1.5,
        strength = 0.2,
    )
    assert _FakeImg2ImgPipe.last_kwargs["strength"] == 0.2
    # 64 * 1.5 = 96, already a multiple of 16.
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (96, 96)


def _png_b64(side: int) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (side, side), (10, 20, 30)).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_image_rejects_oversized(fake_runtime, tmp_path):
    """An input image larger than the per-side cap is rejected with a clear error (protects
    img2img / inpaint / reference from decompression-bomb / OOM inputs), not a 500."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    with pytest.raises(ValueError, match = "too large"):
        backend.generate(prompt = "x", steps = 4, init_image = _png_b64(4112))  # > 4096/side


def test_upscale_output_is_capped(fake_runtime, tmp_path):
    """Upscale bounds the absolute output side to 2048 even when input*factor exceeds it, so a
    large upload at 4x can't OOM the VAE/transformer."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    backend.generate(prompt = "x", steps = 4, seed = 1, init_image = _png_b64(1024), upscale = 4.0)
    # 1024 * 4 = 4096 -> clamped to 2048 (longest side), still a multiple of 16.
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (2048, 2048)


def _mask_b64(side: int) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("L", (side, side), 0)
    for y in range(side // 4, 3 * side // 4):
        for x in range(side // 4, 3 * side // 4):
            img.putpixel((x, y), 255)
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_img2img_snaps_non_multiple_of_16(fake_runtime, tmp_path):
    """An odd-sized img2img upload (not divisible by 16) is auto-resized to the nearest
    multiple of 16 so the pipeline's divisibility check passes instead of erroring."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    backend.generate(prompt = "x", steps = 4, seed = 1, init_image = _png_b64(186), strength = 0.5)
    # 186 / 16 = 11.625 -> round to 12 -> 192.
    assert _FakeImg2ImgPipe.last_kwargs["image"].size == (192, 192)


def test_inpaint_snaps_image_and_mask_together(fake_runtime, tmp_path):
    """Inpaint snaps the odd-sized input to /16 AND resizes the mask to match, so the image
    and mask stay aligned (a mismatch would crash the inpaint pipeline)."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    backend.generate(
        prompt = "x",
        steps = 4,
        seed = 1,
        init_image = _png_b64(186),
        mask_image = _mask_b64(186),
        strength = 0.5,
    )
    assert _FakeInpaintPipe.last_kwargs["image"].size == (192, 192)
    assert _FakeInpaintPipe.last_kwargs["mask_image"].size == (192, 192)


def test_generate_reference_uses_loaded_pipe_at_slider_size(fake_runtime, tmp_path):
    """A reference family (FLUX.2-klein) advertises txt2img + reference, and a generate with
    an init_image passes it as the loaded pipe's `image` arg (no from_pipe, no strength) while
    the output size stays the REQUESTED slider size (the pipe resizes the reference itself)."""
    import diffusers

    diffusers.Flux2KleinPipeline = _FakePipeline
    diffusers.Flux2KleinInpaintPipeline = _FakeInpaintPipeline
    diffusers.Flux2Transformer2DModel = _FakeTransformer
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "flux.2-klein",
    )
    # FLUX.2-klein: txt2img + reference (own pipe) + inpaint (dedicated pipe). No img2img class,
    # so no img2img/upscale.
    assert backend.status()["workflows"] == ["txt2img", "reference", "inpaint"]

    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a portrait in this style",
        steps = 6,
        guidance = 4.0,
        seed = 5,
        width = 768,
        height = 512,
        init_image = _tiny_png_b64(),
        strength = 0.5,
    )
    assert len(out["images"]) == 1
    call = loaded_pipe.last_kwargs
    assert call["image"] is not None  # reference handed to the loaded pipe
    assert call["width"] == 768 and call["height"] == 512  # OUTPUT size = sliders, not input
    assert "strength" not in call  # reference conditioning has no strength
    assert "mask_image" not in call
    # Guidance flows via guidance_scale (FLUX.2 default behaviour).
    assert call["guidance_scale"] == 4.0

    # Multi-reference: extra reference_images are combined with init_image into a LIST so the
    # model can blend several references (subject + style).
    backend.generate(
        prompt = "combine these",
        steps = 6,
        seed = 9,
        width = 1024,
        height = 1024,
        init_image = _tiny_png_b64(),
        reference_images = [_tiny_png_b64(), _tiny_png_b64()],
    )
    img_arg = loaded_pipe.last_kwargs["image"]
    assert isinstance(img_arg, list) and len(img_arg) == 3  # primary + 2 extras

    # Branch ordering: an init image + MASK on a reference family must route to inpaint (the
    # dedicated pipeline), NOT be swallowed by the reference branch (which ignores the mask).
    backend.generate(
        prompt = "repaint here",
        steps = 6,
        seed = 2,
        init_image = _tiny_png_b64(),
        mask_image = _tiny_mask_b64(),
        strength = 0.8,
    )
    assert _FakeInpaintPipeline.built_from is loaded_pipe  # built via from_pipe off the load
    assert _FakeInpaintPipe.last_kwargs["mask_image"] is not None
    assert _FakeInpaintPipe.last_kwargs["strength"] == 0.8

    # Without an init image the same family does plain txt2img (no image arg).
    backend.generate(prompt = "just text", steps = 6, seed = 1)
    assert backend._state.pipe.last_kwargs.get("image") is None


def _tiny_mask_b64() -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    # A grayscale mask: white square (repaint) on black (keep).
    img = Image.new("L", (64, 64), 0)
    for y in range(16, 48):
        for x in range(16, 48):
            img.putpixel((x, y), 255)
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_generate_inpaint_uses_from_pipe(fake_runtime, tmp_path):
    """An init_image + mask_image routes generate() through the family's inpaint pipeline,
    built via Pipeline.from_pipe around the loaded pipe (no reload), with the decoded image
    + mask + strength passed through and width/height dropped (size derives from the input)."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    loaded_pipe = backend._state.pipe
    out = backend.generate(
        prompt = "a red door",
        steps = 4,
        guidance = 0.0,
        seed = 5,
        init_image = _tiny_png_b64(),
        mask_image = _tiny_mask_b64(),
        strength = 0.7,
    )
    assert len(out["images"]) == 1
    # The inpaint pipe (not img2img) was selected and built from the loaded pipe.
    assert _FakeInpaintPipeline.built_from is loaded_pipe
    assert _FakeImg2ImgPipeline.built_from is None
    call = _FakeInpaintPipe.last_kwargs
    assert call["image"] is not None and call["mask_image"] is not None
    assert call["strength"] == 0.7
    assert "width" not in call and "height" not in call  # inpaint derives size from image


def test_image_conditioned_passes_image_size_not_slider(fake_runtime, tmp_path):
    """When the workflow pipe DOES accept width/height, an image-conditioned call must pass
    the INPUT IMAGE's size, never the txt2img slider size -- otherwise a non-slider-sized
    input (e.g. a 1536px outpaint canvas with a 1024 slider) mismatches the latents
    ("tensor a (128) must match tensor b (192)"). Covers Transform + Extend with any size."""
    import base64
    import io

    from PIL import Image

    class _SizePipe:
        last: dict = {}

        def __call__(
            self,
            *,
            prompt = None,
            image = None,
            strength = None,
            width = None,
            height = None,
            negative_prompt = None,
            callback_on_step_end = None,
            guidance_scale = None,
            true_cfg_scale = None,
            **kwargs,
        ):
            _SizePipe.last = {"width": width, "height": height}
            n = kwargs.get("num_images_per_prompt", 1)
            return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])

    class _SizePipeline:
        @classmethod
        def from_pipe(cls, base_pipe, **kwargs):
            return _SizePipe()

    import diffusers

    diffusers.ZImageImg2ImgPipeline = _SizePipeline
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    buf = io.BytesIO()
    Image.new("RGB", (96, 64), (10, 20, 30)).save(buf, format = "PNG")  # non-square, non-slider
    b64 = base64.b64encode(buf.getvalue()).decode()
    backend.generate(prompt = "x", steps = 4, width = 1024, height = 1024, init_image = b64, strength = 0.5)
    # The pipe got the IMAGE's 96x64, not the 1024x1024 slider.
    assert _SizePipe.last == {"width": 96, "height": 64}


def test_compile_shape_dims_follow_workflow():
    """_compile_shape_dims mirrors generate()'s width/height derivation: slider size for
    txt2img / reference / controlnet, the input image's size for the image-conditioned
    workflows (whose forward runs at init_pil.size, whatever the sliders say)."""
    from PIL import Image

    from core.inference.diffusion import _compile_shape_dims

    img = Image.new("RGB", (96, 64), (10, 20, 30))
    assert _compile_shape_dims("txt2img", None, 1024, 512) == (1024, 512)
    # reference generates at the slider size even though an init image is present.
    assert _compile_shape_dims("reference", img, 1024, 512) == (1024, 512)
    assert _compile_shape_dims("controlnet", None, 768, 768) == (768, 768)
    for wf in ("img2img", "inpaint", "upscale", "edit"):
        assert _compile_shape_dims(wf, img, 1024, 512) == (96, 64)


def test_register_shape_uses_actual_forward_dims(fake_runtime, tmp_path, monkeypatch):
    """The static compile-cache manifest must record the dims the forward ACTUALLY ran
    at: an image-conditioned generate derives its output size from the input image, so
    registering the slider values would mark a never-compiled shape as covered while the
    truly-used shape never re-dirties/saves the bundle (warm restarts keep paying its
    compile)."""
    from core.inference import diffusion as diff

    registered: list = []
    monkeypatch.setattr(
        diff.compile_cache,
        "register_shape",
        lambda ctx, shape, *, static: registered.append(tuple(shape)),
    )
    monkeypatch.setattr(diff.compile_cache, "save", lambda ctx, *, logger = None: True)
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "model.gguf", base_repo = "base/repo", family_override = "z-image"
    )
    # txt2img registers the requested slider size.
    backend.generate(prompt = "x", steps = 4, width = 1024, height = 512, seed = 1)
    assert registered[-1] == (1024, 512, 1)
    # img2img runs at the INPUT image's 64x64; the 1024x512 slider must not be recorded.
    backend.generate(
        prompt = "x",
        steps = 4,
        width = 1024,
        height = 512,
        seed = 1,
        init_image = _tiny_png_b64(),
        strength = 0.5,
    )
    assert registered[-1] == (64, 64, 1)


def test_edit_family_uses_own_pipeline_and_requires_image(fake_runtime, tmp_path):
    """An instruction-editing family (Qwen-Image-Edit) exposes only the 'edit' workflow,
    runs the image through its OWN loaded pipeline (no from_pipe), and rejects a call with
    no input image."""
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Qwen/Qwen-Image-Edit-2511",
        family_override = "qwen-image-edit",
    )
    # Edit families advertise only the edit workflow (no txt2img / img2img / inpaint).
    assert backend.status()["workflows"] == ["edit"]
    loaded_pipe = backend._state.pipe

    out = backend.generate(
        prompt = "make it night",
        steps = 8,
        guidance = 4.0,
        seed = 1,
        init_image = _tiny_png_b64(),
    )
    assert len(out["images"]) == 1
    # The loaded pipe handled it directly -- no from_pipe img2img/inpaint was built.
    assert backend._state.pipe is loaded_pipe
    assert _FakeImg2ImgPipeline.built_from is None and _FakeInpaintPipeline.built_from is None
    assert loaded_pipe.last_kwargs.get("image") is not None

    # An edit model with no input image fails fast with a clear message.
    with pytest.raises(ValueError, match = "image"):
        backend.generate(prompt = "make it night", steps = 8)


def test_load_pipeline_kind_uses_from_pretrained(fake_runtime):
    """A full-pipeline (no single-file) load on an unsloth/* repo builds the pipe with
    pipeline_cls.from_pretrained(repo_id) -- NO single-file transformer build, NO GGUF
    quant config -- so an embedded bnb-4bit config is reloaded by diffusers itself."""
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        "unsloth/Z-Image-Turbo-unsloth-bnb-4bit", family_override = "z-image"
    )
    assert status["loaded"] is True
    assert status["family"] == "z-image"
    # from_pretrained pointed at the repo itself (it IS its own base), with no transformer.
    assert _FakePipeline.last["base"] == "unsloth/Z-Image-Turbo-unsloth-bnb-4bit"
    assert "transformer" not in _FakePipeline.last
    # The GGUF single-file build path was never taken.
    assert _FakeTransformer.last == {}


def test_load_single_file_safetensors_no_gguf_config(fake_runtime, tmp_path):
    """A single-file *.safetensors transformer is built with from_single_file WITHOUT the
    GGUF dequant config (it carries its own dtype), then assembled from the base repo."""
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.safetensors",
        base_repo = "base/repo",
        family_override = "qwen-image",
    )
    assert status["loaded"] is True
    assert _FakeTransformer.last["path"] == str((tmp_path / "model.safetensors").resolve())
    assert _FakeTransformer.last["subfolder"] == "transformer"
    # No GGUF quant config on the safetensors path (the GGUF path sets one).
    assert "quantization_config" not in _FakeTransformer.last
    assert _FakePipeline.last["base"] == "base/repo"
    assert "transformer" in _FakePipeline.last


def test_load_sdxl_pipeline_from_pretrained(fake_runtime):
    """SDXL as a full pipeline (no single-file name) loads via pipeline_cls.from_pretrained
    on the allowlisted official base repo -- no U-Net single-file build, no GGUF config.
    A U-Net family must NOT try to build a transformer from a single file."""
    backend = DiffusionBackend()
    status = backend.load_pipeline("stabilityai/stable-diffusion-xl-base-1.0")
    assert status["loaded"] is True
    assert status["family"] == "sdxl"
    assert _FakePipeline.last["base"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert "transformer" not in _FakePipeline.last
    # Neither single-file path (transformer-only nor whole-pipeline) was taken.
    assert _FakeTransformer.last == {}
    assert _FakePipeline.last_single_file == {}


def test_load_sdxl_single_file_uses_pipeline_from_single_file(fake_runtime, tmp_path):
    """A single-file SDXL *.safetensors is the WHOLE pipeline: it must load via
    pipeline_cls.from_single_file(path, config=base), NOT transformer_cls.from_single_file
    (UNet2DConditionModel has no companion-transformer assembly here)."""
    (tmp_path / "sdxl.safetensors").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "sdxl.safetensors", family_override = "sdxl"
    )
    assert status["loaded"] is True
    assert status["family"] == "sdxl"
    # The whole-pipeline single-file path was taken with the base repo as config.
    assert _FakePipeline.last_single_file["path"] == str((tmp_path / "sdxl.safetensors").resolve())
    assert _FakePipeline.last_single_file["config"] == "stabilityai/stable-diffusion-xl-base-1.0"
    # The transformer-only single-file build was NOT taken.
    assert _FakeTransformer.last == {}


def test_load_sdxl_allowlisted_turbo_repo_is_trusted(fake_runtime):
    """The official sdxl-turbo repo is on the non-GGUF allowlist, so a full-pipeline load
    is permitted even though it is not under unsloth/*."""
    backend = DiffusionBackend()
    status = backend.load_pipeline("stabilityai/sdxl-turbo")
    assert status["loaded"] is True
    assert status["family"] == "sdxl"


def test_load_pipeline_rejects_non_unsloth_repo(fake_runtime):
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "unsloth"):
        backend.load_pipeline("randomorg/Z-Image-bnb-4bit", family_override = "z-image")


def test_load_sdxl_rejects_untrusted_repo(fake_runtime):
    """A random non-allowlisted, non-unsloth repo is still rejected for a full pipeline
    load even when it detects as SDXL -- the allowlist is exact-match only."""
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "unsloth"):
        backend.load_pipeline("randomorg/my-sdxl-merge", family_override = "sdxl")


def test_validate_gates_untrusted_base_repo(fake_runtime, tmp_path):
    # A companion base_repo also loads via from_pretrained, so a trusted GGUF model_path must not
    # smuggle in an arbitrary remote base: base_repo clears the same trust bar as a non-GGUF repo
    # id (mirrors the video loader), and the check runs before any GPU handoff.
    backend = DiffusionBackend()
    with pytest.raises(ValueError, match = "base_repo"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = "evil/companions",
        )
    # A local base_repo dir that is NOT a diffusers pipeline (no model_index.json) is rejected
    # HERE, before the GPU handoff: it passes the any-existing-path trust check but the base loads
    # via from_pretrained (needs model_index.json), so it would else evict the resident model and
    # only then fail in the background load.
    bad_base = tmp_path / "bare-base"
    bad_base.mkdir()
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = str(bad_base),
        )
    # A local base_repo that IS a real pipeline dir (model_index.json) passes the gate.
    (tmp_path / "model_index.json").write_text("{}")
    fam = backend.validate_load_request(
        "unsloth/Qwen-Image-2512-GGUF",
        gguf_filename = "x.gguf",
        model_kind = "gguf",
        base_repo = str(tmp_path),
    )
    assert fam is not None


def test_resolve_local_single_file(tmp_path):
    # A bare single-file safetensors directory (no model_index.json) resolves to that checkpoint's
    # basename, so the images load route can reinterpret an On-Device "pipeline" pick as a
    # single_file load instead of 400ing on the missing model_index.json.
    from core.inference.diffusion import resolve_local_single_file

    d = tmp_path / "solo"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(d)) == "model.safetensors"

    # A real diffusers pipeline dir (has model_index.json) loads as a pipeline unchanged -> None.
    (d / "model_index.json").write_text("{}")
    assert resolve_local_single_file(str(d)) is None

    # Ambiguous (two checkpoints, e.g. a sharded pipeline) or empty dirs -> None (unchanged load).
    d2 = tmp_path / "shards"
    d2.mkdir()
    (d2 / "a.safetensors").write_bytes(b"w")
    (d2 / "b.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(d2)) is None
    assert resolve_local_single_file(str(tmp_path / "empty-nonexistent")) is None
    # A remote repo id (not a local dir) -> None.
    assert resolve_local_single_file("unsloth/Qwen-Image-2512-GGUF") is None

    # A PEFT LoRA adapter folder (adapter_config.json + adapter_model.safetensors), even with a
    # family-token name, is NOT a base checkpoint: from_single_file would fail on the adapter
    # weights AFTER the route evicted the resident model, so it must not be reinterpreted as a
    # single_file pick -> None (the pipeline pick then 400s in validation, before the handoff).
    adapter = tmp_path / "flux-style-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(adapter)) is None
    # A bare adapter_model.safetensors (no config) is likewise not treated as the sole checkpoint.
    adapter2 = tmp_path / "z-image-lora"
    adapter2.mkdir()
    (adapter2 / "adapter_model.safetensors").write_bytes(b"w")
    assert resolve_local_single_file(str(adapter2)) is None


def test_resolve_base_repo_drops_untrusted_card_tag(monkeypatch):
    # With no base_repo, the base is resolved from the GGUF repo's base_model card tag --
    # attacker-controlled metadata on any remote repo -- then loaded via from_pretrained. An
    # untrusted tag must be dropped for the curated family default, so an attacker GGUF repo can't
    # point the base at an arbitrary repo to be deserialized.
    import core.inference.diffusion as dmod

    fam = detect_family("unsloth/FLUX.1-dev-GGUF")
    # A malicious card tag is ignored -> the family default base is used instead.
    monkeypatch.setattr(dmod, "_hf_base_model", lambda repo_id, hf_token: "attacker/evil-pipeline")
    assert _resolve_base_repo("attacker/flux.1-evil-GGUF", None, fam, None) == fam.base_repo
    # A trusted (allowlisted) card tag is still honoured, so variant resolution is not regressed.
    monkeypatch.setattr(
        dmod, "_hf_base_model", lambda repo_id, hf_token: "black-forest-labs/FLUX.1-dev"
    )
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", None, fam, None)
        == "black-forest-labs/FLUX.1-dev"
    )
    # An explicit trusted base_repo wins over the card tag; an explicit untrusted one is caught
    # earlier at validate_load_request (covered by test_validate_gates_untrusted_base_repo).
    assert (
        _resolve_base_repo("unsloth/FLUX.1-dev-GGUF", "unsloth/custom-base", fam, None)
        == "unsloth/custom-base"
    )


def test_detect_family_rejects_layered():
    # Qwen-Image-Layered needs a dedicated pipeline (additional_t_cond); it must be
    # rejected so it fails fast at load instead of crashing at the first denoise step.
    assert detect_family("unsloth/Qwen-Image-Layered-GGUF") is None
    assert detect_family("unsloth/qwen_image_layered") is None


def test_failed_load_rolls_back_eager_patches(fake_runtime, tmp_path, monkeypatch):
    """A load failure AFTER the eager patches install but BEFORE the _LoadState commit must
    roll the process-wide patches back, so the next bit-identical `off` load is not
    contaminated (the asymmetric-cleanup bug the reviewers flagged)."""
    from core.inference import diffusion as diff_mod
    from core.inference import diffusion_eager_patches as ep

    (tmp_path / "model.gguf").write_bytes(b"x")
    ep.uninstall_patches()  # clean slate

    def _boom(*_a, **_k):
        raise RuntimeError("placement boom")

    # apply_memory_plan runs AFTER the patches are installed, before _LoadState commits.
    monkeypatch.setattr(diff_mod, "apply_memory_plan", _boom)
    backend = DiffusionBackend()
    with pytest.raises(RuntimeError):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.gguf",
            family_override = "z-image",
            base_repo = "base/repo",
            speed_mode = "eager",  # != off -> installs the shared patches
        )
    assert ep.is_installed() is False  # rolled back by the load-failure finally
    assert backend.is_loaded is False


def test_cpu_offload_ignored_off_cuda(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        family_override = "z-image",
        base_repo = "base/repo",
        cpu_offload = True,
    )
    # No CUDA in the stub, so offload is not engaged.
    assert status["cpu_offload"] is False


def test_low_vram_ignored_off_cuda(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        family_override = "z-image",
        base_repo = "base/repo",
        memory_mode = "low_vram",
    )
    # No CUDA in the stub, so offload is not engaged regardless of the request.
    assert status["cpu_offload"] is False


def test_generate_without_load_raises(fake_runtime):
    backend = DiffusionBackend()
    with pytest.raises(RuntimeError):
        backend.generate(prompt = "x")


def test_failed_load_restores_backend_flags(fake_runtime, tmp_path, monkeypatch):
    # A failure AFTER apply_speed_optims (here an OOM in apply_memory_plan) must go through the
    # load's try/finally and restore the process-global TF32 / cudnn flags, so a later `off` load
    # is still bit-identical, and must not commit partial state. Regression: a refactor dropped
    # this guard, leaking the flags on a failed load.
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()

    restored: list = []
    cleared: list = []
    monkeypatch.setattr(
        "core.inference.diffusion.restore_backend_flags", lambda snap: restored.append(snap)
    )
    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", lambda: cleared.append(True))
    monkeypatch.setattr(
        "core.inference.diffusion.apply_memory_plan",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
    )

    with pytest.raises(RuntimeError, match = "out of memory"):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.gguf",
            family_override = "z-image",
            base_repo = "base/repo",
            speed_mode = "max",
        )
    assert restored, "restore_backend_flags was not called on the failed-load path"
    assert cleared, "clear_gpu_cache was not called on the failed-load path (VRAM leak)"
    assert backend._state is None and backend.is_loaded is False


def test_resolve_base_repo_prefers_caller_then_hf_tag_then_fallback(monkeypatch):
    from core.inference import diffusion
    from core.inference.diffusion_families import detect_family

    fam = detect_family("unsloth/Qwen-Image-2512-GGUF")
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo, tok: "Qwen/Qwen-Image-2512")
    # Caller's explicit base wins and the HF tag is not consulted.
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "my/base", fam, None)
        == "my/base"
    )
    # No caller base: the repo's base_model tag (the variant base) is used.
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", None, fam, None)
        == "Qwen/Qwen-Image-2512"
    )
    # No caller base and no tag: the family fallback.
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo, tok: None)
    assert (
        diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "  ", fam, None)
        == fam.base_repo
    )


def test_load_without_gguf_raises():
    backend = DiffusionBackend()
    # No gguf_filename -> a full-pipeline load, gated to unsloth/*; a non-unsloth repo
    # is rejected before any GPU/network work.
    with pytest.raises(ValueError, match = "unsloth"):
        backend.load_pipeline("some-org/Z-Image-bnb-4bit")


def test_load_unknown_family_raises():
    backend = DiffusionBackend()
    with pytest.raises(ValueError):
        backend.load_pipeline("some/unrecognised-repo", gguf_filename = "x.gguf")


# load_progress state machine (no threads / network / real cache)

from core.inference.diffusion import _LoadingState, _LoadState  # noqa: E402


def test_load_progress_idle_and_ready():
    backend = DiffusionBackend()
    assert backend.load_progress()["phase"] is None
    backend._state = _LoadState(object(), None, "r", "b", "cpu", "float32", False)
    assert backend.load_progress()["phase"] == "ready"


def test_load_progress_error():
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = "r", base_repo = "b", error = "boom")
    p = backend.load_progress()
    assert p["phase"] == "error" and p["error"] == "boom"


def test_load_progress_downloading_then_finalizing(monkeypatch):
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = "r", base_repo = "b", expected_bytes = 1000)

    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 150))
    p = backend.load_progress()
    assert p["phase"] == "downloading"
    assert p["bytes_downloaded"] == 300  # summed across repo + base
    assert abs(p["fraction"] - 0.3) < 1e-9

    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 500))
    assert backend.load_progress()["phase"] == "finalizing"  # 1000/1000


def test_base_file_downloaded_excludes_undownloaded():
    # Counted: the pipeline manifest + component subfolders from_pretrained fetches.
    assert _base_file_downloaded("model_index.json")
    assert _base_file_downloaded("text_encoder/model-00001-of-00003.safetensors")
    assert _base_file_downloaded("vae/diffusion_pytorch_model.safetensors")
    # Excluded: the GGUF supplies the transformer; docs/assets and top-level files
    # are never downloaded, so counting them would peg the bar short of 100%.
    assert not _base_file_downloaded(
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors"
    )
    assert not _base_file_downloaded("assets/Z-Image-Gallery.pdf")
    assert not _base_file_downloaded("README.md")
    assert not _base_file_downloaded(".gitattributes")


def test_load_progress_fraction_clamped(monkeypatch):
    # The cache scan can exceed the estimate (e.g. a second cached quant); the
    # reported fraction must still clamp to 1.0 rather than overshoot.
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = "r", base_repo = "b", expected_bytes = 1000)
    monkeypatch.setattr(DiffusionBackend, "_cache_bytes", staticmethod(lambda repo: 900))
    p = backend.load_progress()  # summed 1800 > expected 1000
    assert p["phase"] == "finalizing"
    assert p["fraction"] == 1.0
    assert p["bytes_downloaded"] == 1000  # clamped to the estimate


def test_estimate_eta():
    from core.inference.diffusion import _estimate_eta

    # No rate yet until a step has elapsed since the first.
    assert _estimate_eta(8, 1, first_step_at = 100.0, now = 100.0) is None
    assert _estimate_eta(8, 0, first_step_at = 0.0, now = 100.0) is None
    # 3 steps in 3s since the first ⇒ 1s/step ⇒ 4 steps left ⇒ ~4s.
    assert _estimate_eta(8, 4, first_step_at = 100.0, now = 103.0) == 4.0
    # Last step ⇒ 0 remaining.
    assert _estimate_eta(8, 8, first_step_at = 100.0, now = 107.0) == 0.0


def test_generate_qwen_uses_true_cfg_scale(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Qwen/Qwen-Image",
        family_override = "qwen-image",
    )
    backend.generate(prompt = "a sloth", guidance = 4.0)
    # Qwen-Image's distilled guidance is off; the real CFG must land on true_cfg_scale.
    call = backend._state.pipe.last_kwargs
    assert call["true_cfg_scale"] == 4.0 and call["guidance_scale"] is None


def _load_ideogram(backend, tmp_path):
    # Ideogram 4 loads only as a full pipeline (its two DiTs are assembled per-component
    # by the stubbed load_ideogram4_pipeline); a local pipeline dir is enough here.
    (tmp_path / "model_index.json").write_text("{}")
    backend.load_pipeline(str(tmp_path), family_override = "ideogram-4")


def test_ideogram_rejects_single_file_and_gguf_kinds(fake_runtime, tmp_path):
    # Ideogram 4 needs two DiTs assembled per-component, so there's no transformer-only
    # single-file or GGUF load: the explicit kinds must be rejected up front (before a load evicts
    # a working model), not assembled into a pipeline missing its second DiT.
    backend = DiffusionBackend()
    (tmp_path / "model.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match = "full diffusers pipeline"):
        backend.load_pipeline(
            str(tmp_path), gguf_filename = "model.gguf", family_override = "ideogram-4"
        )
    (tmp_path / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ValueError, match = "full diffusers pipeline"):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.safetensors",
            model_kind = "single_file",
            family_override = "ideogram-4",
        )


def test_generate_ideogram_defaults_keep_recommended_schedule(fake_runtime, tmp_path):
    # Ideogram 4's pipeline defaults to its recommended tapered guidance_schedule (45x7.0 + 3x3.0,
    # valid only at 48 steps) and REJECTS guidance_scale while the schedule is set. At the family's
    # advertised defaults the backend must drop the constant so the recommended taper engages.
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 48, guidance = 7.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] is None  # not passed: the pipe default engages
    assert "guidance_schedule" not in call


def test_generate_ideogram_custom_guidance_nulls_schedule(fake_runtime, tmp_path):
    # Any non-default request must broadcast the constant legally: guidance_scale set AND
    # guidance_schedule explicitly nulled (the pipeline raises when both are set, and its default
    # schedule is non-None).
    backend = DiffusionBackend()
    _load_ideogram(backend, tmp_path)
    backend.generate(prompt = "a sloth", steps = 20, guidance = 5.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 5.0
    assert "guidance_schedule" in call and call["guidance_schedule"] is None


def test_begin_load_rejects_concurrent(monkeypatch):
    backend = DiffusionBackend()
    # The worker resolves the base + downloads, both over the network; stub them
    # so the test is offline.
    monkeypatch.setattr("core.inference.diffusion._hf_base_model", lambda *a, **k: None)
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda self, *a, **k: None)
    monkeypatch.setattr(
        DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: (0, []))
    )
    # Block the spawned worker so the load stays "in progress".
    monkeypatch.setattr(
        DiffusionBackend, "load_pipeline", lambda self, **k: __import__("time").sleep(0.2)
    )
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    with pytest.raises(RuntimeError):
        backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")


def test_unload_cancels_in_flight_load(fake_runtime):
    # An unload (or arbiter eviction, which calls unload) while a load's worker is still
    # resolving/downloading must cancel it: load_pipeline sees the bumped token and aborts, so the
    # evicted load never resurrects a pipeline into VRAM.
    backend = DiffusionBackend()
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")
    token = 7
    backend._load_token = token
    with pytest.raises(RuntimeError, match = "cancelled"):
        # Simulate the worker reaching load_pipeline after unload bumped the token.
        backend._load_token = token + 1
        backend.load_pipeline(
            "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_K_S.gguf",
            base_repo = fam.base_repo,
            _load_token = token,
        )


def test_superseded_load_does_not_cancel_live_generation(fake_runtime):
    # A superseded background load (token bumped by a newer load/unload) that finally reaches
    # load_pipeline must bail WITHOUT signalling the current model's in-flight generation: the
    # token check must run before the cancel is set, or a stale worker aborts an unrelated,
    # still-live denoise.
    import threading as _threading

    backend = DiffusionBackend()
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")
    live_cancel = _threading.Event()
    backend._active_generate_cancel = live_cancel  # a generation from the CURRENT model
    token = 11
    backend._load_token = token + 1  # this load has already been superseded
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.load_pipeline(
            "unsloth/Z-Image-Turbo-GGUF",
            gguf_filename = "z-image-turbo-Q4_K_S.gguf",
            base_repo = fam.base_repo,
            _load_token = token,
        )
    assert not live_cancel.is_set()  # the live generation was left untouched


def test_pick_dtype_bf16_only_on_ampere(fake_runtime, monkeypatch):
    # BF16 only on Ampere+ (cc >= 8); pre-Ampere cards must fall back to FP16.
    torch = sys.modules["torch"]
    backend = DiffusionBackend()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0), raising = False)
    assert backend._pick_device_and_dtype() == ("cuda", torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5), raising = False)
    assert backend._pick_device_and_dtype() == ("cuda", torch.float16)


def test_unload_sets_cancel_event(fake_runtime):
    # unload signals an in-flight download (which runs without the lock) to abort.
    backend = DiffusionBackend()
    assert not backend._cancel_event.is_set()
    backend.unload()
    assert backend._cancel_event.is_set()


def test_prefetch_aborts_when_cancelled(tmp_path):
    # A prefetch interrupted by unload (cancel event set) raises rather than
    # downloading the whole base, so the load can be preempted mid-download.
    backend = DiffusionBackend()
    backend._cancel_event.set()
    # Local gguf path so the transformer download is skipped; the base loop hits
    # the cancel check on its first file (no network).
    (tmp_path / "model.gguf").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "Cancelled"):
        backend._prefetch_files(
            str(tmp_path),
            "model.gguf",
            "Tongyi-MAI/Z-Image-Turbo",
            ["vae/diffusion_pytorch_model.safetensors"],
            None,
        )


def test_prefetch_downloads_gguf_and_base(monkeypatch, tmp_path):
    backend = DiffusionBackend()
    calls: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: (calls.append((repo, fn)), f"/cache/{fn}")[1],
    )
    # Hub repo: the GGUF transformer and each base file are fetched.
    backend._prefetch_files(
        "unsloth/Z-Image-Turbo-GGUF",
        "model.gguf",
        "base/repo",
        ["vae/x.safetensors", "text_encoder/y.safetensors"],
        "hf_tok",
    )
    assert ("unsloth/Z-Image-Turbo-GGUF", "model.gguf") in calls
    assert ("base/repo", "vae/x.safetensors") in calls
    assert ("base/repo", "text_encoder/y.safetensors") in calls
    # Local GGUF path: the transformer download is skipped, base still fetched.
    calls.clear()
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend._prefetch_files(str(tmp_path), "model.gguf", "base/repo", ["vae/x.safetensors"], None)
    assert all(repo != str(tmp_path) for repo, _ in calls)
    assert ("base/repo", "vae/x.safetensors") in calls


# fp16-incompatible guard + dtype promotion


def test_zimage_is_fp16_incompatible():
    # Only Z-Image-class families carry the guard (their activations overflow fp16).
    assert detect_family("unsloth/Z-Image-Turbo-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Z-Image-GGUF").fp16_incompatible is True
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").fp16_incompatible is False
    assert detect_family("unsloth/FLUX.1-schnell-GGUF").fp16_incompatible is False
    assert detect_family("unsloth/FLUX.2-klein-4B-GGUF").fp16_incompatible is False


def test_resolve_compute_dtype_promotes_fp16_for_zimage(fake_runtime):
    torch = sys.modules["torch"]
    z = detect_family("unsloth/Z-Image-GGUF")
    q = detect_family("unsloth/Qwen-Image-GGUF")
    # Z-Image: fp16 -> fp32; bf16 / fp32 pass through unchanged.
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float32
    assert _resolve_diffusion_compute_dtype(z, torch.bfloat16) is torch.bfloat16
    assert _resolve_diffusion_compute_dtype(z, torch.float32) is torch.float32
    # An fp16-compatible family (and None) keep fp16.
    assert _resolve_diffusion_compute_dtype(q, torch.float16) is torch.float16
    assert _resolve_diffusion_compute_dtype(None, torch.float16) is torch.float16


def test_load_promotes_fp16_to_fp32_for_zimage_only(fake_runtime, monkeypatch, tmp_path):
    torch = sys.modules["torch"]
    # Pre-Ampere CUDA -> the resolver picks fp16; the guard must promote Z-Image
    # (and only Z-Image) to fp32 so it doesn't render a black image.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5), raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")

    z = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image"
    )
    assert z["device"] == "cuda" and z["dtype"] == "float32"
    # The promoted dtype reaches the transformer build (and thus the quant config).
    assert str(_FakeTransformer.last["torch_dtype"]) == "torch.float32"

    q = DiffusionBackend().load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "qwen-image"
    )
    assert q["dtype"] == "float16"  # fp16-compatible family keeps fp16 on pre-Ampere


def test_bad_mode_strings_fail_before_eviction(fake_runtime):
    # Every mode normalizer that can raise runs BEFORE the load evicts the previous
    # pipeline, so a bad request never costs the user their working model.
    backend = DiffusionBackend()
    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = object(),
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )
    for kwargs in (
        {"transformer_quant": "int7"},
        {"speed_mode": "warp"},
        {"attention_backend": "bogus"},
        {"transformer_cache": "bogus"},
        {"text_encoder_quant": "fp3"},
    ):
        with pytest.raises(ValueError):
            backend.load_pipeline("unsloth/Z-Image-GGUF", gguf_filename = "m.gguf", **kwargs)
        assert backend._state is not None


# Lock split + mid-denoise cancellation


def test_generate_lock_split_keeps_status_and_unload_responsive(fake_runtime):
    import threading

    backend = DiffusionBackend()
    started = threading.Event()
    release = threading.Event()

    class _BlockingPipe:
        def __call__(self, **kwargs):
            started.set()
            release.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = _BlockingPipe(),
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )

    out: dict = {}

    def _run():
        try:
            out["res"] = backend.generate(prompt = "p", steps = 4)
        except Exception as exc:  # noqa: BLE001
            out["exc"] = exc

    t = threading.Thread(target = _run)
    t.start()
    assert started.wait(5)  # the denoise is in flight, holding only _generate_lock

    # status() / generate_progress() must NOT block behind the denoise.
    assert backend.status()["loaded"] is True
    assert backend.generate_progress()["active"] is True

    cancel_ref = backend._active_generate_cancel
    assert cancel_ref is not None

    # unload() signals THIS generation's cancel event, then waits for the denoise to exit before
    # returning: callers treat its return as "VRAM is free" (the GPU arbiter hands the GPU to chat
    # on it). Release the pipe once the cancel lands, standing in for a real pipeline's step
    # callback.
    releaser = threading.Thread(target = lambda: (cancel_ref.wait(5), release.set()))
    releaser.start()
    backend.unload()
    releaser.join(5)
    assert cancel_ref.is_set()
    assert backend.status()["loaded"] is False

    t.join(5)
    # The cancelled generation raised rather than returning a now-evicted image, and
    # it had already exited (deregistering its cancel) before unload() returned.
    assert "exc" in out and "cancelled" in str(out["exc"]).lower()
    assert backend._active_generate_cancel is None


def test_callback_cancellation_interrupts_denoise(fake_runtime):
    import threading

    backend = DiffusionBackend()
    at_step0 = threading.Event()
    resume = threading.Event()

    class _SteppingPipe:
        def __init__(self) -> None:
            self._interrupt = False
            self.steps_run = 0

        def __call__(
            self,
            *,
            callback_on_step_end = None,
            num_inference_steps = 8,
            **kwargs,
        ):
            for i in range(num_inference_steps):
                if self._interrupt:  # diffusers' interrupt protocol
                    break
                if callback_on_step_end is not None:
                    callback_on_step_end(self, i, 0.0, {})
                self.steps_run = i + 1
                if i == 0:
                    at_step0.set()
                    resume.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    pipe = _SteppingPipe()
    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = pipe,
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )

    out: dict = {}

    def _run():
        try:
            out["res"] = backend.generate(prompt = "p", steps = 8)
        except Exception as exc:  # noqa: BLE001
            out["exc"] = exc

    t = threading.Thread(target = _run)
    t.start()
    assert at_step0.wait(5)  # step 0's callback ran with no cancel pending
    # Simulate an eviction / superseding load signalling THIS generation's cancel.
    assert backend._active_generate_cancel is not None
    backend._active_generate_cancel.set()
    resume.set()
    t.join(5)
    # The next step's callback saw the cancel, flipped pipe._interrupt, and the loop
    # broke early, so the generation raised instead of returning a partial image.
    assert pipe._interrupt is True
    assert pipe.steps_run < 8
    assert "exc" in out and "cancelled" in str(out["exc"]).lower()


def test_validate_load_request(tmp_path):
    backend = DiffusionBackend()
    # No filename + unsloth repo -> a full-pipeline load (allowed for unsloth/*).
    assert backend.validate_load_request("unsloth/Z-Image-Turbo-unsloth-bnb-4bit").name == "z-image"
    # No filename + non-unsloth repo -> a pipeline load, gated to unsloth/* -> rejected.
    with pytest.raises(ValueError, match = "unsloth"):
        backend.validate_load_request("some-org/Z-Image-bnb-4bit")
    # An explicit gguf/single_file kind still requires a single-file name.
    with pytest.raises(ValueError, match = "single-file"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", model_kind = "gguf")
    # A pipeline kind must NOT carry a single-file name.
    with pytest.raises(ValueError, match = "pipeline"):
        backend.validate_load_request(
            "unsloth/Z-Image-Turbo-bnb-4bit", gguf_filename = "q.gguf", model_kind = "pipeline"
        )
    # A single-file safetensors load is also gated to unsloth/* repos.
    with pytest.raises(ValueError, match = "unsloth"):
        backend.validate_load_request("some-org/Z-Image", gguf_filename = "model.safetensors")
    with pytest.raises(ValueError, match = "family"):
        backend.validate_load_request("meta/Llama-3", gguf_filename = "q.gguf")
    # A family-looking repo paired with a non-GGUF single-file name is rejected here, BEFORE the
    # route evicts chat and hands over the GPU (else the background load would be the first to
    # notice README.md is not a checkpoint).
    with pytest.raises(ValueError, match = r"\.gguf"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "README.md")
    assert (
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf").name
        == "z-image"
    )
    # A kind/extension mismatch fails fast here, before the route evicts chat + grabs the
    # GPU only to fail in the background from_single_file path.
    with pytest.raises(ValueError, match = ".gguf"):
        backend.validate_load_request(
            "unsloth/Z-Image-Turbo-GGUF", gguf_filename = "model.safetensors", model_kind = "gguf"
        )
    with pytest.raises(ValueError, match = "gguf"):
        backend.validate_load_request(
            "unsloth/Qwen-Image-2512-FP8", gguf_filename = "q.gguf", model_kind = "single_file"
        )
    # A remote "*-GGUF" repo loaded as a full pipeline (no single-file name) is a single-file GGUF
    # repo, so from_pretrained finds no pipeline manifest and fails after chat is evicted; reject
    # it here before the GPU handoff.
    with pytest.raises(ValueError, match = "GGUF"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", model_kind = "pipeline")
    # A local path with a missing child fails here (before any GPU/network work).
    with pytest.raises(FileNotFoundError):
        backend.validate_load_request(
            str(tmp_path), gguf_filename = "missing.gguf", family_override = "z-image"
        )
    (tmp_path / "m.gguf").write_bytes(b"x")
    assert (
        backend.validate_load_request(
            str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image"
        ).name
        == "z-image"
    )
    # A path-shaped repo_id that does not exist is rejected here (it would otherwise
    # be treated as remote, evict chat, and only fail in the background load).
    with pytest.raises(FileNotFoundError):
        backend.validate_load_request(
            "/tmp/unsloth-definitely-missing-model",
            gguf_filename = "m.gguf",
            family_override = "z-image",
        )


def test_replacement_load_waits_for_inflight_generation(fake_runtime, tmp_path):
    # A superseding load must signal the in-flight generation's cancel AND wait for it to release
    # _generate_lock before allocating, so two pipelines never sit in VRAM at once (unlike
    # unload(), which returns promptly without waiting).
    import threading

    backend = DiffusionBackend()
    started = threading.Event()
    release = threading.Event()

    class _BlockingPipe:
        def __call__(self, **kwargs):
            started.set()
            release.wait(5)
            return types.SimpleNamespace(images = [_FakeImage()])

    fam = detect_family("unsloth/Z-Image-GGUF")
    backend._state = _LoadState(
        pipe = _BlockingPipe(),
        family = fam,
        repo_id = "r",
        base_repo = "b",
        device = "cpu",
        dtype = "float32",
        cpu_offload = False,
    )

    gen_out: dict = {}

    def _gen():
        try:
            backend.generate(prompt = "p", steps = 4)
        except Exception as exc:  # noqa: BLE001
            gen_out["exc"] = exc

    gt = threading.Thread(target = _gen)
    gt.start()
    assert started.wait(5)  # generation in flight, holding _generate_lock

    (tmp_path / "m.gguf").write_bytes(b"x")
    load_done = threading.Event()

    def _load():
        backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
        load_done.set()

    lt = threading.Thread(target = _load)
    lt.start()

    # The load must NOT finish while the generation still holds _generate_lock; it
    # has signalled the generation's cancel and is waiting to allocate.
    assert not load_done.wait(0.5)
    assert backend._active_generate_cancel is not None
    assert backend._active_generate_cancel.is_set()

    release.set()  # the blocked denoise returns; generate() sees cancel and raises
    gt.join(5)
    assert load_done.wait(5)  # only now does the replacement allocate
    assert "exc" in gen_out and "cancelled" in str(gen_out["exc"]).lower()
    assert backend.status()["loaded"] is True
    assert backend.status()["repo_id"] == str(tmp_path)


# ── Phase 2A: memory policy wiring (load -> planner -> placement) ──────────────


def test_load_reports_memory_plan_fields_on_cpu(fake_runtime, tmp_path):
    # The default stub resolves to a CPU target: no offload is possible, but VAE
    # tiling is on (no separate device pool), and status carries the new fields.
    (tmp_path / "m.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert status["offload_policy"] == "none"
    assert status["cpu_offload"] is False
    assert status["vae_tiling"] is True
    assert status["memory_mode"] == "auto"
    pipe = backend._state.pipe
    assert pipe.moved_to == "cpu" and pipe.vae_tiled and pipe.vae_sliced


def _force_cuda_target(backend, monkeypatch):
    """Drive the loader down the CUDA (offload-capable) path under the stub."""
    torch = sys.modules["torch"]
    monkeypatch.setattr(backend, "_pick_device_and_dtype", lambda: ("cuda", torch.bfloat16))


def test_load_memory_mode_balanced_streams_or_falls_back(fake_runtime, tmp_path, monkeypatch):
    # balanced requests streamed block-level (group) offload. Under the stub there's no real
    # diffusers.hooks, so group can't engage and the applier falls back to whole-module offload,
    # reporting the policy actually engaged (the real "group" path is GPU-verified in the bench).
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "balanced"
    )
    assert status["offload_policy"] in ("group", "model") and status["cpu_offload"] is True
    assert status["memory_mode"] == "balanced"
    assert backend._state.pipe.offloaded is True  # model-offload fallback engaged


def test_load_memory_mode_low_vram_engages_model_offload(fake_runtime, tmp_path, monkeypatch):
    # low_vram offloads every component (lowest VRAM); whole-module offload is the
    # robust path and engages directly (no streaming, so no diffusers.hooks needed).
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "low_vram"
    )
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True
    pipe = backend._state.pipe
    assert pipe.offloaded is True and pipe.moved_to is None  # offload owns placement


def test_load_explicit_cpu_offload_engages_model_offload_on_cuda(
    fake_runtime, tmp_path, monkeypatch
):
    # cpu_offload=True with no mode: auto would stay resident (budget unknown under
    # the stub), but the explicit flag forces whole-module offload.
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", cpu_offload = True
    )
    assert status["offload_policy"] == "model" and status["cpu_offload"] is True


def test_load_speed_mode_gguf_auto_defaults_and_explicit(fake_runtime, tmp_path):
    # No speed_mode on a GGUF model -> auto `default` (near-lossless, compile sits below the quant
    # noise floor). compile only engages on CUDA, so on this CPU stub no optim engages, but the
    # resolved mode is `default`.
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert status["speed_mode"] == "default"
    # An explicit "off" opts back into the bit-identical path (engages nothing).
    status_off = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", speed_mode = "off"
    )
    assert status_off["speed_mode"] == "off" and status_off["speed_optims"] == []
    # An explicit speed_mode threads through to status (engaged optims are GPU-verified).
    status2 = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", speed_mode = "max"
    )
    assert status2["speed_mode"] == "max"
    # Text-encoder quant defaults off (None); a requested mode threads through (the
    # actual engagement is GPU-verified, since it needs real torch/torchao).
    assert status2["text_encoder_quant"] is None
    status3 = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        text_encoder_quant = "nvfp4",
    )
    # Under the CPU stub nvfp4 is unsupported, so it engages nothing -> None.
    assert status3["text_encoder_quant"] is None


def test_load_fast_mode_stays_resident_on_cuda(fake_runtime, tmp_path, monkeypatch):
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    status = backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", memory_mode = "fast"
    )
    assert status["offload_policy"] == "none" and status["cpu_offload"] is False
    assert backend._state.pipe.moved_to == "cuda"


# ── transformer quant (opt-in dense fast path) ────────────────────────────────


def _stub_dense_quant(monkeypatch, *, scheme = "fp8"):
    """Force the dense+quant branch hermetically: a supported dense source, a
    from_pretrained on the fake transformer, and a quantizer that engages `scheme`.
    Returns a dict recording the dense-loader / quantizer calls."""
    from core.inference import diffusion as dmod

    calls: dict = {"from_pretrained": 0, "quantize": 0, "quant_mode": None}

    @classmethod
    def _from_pretrained(cls, base, **kwargs):
        calls["from_pretrained"] += 1
        calls["fp_kwargs"] = {"base": base, **kwargs}
        return object()

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _from_pretrained, raising = False)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    # Resolve the scheme without the real GPU smoke probe, and configure no pre-quant
    # checkpoint so the dense materialise+quantise branch is the one exercised.
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: scheme
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)

    def _quantize(pipe, target, *, mode, **kw):
        calls["quantize"] += 1
        calls["quant_mode"] = mode
        return scheme

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)
    return calls


def test_default_load_autos_dense_gate_and_falls_back(fake_runtime, tmp_path, monkeypatch):
    # UNSET Dtype defaults to the hardware ladder: the dense gate IS consulted, and a
    # device without dense support (this fake runtime) falls back to the GGUF build.
    from core.inference import diffusion as dmod

    consulted = {"n": 0}

    def _supported(*a, **k):
        consulted["n"] += 1
        return False

    monkeypatch.setattr(dmod, "dense_transformer_supported", _supported)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert consulted["n"] >= 1
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]  # GGUF from_single_file was used


def test_explicit_off_load_skips_dense_quant_path(fake_runtime, tmp_path, monkeypatch):
    # An EXPLICIT "none" pins running the GGUF as-is: the dense gate is never even
    # consulted (short-circuit), so the pinned-off contract cannot regress.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod,
        "dense_transformer_supported",
        lambda *a, **k: pytest.fail("dense path must not run with an explicit off"),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "none",
    )
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]  # GGUF from_single_file was used


def test_speed_off_load_suppresses_auto_dtype_quant(fake_runtime, tmp_path, monkeypatch):
    # An explicit Speed="off" (bit-exact) load with an UNSET dtype must stay GGUF-as-is: the auto
    # dtype default must NOT promote it to a quantized + compiled build (silently breaking the
    # bit-exact request). The dense gate must never be consulted.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod,
        "dense_transformer_supported",
        lambda *a, **k: pytest.fail("dense path must not run under an explicit Speed=off"),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        speed_mode = "off",
    )
    assert status["transformer_quant"] is None
    assert status["speed_mode"] == "off"
    assert _FakeTransformer.last["path"]  # GGUF from_single_file was used, not a dense build


def test_speed_off_load_suppresses_auto_companion_quant(fake_runtime, tmp_path, monkeypatch):
    # Mirror the DiT suppression for the companions: an explicit Speed="off" load with TE/VAE left at
    # auto must keep them dense (mode "off"), not promote them to auto-quant and silently fp8/int8 the
    # text encoder + VAE, which would break the bit-exact request. Unset speed still auto-quantises.
    from core.inference import diffusion as dmod

    te_modes: list = []
    vae_modes: list = []
    monkeypatch.setattr(
        dmod, "quantize_text_encoders", lambda pipe, target, *, mode, **kw: te_modes.append(mode)
    )
    monkeypatch.setattr(
        dmod, "quantize_vae", lambda pipe, target, *, mode, **kw: vae_modes.append(mode)
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image", speed_mode = "off"
    )
    assert te_modes == ["off"] and vae_modes == ["off"]  # dense, not auto
    backend.unload()
    backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert te_modes[-1] == "auto" and vae_modes[-1] == "auto"  # promoted when speed is not off


def test_speed_off_load_suppresses_explicit_auto_companion_quant(
    fake_runtime, tmp_path, monkeypatch
):
    # auto is backend-owned, exactly like transformer_quant: an EXPLICIT text_encoder_quant/
    # vae_quant="auto" must also go dense under Speed="off", not just an unset default. Otherwise a
    # caller that sends auto + off would silently fp8/int8 the companions and break the bit-exact
    # request. An explicit CONCRETE scheme still forces quant even under off.
    from core.inference import diffusion as dmod

    te_modes: list = []
    vae_modes: list = []
    monkeypatch.setattr(
        dmod, "quantize_text_encoders", lambda pipe, target, *, mode, **kw: te_modes.append(mode)
    )
    monkeypatch.setattr(
        dmod, "quantize_vae", lambda pipe, target, *, mode, **kw: vae_modes.append(mode)
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        speed_mode = "off",
        text_encoder_quant = "auto",
        vae_quant = "auto",
    )
    assert te_modes == ["off"] and vae_modes == ["off"]  # explicit auto suppressed under off
    backend.unload()
    # An explicit concrete scheme is still honoured under off (only auto is backend-owned).
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        speed_mode = "off",
        text_encoder_quant = "fp8",
    )
    assert te_modes[-1] == "fp8"


def test_transformer_quant_dense_path_engaged(fake_runtime, tmp_path, monkeypatch):
    # transformer_quant + a CUDA resident plan -> load the DENSE transformer from the
    # base repo, place it on the device, quantise it, and report the engaged scheme.
    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] == "fp8"
    # No speed_mode was given, but a quantized transformer is ~30x slower eager, so the
    # backend promotes it to `default` (regional compile) instead of the dense `off`.
    assert status["speed_mode"] == "default"
    assert calls["from_pretrained"] == 1 and calls["quantize"] == 1
    assert calls["quant_mode"] == "fp8"
    assert calls["fp_kwargs"]["subfolder"] == "transformer"  # dense transformer subfolder
    # The GGUF single-file path was NOT used for the transformer.
    assert _FakeTransformer.last == {}
    # quantize ran on-device: the dense pipe was placed on cuda (before compile).
    assert backend._state.pipe.moved_to == "cuda"
    assert status["offload_policy"] == "none"


def test_transformer_quant_prequant_path_engaged(fake_runtime, tmp_path, monkeypatch):
    # A configured pre-quant checkpoint -> load the already-quantized transformer directly;
    # the dense from_pretrained and the on-device quantize_transformer are NOT used.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: object())
    prequant_obj = object()
    loaded: dict = {"n": 0}

    def _load_prequant(transformer_cls, base, source, **kw):
        loaded["n"] += 1
        loaded["scheme"] = kw.get("scheme")
        return prequant_obj

    monkeypatch.setattr(dmod, "load_prequantized_transformer", _load_prequant)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense from_pretrained must not run when a prequant checkpoint loads")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    monkeypatch.setattr(
        dmod,
        "quantize_transformer",
        lambda *a, **k: pytest.fail("quantize_transformer must not run on the prequant path"),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
        transformer_prequant_path = str(tmp_path / "zimage_fp8.pt"),
    )
    assert status["transformer_quant"] == "fp8"
    assert loaded["n"] == 1 and loaded["scheme"] == "fp8"
    # The pre-quantized transformer object was assembled into the pipeline...
    assert _FakePipeline.last.get("transformer") is prequant_obj
    # ...and the GGUF single-file path was not used.
    assert _FakeTransformer.last == {}


def test_transformer_quant_prequant_load_fails_falls_back_to_dense(
    fake_runtime, tmp_path, monkeypatch
):
    # A configured prequant source whose load returns None must fall back to the dense
    # materialise+quantise path (not straight to GGUF), preserving the fast mode.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    calls = _stub_dense_quant(monkeypatch, scheme = "fp8")
    # Override the no-prequant default: a source resolves, but its load fails.
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: object())
    monkeypatch.setattr(dmod, "load_prequantized_transformer", lambda *a, **k: None)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] == "fp8"
    assert calls["from_pretrained"] == 1 and calls["quantize"] == 1  # dense path ran
    assert _FakeTransformer.last == {}  # GGUF not used


def test_transformer_quant_falls_back_to_gguf_on_failure(fake_runtime, tmp_path, monkeypatch):
    # A dense/quant failure (here: quantize returns None -> unsupported) must fall back
    # to the GGUF build, not error -- status reports no transformer_quant engaged.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)

    @classmethod
    def _from_pretrained(cls, base, **kwargs):
        return object()

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _from_pretrained, raising = False)
    monkeypatch.setattr(dmod, "quantize_transformer", lambda pipe, target, **kw: None)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["loaded"] is True
    assert status["transformer_quant"] is None  # fell back
    assert _FakeTransformer.last["path"]  # GGUF from_single_file used


def test_transformer_quant_skipped_when_plan_offloads(fake_runtime, tmp_path, monkeypatch):
    # The dense bf16 transformer only fits resident, so when the memory plan would offload (here
    # low_vram) the fast path is skipped and GGUF loads instead -- the dense transformer is never
    # loaded.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense transformer must not load when the plan offloads")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
        memory_mode = "low_vram",
    )
    assert status["transformer_quant"] is None
    assert status["offload_policy"] == "model"
    assert _FakeTransformer.last["path"]  # GGUF path used


def test_dense_quant_skipped_when_dense_transformer_does_not_fit(
    fake_runtime, tmp_path, monkeypatch
):
    # The GGUF fits resident (plan `none`), but the DENSE bf16 transformer the fast path
    # materializes does not. The fast path must be skipped up front (preflighted against the dense
    # transformer, not the GGUF), and GGUF loads RESIDENT -- not evicted, OOMed, then offloaded.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    # A scheme resolves and there is no prequant, so the dense bf16 is materialized and the
    # dense-fit re-check runs against a large (won't-fit) dense transformer.
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_transformer_resident_bytes",
        staticmethod(lambda base: 40 * 1024**3),
    )
    orig_plan = DiffusionBackend._plan_memory

    def plan_wrap(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        # GGUF budget fits (real plan -> none); the dense-transformer preflight does not.
        if transformer_resident_override_mib is not None:
            return types.SimpleNamespace(offload_policy = "model")
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", plan_wrap)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense transformer must not load when it won't fit resident")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["transformer_quant"] is None  # dense quant skipped
    assert status["offload_policy"] == "none"  # GGUF loaded resident, not offloaded
    assert _FakeTransformer.last["path"]  # GGUF path used


def test_dense_quant_prequant_proceeds_but_forbids_dense_fallback(fake_runtime, tmp_path, monkeypatch):
    # With a prequant checkpoint, the fast path loads the small quantized file, so a dense
    # misfit must NOT decline the fast path -- but the dense re-check still runs to gate the
    # in-loader fallback: if the prequant later fails, the loader must raise to GGUF instead of
    # materialising the dense bf16 the plan never budgeted (allow_dense_fallback=False).
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    # usable_ (not resolve_): the re-check site only honours a source the loader would
    # actually accept, so the fake must present a USABLE one (e.g. a hosted repo).
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: "prequant/path")
    # Large dense shards cached: if the re-check ran, it would wrongly decline the fast path.
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_transformer_resident_bytes",
        staticmethod(lambda base: 999 * 1024**3),
    )
    dense_refit_ran = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        if transformer_resident_override_mib is not None:
            dense_refit_ran.append(True)
            # GGUF budget fits (real plan -> none); the dense-transformer preflight does not.
            return types.SimpleNamespace(offload_policy = "model")
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        return None, None  # fall through to GGUF; we only assert the path was reached

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert dense_refit_ran == [True]  # the re-check runs (it gates the fallback)...
    assert attempted == [False]  # ...fast path still attempted, dense fallback forbidden


def test_dense_quant_replan_retries_once_on_transient_free_undercount(
    fake_runtime, tmp_path, monkeypatch
):
    # A transient foreign allocation at snapshot time makes an empty card look full and the
    # candidate replan declines resident -- but the candidate FITS total capacity, so the
    # loader must retry the replan once with a fresh settled snapshot instead of silently
    # falling back to GGUF-as-is (measured: FLUX.2-dev int8 cold load on an idle B200).
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 33_831, companions_mib = 46_157, prequant = True
        ),
    )
    replan_calls = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(self, *a, transformer_resident_override_mib = None, **k):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            # Initial GGUF plan: force offload so the candidate replan branch is entered.
            return dataclasses.replace(real, offload_policy = "model")
        replan_calls.append(True)
        if len(replan_calls) == 1:
            # First replan: the transient undercount. Required fits total capacity
            # (90,228 <= 0.85 * (183,359 - 18,335)), so a retry must follow.
            return types.SimpleNamespace(
                offload_policy = "model",
                estimates = {"resident_required_mib": 90_228, "safe_device_budget_mib": 40_000},
                device_memory = types.SimpleNamespace(
                    total_mib = 183_359, memory_kind = "discrete_vram", free_mib = 60_000
                ),
                reasons = ("companions exceed budget",),
            )
        # Retry: the transient cleared; resident.
        return dataclasses.replace(real, offload_policy = "none")

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    attempted = []

    def fake_dense_load(self, *a, **k):
        attempted.append(k.get("allow_dense_fallback"))
        raise RuntimeError("test: stop after reaching the fast path")

    monkeypatch.setattr(DiffusionBackend, "_load_dense_quant_pipeline", fake_dense_load)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "int8",
    )
    assert replan_calls == [True, True]  # declined once, retried once
    assert attempted == [False]  # fast path attempted; prequant-sized plan forbids dense fallback


def test_dense_quant_replan_no_retry_when_capacity_truly_short(
    fake_runtime, tmp_path, monkeypatch
):
    # When the candidate does NOT fit total capacity, the decline is real: no retry.
    import dataclasses

    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
    )
    monkeypatch.setattr(
        dmod,
        "resolve_dense_quant_candidate",
        lambda **kw: types.SimpleNamespace(
            transient_transformer_mib = 33_831, companions_mib = 46_157, prequant = True
        ),
    )
    replan_calls = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(self, *a, transformer_resident_override_mib = None, **k):
        real = orig_plan(
            self, *a, transformer_resident_override_mib = transformer_resident_override_mib, **k
        )
        if transformer_resident_override_mib is None:
            return dataclasses.replace(real, offload_policy = "model")
        replan_calls.append(True)
        return types.SimpleNamespace(
            offload_policy = "model",
            estimates = {"resident_required_mib": 150_000, "safe_device_budget_mib": 40_000},
            device_memory = types.SimpleNamespace(
                total_mib = 183_359, memory_kind = "discrete_vram", free_mib = 60_000
            ),
            reasons = ("companions exceed budget",),
        )

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "int8",
    )
    assert replan_calls == [True]  # genuine capacity shortfall: declined without a retry


class _BakePipe:
    def __init__(self):
        self.calls: list = []

    def load_lora_weights(self, path, adapter_name = None):
        self.calls.append(("load", path, adapter_name))

    def set_adapters(self, names, adapter_weights = None):
        self.calls.append(("set", tuple(names), tuple(adapter_weights)))


def test_dense_quant_lora_bake_attaches_before_quantize(fake_runtime, monkeypatch):
    # A LoRA bake must (a) skip the prequant shortcut (adapters need the DENSE transformer),
    # (b) attach the adapters BEFORE quantize_transformer (peft's post-quant torchao dispatch
    # TypeErrors on a manually quantized module), and (c) mark the pipe as baked.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
    )
    prequant_consulted = []
    monkeypatch.setattr(
        dmod,
        "resolve_prequant_source",
        lambda *a, **k: prequant_consulted.append(True) or None,
    )
    order: list = []

    class FakeTransformerCls:
        @staticmethod
        def from_pretrained(*a, **k):
            order.append("dense_load")
            return object()

    pipe = _BakePipe()
    monkeypatch.setattr(
        DiffusionBackend, "_assemble_pipe", staticmethod(lambda *a, **k: pipe)
    )
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_lora_set",
        staticmethod(lambda specs, **k: (("sloth", "/adapters/sloth.safetensors", 0.8),)),
    )

    def fake_quantize(p, target, **k):
        order.append("quantize")
        assert any(c[0] == "load" for c in p.calls), "adapters must attach before quantize"
        return "int8"

    monkeypatch.setattr(dmod, "quantize_transformer", fake_quantize)
    got_pipe, scheme = backend._load_dense_quant_pipeline(
        FakeTransformerCls,
        object,
        "base/repo",
        "cuda",
        "bf16",
        None,
        types.SimpleNamespace(device = "cuda", dtype = "bf16"),
        "int8",
        fam = types.SimpleNamespace(name = "z-image"),
        lora_specs = [("sloth", 0.8)],
    )
    assert scheme == "int8"
    assert prequant_consulted == []  # prequant shortcut skipped for the bake
    assert order == ["dense_load", "quantize"]
    assert pipe.calls[0] == ("load", "/adapters/sloth.safetensors", "sloth")
    assert pipe.calls[1] == ("set", ("sloth",), (0.8,))
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 0.8),)
    assert pipe._unsloth_loras_baked is True


def _quant_lora_state(pipe, quant = "int8"):
    return types.SimpleNamespace(
        pipe = pipe,
        transformer_quant = quant,
        kind = "gguf",
        family = types.SimpleNamespace(name = "z-image"),
        hf_token = None,
        speed_optims = ("compiled",),
    )


def test_apply_loras_quant_unbaked_requires_reload(monkeypatch):
    # A quantized pipe built WITHOUT adapters cannot take one at generation time (topology is
    # frozen after quantize_ + compile): clean 400 telling the client to reload.
    backend = DiffusionBackend()
    pipe = _BakePipe()
    with pytest.raises(ValueError, match = "Reload the model with the adapter selection"):
        backend._apply_loras(
            _quant_lora_state(pipe), [("sloth", 1.0)], threading.Event()
        )
    # ...but a no-adapter generation on the same pipe stays a plain no-op.
    backend._apply_loras(_quant_lora_state(pipe), [], threading.Event())
    assert pipe.calls == []


def test_apply_loras_quant_baked_matrix(monkeypatch):
    # Baked pipe: same set -> no-op; weight-only change -> set_adapters (value-level, no
    # topology change); empty -> all scales 0 (reproduces the quantized base); different
    # adapter set -> reload error.
    backend = DiffusionBackend()
    monkeypatch.setattr(
        DiffusionBackend,
        "_resolve_lora_set",
        staticmethod(
            lambda specs, **k: tuple(
                (i, f"/adapters/{i}.safetensors", w) for (i, w) in specs
            )
        ),
    )

    def baked_pipe():
        pipe = _BakePipe()
        pipe._unsloth_loras = (("sloth", "/adapters/sloth.safetensors", 0.8),)
        pipe._unsloth_loras_baked = True
        return pipe

    ev = threading.Event()
    # same set: no-op
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [("sloth", 0.8)], ev)
    assert pipe.calls == []
    # weight-only change: live set_adapters + marker update
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [("sloth", 1.4)], ev)
    assert pipe.calls == [("set", ("sloth",), (1.4,))]
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 1.4),)
    # empty: scale everything to 0 (quantized base output), marker keeps paths
    pipe = baked_pipe()
    backend._apply_loras(_quant_lora_state(pipe), [], ev)
    assert pipe.calls == [("set", ("sloth",), (0.0,))]
    assert pipe._unsloth_loras == (("sloth", "/adapters/sloth.safetensors", 0.0),)
    # empty again after zeroing: no further calls
    backend._apply_loras(_quant_lora_state(pipe), [], ev)
    assert len(pipe.calls) == 1
    # different adapter set: topology change -> reload error
    pipe = baked_pipe()
    with pytest.raises(ValueError, match = "Reload the model with the new adapter selection"):
        backend._apply_loras(_quant_lora_state(pipe), [("other", 1.0)], ev)


def test_assemble_pipe_routes_krea2_per_component(monkeypatch):
    # krea's repo ships transformers-5.x configs and no top-level tokenizer files, so
    # Pipeline.from_pretrained dies in the tokenizer (vocab_file = None). The quant fast
    # path must assemble per-component via load_krea2_pipeline like every other krea load.
    from core.inference import diffusion as dmod

    calls: dict = {}

    class Pipe:
        def to(self, device):
            calls["device"] = device
            return self

    def fake_loader(base, dtype, hf_token = None, transformer = None):
        calls["base"] = base
        calls["transformer"] = transformer
        return Pipe()

    monkeypatch.setattr(dmod, "load_krea2_pipeline", fake_loader)

    class ExplodingPipeline:
        @staticmethod
        def from_pretrained(*a, **k):
            raise AssertionError("krea-2 must not go through Pipeline.from_pretrained")

    marker = object()
    pipe = dmod.DiffusionBackend._assemble_pipe(
        ExplodingPipeline,
        "krea/Krea-2-Turbo",
        marker,
        "bf16",
        None,
        "cuda:0",
        fam = types.SimpleNamespace(name = "krea-2"),
    )
    assert isinstance(pipe, Pipe)
    assert calls == {"base": "krea/Krea-2-Turbo", "transformer": marker, "device": "cuda:0"}


def test_dense_quant_unusable_prequant_path_runs_dense_refit(fake_runtime, tmp_path, monkeypatch):
    # A request-supplied transformer_prequant_path the loader refuses (missing, or outside
    # UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH) resolves to NO usable prequant source, so the
    # dense-transformer fit re-check MUST run: with real device budgets it declines the fast path
    # up front instead of evicting the resident pipeline and OOMing in the dense bf16 fallback.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: "fp8"
    )
    # The REAL usable_prequant_source refuses a non-allowlisted path (unit-tested in
    # test_diffusion_prequant.py); returning None here pins that outcome at this site.
    monkeypatch.setattr(dmod, "usable_prequant_source", lambda fam, scheme, **kw: None)
    monkeypatch.setattr(
        DiffusionBackend,
        "_dense_transformer_resident_bytes",
        staticmethod(lambda base: 999 * 1024**3),
    )
    dense_refit_ran = []
    orig_plan = DiffusionBackend._plan_memory

    def spy_plan(
        self,
        *a,
        transformer_resident_override_mib = None,
        **k,
    ):
        if transformer_resident_override_mib is not None:
            dense_refit_ran.append(True)
        return orig_plan(self, *a, **k)

    monkeypatch.setattr(DiffusionBackend, "_plan_memory", spy_plan)
    monkeypatch.setattr(
        DiffusionBackend, "_load_dense_quant_pipeline", lambda self, *a, **k: (None, None)
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
        transformer_prequant_path = str(tmp_path / "not-allowlisted.pt"),
    )
    # Unusable path -> no prequant shortcut -> the dense fit re-check ran.
    assert dense_refit_ran == [True]
    assert backend.status()["loaded"] is True


def test_transformer_quant_unsupported_scheme_skips_dense_download(
    fake_runtime, tmp_path, monkeypatch
):
    # An explicit unsupported scheme (select_transformer_quant_scheme -> None) must fail the dense
    # path BEFORE materialising the multi-GB dense transformer, then fall back to GGUF -- else the
    # download runs under the load lock during finalization after the old model was evicted, only
    # to fail at quantize.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    monkeypatch.setattr(dmod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        dmod, "select_transformer_quant_scheme", lambda target, mode, family = None: None
    )
    monkeypatch.setattr(dmod, "resolve_prequant_source", lambda fam, scheme, **kw: None)

    @classmethod
    def _fp_fail(cls, *a, **k):
        pytest.fail("dense transformer must not download when the scheme is unsupported")

    monkeypatch.setattr(_FakeTransformer, "from_pretrained", _fp_fail, raising = False)
    (tmp_path / "m.gguf").write_bytes(b"x")
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "m.gguf",
        family_override = "z-image",
        transformer_quant = "fp8",
    )
    assert status["loaded"] is True
    assert status["transformer_quant"] is None  # fell back to GGUF
    assert _FakeTransformer.last["path"]  # GGUF from_single_file used


def test_base_file_downloaded_include_transformer_flag():
    # Default: transformer/ shards are the GGUF's job, so they are excluded from
    # the prefetch list; the dense transformer-quant path opts them back in.
    from core.inference.diffusion import _base_file_downloaded

    assert _base_file_downloaded("transformer/diffusion_pytorch_model-00001.safetensors") is False
    assert (
        _base_file_downloaded(
            "transformer/diffusion_pytorch_model-00001.safetensors", include_transformer = True
        )
        is True
    )
    # The flag must not admit anything else that is normally excluded.
    assert _base_file_downloaded("assets/teaser.png", include_transformer = True) is False
    assert _base_file_downloaded("README.md", include_transformer = True) is False


def test_dense_quant_prefetch_needed_gates(fake_runtime, monkeypatch):
    # The transformer/ prefetch widens exactly when load_pipeline takes the dense-quant path: it
    # defers to resolve_dense_quant_candidate (quant requested + device supported + scheme
    # resolvable + no prequant checkpoint + disk for the extra bf16 shards). An explicit
    # Speed="off" (bit-exact) load never widens.
    from core.inference import diffusion as dmod

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    fam = detect_family("unsloth/Z-Image-Turbo-GGUF")

    seen: list = []

    def fake_candidate(
        *,
        fam,
        target,
        requested,
        base_repo = None,
        prequant_path = None,
        force_dense = False,
        logger = None,
    ):
        seen.append(requested)
        # A real (non-prequant) dense-quant candidate: scheme resolves AND disk fits, so the
        # loader takes the dense build that needs the base repo's bf16 transformer/ shards.
        return types.SimpleNamespace(prequant = False)

    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", fake_candidate)

    # Explicit fp8 -> widens; the resolved mode is threaded through to the candidate resolver.
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is True
    assert seen[-1] == "fp8"
    # UNSET defaults to the hardware ladder (Dtype default-auto) -> widens, threading auto.
    assert backend._dense_quant_prefetch_needed(fam, {}) is True
    assert seen[-1] == "auto"
    # A definite-offload memory policy forces load_pipeline onto offload regardless of the dense
    # candidate's smaller footprint, so the dense build never runs and the widened prefetch would
    # download base transformer/ shards the offloaded GGUF path never uses (disk-full there has no
    # GGUF fallback). balanced / low_vram (and the legacy cpu_offload flag absent a memory_mode)
    # must NOT widen, even though the candidate is dense-viable.
    before = len(seen)
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "balanced"}
        )
        is False
    )
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "low_vram"}
        )
        is False
    )
    assert (
        backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8", "cpu_offload": True})
        is False
    )
    # The gate short-circuits BEFORE resolving the candidate (no wasted resolve).
    assert len(seen) == before
    # An explicit memory_mode still consulting the candidate: fast/auto can flip resident, so they
    # widen when the candidate is dense-viable (memory_mode="fast" does not force offload).
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "fast"}
        )
        is True
    )
    # A cpu_offload flag is overridden by an explicit resident memory_mode, so it still widens.
    assert (
        backend._dense_quant_prefetch_needed(
            fam, {"transformer_quant": "fp8", "memory_mode": "fast", "cpu_offload": True}
        )
        is True
    )
    # An explicit off pins running the GGUF as-is -> never widen (mode resolves to None first).
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "none"}) is False
    # An explicit Speed="off" (bit-exact) load suppresses the dense path -> never widen.
    assert (
        backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8", "speed_mode": "off"})
        is False
    )
    # A PREQUANT candidate loads the small pre-quantized checkpoint (+ config / companions), NOT
    # the base repo's dense transformer/ shards, so the widened prefetch must NOT fire -- else it
    # defeats the prequant savings and can hard-fail begin_load (no GGUF fallback) on a disk-full.
    monkeypatch.setattr(
        dmod, "resolve_dense_quant_candidate", lambda **kw: types.SimpleNamespace(prequant = True)
    )
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False
    # No viable candidate (unsupported scheme / no disk room) -> never widen. The disk guard here
    # averts filling the cache volume and hard-failing the load instead of falling back to GGUF.
    monkeypatch.setattr(dmod, "resolve_dense_quant_candidate", lambda **kw: None)
    assert backend._dense_quant_prefetch_needed(fam, {"transformer_quant": "fp8"}) is False


def test_diffusion_status_response_carries_resolved():
    # The backend records per-control auto-policy provenance (build_resolved_record) on
    # state.resolved; the response model must DECLARE the field or Pydantic's extra='ignore' drops
    # it, leaving that plumbing dead (never reaching a client).
    from models.inference import DiffusionStatusResponse

    rec = {"transformer_quant": {"value": "fp8", "source": "auto", "reason": "blackwell"}}
    resp = DiffusionStatusResponse(loaded = True, resolved = rec)
    # The typed field coerces the plain record into DiffusionResolvedControl objects; the
    # serialized form must round-trip back to the record, proving the field is DECLARED and not
    # dropped by Pydantic's extra='ignore'.
    assert resp.model_dump()["resolved"] == rec
    # Absent by default (nothing resolved / native engine).
    assert DiffusionStatusResponse(loaded = False).resolved is None


def test_companion_cache_bytes_local_dir_excludes_transformer(tmp_path):
    # A LOCAL diffusers base: sum the on-disk VAE / text-encoder weights so auto memory planning
    # sees the resident companions, but exclude transformer/ (the GGUF supplies it) and non-weight
    # files. A folded-to-zero companion could OOM a resident plan.
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "text_encoder").mkdir()
    (tmp_path / "text_encoder" / "model.safetensors").write_bytes(b"y" * 50)
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"z" * 9999)
    (tmp_path / "model_index.json").write_bytes(b"{}")  # non-weight file, ignored
    total = DiffusionBackend._companion_cache_bytes(str(tmp_path))
    assert total == 150  # vae + text_encoder only; transformer/ and json excluded


def test_plan_memory_dense_replan_does_not_double_count_prefetched_transformer(monkeypatch):
    # Re-planning the dense transformer-quant candidate: the dense path prefetches the base repo's
    # transformer/ shards into the SAME blob cache _companion_cache_bytes sums. If the re-plan read
    # that cache it would count the transformer TWICE (as transformer_resident_override_mib and as
    # a "companion") and force offload even when the quantised artifact fits resident. The re-plan
    # must use the auto-policy's companion estimate. Here the cache is stubbed to the inflated
    # value; the plan must still stay resident.
    from core.inference import diffusion as dmod
    from core.inference.diffusion_memory import OFFLOAD_NONE, DeviceMemory

    backend = DiffusionBackend()
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    # 40 GiB discrete card, 40000 MiB free: comfortably fits transformer + real
    # companions + headroom, but NOT a second copy of the bf16 transformer.
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 40000, 40960),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 4000)
    # The cache is inflated by the prefetched bf16 transformer (~24000) on top of the
    # ~8000 real companions; if the re-plan consulted it the plan would offload.
    monkeypatch.setattr(
        DiffusionBackend,
        "_companion_cache_bytes",
        staticmethod(lambda base: (8000 + 24000) * 1024 * 1024),
    )
    fam = types.SimpleNamespace(name = "z-image")
    plan = backend._plan_memory(
        target,
        None,
        "org/base",
        fam,
        None,
        False,
        kind = "gguf",
        transformer_resident_override_mib = 12000,  # int8 candidate transient (~half bf16)
        companion_override_mib = 8000,  # auto-policy text-encoder + VAE estimate
    )
    # 12000 + 8000 + 4000 + 2048 overhead = 26048 MiB, fits the ~36 GiB budget.
    # A double-count (12000 + [8000+24000] + ...) would have exceeded it and offloaded.
    assert plan.offload_policy == OFFLOAD_NONE


def test_reset_step_cache_helper_is_best_effort():
    # Prefers the real diffusers CacheMixin hook (_reset_stateful_cache): on a genuine Flux /
    # QwenImage transformer that is the reset entry point, and reset_stateful_hooks lives only on
    # the HookRegistry (getattr on the transformer returns None), so the old lookup was a silent
    # no-op that left stale FBCache residuals for the next generation.
    calls = []
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(_reset_stateful_cache = lambda: calls.append("real"))
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["real"]
    # _reset_stateful_cache wins when both are present.
    calls.clear()
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(
            _reset_stateful_cache = lambda: calls.append("real"),
            reset_stateful_hooks = lambda: calls.append("fallback"),
        )
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["real"]
    # Falls back to reset_stateful_hooks for a transformer that exposes only that.
    calls.clear()
    pipe = types.SimpleNamespace(
        transformer = types.SimpleNamespace(reset_stateful_hooks = lambda: calls.append("fallback"))
    )
    DiffusionBackend._reset_step_cache(pipe)
    assert calls == ["fallback"]
    # No transformer, or a transformer without either hook -> silent no-op (never raises).
    DiffusionBackend._reset_step_cache(types.SimpleNamespace())
    DiffusionBackend._reset_step_cache(types.SimpleNamespace(transformer = object()))


def test_generate_resets_step_cache_only_when_engaged(fake_runtime, tmp_path):
    # FBCache residuals live on the resident transformer across generations, so each
    # generate() must reset the stateful cache first -- but only when a cache is engaged.
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
    )
    resets = []
    # Use the real diffusers CacheMixin entry point (_reset_stateful_cache); a genuine
    # Flux/QwenImage transformer exposes this, not reset_stateful_hooks.
    backend._state.pipe.transformer = types.SimpleNamespace(
        _reset_stateful_cache = lambda: resets.append(True)
    )
    # No cache engaged (transformer_cache is None) -> reset must NOT run.
    backend.generate(prompt = "a sloth")
    assert resets == []
    # Engage a cache; every subsequent generation resets the stateful cache first.
    object.__setattr__(backend._state, "transformer_cache", "fbcache")
    backend.generate(prompt = "a sloth")
    backend.generate(prompt = "another sloth")
    assert resets == [True, True]


def test_prefetch_returns_snapshot_dir_for_manifest(monkeypatch):
    # The prefetched pipeline manifest's directory is the local snapshot root; a
    # config-only base list (no manifest) returns None so the hub id stays in use.
    backend = DiffusionBackend()
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **k: f"/cache/snap/{fn}",
    )
    root = backend._prefetch_files(
        "base/repo", None, "base/repo", ["model_index.json", "vae/x.safetensors"], None
    )
    assert root == "/cache/snap"
    assert (
        backend._prefetch_files("base/repo", None, "base/repo", ["vae/x.safetensors"], None) is None
    )


def test_pipeline_load_uses_predownloaded_dir(fake_runtime, tmp_path):
    # With a prefetched snapshot, from_pretrained must receive the local dir -- its own hub sweep
    # would re-download the root packaged singles the scoped prefetch skips (24 GB per FLUX.1 repo).
    backend = DiffusionBackend()
    backend.load_pipeline(
        "unsloth/Qwen-Image-2512-bnb-4bit",
        model_kind = "pipeline",
        _base_local_dir = str(tmp_path),
    )
    assert _FakePipeline.last["base"] == str(tmp_path)
    backend.unload()


def test_unload_waits_for_in_flight_denoise_before_teardown():
    # Regression: unload() must wait for a running denoise to exit (acquire _generate_lock) before
    # _unload_locked() tears down process-wide state the denoise still depends on.
    import threading

    backend = DiffusionBackend()

    denoise_active = {"v": False}
    teardown_saw = []  # records denoise_active at the moment _unload_locked runs

    cancel = threading.Event()
    backend._active_generate_cancel = cancel
    started = threading.Event()
    finish = threading.Event()

    # _generate_lock is the only lock a real denoise holds for its whole body.
    def _denoise():
        with backend._generate_lock:
            denoise_active["v"] = True
            started.set()
            cancel.wait(2.0)  # unload signals this
            finish.wait(2.0)  # the test lets us finish
            denoise_active["v"] = False  # about to release _generate_lock

    def _fake_unload_locked():
        teardown_saw.append(denoise_active["v"])

    backend._unload_locked = _fake_unload_locked  # instance attr shadows the method

    d = threading.Thread(target = _denoise)
    d.start()
    assert started.wait(2.0)  # denoise holds _generate_lock

    unloaded = threading.Event()

    def _unload():
        backend.unload()
        unloaded.set()

    u = threading.Thread(target = _unload)
    u.start()
    assert cancel.wait(2.0)  # unload has signalled the denoise and is now waiting on _generate_lock
    # unload must NOT have torn down yet -- it is blocked on the denoise's _generate_lock.
    assert teardown_saw == []
    assert not unloaded.wait(0.3)

    finish.set()  # let the denoise release _generate_lock
    d.join(2.0)
    u.join(2.0)
    assert unloaded.is_set()
    # Teardown ran exactly once, and only AFTER the denoise had exited (denoise_active was False).
    assert teardown_saw == [False]
