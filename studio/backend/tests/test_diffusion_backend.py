# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

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
import sys
import types
from typing import Any

import pytest


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
    # "FLUX.2-klein" slug that does not exist). The frontend curated
    # picker still passes base_repo explicitly per size so this default
    # only fires for the "custom HF repo" mode.
    assert fam.base_repo == "black-forest-labs/FLUX.2-klein-base-4B"


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


def test_supported_families_payload_shape():
    from core.inference.diffusion import supported_families

    payload = supported_families()
    assert isinstance(payload, list)
    assert len(payload) >= 4
    for entry in payload:
        assert set(entry.keys()) == {"name", "pipeline_class", "base_repo"}


# ── singleton ───────────────────────────────────────────────────


def test_get_diffusion_backend_singleton():
    from core.inference.diffusion import get_diffusion_backend

    a = get_diffusion_backend()
    b = get_diffusion_backend()
    assert a is b


# ── status() shape ──────────────────────────────────────────────


def test_status_shape_unloaded():
    from core.inference.diffusion import get_diffusion_backend

    s = get_diffusion_backend().status()
    expected_keys = {
        "is_loaded",
        "is_loading",
        "repo_id",
        "family",
        "pipeline_class",
        "base_repo",
        "gguf_path",
        "device",
        "dtype",
        "loaded_at",
        "last_error",
        "supported_families",
    }
    assert expected_keys.issubset(s.keys())
    assert s["is_loaded"] is False
    assert s["repo_id"] is None


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

    class _FakeTransformer:
        @classmethod
        def from_single_file(cls, path, quantization_config = None, torch_dtype = None):
            inst = cls()
            inst.path = path
            inst.qc = quantization_config
            inst.dtype = torch_dtype
            return inst

    class _FakePipeline:
        @classmethod
        def from_pretrained(cls, base_repo, **kwargs):
            if raise_on_pipeline:
                raise RuntimeError("simulated load failure")
            inst = cls()
            inst.base_repo = base_repo
            inst.kwargs = kwargs
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
            return self

    fake.GGUFQuantizationConfig = _FakeQuantConfig
    fake.Flux2KleinPipeline = _FakePipeline
    fake.Flux2Transformer2DModel = _FakeTransformer
    fake.Flux2Pipeline = _FakePipeline
    fake.FluxPipeline = _FakePipeline
    fake.FluxTransformer2DModel = _FakeTransformer
    fake.QwenImagePipeline = _FakePipeline
    fake.QwenImageTransformer2DModel = _FakeTransformer
    fake.SD3Transformer2DModel = _FakeTransformer
    fake.StableDiffusion3Pipeline = _FakePipeline
    fake.StableDiffusionXLPipeline = _FakePipeline

    monkeypatch.setitem(sys.modules, "diffusers", fake)

    # Pretend HF Hub gave us a local file without actually fetching.
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = (
        lambda repo_id, filename, token = None: f"/fake/{repo_id}/{filename}"
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    # Force CPU dtype so the test does not need CUDA.
    import core.inference.diffusion as d

    monkeypatch.setattr(
        d.DiffusionBackend,
        "_pick_device_and_dtype",
        lambda self: ("cpu", "fake_dtype"),
    )

    return fake


def test_load_model_unknown_family(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    with pytest.raises(RuntimeError, match = "Could not infer"):
        backend.load_model("private/random-repo")


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
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-4B"
    assert status["gguf_path"] == (
        "/fake/unsloth/FLUX.2-klein-4B-GGUF/flux-2-klein-4b-Q4_K_S.gguf"
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
    assert status["repo_id"] == "owner/FLUX.1-finetune-diffusers"
    # And the fake pipeline records what we called from_pretrained with.
    assert backend._pipe.base_repo == "owner/FLUX.1-finetune-diffusers"


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
    assert active["max"] == 1, (
        f"Expected concurrent loads to serialise; max_active={active['max']}"
    )


def test_pipe_accepts_kwarg_filter():
    """The negative_prompt filter must drop the kwarg on classes that
    do not accept it (FLUX.2 / FLUX.2 klein) and keep it on the rest."""
    from core.inference.diffusion import _pipe_accepts_kwarg

    class _NoNeg:
        def __call__(self, *, prompt, num_inference_steps, guidance_scale, width, height):
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
    backend._family = d._FAMILIES[2]  # flux.1 supports negative_prompt
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
