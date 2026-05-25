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
from types import SimpleNamespace
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


def test_detect_family_sd35_is_not_sd3():
    """SD3.5 must NOT be matched as SD3 Medium. Pairing SD3.5 GGUFs
    with the Medium base produces a misleading load."""
    from core.inference.diffusion import detect_family

    assert detect_family("unsloth/SD3.5-large-GGUF") is None
    assert detect_family("unsloth/stable-diffusion-3.5-large-GGUF") is None


def test_detect_family_qwen_image_edit_is_not_qwen_image():
    """Qwen-Image-Edit must NOT be matched as Qwen-Image. The Edit
    variant uses a different pipeline (image-to-image)."""
    from core.inference.diffusion import detect_family

    assert detect_family("unsloth/Qwen-Image-Edit-GGUF") is None
    assert detect_family("unsloth/Qwen-Image-Edit-2509-GGUF") is None
    # Underscore spellings on the Hub must also be excluded; otherwise
    # qwen_image_edit-GGUF silently matches the base Qwen-Image family.
    assert detect_family("unsloth/qwen_image_edit-GGUF") is None
    assert detect_family("unsloth/QwenImageEdit-GGUF") is None


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
    """Public status() (the browser-facing payload) must NOT contain
    the guard-only ``active_*`` / ``pending_*`` fields (round 16
    P1 #5)."""
    from core.inference.diffusion import get_diffusion_backend

    s = get_diffusion_backend().status()
    expected_keys = {
        "is_loaded",
        "is_loading",
        "repo_id",
        "family",
        "pipeline_class",
        "base_repo",
        "gguf_filename",
        "device",
        "dtype",
        "loaded_at",
        "last_error",
        "supported_families",
    }
    assert expected_keys.issubset(s.keys())
    # Guard-facing fields are gated behind include_internal=True.
    for guard_key in (
        "active_repo_id",
        "active_base_repo",
        "active_gguf_filename",
        "pending_repo_id",
        "pending_base_repo",
        "pending_gguf_filename",
    ):
        assert guard_key not in s, f"public status() must not expose {guard_key}"
    assert s["is_loaded"] is False
    assert s["repo_id"] is None

    # Internal status() exposes the guard fields for delete/route use.
    s_internal = get_diffusion_backend().status(include_internal = True)
    assert s_internal["active_gguf_filename"] is None
    assert s_internal["pending_gguf_filename"] is None


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
        def from_single_file(cls, path, **kw):
            inst = cls()
            inst.path = path
            inst.qc = kw.get("quantization_config")
            inst.dtype = kw.get("torch_dtype")
            inst.config = kw.get("config")
            inst.subfolder = kw.get("subfolder")
            inst.token = kw.get("token")
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
    # _smart_base_repo picks the distilled 4B (not the Base) for the
    # "FLUX.2-klein-4B-GGUF" repo name. The Base variant kicks in only
    # when "base" is part of the repo id.
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-4B"
    assert status["gguf_filename"] == "flux-2-klein-4b-Q4_K_S.gguf"


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


def test_smart_base_repo_picks_base_9b(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-base-9B-GGUF",
        gguf_filename = "flux-2-klein-base-9b-Q4_K_S.gguf",
    )
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-9B"


def test_smart_base_repo_picks_base_4b(monkeypatch):
    _install_fake_diffusers(monkeypatch)
    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    status = backend.load_model(
        "unsloth/FLUX.2-klein-base-4B-GGUF",
        gguf_filename = "flux-2-klein-base-4b-Q4_K_S.gguf",
    )
    assert status["base_repo"] == "black-forest-labs/FLUX.2-klein-base-4B"


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
    backend._family = d._FAMILIES[2]
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
    backend._family = d._FAMILIES[2]
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


def test_load_publishes_pending_target_during_loading(monkeypatch):
    """status() must expose the pending repo_id / base_repo / gguf
    file while is_loading=True so cache- and finetuned-delete guards
    can refuse to rmtree the repo being downloaded right now."""
    import threading
    import core.inference.diffusion as d
    from PIL import Image

    fake = _install_fake_diffusers(monkeypatch)

    pending_seen: dict = {}
    pretrained_blocked = threading.Event()
    pretrained_release = threading.Event()

    class _SlowPipeline:
        @classmethod
        def from_pretrained(cls, base_repo, **kwargs):
            pretrained_blocked.set()
            # Capture status() output while the load is blocked.
            backend = d.get_diffusion_backend()
            pending_seen.update(backend.status())
            pretrained_release.wait(timeout = 5)
            inst = cls()
            inst.base_repo = base_repo
            return inst

        def __call__(self, **kwargs):
            class _Out:
                pass

            o = _Out()
            o.images = [Image.new("RGB", (kwargs["width"], kwargs["height"]))]
            return o

        def enable_model_cpu_offload(self):
            pass

        def to(self, device):
            return self

    fake.Flux2KleinPipeline = _SlowPipeline

    backend = d.get_diffusion_backend()
    backend.unload_model()

    def do_load():
        try:
            backend.load_model(
                "unsloth/FLUX.2-klein-4B-GGUF",
                gguf_filename = "flux-2-klein-4b-Q4_K_S.gguf",
            )
        except Exception:
            pass

    t = threading.Thread(target = do_load)
    t.start()
    try:
        assert pretrained_blocked.wait(timeout = 5)
        # While blocked inside from_pretrained, status reads should
        # already see the pending repo so deletes can be refused.
        assert pending_seen.get("is_loading") is True
        assert pending_seen.get("repo_id") == "unsloth/FLUX.2-klein-4B-GGUF"
        assert pending_seen.get("base_repo") == "black-forest-labs/FLUX.2-klein-4B"
    finally:
        pretrained_release.set()
        t.join(timeout = 5)


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

    def _boom(**_):
        raise AssertionError("hf_hub_download must not run for a local dir")

    fake_hub = SimpleNamespace(hf_hub_download = _boom)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = d.DiffusionBackend()
    backend.load_model(
        repo_id = "exports/my-flux",
        gguf_filename = "model.gguf",
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

    assert detect_family(repo_id) is None


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
