# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the diffusion backend.

The family helpers are pure functions, tested directly. The backend lifecycle is
exercised with ``torch`` / ``diffusers`` stubbed via ``sys.modules`` so no real
GPU, weights, or network access is needed (sub-second, CI-friendly).
"""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.diffusion import (
    DiffusionBackend,
    _base_file_downloaded,
)
from core.inference.diffusion_families import (
    detect_family,
    resolve_base_repo,
    resolve_local_gguf_child,
)


# Pure family helpers


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
    # Only klein is wired up; the Mistral-based FLUX.2-dev base repo is gated.
    assert detect_family("unsloth/FLUX.2-dev-GGUF") is None
    # Qwen-Image guides via true_cfg_scale, not guidance_scale.
    assert detect_family("unsloth/Qwen-Image-2512-GGUF").cfg_kwarg == "true_cfg_scale"
    assert detect_family("unsloth/Z-Image-GGUF").cfg_kwarg == "guidance_scale"
    # Image-editing checkpoints are rejected (text-to-image backend only).
    assert detect_family("unsloth/Qwen-Image-Edit-2511-GGUF") is None
    assert detect_family("unsloth/FLUX.1-Kontext-dev-GGUF") is None
    assert detect_family("meta-llama/Llama-3-8B") is None


def test_detect_family_override():
    assert detect_family("local/path", override = "z-image").name == "z-image"
    assert detect_family("local/path", override = "zimage").name == "z-image"
    assert detect_family("local/path", override = "not-a-family") is None


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
        self.last_kwargs = None

    def to(self, device):
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self) -> None:
        self.offloaded = True

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        n = kwargs.get("num_images_per_prompt", 1)
        return types.SimpleNamespace(images = [_FakeImage() for _ in range(n)])


class _FakePipeline:
    last: dict = {}

    @classmethod
    def from_pretrained(cls, base, **kwargs):
        _FakePipeline.last = {"base": base, **kwargs}
        return _FakePipe()


class _FakeTransformer:
    last: dict = {}

    @classmethod
    def from_single_file(cls, path, **kwargs):
        _FakeTransformer.last = {"path": path, **kwargs}
        return object()


@pytest.fixture
def fake_runtime(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = _FakeDtype("bfloat16")
    torch.float16 = _FakeDtype("float16")
    torch.float32 = _FakeDtype("float32")
    torch.Generator = _FakeGenerator
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    torch.backends = types.SimpleNamespace(mps = None)

    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)
    diffusers.ZImagePipeline = _FakePipeline
    diffusers.ZImageTransformer2DModel = _FakeTransformer

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    # The backend imports clear_gpu_cache by reference; no-op it so unload doesn't
    # run real hardware detection against the stubbed torch.
    monkeypatch.setattr("core.inference.diffusion.clear_gpu_cache", lambda: None)
    _FakePipeline.last = {}
    _FakeTransformer.last = {}
    yield


def test_load_generate_unload_gguf(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = DiffusionBackend()

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "base/repo",
        family_override = "z-image",
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
    assert _FakePipeline.last["base"] == "base/repo"
    assert "transformer" in _FakePipeline.last

    gen = backend.generate(prompt = "a sloth", width = 512, height = 512, steps = 4, guidance = 3.0)
    assert gen["seed"] == 4242  # random seed reported back
    assert gen["repo_id"] == str(tmp_path)  # echoed so the route can record the model
    assert len(gen["images"]) == 1  # PIL images handed to the route for persistence

    gen2 = backend.generate(prompt = "again", seed = 99)
    assert gen2["seed"] == 99

    # batch_size produces that many images in one call, all sharing the seed.
    batch = backend.generate(prompt = "batch", seed = 7, batch_size = 3)
    assert len(batch["images"]) == 3 and batch["seed"] == 7

    assert backend.unload()["loaded"] is False
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


def test_generate_without_load_raises(fake_runtime):
    backend = DiffusionBackend()
    with pytest.raises(RuntimeError):
        backend.generate(prompt = "x")


def test_resolve_base_repo_prefers_caller_then_hf_tag_then_fallback(monkeypatch):
    from core.inference import diffusion
    from core.inference.diffusion_families import detect_family

    fam = detect_family("unsloth/Qwen-Image-2512-GGUF")
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo, tok: "Qwen/Qwen-Image-2512")
    # Caller's explicit base wins and the HF tag is not consulted.
    assert diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "my/base", fam, None) == "my/base"
    # No caller base: the repo's base_model tag (the variant base) is used.
    assert diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", None, fam, None) == "Qwen/Qwen-Image-2512"
    # No caller base and no tag: the family fallback.
    monkeypatch.setattr(diffusion, "_hf_base_model", lambda repo, tok: None)
    assert diffusion._resolve_base_repo("unsloth/Qwen-Image-2512-GGUF", "  ", fam, None) == fam.base_repo


def test_load_without_gguf_raises():
    backend = DiffusionBackend()
    with pytest.raises(ValueError):
        backend.load_pipeline("unsloth/Z-Image-Turbo-GGUF")  # no gguf_filename


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
    assert not _base_file_downloaded("transformer/diffusion_pytorch_model-00001-of-00003.safetensors")
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


def test_begin_load_rejects_concurrent(monkeypatch):
    backend = DiffusionBackend()
    monkeypatch.setattr(DiffusionBackend, "_estimate_download_bytes", staticmethod(lambda *a, **k: 0))
    # Block the spawned worker so the load stays "in progress".
    monkeypatch.setattr(DiffusionBackend, "load_pipeline", lambda self, **k: __import__("time").sleep(0.2))
    backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
    with pytest.raises(RuntimeError):
        backend.begin_load("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "z-image-turbo-Q4_K_S.gguf")
