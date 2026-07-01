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
import types

import pytest

from core.inference.diffusion import (
    DiffusionBackend,
    _LoadState,
    _base_file_downloaded,
    _resolve_diffusion_compute_dtype,
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
        self.sequential_offloaded = False
        self.vae_tiled = False
        self.vae_sliced = False
        self.last_kwargs = None

    def to(self, device):
        self.moved_to = device
        return self

    def enable_model_cpu_offload(self) -> None:
        self.offloaded = True

    def enable_sequential_cpu_offload(self) -> None:
        self.sequential_offloaded = True

    def enable_vae_tiling(self) -> None:
        self.vae_tiled = True

    def enable_vae_slicing(self) -> None:
        self.vae_sliced = True

    # Explicit signature (not just **kwargs) so generate()'s signature-gated
    # guards for negative_prompt / callback_on_step_end actually take effect —
    # a **kwargs-only fake would make `"negative_prompt" in signature` always False.
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
    # generate() wraps the pipe call in torch.inference_mode(); a no-op CM here.
    torch.inference_mode = lambda: contextlib.nullcontext()

    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)
    diffusers.ZImagePipeline = _FakePipeline
    diffusers.ZImageTransformer2DModel = _FakeTransformer
    # Qwen-Image too, so the true_cfg_scale cfg-kwarg path is exercisable.
    diffusers.QwenImagePipeline = _FakePipeline
    diffusers.QwenImageTransformer2DModel = _FakeTransformer

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
    # An unload (or an arbiter eviction, which calls unload) while a load's worker
    # is still resolving/downloading must cancel it: load_pipeline sees the bumped
    # token and aborts, so the evicted load never resurrects a pipeline into VRAM.
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

    # unload() must return promptly (it does not wait on _generate_lock) and signal
    # THIS in-flight generation's cancel event.
    backend.unload()
    assert backend._active_generate_cancel is not None
    assert backend._active_generate_cancel.is_set()
    assert backend.status()["loaded"] is False

    release.set()
    t.join(5)
    # The cancelled generation raised rather than returning a now-evicted image.
    assert "exc" in out and "cancelled" in str(out["exc"]).lower()


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
    with pytest.raises(ValueError, match = "gguf_filename"):
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF")
    with pytest.raises(ValueError, match = "family"):
        backend.validate_load_request("meta/Llama-3", gguf_filename = "q.gguf")
    assert (
        backend.validate_load_request("unsloth/Z-Image-Turbo-GGUF", gguf_filename = "q.gguf").name
        == "z-image"
    )
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
    # A superseding load must signal the in-flight generation's cancel AND wait for
    # it to release _generate_lock before allocating, so two pipelines never sit in
    # VRAM at once (unlike unload(), which returns promptly without waiting).
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
    # balanced requests streamed block-level (group) offload. Under the stub there is
    # no real diffusers.hooks, so group can't engage and the applier falls back to
    # whole-module offload, reporting the policy actually engaged (the real "group"
    # path is GPU-verified in the bench).
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
    # No speed_mode on a GGUF model -> auto `default` (near-lossless, compile sits
    # below the quant noise floor). compile itself only engages on CUDA, so on this
    # CPU stub no optim need engage, but the resolved mode is `default`.
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

    def _quantize(pipe, target, *, mode, **kw):
        calls["quantize"] += 1
        calls["quant_mode"] = mode
        return scheme

    monkeypatch.setattr(dmod, "quantize_transformer", _quantize)
    return calls


def test_default_load_skips_dense_quant_path(fake_runtime, tmp_path, monkeypatch):
    # With no transformer_quant flag the GGUF path is taken and the dense gate is
    # never even consulted (short-circuit), so the default cannot regress.
    from core.inference import diffusion as dmod

    monkeypatch.setattr(
        dmod,
        "dense_transformer_supported",
        lambda *a, **k: pytest.fail("dense path must not run without the flag"),
    )
    (tmp_path / "m.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    status = backend.load_pipeline(str(tmp_path), gguf_filename = "m.gguf", family_override = "z-image")
    assert status["transformer_quant"] is None
    assert _FakeTransformer.last["path"]  # GGUF from_single_file was used


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
    # The dense bf16 transformer only fits resident, so when the memory plan would
    # offload (here low_vram) the fast path is skipped and GGUF loads instead -- the
    # dense transformer is never even loaded.
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
