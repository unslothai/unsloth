# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VideoBackend lifecycle on a faked torch/diffusers runtime (CPU-only, offline).
Mirrors test_diffusion_backend's fake_runtime pattern: explicit fake signatures so
the signature-gated kwargs actually exercise, sys.modules stubs so no real ML
stack loads."""

import builtins
import contextlib
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from core.inference.video import (
    VideoBackend,
    _detect_load_family,
    get_video_backend,
    resolve_video_model_kind,
)
from core.inference.video_families import VIDEO_CANCELLED_MSG, VIDEO_NOT_LOADED_MSG


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


class _FakeVae:
    def __init__(self) -> None:
        self.tiled = False

    def enable_tiling(self) -> None:
        self.tiled = True


class _FakePipe:
    def __init__(self) -> None:
        self.moved_to = None
        self.vae = _FakeVae()
        self.last_kwargs = None
        self._interrupt = False

    def to(self, device):
        self.moved_to = device
        return self

    def enable_vae_tiling(self) -> None:
        self.vae.tiled = True

    # Explicit signature so generate() signature-gated kwargs actually engage; **kwargs would defeat the gates.
    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        num_inference_steps = None,
        guidance_scale = None,
        width = None,
        height = None,
        num_frames = None,
        frame_rate = None,
        generator = None,
        sigmas = None,
        callback_on_step_end = None,
        **kwargs,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "sigmas": sigmas,
            **kwargs,
        }
        if callback_on_step_end is not None:
            for step in range(int(num_inference_steps or 1)):
                callback_on_step_end(self, step, 0, {})
                if self._interrupt:
                    break
        frames = [[object() for _ in range(int(num_frames or 1))]]
        return types.SimpleNamespace(frames = frames, audio = None)


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


# Wan2.2 fakes: a per-DiT trackable transformer so the dual-DiT tests can assert speed / cache / attention on BOTH
# experts. The MoE __call__ carries guidance_scale_2 and the single-DiT one omits it, so the cfg2 gate is exercised.


class _FakeWanDiT:
    """One Wan denoiser. Records which optimisation helpers touched it (the loader
    applies each once per expert on an MoE load), so a test can prove BOTH experts
    were covered. compile_repeated_blocks / enable_cache / set_attention_backend are
    exactly the attribute names the imported helpers look for."""

    def __init__(self) -> None:
        self.compiled = False
        self.cache_config = None
        self.attention = None

    def compile_repeated_blocks(self, **kwargs) -> None:
        self.compiled = True

    def enable_cache(self, config) -> None:
        self.cache_config = config

    def disable_cache(self) -> None:
        self.cache_config = None

    def set_attention_backend(self, backend) -> None:
        self.attention = backend

    @contextlib.contextmanager
    def cache_context(self, name):
        # Real pipelines open a cache_context around the denoise loop for the First-Block-Cache hook, so the fake provides one.
        yield


class _FakeWanVae:
    def __init__(self) -> None:
        self.tiled = False

    def enable_tiling(self) -> None:
        self.tiled = True

    def to(self, *args, **kwargs):
        return self


class _FakeWanPipeBase:
    """Shared Wan pipeline state. Subclasses provide the __call__ with the right
    explicit signature (with/without guidance_scale_2) so the generate() cfg2 and
    frame_rate signature-gates actually exercise -- ``**kwargs`` alone would hide the
    parameter names inspect.signature reads."""

    moe: bool = False

    def __init__(self) -> None:
        self.vae = _FakeWanVae()
        self.transformer = _FakeWanDiT()
        self.transformer_2 = _FakeWanDiT() if self.moe else None
        self.components = {"transformer": self.transformer, "vae": self.vae}
        if self.moe:
            self.components["transformer_2"] = self.transformer_2
        self.moved_to = None
        self.last_kwargs = None
        self._interrupt = False

    def to(self, device):
        self.moved_to = device
        return self

    def enable_vae_tiling(self) -> None:
        self.vae.tiled = True

    def _finish(self, num_inference_steps, num_frames, callback_on_step_end):
        if callback_on_step_end is not None:
            for step in range(int(num_inference_steps or 1)):
                callback_on_step_end(self, step, 0, {})
                if self._interrupt:
                    break
        frames = [[object() for _ in range(int(num_frames or 1))]]
        return types.SimpleNamespace(frames = frames, audio = None)


class _FakeWanPipeSingle(_FakeWanPipeBase):
    """Single-DiT Wan pipeline (TI2V-5B): NO guidance_scale_2 in the signature, so the
    cfg2 gate must not thread it."""

    moe = False

    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        num_inference_steps = None,
        guidance_scale = None,
        width = None,
        height = None,
        num_frames = None,
        generator = None,
        callback_on_step_end = None,
        **kwargs,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            **kwargs,
        }
        with self.transformer.cache_context("cond"):  # real Wan pipeline wraps the denoise loop
            pass
        return self._finish(num_inference_steps, num_frames, callback_on_step_end)


class _FakeWanPipeMoE(_FakeWanPipeBase):
    """Dual-DiT MoE Wan pipeline (A14B): guidance_scale_2 IS in the signature, matching
    WanPipeline.__call__ in diffusers 0.39, so the cfg2 gate threads it."""

    moe = True

    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        num_inference_steps = None,
        guidance_scale = None,
        guidance_scale_2 = None,
        width = None,
        height = None,
        num_frames = None,
        generator = None,
        callback_on_step_end = None,
        **kwargs,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "guidance_scale_2": guidance_scale_2,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            **kwargs,
        }
        with self.transformer.cache_context("cond"):  # real Wan pipeline wraps the denoise loop
            pass
        return self._finish(num_inference_steps, num_frames, callback_on_step_end)


class _FakeWanPipelineSingle:
    """WanPipeline fake (from_pretrained). One class serves both families and picks the
    single-DiT / dual-DiT pipe by the repo id, exactly as diffusers dispatches on the
    repo's model_index.json (A14B lists transformer_2, TI2V-5B does not)."""

    last: dict = {}

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        _FakeWanPipelineSingle.last = {"repo": repo, **kwargs}
        moe = "a14b" in str(repo).lower()
        return _FakeWanPipeMoE() if moe else _FakeWanPipeSingle()


# HunyuanVideo-1.5 fakes: no guidance kwarg, no step callback, a guider carrying the CFG scale, and a scheduler.step-driven loop.


class _FakeHV15Scheduler:
    def __init__(self) -> None:
        self.calls = 0
        # Test hook fired from the ORIGINAL step, letting a test cancel mid-denoise exactly as a user request would.
        self.on_step = None

    def step(self, *args, **kwargs):
        self.calls += 1
        if self.on_step is not None:
            self.on_step(self.calls)
        return object()


class _FakeHV15Pipe:
    def __init__(self) -> None:
        self.vae = _FakeWanVae()
        self.transformer = _FakeWanDiT()
        self.scheduler = _FakeHV15Scheduler()
        self.guider = types.SimpleNamespace(guidance_scale = 6.0)
        self.components = {"transformer": self.transformer, "vae": self.vae}
        self.moved_to = None
        self.last_kwargs = None
        self.hooks_freed = 0

    def maybe_free_model_hooks(self):
        self.hooks_freed += 1

    def to(self, device):
        self.moved_to = device
        return self

    def enable_vae_tiling(self) -> None:
        self.vae.tiled = True

    def __call__(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        height = None,
        width = None,
        num_frames = None,
        num_inference_steps = None,
        generator = None,
        **kwargs,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            **kwargs,
        }
        with self.transformer.cache_context("cond"):  # real HV15 pipeline wraps the denoise loop
            pass
        for _ in range(int(num_inference_steps or 1)):
            self.scheduler.step()
        frames = [[object() for _ in range(int(num_frames or 1))]]
        return types.SimpleNamespace(frames = frames, audio = None)


class _FakeHV15Pipeline:
    last: dict = {}
    instance = None

    @classmethod
    def from_pretrained(cls, repo, **kwargs):
        _FakeHV15Pipeline.last = {"repo": repo, **kwargs}
        _FakeHV15Pipeline.instance = _FakeHV15Pipe()
        return _FakeHV15Pipeline.instance


@pytest.fixture
def fake_runtime(monkeypatch):
    torch = types.ModuleType("torch")
    torch.bfloat16 = _FakeDtype("bfloat16")
    torch.float16 = _FakeDtype("float16")
    torch.float32 = _FakeDtype("float32")
    torch.Generator = _FakeGenerator
    torch.cuda = types.SimpleNamespace(is_available = lambda: False)
    torch.backends = types.SimpleNamespace(mps = None)
    torch.inference_mode = lambda: contextlib.nullcontext()

    diffusers = types.ModuleType("diffusers")
    diffusers.GGUFQuantizationConfig = lambda compute_dtype = None: ("quant", compute_dtype)
    diffusers.LTX2Pipeline = _FakePipeline
    diffusers.LTX2VideoTransformer3DModel = _FakeTransformer
    # Wan2.2: one pipeline class serves both families (it dispatches on the repo id).
    diffusers.WanPipeline = _FakeWanPipelineSingle
    diffusers.WanTransformer3DModel = _FakeTransformer
    diffusers.HunyuanVideo15Pipeline = _FakeHV15Pipeline
    diffusers.HunyuanVideo15Transformer3DModel = _FakeTransformer
    diffusers.FirstBlockCacheConfig = lambda threshold = None: ("fbcache", threshold)

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setattr("core.inference.video.clear_gpu_cache", lambda: None)
    # MP4 encode needs real frames + PyAV; the contract under test is the byte handoff, so stub the encoder.
    monkeypatch.setattr(
        VideoBackend, "_encode_mp4", staticmethod(lambda frames, fps, audio, pipe: b"MP4")
    )
    _FakePipeline.last = {}
    _FakeTransformer.last = {}
    yield


def _load_gguf(backend, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"weights")
    return backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )


def test_resolve_kind():
    assert resolve_video_model_kind("x.gguf", None) == "gguf"
    assert resolve_video_model_kind("x.safetensors", None) == "single_file"
    assert resolve_video_model_kind(None, None) == "pipeline"
    with pytest.raises(ValueError):
        resolve_video_model_kind(None, "bogus")


def test_validate_rejects_unknown_and_untrusted():
    backend = VideoBackend()
    with pytest.raises(ValueError, match = "not a supported"):
        backend.validate_load_request("someorg/some-image-model")
    # A known family but an untrusted repo id must not open from_pretrained.
    with pytest.raises(ValueError, match = "limited to"):
        backend.validate_load_request("evil/ltx-2-repack")
    # GGUF loads stay open to any repo (single-file read, no pickle).
    fam = backend.validate_load_request(
        "anyorg/ltx-2-GGUF", gguf_filename = "x.gguf", model_kind = "gguf"
    )
    assert fam.name == "ltx-2"
    with pytest.raises(ValueError, match = "filename"):
        backend.validate_load_request("unsloth/LTX-2.3-GGUF", model_kind = "gguf")


def test_validate_gates_base_repo_and_local_paths(tmp_path):
    backend = VideoBackend()
    # An arbitrary remote base_repo must not reach from_pretrained via a GGUF pick.
    with pytest.raises(ValueError, match = "base_repo"):
        backend.validate_load_request(
            "unsloth/LTX-2.3-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = "evil/companions",
        )
    # The family base and local dirs stay allowed.
    fam = backend.validate_load_request(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "x.gguf",
        model_kind = "gguf",
        base_repo = "Lightricks/LTX-2",
    )
    assert fam.name == "ltx-2"
    # A local dir without the picked checkpoint fails BEFORE the GPU handoff.
    with pytest.raises(ValueError):
        backend.validate_load_request(
            str(tmp_path), gguf_filename = "missing.gguf", family_override = "ltx-2"
        )
    # A path-shaped repo id that does not exist fails validation too.
    with pytest.raises(ValueError, match = "does not exist"):
        backend.validate_load_request(
            str(tmp_path / "nope" / "model.gguf"),
            gguf_filename = "model.gguf",
            family_override = "ltx-2",
        )


def test_validate_rejects_kind_extension_mismatch(tmp_path):
    backend = VideoBackend()
    # A kind/extension mismatch must be rejected BEFORE the GPU handoff, not fail in the wrong single-file loader after eviction.
    with pytest.raises(ValueError, match = "needs model_kind 'gguf'"):
        backend.validate_load_request(
            "unsloth/LTX-2.3-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "single_file",
            family_override = "ltx-2",
        )
    with pytest.raises(ValueError, match = "requires a .gguf"):
        backend.validate_load_request(
            "unsloth/LTX-2.3",
            gguf_filename = "x.safetensors",
            model_kind = "gguf",
            family_override = "ltx-2",
        )
    with pytest.raises(ValueError, match = "not a loadable single-file checkpoint"):
        backend.validate_load_request(
            "unsloth/LTX-2.3",
            gguf_filename = "readme.md",
            model_kind = "single_file",
            family_override = "ltx-2",
        )


def test_validate_rejects_local_file_suffix_kind_mismatch(tmp_path):
    backend = VideoBackend()
    # A local FILE is handed straight to the loader (_resolve_checkpoint_path IGNORES gguf_filename), so the file's own
    # suffix must match the kind. Such a mismatch slips past the filename checks, so reject it before eviction.
    gguf_file = tmp_path / "ltx.gguf"
    gguf_file.write_bytes(b"weights")
    safetensors_file = tmp_path / "ltx.safetensors"
    safetensors_file.write_bytes(b"weights")
    with pytest.raises(ValueError, match = "not a .safetensors file"):
        backend.validate_load_request(
            str(gguf_file),
            gguf_filename = "ltx.safetensors",
            model_kind = "single_file",
            family_override = "ltx-2",
        )
    with pytest.raises(ValueError, match = "not a .gguf file"):
        backend.validate_load_request(
            str(safetensors_file),
            gguf_filename = "ltx.gguf",
            model_kind = "gguf",
            family_override = "ltx-2",
        )
    # Matching pairs still validate: the local file's suffix agrees with the resolved kind.
    assert (
        backend.validate_load_request(
            str(gguf_file),
            gguf_filename = "ltx.gguf",
            model_kind = "gguf",
            family_override = "ltx-2",
        ).name
        == "ltx-2"
    )
    assert (
        backend.validate_load_request(
            str(safetensors_file),
            gguf_filename = "ltx.safetensors",
            model_kind = "single_file",
            family_override = "ltx-2",
        ).name
        == "ltx-2"
    )


def test_validate_rejects_windows_shaped_missing_checkpoint(tmp_path):
    backend = VideoBackend()
    # A missing Windows-shaped local pick must fail HERE, not be treated as a Hub repo and fail after eviction.
    with pytest.raises(ValueError, match = "does not exist"):
        backend.validate_load_request(
            "C:\\models\\ltx.gguf",
            gguf_filename = "ltx.gguf",
            family_override = "ltx-2",
        )
    # A bare "org/name" Hub id (no path shape) is still left for the background load to resolve.
    fam = backend.validate_load_request(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx.gguf",
        family_override = "ltx-2",
    )
    assert fam.name == "ltx-2"


def test_validate_rejects_local_pipeline_without_model_index(tmp_path):
    backend = VideoBackend()
    d = tmp_path / "ltx-local"
    (d / "transformer").mkdir(parents = True)
    (d / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    # A local dir missing model_index.json is not a loadable pipeline; it must fail preflight BEFORE eviction.
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(str(d), family_override = "ltx-2")
    # With a model_index.json it is a valid local pipeline pick and passes preflight.
    (d / "model_index.json").write_text("{}")
    fam = backend.validate_load_request(str(d), family_override = "ltx-2")
    assert fam.name == "ltx-2"


def test_validate_rejects_local_file_picked_as_pipeline(tmp_path):
    backend = VideoBackend()
    # A local FILE sent as a pipeline is not a diffusers directory, so gate on .exists() (not .is_dir()) to catch files too.
    f = tmp_path / "ltx-2.safetensors"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(str(f), model_kind = "pipeline", family_override = "ltx-2")


def test_validate_rejects_local_base_repo_without_model_index(tmp_path):
    backend = VideoBackend()
    # A local base_repo dir that is NOT a diffusers pipeline passes the trust check but loads via from_pretrained, so reject it here.
    bad_base = tmp_path / "bare-base"
    bad_base.mkdir()
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(
            "unsloth/LTX-2.3-GGUF",
            gguf_filename = "x.gguf",
            model_kind = "gguf",
            base_repo = str(bad_base),
        )
    # A local base_repo that IS a real pipeline dir passes the gate.
    (bad_base / "model_index.json").write_text("{}")
    fam = backend.validate_load_request(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "x.gguf",
        model_kind = "gguf",
        base_repo = str(bad_base),
    )
    assert fam.name == "ltx-2"


def test_validate_rejects_gguf_repo_as_pipeline():
    backend = VideoBackend()
    # A -GGUF repo with no quant filename resolves to pipeline kind and would only fail minutes later, after eviction.
    with pytest.raises(ValueError, match = "pick one of its .gguf files"):
        backend.validate_load_request("unsloth/LTX-2.3-GGUF")
    with pytest.raises(ValueError, match = "pick one of its .gguf files"):
        backend.validate_load_request("unsloth/Wan2.2-TI2V-5B-GGUF/")


def test_detect_load_family_filename_fallback():
    # Repo id alone carries the family.
    fam = _detect_load_family("Lightricks/LTX-2", None, None)
    assert fam is not None and fam.name == "ltx-2"
    # Repo id is opaque but the picked filename carries the family: fall back to the combined path so validate and _run_load agree.
    fam = _detect_load_family("someorg/quants", "ltx-2-19b-Q4_K_M.gguf", None)
    assert fam is not None and fam.name == "ltx-2"
    # No filename and no recognisable repo id: no family.
    assert _detect_load_family("someorg/quants", None, None) is None
    # An explicit override resolves by name/alias and skips the filename fallback: a bogus override stays None.
    fam = _detect_load_family("someorg/quants", "ltx-2-19b-Q4_K_M.gguf", "ltxv")
    assert fam is not None and fam.name == "ltx-2"
    assert _detect_load_family("someorg/quants", "ltx-2-19b-Q4_K_M.gguf", "bogus") is None


def test_detect_load_family_cached_hub_arch_fallback(monkeypatch):
    # A cached HUB GGUF is admitted to the picker by its architecture, but an opaque repo id + renamed file carry no family token, so the cache fallback keeps a supported pick loadable.
    import huggingface_hub

    import utils.models.gguf_metadata as gguf_meta

    # No local file at Path(repo_id)/filename; resolve the arch from the cached blob instead.
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda repo_id, filename, **kw: "/fake/cache/blobs/model.gguf",
    )
    monkeypatch.setattr(
        gguf_meta, "read_gguf_general_metadata", lambda path: {"general.architecture": "ltxv"}
    )
    fam = _detect_load_family("someorg/opaque-quants", "model.gguf", None)
    assert fam is not None and fam.name == "ltx-2"

    # A cache MISS (blob not present -> None) still yields None (400 exactly as before).
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    assert _detect_load_family("someorg/opaque-quants", "model.gguf", None) is None

    # A recognised-but-unsupported video arch stays None, so an unsupported cached pick 400s like the local-dir case.
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/fake/cache/blobs/model.gguf"
    )
    monkeypatch.setattr(
        gguf_meta, "read_gguf_general_metadata", lambda path: {"general.architecture": "wan"}
    )
    assert _detect_load_family("someorg/opaque-quants", "model.gguf", None) is None

    # The blob lives in a NON-active cache root: the active probe misses, but the per-root probe finds it.
    import hub.utils.paths as hub_paths

    monkeypatch.setattr(hub_paths, "legacy_hf_cache_dir", lambda: "/fake/legacy")
    monkeypatch.setattr(hub_paths, "hf_default_cache_dir", lambda: "/fake/default")
    monkeypatch.setattr(
        gguf_meta, "read_gguf_general_metadata", lambda path: {"general.architecture": "ltxv"}
    )
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        # Active root (cache_dir absent) misses; only the legacy/default roots have the blob.
        lambda repo_id, filename, cache_dir = None: (
            "/fake/legacy/blobs/model.gguf" if cache_dir else None
        ),
    )
    fam = _detect_load_family("someorg/opaque-quants", "model.gguf", None)
    assert fam is not None and fam.name == "ltx-2"


def test_loading_repo_ids_guards_in_flight_delete():
    # During a background load status() is still False but the repo is downloading, so the delete guard needs loading_repo_ids.
    from core.inference.video import _VideoLoadingState

    backend = VideoBackend()
    assert backend.loading_repo_ids() == ()  # idle: nothing to guard
    backend._loading = _VideoLoadingState(repo_id = "org/ckpt", base_repo = "Lightricks/LTX-2")
    assert set(backend.loading_repo_ids()) == {"org/ckpt", "Lightricks/LTX-2"}
    # An errored load is no longer in flight, so the files are safe to delete.
    backend._loading = _VideoLoadingState(
        repo_id = "org/ckpt", base_repo = "Lightricks/LTX-2", error = "boom"
    )
    assert backend.loading_repo_ids() == ()
    # A load whose base equals the repo (or is empty) yields just the one id.
    backend._loading = _VideoLoadingState(repo_id = "org/ckpt", base_repo = "")
    assert backend.loading_repo_ids() == ("org/ckpt",)


def test_load_generate_unload_gguf(fake_runtime, tmp_path):
    backend = VideoBackend()
    status = _load_gguf(backend, tmp_path)
    assert status["loaded"] is True and status["family"] == "ltx-2"
    assert status["model_kind"] == "gguf"
    assert status["has_audio"] is True
    # The GGUF transformer is dequant-configured and assembled onto the base repo.
    assert _FakeTransformer.last["path"].endswith("model.gguf")
    assert _FakeTransformer.last["quantization_config"][0] == "quant"
    assert _FakePipeline.last["base"] == "Lightricks/LTX-2"
    assert "transformer" in _FakePipeline.last
    # Video decode is the memory peak: tiling is always on.
    assert status["vae_tiling"] is True
    assert status["defaults"]["frame_step"] == 8

    result = backend.generate(
        prompt = "a sloth surfing", width = 1000, height = 700, num_frames = 120, fps = 24
    )
    call = backend._state.pipe.last_kwargs
    # Shape snapping happened BEFORE the pipe call: /32 sizes, 8k+1 frames.
    assert (call["width"], call["height"]) == (992, 672)
    assert call["num_frames"] == 113
    assert call["frame_rate"] == 24.0
    assert result["mp4_bytes"] == b"MP4"
    assert result["num_frames"] == 113 and result["fps"] == 24
    assert result["has_audio"] is False  # fake pipe returned no audio track
    assert 0 <= result["seed"] < 2**53

    status = backend.unload()
    assert status["loaded"] is False


def test_load_holds_generate_lock_across_placement(fake_runtime, tmp_path, monkeypatch):
    # The video load must hold _generate_lock across GPU placement so an unload -- which barriers on that lock -- cannot
    # hand the GPU away mid-move. unload() must block until placement releases it, and the superseded load then aborts.
    import threading

    from core.inference import video as video_mod

    backend = VideoBackend()
    placement_started = threading.Event()
    release_placement = threading.Event()
    real_apply = video_mod.apply_memory_plan

    def blocking_apply(pipe, plan, **kw):
        placement_started.set()
        assert release_placement.wait(timeout = 5), "test placement barrier never released"
        return real_apply(pipe, plan, **kw)

    monkeypatch.setattr(video_mod, "apply_memory_plan", blocking_apply)

    load_exc = []

    def do_load():
        try:
            _load_gguf(backend, tmp_path)
        except Exception as e:  # noqa: BLE001 -- the concurrent unload supersedes this load
            load_exc.append(e)

    load_thread = threading.Thread(target = do_load)
    load_thread.start()
    assert placement_started.wait(timeout = 5), "load never reached placement"

    # Placement is in flight, holding _generate_lock, so unload() must block on its barrier.
    unload_done = []

    def do_unload():
        backend.unload()
        unload_done.append(True)

    unload_thread = threading.Thread(target = do_unload)
    unload_thread.start()
    unload_thread.join(timeout = 0.5)
    assert not unload_done, "unload() returned while placement still held _generate_lock (the race)"

    # Release placement; unload()'s barrier then passes and its teardown runs strictly AFTER the load's commit.
    release_placement.set()
    unload_thread.join(timeout = 5)
    load_thread.join(timeout = 5)
    assert unload_done, "unload() did not complete after placement released _generate_lock"
    assert not load_thread.is_alive() and not load_exc
    assert backend._state is None  # unload's teardown ran after the load, leaving nothing resident


def test_load_records_engaged_speed_optims(fake_runtime, tmp_path, monkeypatch):
    # Regression: the load tail once re-ran the already-filtered speed_optims tuple through ``.items()``, crashing every
    # real-GPU load. The fake runtime forces every optim False, so this only reproduces when one is made to engage.
    from core.inference import video as video_mod

    monkeypatch.setattr(
        video_mod,
        "apply_speed_optims",
        lambda *a, **k: {"channels_last": True, "cudnn_benchmark": False},
    )
    backend = VideoBackend()
    status = _load_gguf(backend, tmp_path)
    assert status["loaded"] is True
    assert status["speed_optims"] == ["channels_last"]


def test_generate_defaults_from_variant(fake_runtime, tmp_path):
    # A distilled GGUF pick defaults to the few-step no-CFG schedule.
    (tmp_path / "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf").write_bytes(b"w")
    backend = VideoBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )
    backend.generate(prompt = "a sloth")
    call = backend._state.pipe.last_kwargs
    assert call["num_inference_steps"] == 8
    assert call["guidance_scale"] == 1.0
    # At the distilled default step count the calibrated ltx_core curve is passed verbatim (the scheduler's own spacing lands far off).
    from core.inference.video_ltx2 import LTX23_DISTILLED_SIGMAS

    assert call["sigmas"] == list(LTX23_DISTILLED_SIGMAS)


def test_ltx23_load_forwards_the_precast_encoder(fake_runtime, tmp_path, monkeypatch):
    # The wiring half of the same bug: the 2.3 branch does not use pipe_kwargs, so the loader must pass the pre-cast encoder across explicitly.
    from core.inference import diffusion_te_prequant, video_ltx2

    precast = object()
    monkeypatch.setattr(
        diffusion_te_prequant, "te_prequant_pipe_kwargs", lambda *a, **k: {"text_encoder": precast}
    )
    monkeypatch.setattr(video_ltx2, "is_ltx23_checkpoint", lambda path: True)
    seen: dict = {}

    def _assemble(checkpoint_path, **kwargs):
        seen.update(kwargs)
        return _FakePipeline.from_pretrained("Lightricks/LTX-2")

    monkeypatch.setattr(video_ltx2, "load_ltx23_pipeline", _assemble)

    (tmp_path / "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf").write_bytes(b"w")
    backend = VideoBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
        text_encoder_quant = "fp8",
    )
    assert seen["text_encoder"] is precast
    backend.unload()


def test_generate_distilled_custom_steps_keep_scheduler_spacing(fake_runtime, tmp_path):
    # A non-default step count has no calibrated list, so the scheduler spacing applies and no sigmas kwarg is injected.
    (tmp_path / "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf").write_bytes(b"w")
    backend = VideoBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )
    backend.generate(prompt = "a sloth", steps = 12)
    call = backend._state.pipe.last_kwargs
    assert call["num_inference_steps"] == 12
    assert call["sigmas"] is None


def test_generate_dev_base_never_gets_distilled_sigmas(fake_runtime, tmp_path):
    # The dev/base DiT uses the resolution-shifted scheduler spacing even at 8 steps: the calibrated list is distilled-only.
    (tmp_path / "ltx-2.3-22b-dev-Q4_K_M.gguf").write_bytes(b"w")
    backend = VideoBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "ltx-2.3-22b-dev-Q4_K_M.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )
    backend.generate(prompt = "a sloth", steps = 8)
    call = backend._state.pipe.last_kwargs
    assert call["num_inference_steps"] == 8
    assert call["sigmas"] is None


def test_ltx23_verbatim_sigmas_restores_scheduler_config():
    # The context manager must neutralise exactly the transforms that distort explicit sigmas, and restore them even on error.
    from core.inference.video_ltx2 import ltx23_verbatim_sigmas

    class _Cfg(dict):
        pass

    class _Sched:
        def __init__(self):
            self.config = _Cfg(use_dynamic_shifting = True, shift = 1.0, shift_terminal = 0.1)

        def register_to_config(self, **kw):
            self.config.update(kw)

    pipe = types.SimpleNamespace(scheduler = _Sched())
    with ltx23_verbatim_sigmas(pipe):
        assert pipe.scheduler.config["use_dynamic_shifting"] is False
        assert pipe.scheduler.config["shift_terminal"] is None
    assert pipe.scheduler.config["use_dynamic_shifting"] is True
    assert pipe.scheduler.config["shift_terminal"] == 0.1
    with pytest.raises(RuntimeError):
        with ltx23_verbatim_sigmas(pipe):
            raise RuntimeError("boom")
    assert pipe.scheduler.config["use_dynamic_shifting"] is True
    # A pipe without a scheduler is a no-op, not a crash.
    with ltx23_verbatim_sigmas(types.SimpleNamespace()):
        pass


def test_generate_resets_step_cache_only_when_engaged(fake_runtime, tmp_path):
    # FBCache residuals live on the long-lived DiT(s) and survive a generation, so the next clip at a new resolution would
    # crash on stale state. generate must reset them when a cache is engaged, and transformer_2 too when present.
    import dataclasses

    (tmp_path / "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf").write_bytes(b"w")
    backend = VideoBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )
    resets = []
    backend._state.pipe.transformer = types.SimpleNamespace(
        _reset_stateful_cache = lambda: resets.append("transformer")
    )
    backend._state.pipe.transformer_2 = types.SimpleNamespace(
        _reset_stateful_cache = lambda: resets.append("transformer_2")
    )
    # No cache engaged -> no reset.
    backend.generate(prompt = "a sloth")
    assert resets == []
    # Cache engaged: both resident DiTs reset before the pipe call.
    backend._state = dataclasses.replace(backend._state, transformer_cache = "fbcache")
    backend.generate(prompt = "a sloth")
    assert resets == ["transformer", "transformer_2"]


def test_is_ltx23_checkpoint_gguf(monkeypatch, tmp_path):
    # diffusers maps every LTX-2 single file to the 2.0 config, so a 2.3 checkpoint must be detected from its header; an unreadable header falls back, never raises.
    from core.inference.video_ltx2 import is_ltx23_checkpoint

    def _reader_for(shapes):
        tensors = [types.SimpleNamespace(name = n, shape = s) for n, s in shapes.items()]
        return lambda path: types.SimpleNamespace(tensors = tensors)

    gguf = types.ModuleType("gguf")
    # GGUF headers store dims in GGML (reversed) order.
    gguf.GGUFReader = _reader_for(
        {
            "model.diffusion_model.transformer_blocks.0.scale_shift_table": (4096, 9),
        }
    )
    monkeypatch.setitem(sys.modules, "gguf", gguf)
    path = tmp_path / "ltx23.gguf"
    path.write_bytes(b"x")
    assert is_ltx23_checkpoint(path) is True

    gguf.GGUFReader = _reader_for(
        {
            "model.diffusion_model.transformer_blocks.0.scale_shift_table": (4096, 6),
        }
    )
    assert is_ltx23_checkpoint(path) is False

    def _boom(path):
        raise RuntimeError("bad magic")

    gguf.GGUFReader = _boom
    assert is_ltx23_checkpoint(path) is False


def test_is_ltx23_checkpoint_safetensors(monkeypatch, tmp_path):
    from core.inference.video_ltx2 import is_ltx23_checkpoint

    class _FakeSlice:
        def __init__(self, shape):
            self._shape = shape

        def get_shape(self):
            return self._shape

    class _FakeSafe:
        def __init__(self, shapes):
            self._shapes = shapes

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def keys(self):
            return list(self._shapes)

        def get_slice(self, name):
            return _FakeSlice(self._shapes[name])

    shapes = {
        "model.diffusion_model.transformer_blocks.0.scale_shift_table": (9, 4096),
    }
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda path, framework = None: _FakeSafe(shapes)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    path = tmp_path / "ltx23.safetensors"
    path.write_bytes(b"x")
    assert is_ltx23_checkpoint(path) is True


def test_ltx23_split_and_variant(tmp_path):
    # Pure functions: combined-checkpoint partitioning and companion-set choice.
    from core.inference.video_ltx2 import _split_checkpoint, checkpoint_variant

    state = {
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": 1,
        "model.diffusion_model.video_embeddings_connector.learnable_registers": 2,
        "model.diffusion_model.prompt_adaln_single.linear.weight": 3,
        "text_embedding_projection.video_aggregate_embed.weight": 4,
        "vae.decoder.conv_in.weight": 5,
        "audio_vae.encoder.conv_in.weight": 6,
        "vocoder.bwe_generator.conv_pre.weight": 7,
    }
    groups = _split_checkpoint(state)
    assert set(groups["dit"]) == {
        "transformer_blocks.0.attn1.to_q.weight",
        "prompt_adaln_single.linear.weight",
    }
    assert set(groups["connectors"]) == {
        "video_embeddings_connector.learnable_registers",
        "text_embedding_projection.video_aggregate_embed.weight",
    }
    assert groups["vae"] == {"decoder.conv_in.weight": 5}
    assert groups["audio_vae"] == {"encoder.conv_in.weight": 6}
    assert groups["vocoder"] == {"bwe_generator.conv_pre.weight": 7}

    assert checkpoint_variant("x/ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf") == "distilled"
    assert checkpoint_variant("x/ltx-2.3-22b-dev-Q8_0.gguf") == "dev"


def test_ltx23_scaled_fp8_refused(monkeypatch, tmp_path):
    # The Lightricks fp8 files carry .weight_scale/.input_scale companions, so a plain dtype cast would corrupt them and the loader must refuse.
    from core.inference import video_ltx2

    # Stub the module tree so this also runs under the CI sim, which blocks the real diffusers import.
    diffusers = types.ModuleType("diffusers")
    diffusers.LTX2Pipeline = object
    loaders = types.ModuleType("diffusers.loaders")
    sfu = types.ModuleType("diffusers.loaders.single_file_utils")
    sfu.load_single_file_checkpoint = lambda path: {
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": object(),
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight_scale": object(),
    }
    diffusers.loaders = loaders
    loaders.single_file_utils = sfu
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.loaders", loaders)
    monkeypatch.setitem(sys.modules, "diffusers.loaders.single_file_utils", sfu)
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

    path = tmp_path / "ltx-2.3-22b-distilled-fp8.safetensors"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match = "scaled fp8"):
        video_ltx2.load_ltx23_pipeline(
            path, base_repo = "Lightricks/LTX-2", torch_dtype = None, is_gguf = False
        )


def _ltx23_assembly_stubs(monkeypatch, tmp_path):
    """Module tree + component loaders for a 2.3 assembly, so only the encoder choice is under test."""
    from core.inference import video_ltx2

    class _Loaded:
        def __init__(self, what: str) -> None:
            self.what = what

        @classmethod
        def from_pretrained(
            cls,
            base,
            subfolder = None,
            token = None,
            **extra,
        ):
            _Loaded.calls.append(subfolder)
            return cls(subfolder or "?")

    _Loaded.calls = []

    class _FakeLTX2Pipeline:
        last: dict = {}

        def __init__(self, **kwargs):
            _FakeLTX2Pipeline.last = kwargs

        @staticmethod
        def load_config(base_repo, token = None):
            return {
                "scheduler": ["diffusers", "_Loaded"],
                "tokenizer": ["transformers", "_Loaded"],
                "text_encoder": ["transformers", "_Loaded"],
            }

    diffusers = types.ModuleType("diffusers")
    diffusers.LTX2Pipeline = _FakeLTX2Pipeline
    diffusers._Loaded = _Loaded
    loaders = types.ModuleType("diffusers.loaders")
    sfu = types.ModuleType("diffusers.loaders.single_file_utils")
    sfu.load_single_file_checkpoint = lambda path: {
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": object(),
    }
    diffusers.loaders = loaders
    loaders.single_file_utils = sfu
    transformers = types.ModuleType("transformers")
    transformers._Loaded = _Loaded
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.loaders", loaders)
    monkeypatch.setitem(sys.modules, "diffusers.loaders.single_file_utils", sfu)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    monkeypatch.setattr(video_ltx2, "load_ltx23_transformer", lambda *a, **k: "dit")
    monkeypatch.setattr(video_ltx2, "load_ltx23_connectors", lambda *a, **k: "connectors")
    monkeypatch.setattr(video_ltx2, "load_ltx23_vae", lambda *a, **k: "vae")
    monkeypatch.setattr(
        video_ltx2, "load_ltx23_audio_vae_and_vocoder", lambda *a, **k: ("audio_vae", "vocoder")
    )

    path = tmp_path / "ltx-2.3-22b-distilled-Q4_K_M.gguf"
    path.write_bytes(b"x")
    return video_ltx2, _FakeLTX2Pipeline, _Loaded, path


def test_ltx23_assembly_takes_a_supplied_text_encoder(monkeypatch, tmp_path):
    # The 2.3 assembly builds every component itself, so an fp8 request only reaches it through this argument.
    video_ltx2, pipeline_cls, loaded, path = _ltx23_assembly_stubs(monkeypatch, tmp_path)
    precast = object()

    video_ltx2.load_ltx23_pipeline(
        path,
        base_repo = "Lightricks/LTX-2",
        torch_dtype = None,
        is_gguf = True,
        text_encoder = precast,
    )
    assert pipeline_cls.last["text_encoder"] is precast
    # Only the scheduler and tokenizer were fetched from the base repo.
    assert loaded.calls == ["scheduler", "tokenizer"]

    # No pre-cast encoder -> the dense one still loads from the base repo, as before.
    loaded.calls.clear()
    video_ltx2.load_ltx23_pipeline(
        path, base_repo = "Lightricks/LTX-2", torch_dtype = None, is_gguf = True
    )
    assert loaded.calls == ["scheduler", "tokenizer", "text_encoder"]
    assert isinstance(pipeline_cls.last["text_encoder"], loaded)


def test_generate_without_load_raises(fake_runtime):
    backend = VideoBackend()
    with pytest.raises(RuntimeError, match = VIDEO_NOT_LOADED_MSG):
        backend.generate(prompt = "x")


def test_generate_progress_and_cancel_idle(fake_runtime):
    backend = VideoBackend()
    # The idle shape carries the image-endpoint aliases (total_steps / fraction) so one poller works against both APIs.
    assert backend.generate_progress() == {"active": False, "total_steps": 0, "fraction": 0.0}
    assert backend.cancel_generate() is False


def test_generate_progress_derives_total_steps_and_fraction(fake_runtime):
    # A mid-denoise poll must report fraction = step / total under BOTH field names.
    backend = VideoBackend()
    backend._gen = {"active": True, "phase": "denoise", "step": 5, "total": 20}
    gen = backend.generate_progress()
    assert gen["total"] == 20 and gen["total_steps"] == 20
    assert gen["step"] == 5 and gen["fraction"] == 0.25


def test_failed_background_generate_retains_terminal_error(fake_runtime, tmp_path, monkeypatch):
    # A page mounted AFTER a background job failed reads the outcome from this retained terminal record, so a failure must stay pollable until the next job.
    backend = VideoBackend()
    _load_gguf(backend, tmp_path)

    def _boom(
        self,
        *,
        prompt = None,
        negative_prompt = None,
        num_inference_steps = None,
        guidance_scale = None,
        width = None,
        height = None,
        num_frames = None,
        frame_rate = None,
        generator = None,
        callback_on_step_end = None,
        **kwargs,
    ):
        raise ValueError("frames exceed the device memory")

    monkeypatch.setattr(type(backend._state.pipe), "__call__", _boom)
    backend.begin_generate(prompt = "a clip")
    deadline = time.monotonic() + 10
    while backend.generate_progress()["active"] and time.monotonic() < deadline:
        time.sleep(0.01)
    gen = backend.generate_progress()
    assert gen["active"] is False
    assert gen["phase"] == "failed"
    assert gen["error"] == "frames exceed the device memory"
    # Re-poll: a mount-time probe is a second read of the same record, not a one-shot drain.
    assert backend.generate_progress()["phase"] == "failed"


def test_cache_bytes_counts_incomplete_blobs(fake_runtime, tmp_path, monkeypatch):
    # scan_cache_dir skips in-flight *.incomplete blobs, so the counter froze for the whole shard pull. The walk must count both, without double-counting symlinks.
    import core.inference.video as video_mod

    repo_dir = tmp_path / "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "aa11").write_bytes(b"x" * 1000)  # completed blob
    (blobs / "bb22.incomplete").write_bytes(b"y" * 500)  # in-flight shard
    snap = repo_dir / "snapshots" / "deadbeef"
    snap.mkdir(parents = True)
    (snap / "model_index.json").symlink_to(blobs / "aa11")  # must not double-count
    # The live cache root, not huggingface_hub's import-time constant: the counter must follow a mid-session cache change.
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(tmp_path))

    backend = VideoBackend()
    assert backend._cache_bytes("Wan-AI/Wan2.2-TI2V-5B-Diffusers") == 1500
    assert backend._cache_bytes("Wan-AI/absent-repo") == 0
    assert backend._cache_bytes(None) == 0


def test_hv15_guider_and_scheduler_progress(fake_runtime):
    # HunyuanVideo-1.5: no guidance kwarg (CFG on the guider), no step callback (progress via the scheduler.step wrapper).
    backend = VideoBackend()
    status = backend.load_pipeline(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        model_kind = "pipeline",
    )
    assert status["family"] == "hunyuanvideo-1.5"
    assert status["has_audio"] is False
    assert status["defaults"]["frame_step"] == 4

    pipe = _FakeHV15Pipeline.instance
    result = backend.generate(
        prompt = "a fox in the snow", steps = 4, guidance = 3.5, num_frames = 9, fps = 24
    )
    assert "guidance_scale" not in pipe.last_kwargs
    assert "callback_on_step_end" not in pipe.last_kwargs
    assert pipe.guider.guidance_scale == 3.5
    # One wrapped tick per denoise step, then the original method back in place.
    assert pipe.scheduler.calls == 4
    assert pipe.scheduler.step.__func__ is _FakeHV15Scheduler.step
    assert result["num_frames"] == 9 and result["has_audio"] is False


def test_hv15_cancel_unwinds_scheduler_loop(fake_runtime):
    backend = VideoBackend()
    backend.load_pipeline(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        model_kind = "pipeline",
    )
    pipe = _FakeHV15Pipeline.instance
    # The cancel lands during the FIRST real step; the next wrapped call must raise out of the loop and generate() must surface the sentinel.
    pipe.scheduler.on_step = lambda n: backend.cancel_generate() if n == 1 else None
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(prompt = "a fox", steps = 4)
    assert pipe.scheduler.calls == 1
    # The wrapper must restore scheduler.step even on the exception path.
    assert pipe.scheduler.step.__func__ is _FakeHV15Scheduler.step
    # The exception unwound pipe.__call__ before its own cleanup, so generate() must have freed the offload hooks itself.
    assert pipe.hooks_freed == 1


def test_cancel_during_export_discards_clip(fake_runtime, monkeypatch):
    # A cancel during the blocking export/mux must still discard the clip: cancel_generate() already reported success.
    backend = VideoBackend()
    backend.load_pipeline(
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        model_kind = "pipeline",
    )

    def _encode_and_cancel(frames, fps, audio, pipe):
        backend.cancel_generate()  # cancel arrives mid-mux, after the last denoise-step check
        return b"MP4"

    monkeypatch.setattr(VideoBackend, "_encode_mp4", staticmethod(_encode_and_cancel))
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(prompt = "a fox", steps = 4)


def test_singleton():
    assert get_video_backend() is get_video_backend()


# ── Wan2.2 ─────────────────────────────────────────────────────────────────────


def test_load_wan_ti2v_5b_pipeline(fake_runtime):
    # A full-pipeline load of the single-DiT TI2V-5B repo: WanPipeline.from_pretrained, no audio, tiling forced on, 4k+1 frame lattice.
    backend = VideoBackend()
    status = backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert status["loaded"] is True
    assert status["family"] == "wan2.2-ti2v-5b"
    assert status["model_kind"] == "pipeline"
    assert status["has_audio"] is False
    assert status["vae_tiling"] is True
    assert status["defaults"]["frame_step"] == 4
    assert status["transformer_quant"] is None
    assert _FakeWanPipelineSingle.last["repo"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def test_video_dense_speed_defaults_to_compile_profile(fake_runtime):
    # A clip denoise amortises the compile within one run, so an UNSET speed on a dense load resolves to `default` -- never max, never off.
    backend = VideoBackend()
    status = backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert status["speed_mode"] == "default"
    assert status["resolved"]["speed_mode"]["source"] == "auto"
    backend.unload()
    status_off = backend.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline", speed_mode = "off"
    )
    assert status_off["speed_mode"] == "off"
    assert status_off["resolved"]["speed_mode"]["source"] == "explicit"


def test_video_speed_off_suppresses_auto_dtype_quant(fake_runtime, monkeypatch):
    # An explicit Speed="off" load with Precision on auto must NOT promote to auto-quant, which would break the bit-exact request.
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    calls: list = []
    monkeypatch.setattr(
        video_mod, "quantize_transformer", lambda view, target, **kw: calls.append(True) or "int8"
    )

    backend = VideoBackend()
    # speed=off + precision auto (unset): no auto-quant, speed stays off (bit-exact).
    status = backend.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline", speed_mode = "off"
    )
    assert calls == []  # quantize_transformer never ran
    assert status["transformer_quant"] is None
    assert status["speed_mode"] == "off"
    backend.unload()

    # Control: with speed NOT off the auto precision promotion still engages, so the suppression above is specific to speed=off.
    backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert calls == [True]


def test_video_step_cache_auto_from_default_schedule(fake_runtime, tmp_path):
    # Unset step cache is AUTO, from the model's default schedule: Wan's 50-step default engages FBCache, LTX's 8-step does not.
    backend = VideoBackend()
    status = backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert status["transformer_cache"] == "fbcache"
    assert status["resolved"]["transformer_cache"]["source"] == "auto"
    backend.unload()

    (tmp_path / "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf").write_bytes(b"w")
    status2 = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )
    assert status2["transformer_cache"] is None
    assert status2["resolved"]["transformer_cache"]["source"] == "auto"
    backend.unload()


def test_video_step_cache_auto_toggles_on_actual_steps(fake_runtime):
    # The AUTO decision follows each generation's ACTUAL step count; an explicit "off" never toggles.
    backend = VideoBackend()
    backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert backend.status()["transformer_cache"] == "fbcache"
    backend.generate(prompt = "a sloth", steps = 8)
    assert backend.status()["transformer_cache"] is None
    backend.generate(prompt = "a sloth", steps = 30)
    assert backend.status()["transformer_cache"] == "fbcache"
    backend.unload()

    backend.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline", transformer_cache = "off"
    )
    assert backend.status()["transformer_cache"] is None
    backend.generate(prompt = "a sloth", steps = 30)
    assert backend.status()["transformer_cache"] is None
    backend.unload()


def test_wan_frame_snapping_4k_plus_1(fake_runtime):
    # Wan snaps num_frames to 4k+1 (temporal factor 4), unlike LTX-2's 8k+1.
    backend = VideoBackend()
    backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    backend.generate(prompt = "a sloth", width = 1000, height = 700, num_frames = 120)
    call = backend._state.pipe.last_kwargs
    assert call["num_frames"] == 117  # 4*29 + 1
    # /32 spatial snap for TI2V-5B: its VAE is 16x spatial * patch 2 = 32, so 1000x700 gives 992x672 (not the /16 992x688).
    assert (call["width"], call["height"]) == (992, 672)


def test_wan_ti2v_defaults_applied(fake_runtime):
    # No steps/guidance passed -> the Wan pipeline defaults (50 / 5.0).
    backend = VideoBackend()
    backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    backend.generate(prompt = "a sloth")
    call = backend._state.pipe.last_kwargs
    assert call["num_inference_steps"] == 50
    assert call["guidance_scale"] == 5.0


def test_wan_ti2v_does_not_thread_cfg2(fake_runtime):
    # The single-DiT TI2V pipeline has no guidance_scale_2, so a request value must NOT be threaded (WanPipeline raises).
    backend = VideoBackend()
    backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    backend.generate(prompt = "a sloth", guidance_2 = 3.5)
    call = backend._state.pipe.last_kwargs
    assert "guidance_scale_2" not in call


def test_wan_a14b_dual_dit_pipeline_loads(fake_runtime):
    # The A14B repo builds a dual-DiT MoE pipeline (transformer + transformer_2).
    backend = VideoBackend()
    status = backend.load_pipeline("Wan-AI/Wan2.2-T2V-A14B-Diffusers", model_kind = "pipeline")
    assert status["loaded"] is True and status["family"] == "wan2.2-t2v-a14b"
    pipe = backend._state.pipe
    assert pipe.transformer is not None and pipe.transformer_2 is not None


def test_wan_a14b_cfg2_threaded_when_signature_has_it(fake_runtime):
    # The MoE pipeline __call__ carries guidance_scale_2, so an explicit guidance_2 is threaded as that kwarg.
    backend = VideoBackend()
    backend.load_pipeline("Wan-AI/Wan2.2-T2V-A14B-Diffusers", model_kind = "pipeline")
    backend.generate(prompt = "a sloth", guidance = 5.0, guidance_2 = 3.0)
    call = backend._state.pipe.last_kwargs
    assert call["guidance_scale"] == 5.0
    assert call["guidance_scale_2"] == 3.0

    # A None guidance_2 must NOT be threaded, so the pipeline defaults it itself.
    backend.generate(prompt = "a sloth", guidance = 5.0)
    call2 = backend._state.pipe.last_kwargs
    assert call2["guidance_scale_2"] is None


def test_wan_a14b_step_cache_applies_to_both_dits(fake_runtime):
    # A dual-DiT MoE load must engage the step cache on BOTH experts: transformer_2 would otherwise run uncached.
    backend = VideoBackend()
    status = backend.load_pipeline(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        model_kind = "pipeline",
        transformer_cache = "fbcache",
    )
    pipe = backend._state.pipe
    assert pipe.transformer.cache_config is not None
    assert pipe.transformer_2.cache_config is not None
    assert status["transformer_cache"] == "fbcache"


def test_wan_a14b_attention_applies_to_both_dits(fake_runtime, monkeypatch):
    # An explicit attention backend must be set on both experts. The fake runtime is CPU, where the NVIDIA gate drops explicit kernels, so pin it open.
    from core.inference import diffusion_attention as attn_mod

    monkeypatch.setattr(attn_mod, "_is_cuda_nvidia", lambda target: True)
    backend = VideoBackend()
    backend.load_pipeline(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        model_kind = "pipeline",
        attention_backend = "cudnn",
    )
    pipe = backend._state.pipe
    assert pipe.transformer.attention is not None
    assert pipe.transformer_2.attention is not None
    # Both experts got the SAME kernel.
    assert pipe.transformer.attention == pipe.transformer_2.attention


def test_wan_ti2v_single_dit_only_touches_one(fake_runtime):
    # A single-DiT load must not fabricate a second expert or try to optimise one.
    backend = VideoBackend()
    backend.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_kind = "pipeline",
        transformer_cache = "fbcache",
    )
    pipe = backend._state.pipe
    assert pipe.transformer_2 is None
    assert pipe.transformer.cache_config is not None


def test_wan_a14b_dense_quant_applies_to_both_dits(fake_runtime, monkeypatch):
    # transformer_quant on a pipeline load quantises the dense DiT(s). Stub the quant seams to record which pipe view each helper saw: BOTH experts must be quantised.
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    quantised = []

    def _fake_quant(
        view,
        target,
        *,
        mode,
        family,
        logger = None,
    ):
        # The helper reads view.transformer; record what it would quantise to prove the second expert was reached.
        quantised.append(view.transformer)
        return "int8"

    monkeypatch.setattr(video_mod, "quantize_transformer", _fake_quant)

    backend = VideoBackend()
    status = backend.load_pipeline(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        model_kind = "pipeline",
        transformer_quant = "int8",
    )
    pipe = backend._state.pipe
    # Both experts were passed to quantize_transformer, in that order.
    assert quantised == [pipe.transformer, pipe.transformer_2]
    assert status["transformer_quant"] == "int8"


def test_dense_quant_skipped_under_offload(fake_runtime, monkeypatch):
    # Offload hooks move modules with Module.to(), which torchao tensors reject, so any offload policy must SKIP quant: the load succeeds dense and the record explains why.
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    quantised = []

    def _fake_quant(
        view,
        target,
        *,
        mode,
        family,
        logger = None,
    ):
        quantised.append(view.transformer)
        return "int8"

    monkeypatch.setattr(video_mod, "quantize_transformer", _fake_quant)
    # The CPU fake target never plans an offload, so force one at the plan seam and stub the apply step.
    import dataclasses

    real_plan = video_mod.plan_diffusion_memory
    monkeypatch.setattr(
        video_mod,
        "plan_diffusion_memory",
        lambda **kwargs: dataclasses.replace(real_plan(**kwargs), offload_policy = "model"),
    )
    monkeypatch.setattr(
        video_mod,
        "apply_memory_plan",
        lambda pipe, plan, device = None, logger = None: ("model", True),
    )

    backend = VideoBackend()
    status = backend.load_pipeline(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        model_kind = "pipeline",
        transformer_quant = "int8",
    )
    assert status["offload_policy"] == "model"
    assert quantised == []
    assert status["transformer_quant"] is None
    assert "offload moves the DiT" in status["resolved"]["transformer_quant"]["reason"]


def test_wan_a14b_partial_quant_fails_the_load(fake_runtime, monkeypatch):
    # If the first expert quantises but the second does not, the pipe is left at mismatched precision with no way back, so the load must fail cleanly.
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    outcomes = iter(["int8", None])
    monkeypatch.setattr(
        video_mod,
        "quantize_transformer",
        lambda view, target, *, mode, family, logger = None: next(outcomes),
    )

    backend = VideoBackend()
    with pytest.raises(RuntimeError, match = "1/2 experts"):
        backend.load_pipeline(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            model_kind = "pipeline",
            transformer_quant = "int8",
        )
    assert backend.status()["loaded"] is False


def test_wan_ti2v_dense_quant_applies_to_single_dit(fake_runtime, monkeypatch):
    # A single-DiT pipeline load quantises exactly one transformer.
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    quantised = []

    def _fake_quant(
        view,
        target,
        *,
        mode,
        family,
        logger = None,
    ):
        quantised.append(view.transformer)
        return "fp8"

    monkeypatch.setattr(video_mod, "quantize_transformer", _fake_quant)

    backend = VideoBackend()
    status = backend.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_kind = "pipeline",
        transformer_quant = "fp8",
    )
    assert quantised == [backend._state.pipe.transformer]
    assert status["transformer_quant"] == "fp8"


def test_wan_validate_trusted_repos(fake_runtime):
    # The two Wan base repos are trusted for non-GGUF (pipeline) loads; an unrelated repo carrying the family name is not.
    backend = VideoBackend()
    fam = backend.validate_load_request("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert fam.name == "wan2.2-ti2v-5b"
    fam2 = backend.validate_load_request("Wan-AI/Wan2.2-T2V-A14B-Diffusers", model_kind = "pipeline")
    assert fam2.name == "wan2.2-t2v-a14b"
    with pytest.raises(ValueError, match = "limited to"):
        backend.validate_load_request("evil/wan2.2-ti2v-5b-repack", model_kind = "pipeline")
    # A bad transformer_quant scheme is rejected cheaply at validate time.
    with pytest.raises(ValueError, match = "transformer_quant"):
        backend.validate_load_request(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            model_kind = "pipeline",
            transformer_quant = "bogus",
        )


def test_wan_a14b_refuses_single_file_loads(fake_runtime):
    # A single checkpoint carries only one of the A14B experts and the other would load dense outside the memory plan, so validate refuses it.
    backend = VideoBackend()
    with pytest.raises(ValueError, match = "dual-expert"):
        backend.validate_load_request(
            "QuantStack/Wan2.2-T2V-A14B-GGUF",
            gguf_filename = "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
        )
    # The single-DiT 5B family still accepts GGUF.
    fam = backend.validate_load_request(
        "unsloth/Wan2.2-TI2V-5B-GGUF",
        gguf_filename = "Wan2.2-TI2V-5B-Q4_K_M.gguf",
    )
    assert fam.name == "wan2.2-ti2v-5b"


def test_second_dit_view_write_through():
    # Attribute writes on the proxy must land on the real pipe; a ``transformer`` write mirrors onto the second expert.
    from core.inference.video import _SecondDiTView

    pipe = types.SimpleNamespace(transformer = "t1", transformer_2 = "t2", flag = None)
    view = _SecondDiTView(pipe)
    assert view.transformer == "t2"
    view.transformer = "t2-compiled"
    assert pipe.transformer_2 == "t2-compiled" and pipe.transformer == "t1"
    view.flag = "set"
    assert pipe.flag == "set"


# ── scoped base-repo download ─────────────────────────────────────────────────


def _sibling(name, size):
    return types.SimpleNamespace(rfilename = name, size = size)


_LTX2_SIBLINGS = [
    _sibling("model_index.json", 10),
    _sibling("ltx-2-19b-packaged-fp8.safetensors", 170),
    _sibling("transformer/config.json", 1),
    _sibling("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 20),
    _sibling("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 18),
    _sibling("text_encoder/model-00001-of-00002.safetensors", 25),
    _sibling("text_encoder/model-00002-of-00002.safetensors", 25),
    _sibling("text_encoder/diffusion_pytorch_model-00001-of-00002.safetensors", 25),
    _sibling("text_encoder/diffusion_pytorch_model-00002-of-00002.safetensors", 25),
    _sibling("vae/diffusion_pytorch_model.safetensors", 3),
    _sibling("tokenizer/tokenizer.model", 1),
    _sibling("tokenizer/chat_template.jinja", 1),
    _sibling("assets/example.mp4", 500),
]


def test_base_download_files_scopes_pipeline_pull():
    # A pipeline load skips the packaged root checkpoint, duplicate encoder shards and non-weight assets.
    info = types.SimpleNamespace(siblings = _LTX2_SIBLINGS)
    files = dict(VideoBackend._base_download_files(info, "pipeline"))
    assert "ltx-2-19b-packaged-fp8.safetensors" not in files
    assert "text_encoder/diffusion_pytorch_model-00001-of-00002.safetensors" not in files
    assert "assets/example.mp4" not in files
    assert files["text_encoder/model-00001-of-00002.safetensors"] == 25
    assert files["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"] == 20
    # The standalone chat template must survive the whitelist: apply_chat_template reads it at generation time.
    assert "tokenizer/chat_template.jinja" in files
    assert sum(files.values()) == 10 + 1 + 20 + 18 + 25 + 25 + 3 + 1 + 1


def test_base_download_files_gguf_drops_transformer():
    # A GGUF/single-file checkpoint replaces the DiT: the base transformer never pulls.
    info = types.SimpleNamespace(siblings = _LTX2_SIBLINGS)
    names = [n for n, _ in VideoBackend._base_download_files(info, "gguf")]
    assert not any(n.startswith("transformer/") for n in names)
    assert "text_encoder/model-00001-of-00002.safetensors" in names


def test_load_progress_clamps_overshoot(fake_runtime, monkeypatch):
    # The cache scan counts blobs a broader previous pull left behind, so the counter must never exceed the scoped estimate.
    backend = VideoBackend()
    backend._loading = types.SimpleNamespace(
        repo_id = "Lightricks/LTX-2", base_repo = None, expected_bytes = 100, error = None
    )
    monkeypatch.setattr(VideoBackend, "_cache_bytes", lambda self, repo: 150)
    progress = backend.load_progress()
    assert progress["phase"] == "finalizing"
    assert progress["downloaded_bytes"] == 100
    assert progress["expected_bytes"] == 100


def test_pipeline_load_uses_predownloaded_dir(fake_runtime, tmp_path):
    # When the scoped pre-download produced a local snapshot, from_pretrained must receive that dir, keeping its own sweep off the hub.
    backend = VideoBackend()
    backend.load_pipeline(
        "Lightricks/LTX-2",
        model_kind = "pipeline",
        _base_local_dir = str(tmp_path),
    )
    assert _FakePipeline.last["base"] == str(tmp_path)
    backend.unload()


def test_base_download_files_ltx23_keeps_only_shared_components():
    # A 2.3 checkpoint supplies the DiT, connectors, both VAEs and the vocoder, so the base pull shrinks to scheduler + TE + tokenizer.
    siblings = _LTX2_SIBLINGS + [
        _sibling("scheduler/scheduler_config.json", 1),
        _sibling("connectors/diffusion_pytorch_model.safetensors", 3),
        _sibling("latent_upsampler/diffusion_pytorch_model.safetensors", 1),
    ]
    info = types.SimpleNamespace(siblings = siblings)
    names = [n for n, _ in VideoBackend._base_download_files(info, "gguf", ltx23 = True)]
    assert "model_index.json" in names
    assert "scheduler/scheduler_config.json" in names
    assert "text_encoder/model-00001-of-00002.safetensors" in names
    assert "tokenizer/tokenizer.model" in names
    assert not any(
        n.startswith(("vae/", "connectors/", "latent_upsampler/", "transformer/")) for n in names
    )


def test_hv15_720p_repo_gets_720p_family_defaults():
    # The 720p repack is trusted but must resolve its OWN family entry: the generic entry would default to 832x480.
    from core.inference.video_families import detect_video_family

    fam = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v")
    assert fam is not None and fam.name == "hunyuanvideo-1.5-720p"
    assert fam.resolution_presets[0] == (1280, 720)
    assert fam.base_repo.endswith("720p_t2v")
    # The 480p repo keeps the original entry.
    fam480 = detect_video_family("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    assert fam480 is not None and fam480.name == "hunyuanvideo-1.5"
    assert fam480.resolution_presets[0] == (832, 480)


def test_predownload_base_honors_cancel_between_files(monkeypatch):
    # A warm-cache sweep returns each file without consulting the event, so the loop must check it explicitly.
    backend = VideoBackend()
    backend._cancel_event.set()
    calls: list = []
    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, fn, tok, **kw: (calls.append(fn), f"/cache/{fn}")[1],
    )

    class _Api:
        def __init__(self, token = None):
            pass

        def model_info(
            self,
            repo,
            files_metadata = True,
        ):
            return types.SimpleNamespace(
                siblings = [
                    _sibling("model_index.json", 1),
                    _sibling("vae/diffusion_pytorch_model.safetensors", 2),
                ]
            )

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend._predownload_base("base/repo", None, "pipeline")
    assert calls == []


def test_detect_load_family_arch_fallback_for_local_gguf(tmp_path, monkeypatch):
    # A local GGUF may carry no family token in its path, so the loader must resolve the family from the arch, not the name.
    from core.inference import video as vid
    from core.inference.video_families import detect_video_family

    d = tmp_path / "my-videos"
    d.mkdir()
    (d / "model.gguf").write_bytes(b"GGUF")  # exists; content irrelevant (reader is patched)

    # Name-only detection misses it (no "ltx" token in the path or filename).
    assert detect_video_family(str(d)) is None
    assert detect_video_family(f"{d}/model.gguf") is None

    # ltxv arch resolves to the ltx-2 family via the arch fallback.
    monkeypatch.setattr(
        "utils.models.gguf_metadata.read_gguf_general_metadata",
        lambda p: {"general.architecture": "ltxv"},
    )
    fam = vid._detect_load_family(str(d), "model.gguf", None)
    assert fam is not None and fam.name == "ltx-2"

    # A video arch with no backend family (wan) stays None, so the loader 400s as before.
    monkeypatch.setattr(
        "utils.models.gguf_metadata.read_gguf_general_metadata",
        lambda p: {"general.architecture": "wan"},
    )
    assert vid._detect_load_family(str(d), "model.gguf", None) is None

    # An explicit family_override skips the arch read entirely (worker parity).
    monkeypatch.setattr(
        "utils.models.gguf_metadata.read_gguf_general_metadata",
        lambda p: {"general.architecture": "ltxv"},
    )
    assert vid._detect_load_family(str(d), "model.gguf", "ltx-2").name == "ltx-2"


class _PlanSibling:
    def __init__(self, rfilename: str, size: int) -> None:
        self.rfilename = rfilename
        self.size = size


class _PlanInfo:
    def __init__(self, siblings) -> None:
        self.siblings = siblings


_LTX_BASE_SIBLINGS = [
    _PlanSibling("model_index.json", 1000),
    _PlanSibling("scheduler/scheduler_config.json", 1000),
    _PlanSibling("tokenizer/tokenizer.json", 5_000_000),
    _PlanSibling("text_encoder/model-00001-of-00005.safetensors", 10_000_000_000),
    _PlanSibling("vae/diffusion_pytorch_model.safetensors", 2_400_000_000),
    _PlanSibling("vocoder/diffusion_pytorch_model.safetensors", 200_000_000),
    _PlanSibling("connectors/diffusion_pytorch_model.safetensors", 2_900_000_000),
    _PlanSibling("transformer/diffusion_pytorch_model.safetensors", 37_800_000_000),
]

_LTX23_REPO_SIBLINGS = [
    _PlanSibling("ltx-2.3-22b-distilled.gguf", 12_000_000_000),
    _PlanSibling("vae/ltx-2.3-22b-distilled_video_vae.safetensors", 2_400_000_000),
    _PlanSibling("vae/ltx-2.3-22b-distilled_audio_vae.safetensors", 200_000_000),
    _PlanSibling(
        "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors", 900_000_000
    ),
    _PlanSibling("vae/ltx-2.3-22b-dev_video_vae.safetensors", 2_400_000_000),
]


def _plan_api(monkeypatch, repos):
    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _PlanInfo(repos[repo_id])

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())


def test_h3_native_download_plan_stages_the_complete_runtime(monkeypatch):
    # The mirror carries the Qwen3-VL encoder quants alongside the denoisers, so one repo covers
    # both halves of the runtime; the VAEs still come from the component repo.
    _plan_api(
        monkeypatch,
        {
            "unsloth/MiniMax-H3-GGUF": [
                _PlanSibling("minimax_h3_fl2va-Q4_K_M.gguf", 19),
                _PlanSibling("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", 18),
            ],
            "Comfy-Org/MiniMax-H3": [
                _PlanSibling("vae/minimax_h3_video_vae_fp16.safetensors", 5),
                _PlanSibling("vae/minimax_h3_audio_vae_fp32.safetensors", 1),
            ],
        },
    )

    plan = VideoBackend().download_plan(
        "unsloth/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        family_override = "minimax-h3",
        model_kind = "gguf",
    )

    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert by_repo["unsloth/MiniMax-H3-GGUF"]["files"] == [
        "minimax_h3_fl2va-Q4_K_M.gguf",
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    ]
    assert by_repo["Comfy-Org/MiniMax-H3"]["files"] == [
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ]
    assert plan["total_bytes"] == 43


def test_direct_h3_native_load_uses_sd_cpp_path(monkeypatch):
    backend = VideoBackend()
    calls = []
    monkeypatch.setattr("core.inference.video._ensure_mp4_encoder_available", lambda: None)
    monkeypatch.setattr(
        backend,
        "_run_load_h3_native",
        lambda **kwargs: calls.append(kwargs),
    )

    result = backend.load_pipeline(
        "leejet/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        family_override = "minimax-h3",
        model_kind = "gguf",
    )

    assert result["loaded"] is False
    assert len(calls) == 1
    assert calls[0]["fam"].name == "minimax-h3"
    assert calls[0]["gguf_filename"] == "minimax_h3_fl2va-Q4_K_M.gguf"


def test_h3_native_load_honors_install_switch_and_maps_xpu_to_vulkan(monkeypatch, tmp_path):
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend, sd_cpp_engine

    class _Api:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, *_args, **_kwargs):
            return _PlanInfo([])

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "xpu", device = "xpu", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: False)

    install_calls = []

    def _ensure(*, allow_install, accelerator):
        install_calls.append((allow_install, accelerator))
        return "/existing/sd-cli" if accelerator == "cpu" else None

    monkeypatch.setattr(sd_cpp_backend, "ensure_sd_cpp_binary", _ensure)

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            # Only "not None" matters here: the load path treats that as "the engine started".
            # Deliberately not a real tag, so nobody reads this as a second pin competing with
            # install_sd_cpp_prebuilt.DEFAULT_TAG, which is what the previous literal looked like
            # once that pin moved on without it.
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    def _download(_repo, wanted, *_args, **_kwargs):
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

    backend = VideoBackend()
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    backend._run_load_h3_native(
        fam = fam,
        token = None,
        cancel_event = threading.Event(),
        repo_id = "leejet/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
    )

    assert install_calls == [(False, "vulkan"), (False, "cpu")]
    assert backend._state is not None
    assert backend._state.device == "cpu"


def test_h3_native_generation_dispatch_does_not_import_torch(monkeypatch):
    from core.inference.video import _VideoLoadState

    backend = VideoBackend()
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    backend._state = _VideoLoadState(
        pipe = object(),
        family = fam,
        repo_id = "leejet/MiniMax-H3-GGUF",
        base_repo = fam.base_repo,
        device = "cpu",
        dtype = "Q4_K_M",
        kind = "gguf",
        engine = "sd_cpp",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
    )
    expected = {"mp4_bytes": b"native"}
    monkeypatch.setattr(backend, "_generate_h3_native", lambda **_kwargs: expected)

    original_import = builtins.__import__

    def _no_torch_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("native generation imported torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch_import)
    assert backend.generate(prompt = "a fox") is expected


def test_h3_native_transcode_is_torch_free_and_keeps_audio(monkeypatch, tmp_path):
    import io
    import math
    from fractions import Fraction

    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    from core.inference.video_minimax_h3 import transcode_video_to_mp4

    source = tmp_path / "native.webm"
    fps, sample_rate = 8, 48_000
    with av.open(str(source), mode = "w", format = "webm") as output:
        video = output.add_stream("libvpx-vp9", rate = fps)
        video.width = video.height = 32
        video.pix_fmt = "yuv420p"
        audio = output.add_stream("libopus", rate = sample_rate)
        audio.layout = "stereo"
        for value in (0, 128):
            frame = av.VideoFrame.from_ndarray(
                np.full((32, 32, 3), value, dtype = np.uint8), format = "rgb24"
            )
            for packet in video.encode(frame):
                output.mux(packet)
        written = 0
        while written < sample_rate // 4:
            count = min(960, sample_rate // 4 - written)
            tone = np.array(
                [
                    int(12_000 * math.sin(2 * math.pi * 440 * (written + i) / sample_rate))
                    for i in range(count)
                ],
                dtype = np.int16,
            )
            frame = av.AudioFrame.from_ndarray(
                np.repeat(tone, 2).reshape(1, count * 2),
                format = "s16",
                layout = "stereo",
            )
            frame.sample_rate = sample_rate
            frame.pts = written
            frame.time_base = Fraction(1, sample_rate)
            for packet in audio.encode(frame):
                output.mux(packet)
            written += count
        for packet in video.encode():
            output.mux(packet)
        for packet in audio.encode():
            output.mux(packet)

    original_import = builtins.__import__

    def _no_ml_stack_import(name, *args, **kwargs):
        if name == "torch" or name.startswith(("torch.", "diffusers")):
            raise AssertionError(f"native transcode imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_ml_stack_import)
    mp4 = transcode_video_to_mp4(source, fps = fps)

    with av.open(io.BytesIO(mp4)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.audio[0].codec_context.name == "aac"
        assert sum(frame.samples for frame in container.decode(audio = 0)) > 0


def test_download_plan_narrows_an_ltx23_pick_and_stages_its_extras(monkeypatch):
    # A 2.3 checkpoint brings its own VAEs, vocoder and connectors, so staging the 2.0 base copies downloads gigabytes the
    # pipeline never opens -- and the companions it DOES read were missing from the plan, so they were pulled inline.
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    # Checkpoint and extras share one repo, so they must be ONE entry: two jobs for the same repo would collide on the scoped job key.
    assert set(by_repo) == {"unsloth/LTX-2.3-GGUF", "Lightricks/LTX-2"}
    ckpt = by_repo["unsloth/LTX-2.3-GGUF"]
    assert ckpt["gguf_filename"] == "ltx-2.3-22b-distilled.gguf"
    assert "vae/ltx-2.3-22b-distilled_video_vae.safetensors" in ckpt["files"]
    assert "vae/ltx-2.3-22b-distilled_audio_vae.safetensors" in ckpt["files"]
    assert "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors" in ckpt["files"]
    # The other variant's companions are not this checkpoint's.
    assert "vae/ltx-2.3-22b-dev_video_vae.safetensors" not in ckpt["files"]

    base = by_repo["Lightricks/LTX-2"]
    assert "scheduler/scheduler_config.json" in base["files"]
    assert any(f.startswith("text_encoder/") for f in base["files"])
    for dropped in ("vae/", "vocoder/", "connectors/", "transformer/"):
        assert not any(f.startswith(dropped) for f in base["files"]), dropped
    assert plan["total_bytes"] == ckpt["bytes"] + base["bytes"]


def _cuda_bf16_target(monkeypatch):
    """Pretend the box can run layerwise fp8, so the pre-cast encoder resolves off-GPU."""
    import torch
    monkeypatch.setattr(
        "core.inference.video.resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16),
    )


_LTX2_FP8_SIBLINGS = [_PlanSibling("LTX-2-text_encoder-FP8.pt", 13_000_000_000)]


def test_base_download_files_skips_precast_text_encoder_weights():
    # The pre-cast checkpoint supplies the encoder WEIGHTS only: the config and shard index stay, since the loader meta-inits from them.
    siblings = _LTX2_SIBLINGS + [
        _sibling("text_encoder/config.json", 1),
        _sibling("text_encoder/model.safetensors.index.json", 1),
    ]
    info = types.SimpleNamespace(siblings = siblings)
    names = [
        n
        for n, _ in VideoBackend._base_download_files(
            info, "gguf", skip_te_components = ("text_encoder",)
        )
    ]
    assert not any(n.endswith(".safetensors") and n.startswith("text_encoder/") for n in names)
    assert "text_encoder/config.json" in names
    assert "text_encoder/model.safetensors.index.json" in names
    # A different component's weights are untouched.
    assert "vae/diffusion_pytorch_model.safetensors" in names


def test_download_plan_swaps_the_dense_encoder_for_the_precast_checkpoint(monkeypatch):
    # An fp8 encoder request loads unsloth/LTX-2-FP8, so staging the dense Gemma3 downloads ~49 GB the pipeline never opens, and the checkpoint it DOES read was missing from the plan.
    _cuda_bf16_target(monkeypatch)
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
            "unsloth/LTX-2-FP8": _LTX2_FP8_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
        text_encoder_quant = "fp8",
    )

    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    assert by_repo["unsloth/LTX-2-FP8"]["files"] == ["LTX-2-text_encoder-FP8.pt"]
    base = by_repo["Lightricks/LTX-2"]
    assert not any(f.startswith("text_encoder/") for f in base["files"])
    # The rest of the scoped base list is unchanged.
    assert "scheduler/scheduler_config.json" in base["files"]
    assert "tokenizer/tokenizer.json" in base["files"]
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_download_plan_keeps_the_dense_encoder_without_an_fp8_request(monkeypatch):
    # No fp8 request means the dense encoder IS the encoder: dropping it would break the load.
    _cuda_bf16_target(monkeypatch)
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    assert "unsloth/LTX-2-FP8" not in by_repo
    assert any(f.startswith("text_encoder/") for f in by_repo["Lightricks/LTX-2"]["files"])


def test_download_plan_keeps_the_dense_encoder_when_the_precast_repo_is_missing(monkeypatch):
    # The hosted artifact can be unpublished, gated or renamed. That must neither drop the dense encoder nor sink the whole plan, which is what an unguarded lookup did.
    _cuda_bf16_target(monkeypatch)
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
        text_encoder_quant = "fp8",
    )

    by_repo = {e["repo_id"]: e for e in plan["entries"]}
    assert set(by_repo) == {"unsloth/LTX-2.3-GGUF", "Lightricks/LTX-2"}
    assert any(f.startswith("text_encoder/") for f in by_repo["Lightricks/LTX-2"]["files"])
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"]) > 0


def test_fetch_te_prequant_only_reports_what_it_downloaded(monkeypatch):
    # The dense skip is earned by the pre-cast file being on disk; an unreachable checkpoint must report nothing.
    backend = VideoBackend()
    source = types.SimpleNamespace(
        kind = "repo", location = "unsloth/LTX-2-FP8", filename = "LTX-2-text_encoder-FP8.pt"
    )

    def _boom(
        repo,
        filename,
        token,
        cancel_event = None,
    ):
        raise OSError("404")

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _boom)
    assert backend._fetch_te_prequant({"text_encoder": source}, None) == ()

    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, filename, token, cancel_event = None: "/tmp/precast.pt",
    )
    assert backend._fetch_te_prequant({"text_encoder": source}, None) == ("text_encoder",)
    # A local path override is the injection's business (allowlist), and nothing is fetched for it.
    local = types.SimpleNamespace(kind = "path", location = "/tmp/x.pt", filename = None)
    assert backend._fetch_te_prequant({"text_encoder": local}, None) == ()


def test_load_pipeline_tops_up_the_dense_encoder_when_injection_fails(fake_runtime, tmp_path):
    # Injection is best-effort, but the pre-download already dropped the dense shards, so a failed injection must restore them rather than crash the load.
    backend = VideoBackend()
    calls: list[dict] = []
    backend._predownload_base = lambda *a, **k: (  # type: ignore[method-assign]
        calls.append({"args": a, "kwargs": k}) or str(tmp_path)
    )
    backend.load_pipeline(
        "Lightricks/LTX-2",
        model_kind = "pipeline",
        _base_local_dir = str(tmp_path),
        _te_prequant_skipped = ("text_encoder",),
    )
    assert len(calls) == 1 and calls[0]["kwargs"]["ltx23"] is False
    backend.unload()

    # Nothing was skipped -> no second pull.
    calls.clear()
    backend.load_pipeline("Lightricks/LTX-2", model_kind = "pipeline", _base_local_dir = str(tmp_path))
    assert calls == []
    backend.unload()


def test_download_plan_keeps_the_wide_base_for_a_plain_ltx2_pick(monkeypatch):
    # Only 2.3 checkpoints supply their own companions; a 2.0 pick still needs the base's.
    _plan_api(
        monkeypatch,
        {
            "someone/LTX-2-GGUF": [_PlanSibling("ltx-2-dev.gguf", 12_000_000_000)],
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "someone/LTX-2-GGUF", gguf_filename = "ltx-2-dev.gguf", family_override = "ltx-2"
    )

    base = next(e for e in plan["entries"] if e["repo_id"] == "Lightricks/LTX-2")
    assert any(f.startswith("vae/") for f in base["files"])
    assert any(f.startswith("connectors/") for f in base["files"])
    # The GGUF still replaces the DiT, so the transformer shards stay out.
    assert not any(f.startswith("transformer/") for f in base["files"])


def test_each_video_load_gets_its_own_cancel_event(monkeypatch):
    """A cancelled load must STAY cancelled once the next one starts.

    unload() sets the event the running worker holds and drops _loading, so the next begin_load
    could arrive before that worker had exited. Clearing a shared event there un-cancelled the old
    worker, and its multi-gigabyte checkpoint pull resumed alongside the replacement load until the
    token check at the very end.
    """
    import threading
    from types import SimpleNamespace

    from core.inference.video import VideoBackend

    backend = VideoBackend.__new__(VideoBackend)
    backend._lock = threading.RLock()
    backend._generate_lock = threading.RLock()
    backend._cancel_event = threading.Event()
    backend._load_token = 0
    backend._loading = None
    backend._active_generate_cancel = None
    backend._state = None

    started: list[threading.Event] = []
    monkeypatch.setattr(
        threading, "Thread", lambda *a, **k: SimpleNamespace(start = lambda: None, daemon = True)
    )
    fam = SimpleNamespace(base_repo = "org/base", name = "wan2.2-ti2v-5b")
    monkeypatch.setattr(VideoBackend, "validate_load_request", lambda self, *a, **k: fam)
    monkeypatch.setattr(VideoBackend, "status", lambda self: {})

    backend.begin_load("org/base")
    first = backend._cancel_event
    started.append(first)

    # unload() signals the in-flight worker and clears _loading, letting a new load through.
    backend._teardown_waiters = 0
    backend._teardown_state_locked = lambda: None
    backend.unload()
    assert first.is_set(), "unload must cancel the in-flight load"

    backend.begin_load("org/other")
    second = backend._cancel_event
    assert second is not first, "each load needs its own event"
    assert first.is_set(), "the replaced load must stay cancelled"
    assert not second.is_set(), "a fresh load starts uncancelled"


# ── teardown fence ────────────────────────────────────────────────────────────


class _HookedLock:
    """threading.Lock wrapper that fires ``on_release`` after a release made by the
    named thread. Lets a test stand in the shoes of a generation parked on
    _generate_lock: it is admitted at the exact instant the teardown releases it."""

    def __init__(self, on_release, thread_name: str) -> None:
        self._lock = threading.Lock()
        self._on_release = on_release
        self._thread_name = thread_name

    def acquire(self, *args, **kwargs):
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()
        if threading.current_thread().name == self._thread_name:
            self._on_release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()


def _run_teardown_race(backend, teardown):
    """Park a generation behind ``teardown``'s _generate_lock barrier, admit it the
    instant that barrier releases the lock, and report what it did."""
    queued: dict = {}
    admitted = threading.Event()
    finished = threading.Event()

    def queued_generate():
        assert admitted.wait(timeout = 5), "queued generation never admitted"
        try:
            queued["out"] = backend.generate(prompt = "queued", steps = 2)
        except RuntimeError as exc:
            queued["error"] = str(exc)
        finally:
            finished.set()

    waiter = threading.Thread(target = queued_generate, daemon = True)
    waiter.start()

    def on_release():
        # The barrier just released _generate_lock: hold the teardown here until the queued generation has had its turn.
        admitted.set()
        finished.wait(timeout = 5)

    backend._generate_lock = _HookedLock(on_release, "teardown-under-test")
    runner = threading.Thread(target = teardown, name = "teardown-under-test")
    runner.start()
    runner.join(timeout = 10)
    assert not runner.is_alive(), "teardown never finished"
    waiter.join(timeout = 5)
    assert not waiter.is_alive(), "queued generation never finished"
    return queued


def test_unload_fences_a_generation_queued_behind_its_barrier(fake_runtime, tmp_path):
    # A generation queued behind unload's barrier holds no cancel event, so unload's signal cannot reach it. Python locks
    # are not FIFO, so it won the lock the moment the barrier let go and denoised on the pipeline the teardown then freed.
    backend = VideoBackend()
    _load_gguf(backend, tmp_path)

    queued = _run_teardown_race(backend, backend.unload)

    assert (
        "out" not in queued
    ), "a generation queued behind the unload barrier ran against a pipeline being torn down"
    assert queued.get("error") in (VIDEO_NOT_LOADED_MSG, VIDEO_CANCELLED_MSG), queued
    assert backend._state is None
    assert backend._teardown_waiters == 0  # the fence drained


def test_superseding_load_fences_a_generation_queued_behind_its_barrier(fake_runtime, tmp_path):
    # The load path takes the same barrier before tearing the old model down, so it has the same hole.
    backend = VideoBackend()
    _load_gguf(backend, tmp_path)

    queued = _run_teardown_race(backend, lambda: _load_gguf(backend, tmp_path))

    assert (
        "out" not in queued
    ), "a generation queued behind the load barrier ran against a pipeline being torn down"
    assert queued.get("error") in (VIDEO_NOT_LOADED_MSG, VIDEO_CANCELLED_MSG), queued
    assert backend._teardown_waiters == 0  # the fence drained


def test_generation_refuses_while_a_teardown_is_waiting(fake_runtime, tmp_path):
    # The fence's effect: with a teardown waiting on _generate_lock, a generation that wins the lock refuses instead of denoising.
    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    assert backend.generate(prompt = "before", steps = 2)["mp4_bytes"] == b"MP4"

    backend._teardown_waiters = 1
    with pytest.raises(RuntimeError, match = "cancelled"):
        backend.generate(prompt = "during", steps = 2)
    # Still loaded: the refusal is about the pending teardown, not a missing model.
    assert backend._state is not None

    backend._teardown_waiters = 0
    assert backend.generate(prompt = "after", steps = 2)["mp4_bytes"] == b"MP4"


def test_a_raising_teardown_still_drains_the_fence(fake_runtime, tmp_path, monkeypatch):
    # _teardown_state_locked ends in clear_gpu_cache(), which raises on a sticky CUDA fault. Without the finally the fence stayed up forever.
    from core.inference import video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)

    def _sticky():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(video_mod, "clear_gpu_cache", _sticky)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        backend.unload()
    assert backend._teardown_waiters == 0, "a failed teardown must not leave the fence up"

    monkeypatch.setattr(video_mod, "clear_gpu_cache", lambda: None)
    _load_gguf(backend, tmp_path)
    assert backend.generate(prompt = "after", steps = 2)["mp4_bytes"] == b"MP4"


# ── the H3 native path and the audio VAE ─────────────────────────────────────
def test_the_h3_native_load_never_puts_the_vae_on_the_cpu():
    """`low_vram` maps to the `model` policy, which emits `--vae-on-cpu` for everyone else.

    On H3 that aborts: the audio VAE's 1-D convolutions reach a CPU path that asserts the
    kernel is F16 while sd.cpp lets the F32 one through, so the process dies with
    `GGML_ASSERT(src0->type == GGML_TYPE_F16) failed`, SIGABRT, exit 134. Converting the
    checkpoint to fp16 does not avoid it, so the flag has to not be emitted.

    Source-level, because reproducing it needs a built sd-cli and the H3 weights. The point
    is that the H3 call site passes the opt-out, and that the flag is what is opted out of.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "offload_flags"
    ]
    assert calls, "video.py no longer builds native offload flags"
    for call in calls:
        opt_out = [kw for kw in call.keywords if kw.arg == "vae_on_cpu"]
        assert opt_out, "the H3 native path must ask for the VAE to stay off the CPU"
        assert isinstance(opt_out[0].value, ast.Constant) and opt_out[0].value.value is False

    # And the flag the opt-out removes is still the one this is about.
    from core.inference.sd_cpp_args import OFFLOAD_MODEL, offload_flags

    assert "--vae-on-cpu" in offload_flags(OFFLOAD_MODEL)
    assert "--vae-on-cpu" not in offload_flags(OFFLOAD_MODEL, vae_on_cpu = False)


def test_h3_rejects_companion_checkpoints_as_the_transformer():
    """Only an fl2va denoiser is a valid T2VA pick, and the mirror ships more than that.

    The Qwen3-VL encoder quants now live in the same repo as the denoisers, so the picker lists
    both and a user can name either. Loading the encoder as the transformer would waste a ~12 GB
    download and fail deep inside sd-cli rather than at the boundary. Ref2VA is a different
    workflow entirely.

    The accept cases include the dynamic rung names specifically: the guard is a prefix/suffix
    check, and `-UD-Q2_K_XL` is a shape it had never seen when it was written.
    """
    from core.inference.video_minimax_h3 import validate_h3_transformer_filename

    for good in (
        "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
        "minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf",
        "minimax_h3_fl2va_pruned-Q4_K.gguf",
        "minimax_h3_fl2va-Q4_K_M.gguf",
    ):
        validate_h3_transformer_filename(good)

    for bad in (
        "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        "minimax_h3_fl2va_pruned_bf16.safetensors",
    ):
        with pytest.raises(ValueError):
            validate_h3_transformer_filename(bad)


def test_the_h3_native_repo_matches_the_family_gguf_repo():
    """The declared pick and the actual download must be the same repo.

    Main's `test_curated_gguf_repos_are_unsloth_mirrors` only inspects `VideoFamily.gguf_repo`,
    but H3's native path downloads from its own `H3_GGUF_REPO` constant and never reads the
    family field. So the two can drift apart and that test still passes while the one-click pick
    resolves to a community repack, which is exactly the failure it exists to prevent.

    Also asserts the transformer and the text encoder come from the same repo, since the mirror
    has to carry both for the pick to be self-contained.
    """
    from core.inference.video_families import detect_video_family
    from core.inference.video_minimax_h3 import (
        H3_GGUF_REPO,
        h3_native_hub_files,
    )

    family = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert family is not None
    assert H3_GGUF_REPO == family.gguf_repo, (
        f"H3 native downloads from {H3_GGUF_REPO} but the family advertises "
        f"{family.gguf_repo}; the curated-mirror test would pass vacuously"
    )

    files = h3_native_hub_files("minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf")
    transformer_repo, transformer_name = files[0]
    encoder_repo, encoder_name = files[1]
    assert transformer_repo == encoder_repo == H3_GGUF_REPO
    assert transformer_name == "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf"
    # The dynamic rung names must still route to the matching encoder tier.
    assert encoder_name == "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"
    assert h3_native_hub_files("minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf")[1][1] == (
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    )


def test_h3_names_the_component_when_a_download_is_refused():
    """A 401 on one H3 component must not surface as the Hub's raw "Repository Not Found".

    H3 pulls four files from two repos, and the Hub returns the same message for "does not exist",
    "is private" and "your token does not cover it". A user reading it has no way to tell which of
    the four failed, and "Repository Not Found" actively points away from the real cause when the
    repo exists but is private. This matters most while the GGUF mirror is unpublished: without
    this, picking H3 fails with a message that suggests the wrong fix.
    """
    from unittest.mock import Mock

    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    from core.inference.video_minimax_h3 import (
        H3_AUDIO_VAE,
        H3_QWEN_Q2,
        h3_download_error,
    )

    def _hub_error(cls):
        response = Mock()
        response.status_code = 401
        response.headers = {}
        return cls("Repository Not Found for url: ...", response = response)

    # Private / unpublished: name the repo and the component, and do not blame the token alone.
    denoiser = h3_download_error(
        "unsloth/MiniMax-H3-GGUF",
        "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
        _hub_error(RepositoryNotFoundError),
    )
    assert isinstance(denoiser, RuntimeError)
    text = str(denoiser)
    assert "unsloth/MiniMax-H3-GGUF" in text
    assert "denoiser" in text
    assert "not published yet" in text

    # The same repo serves the encoder, so the component name has to come from the FILE.
    encoder = str(
        h3_download_error(
            "unsloth/MiniMax-H3-GGUF", H3_QWEN_Q2, _hub_error(RepositoryNotFoundError)
        )
    )
    assert "text encoder" in encoder and "denoiser" not in encoder

    # A gated repo has a different remedy, so it must not reuse the private wording.
    gated = str(h3_download_error("Comfy-Org/MiniMax-H3", H3_AUDIO_VAE, _hub_error(GatedRepoError)))
    assert "audio VAE" in gated
    assert "accept its licence" in gated
    assert "not published yet" not in gated

    # Anything that is not an access error is passed straight back, so a timeout still reads as a
    # timeout rather than being reworded into a permissions problem.
    timeout = TimeoutError("read timed out")
    assert h3_download_error("unsloth/MiniMax-H3-GGUF", H3_QWEN_Q2, timeout) is timeout


def test_the_h3_native_path_pins_cfg_scale_to_one():
    """H3 aborts at any cfg above 1.0, so the guidance slider must never reach sd-cli.

    H3 is distilled and CFG-free: the empty unconditional prompt encodes to zero tokens, and the
    resulting transposed tensor trips `GGML_ASSERT(!ggml_is_transposed(a))` in ggml.c. SIGABRT,
    exit 134. Measured: cfg 1.0 renders, cfg 1.5 and cfg 4.0 both abort. sd.cpp's own default is
    7.0, so this is a crash a plausible refactor reintroduces by simply forwarding `guidance` the
    way every other family does.

    The family already sets `supports_cfg = False`, but that only gates the diffusers path; the
    native path builds its own params object. Asserted at the source level because reproducing the
    abort needs a built sd-cli and the H3 weights.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SdCppVideoGenParams"
    ]
    assert calls, "video.py no longer builds native video params"
    for call in calls:
        pinned = [kw for kw in call.keywords if kw.arg == "cfg_scale"]
        assert (
            pinned
        ), "the H3 native path must pass cfg_scale explicitly, not fall back to a default"
        value = pinned[0].value
        assert isinstance(value, ast.Constant), (
            "cfg_scale must be a literal 1.0 on the H3 native path; forwarding the request's "
            "guidance reintroduces GGML_ASSERT(!ggml_is_transposed(a)), SIGABRT exit 134"
        )
        assert value.value == 1.0, value.value

    # And the family keeps declaring it has no CFG, so the UI does not offer the slider either.
    from core.inference.video_families import detect_video_family

    assert detect_video_family("MiniMaxAI/MiniMax-H3").supports_cfg is False


def _trim_spy(monkeypatch):
    """Record every install_hunyuan_attention_trim call and report it as engaged."""
    from core.inference import video as video_mod

    calls: list = []

    def _fake(
        pipe,
        family,
        *,
        logger = None,
    ):
        calls.append(getattr(family, "name", None))
        return True

    monkeypatch.setattr(video_mod, "install_hunyuan_attention_trim", _fake)
    return calls


def test_attention_trim_installed_and_reported(fake_runtime, monkeypatch):
    # The padded-text trim is installed once per pipe (the installer fans out over the DiTs itself)
    # and, when it engages, is reported as a speed optim.
    calls = _trim_spy(monkeypatch)
    backend = VideoBackend()
    status = backend.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline", speed_mode = "default"
    )
    assert status["loaded"] is True
    assert len(calls) == 1
    assert "hunyuan_attn_trim" in status["speed_optims"]


def test_attention_trim_skipped_for_static_shape_and_off_tiers(fake_runtime, monkeypatch):
    # speed=off must stay bit-identical, and speed=max compiles the blocks with dynamic=False,
    # where the prompt-dependent trimmed text length would make every prompt a fresh graph.
    for mode in ("off", "max"):
        calls = _trim_spy(monkeypatch)
        backend = VideoBackend()
        status = backend.load_pipeline(
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline", speed_mode = mode
        )
        assert status["loaded"] is True, mode
        assert calls == [], mode
        assert "hunyuan_attn_trim" not in status["speed_optims"], mode
