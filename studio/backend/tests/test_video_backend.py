# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VideoBackend lifecycle on a faked torch/diffusers runtime (CPU-only, offline).
Mirrors test_diffusion_backend's fake_runtime pattern: explicit fake signatures so
the signature-gated kwargs actually exercise, sys.modules stubs so no real ML
stack loads."""

import builtins
import contextlib
import dataclasses
import sys
import threading
import time
import types
from dataclasses import replace
from pathlib import Path

import pytest

from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.video import (
    VideoBackend,
    _detect_load_family,
    get_video_backend,
    resolve_video_model_kind,
)
from core.inference.video_families import VIDEO_CANCELLED_MSG, VIDEO_NOT_LOADED_MSG


@pytest.fixture(autouse = True)
def _assume_the_restricted_load_is_available(monkeypatch):
    """A checkpoint only deserializes where torchao is importable, which here it may not be. These
    tests are about the load/plan decisions; the capability is covered in
    test_diffusion_prequant.py."""
    import core.inference.diffusion_prequant as _pq
    monkeypatch.setattr(_pq, "restricted_prequant_load_supported", lambda scheme = None: True)


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
            "generator": generator,
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


class _FakeWanDecoder:
    def __init__(self) -> None:
        self.hooks: list = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)


class _FakeWanVae:
    def __init__(self) -> None:
        self.tiled = False
        self.decoder = _FakeWanDecoder()

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


# MiniMax-H3 BF16 fakes: the Modular Diffusers workflow. ModularPipeline takes no step callback
# (an unknown input is only warned about) and its denoise loop drives components.scheduler.step,
# i.e. pipe.scheduler, once per step -- exactly what the HV15 wrapper hooks.


class _FakeH3Scheduler:
    def __init__(self) -> None:
        self.calls = 0
        # Fired from the ORIGINAL step, so a test can cancel mid-denoise as a user request would.
        self.on_step = None

    def step(self, *args, **kwargs):
        self.calls += 1
        if self.on_step is not None:
            self.on_step(self.calls)
        return object()


class _FakeComponentsManager:
    def __init__(self) -> None:
        self.offload = None

    def enable_auto_cpu_offload(self, **kwargs):
        self.offload = kwargs


class _FakeModularPipe:
    def __init__(self) -> None:
        self.scheduler = _FakeH3Scheduler()
        self.load_kwargs = None
        self.last_kwargs = None

    def load_components(self, **kwargs):
        self.load_kwargs = kwargs

    def __call__(
        self,
        *,
        prompt = None,
        num_inference_steps = None,
        width = None,
        height = None,
        num_frames = None,
        generator = None,
        output = None,
        **kwargs,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "output": output,
            **kwargs,
        }
        for _ in range(int(num_inference_steps or 1)):
            self.scheduler.step(object(), 0, object())
        return {"videos": [[object() for _ in range(int(num_frames or 1))]], "audio": None}


class _FakeModularPipeline:
    last: dict = {}
    instance = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        _FakeModularPipeline.last = {"path": path, **kwargs}
        _FakeModularPipeline.instance = _FakeModularPipe()
        return _FakeModularPipeline.instance


def _load_h3_modular(backend, *, hf_token = None):
    """Commit the H3 BF16 state exactly as load_pipeline's Modular Diffusers branch does."""
    diffusers = sys.modules["diffusers"]
    diffusers.ComponentsManager = _FakeComponentsManager
    diffusers.ModularPipeline = _FakeModularPipeline
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    assert fam is not None
    backend._load_h3_modular_pipeline(
        diffusers = diffusers,
        torch = sys.modules["torch"],
        fam = fam,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base = fam.base_repo,
        kind = "pipeline",
        dtype = sys.modules["torch"].bfloat16,
        device = "cpu",
        hf_token = hf_token,
        memory_mode = None,
        _load_token = None,
        _base_local_dir = None,
    )
    return _FakeModularPipeline.instance


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


def _stub_apply_memory_plan(
    monkeypatch,
    video_mod,
    *,
    policy = "model",
    vae_tiling = True,
) -> list:
    """Stand in for ``apply_memory_plan``, recording the placement kwargs of every call.

    The keywords are spelled out rather than ``**kwargs`` on purpose: a double that swallows the
    signature keeps passing once the load hands over an argument it never reads, which is how
    ``placement_device`` (#8645) became a TypeError on CI.
    """
    calls = []

    def _fake(
        pipe,
        plan,
        *,
        device = None,
        placement_device = None,
        logger = None,
    ):
        calls.append({"device": device, "placement_device": placement_device})
        return (policy, vae_tiling)

    monkeypatch.setattr(video_mod, "apply_memory_plan", _fake)
    return calls


def _assert_placement_follows_the_target(calls, video_mod):
    """Placement gets the INDEXED device off the resolved target, the policy string stays bare."""
    target = video_mod.resolve_diffusion_device_target()
    assert calls, "apply_memory_plan was never called"
    assert calls[0]["placement_device"] == target.torch_device
    assert calls[0]["device"] == target.device


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
    monkeypatch.setattr(gguf_meta, "read_gguf_architecture", lambda path: "ltxv")
    fam = _detect_load_family("someorg/opaque-quants", "model.gguf", None)
    assert fam is not None and fam.name == "ltx-2"

    # A cache MISS (blob not present -> None) still yields None (400 exactly as before).
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    assert _detect_load_family("someorg/opaque-quants", "model.gguf", None) is None

    # A recognised-but-unsupported video arch stays None, so an unsupported cached pick 400s like the local-dir case.
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/fake/cache/blobs/model.gguf"
    )
    monkeypatch.setattr(gguf_meta, "read_gguf_architecture", lambda path: "wan")
    assert _detect_load_family("someorg/opaque-quants", "model.gguf", None) is None

    # The blob lives in a NON-active cache root: the active probe misses, but the per-root probe finds it.
    import hub.utils.paths as hub_paths

    monkeypatch.setattr(hub_paths, "legacy_hf_cache_dir", lambda: "/fake/legacy")
    monkeypatch.setattr(hub_paths, "hf_default_cache_dir", lambda: "/fake/default")
    monkeypatch.setattr(gguf_meta, "read_gguf_architecture", lambda path: "ltxv")
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


@pytest.mark.parametrize("device, expected", [("mps", "cpu"), ("cuda", "cuda")])
def test_generate_seeds_metal_from_a_cpu_generator(fake_runtime, tmp_path, device, expected):
    # Metal reproduces a seed only through a CPU generator, and this also keeps the path off
    # whatever torch.Generator(device="mps") does on the older torch releases install.sh keeps.
    (tmp_path / "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf").write_bytes(b"w")
    backend = VideoBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )
    backend._state = dataclasses.replace(backend._state, device = device)
    backend.generate(prompt = "a sloth", seed = 7)
    assert backend._state.pipe.last_kwargs["generator"].device == expected


def test_ltx23_load_forwards_the_precast_encoder(fake_runtime, tmp_path, monkeypatch):
    # The wiring half of the same bug: the 2.3 branch does not use pipe_kwargs, so the loader must pass the pre-cast encoder across explicitly.
    from core.inference import diffusion_te_prequant, video_ltx2

    # The CPU stub cannot cast the encoder, which the strict default now refuses; this test is
    # about the WIRING, so keep the legacy silent-fallback behaviour for it.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")

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
        # An exact hand-written signature, so it follows the production one: the assembly bypasses
        # the caller's guarded pipe_kwargs, and this base-repo read is the first of the calls that
        # a load nobody asked for must not turn into a fetch.
        def load_config(
            base_repo,
            token = None,
            local_files_only = False,
            cache_dir = None,
        ):
            _FakeLTX2Pipeline.last_config_kwargs = {
                "local_files_only": local_files_only,
                # Pinned to Unsloth's LIVE root: unset, this resolves through huggingface_hub's
                # import-time constant, which a mid-session cache-folder change leaves stale.
                "cache_dir": cache_dir,
            }
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


@pytest.mark.parametrize(
    "video, error, total, expected",
    [
        (
            {"id": "clip-1"},
            None,
            12,
            {
                "phase": "completed",
                "percent": 100,
                "step": 12,
                "total_steps": 12,
                "video_id": "clip-1",
            },
        ),
        (
            None,
            "negative_prompt is not supported by this family.",
            0,
            {
                "phase": "failed",
                "percent": 0,
                "step": 0,
                "total_steps": 0,
                "error": "negative_prompt is not supported by this family.",
            },
        ),
    ],
)
def test_finish_generate_job_logs_each_terminal_outcome_once(
    fake_runtime, monkeypatch, video, error, total, expected
):
    import core.inference.video as video_mod

    events = []

    class _Recorder:
        def info(self, event, **fields):
            events.append((event, fields))

    monkeypatch.setattr(video_mod, "logger", _Recorder())
    backend = VideoBackend()
    token = object()
    backend._generate_job_token = token
    backend._generate_job_active = True

    backend._finish_generate_job(job_token = token, video = video, error = error, total = total)
    backend._finish_generate_job(job_token = token, video = video, error = error, total = total)

    assert events == [("video_generation_progress", expected)]


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
    # Nothing to sync on this CPU-resolved target; the hook must not cost non-MPS loads a thing.
    assert backend._state.pipe.vae.decoder.hooks == []


def _fits_in_memory_snapshot(device):
    """A memory snapshot big enough that no load is ever refused for size, so a test forcing a
    device does not inherit the runner's own free memory as a precondition."""
    from core.inference.diffusion_memory import DeviceMemory

    total = 512 * 1024
    kind = "unified_memory" if device == "mps" else "discrete_vram"
    return lambda target: DeviceMemory(device, device, kind, int(total * 0.80), total)


@pytest.mark.parametrize("device,hooked", [("mps", 1), ("cuda", 0)])
def test_load_installs_the_pressure_gated_decoder_sync_on_mps(
    fake_runtime, monkeypatch, device, hooked
):
    # Tiling alone does not bound a Wan decode on MPS: intermediates accumulate within a single
    # tile until the OS kills the process. The load must arm the sync.
    torch = sys.modules["torch"]
    monkeypatch.setattr(
        torch,
        device,
        types.SimpleNamespace(
            synchronize = lambda: None,
            recommended_max_memory = lambda: 64 * 1024**3,
            driver_allocated_memory = lambda: 0,
        ),
        raising = False,
    )
    monkeypatch.setattr(
        "core.inference.video.resolve_diffusion_device_target",
        lambda: DiffusionDeviceTarget(
            device = device,
            dtype = torch.bfloat16,
            backend = device,
            vendor = None,
            supports_model_cpu_offload = False,
            supports_default_torch_compile = False,
            supports_pinned_transfer = False,
        ),
    )
    # Forcing device="mps" also forces the unified-memory placement, and that snapshot is read
    # from the HOST's free RAM (snapshot_device_memory sends mps to _system_memory_mib), not from
    # the torch.mps stub above. A runner with little free RAM therefore refuses this 25 GB load
    # before the hook is ever installed, which is correct behaviour and nothing to do with what
    # is being asserted here. Pin a pool with room to spare so the result does not depend on how
    # busy the machine is.
    monkeypatch.setattr(
        "core.inference.video.settled_snapshot_device_memory",
        _fits_in_memory_snapshot(device),
    )
    backend = VideoBackend()
    backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert len(backend._state.pipe.vae.decoder.hooks) == hooked


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

    # ...and the RECORD still says the user asked for nothing. The suppression rewrites the
    # internal value to "off" before the resolved record is built, and reporting that as the
    # request makes the record claim an explicit pin: the Auto badge disappears and the Precision
    # select reseeds to none, so after the user changes Speed and reloads, quantisation stays
    # pinned off for no reason they can see.
    resolved = status["resolved"]["transformer_quant"]
    assert (
        resolved["requested"] is None
    ), f"the record reports {resolved['requested']!r} as the user's request; nothing was asked for"
    assert resolved["source"] == "auto"

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


def test_video_gguf_status_reports_selected_quant_instead_of_only_compute_dtype(
    fake_runtime, tmp_path
):
    filename = "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf"
    (tmp_path / filename).write_bytes(b"w")
    backend = VideoBackend()

    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = filename,
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )

    assert status["gguf_variant"] == "Q4_K_M"
    assert backend.unload()["gguf_variant"] is None
    # A pipeline load has no checkpoint quant to name.
    assert (
        backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")[
            "gguf_variant"
        ]
        is None
    )
    backend.unload()


def test_video_status_response_carries_gguf_variant():
    from models.inference import VideoStatusResponse
    assert VideoStatusResponse(loaded = True, gguf_variant = "Q4_K_M").gguf_variant == "Q4_K_M"


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
    # Offload hooks move modules with Module.to(), which torchao tensors reject, so any offload
    # policy must SKIP quant. With the legacy escape hatch set the load still succeeds dense and the
    # record explains why; the strict default refuses instead (see the test below).
    import core.inference.video as video_mod

    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
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
    placements = _stub_apply_memory_plan(monkeypatch, video_mod)

    backend = VideoBackend()
    status = backend.load_pipeline(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        model_kind = "pipeline",
        transformer_quant = "int8",
    )
    _assert_placement_follows_the_target(placements, video_mod)
    assert status["offload_policy"] == "model"
    assert quantised == []
    assert status["transformer_quant"] is None
    resolved = status["resolved"]["transformer_quant"]
    assert "moves the DiT" in resolved["reason"]
    # BOTH sides of the story survive: the ask, the outcome, and that they disagree.
    assert resolved["requested"] == "int8"
    assert resolved["value"] == "off"
    assert resolved["status"] == "fell_back"


def test_the_video_load_places_on_the_selected_card_not_a_bare_device(fake_runtime, monkeypatch):
    # #8645 at the video seam: ``enable_model_cpu_offload`` reads the ordinal off the device and
    # falls back to ``_offload_gpu_id = 0`` without one, so the load passes the INDEXED string as
    # ``placement_device``; ``device`` stays bare because the memory/speed/attention policies
    # compare it against "cuda". The two must not be swapped.
    import dataclasses

    import core.inference.video as video_mod

    real_target = VideoBackend._device_target

    def _selected(self, ordinal = None):
        return dataclasses.replace(real_target(self, ordinal), ordinal = 1)

    monkeypatch.setattr(VideoBackend, "_device_target", _selected)
    real_plan = video_mod.plan_diffusion_memory
    monkeypatch.setattr(
        video_mod,
        "plan_diffusion_memory",
        lambda **kwargs: dataclasses.replace(real_plan(**kwargs), offload_policy = "model"),
    )
    placements = _stub_apply_memory_plan(monkeypatch, video_mod)

    VideoBackend().load_pipeline("Lightricks/LTX-2", model_kind = "pipeline")

    assert placements, "apply_memory_plan was never called"
    assert placements[0]["placement_device"] == "cpu:1"
    assert placements[0]["device"] == "cpu"


def test_explicit_dense_quant_refuses_under_offload(fake_runtime, monkeypatch):
    # Strict default (no escape hatch): an explicit int8 the offload plan cannot honor stops the
    # load, rather than denoising at bf16 while the Precision dropdown still reads INT8.
    import dataclasses

    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        video_mod,
        "quantize_transformer",
        lambda view, target, *, mode, family, logger = None: pytest.fail("must not quantise"),
    )
    real_plan = video_mod.plan_diffusion_memory
    monkeypatch.setattr(
        video_mod,
        "plan_diffusion_memory",
        lambda **kwargs: dataclasses.replace(real_plan(**kwargs), offload_policy = "model"),
    )
    backend = VideoBackend()
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            model_kind = "pipeline",
            transformer_quant = "int8",
        )
    assert "transformer_quant='int8' could not be used" in str(excinfo.value)
    assert "resident memory mode" in str(excinfo.value)
    assert backend.status()["loaded"] is False


def test_the_video_refusal_also_names_a_broken_torchao_rather_than_the_gpu(
    fake_runtime, monkeypatch
):
    # The image twin's finding, shared through explain_unusable_scheme: a torchao that cannot
    # import looks exactly like a GPU without the kernels to the selector, and a refusal that
    # blames the GPU sends the user after hardware for what a reinstall fixes.
    import core.inference.diffusion_transformer_quant as tq
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(video_mod, "select_transformer_quant_scheme", lambda *a, **k: None)
    monkeypatch.setattr(tq, "_TORCHAO_UNAVAILABLE", ("ImportError: no torchao kernels",))
    with pytest.raises(RuntimeError) as excinfo:
        video_mod.assert_video_precision_available(
            types.SimpleNamespace(name = "ltx-2"),
            model_kind = "pipeline",
            transformer_quant = "nvfp4",
        )
    message = str(excinfo.value)
    assert "transformer_quant='nvfp4' could not be used" in message
    assert "no torchao kernels" in message and "not a limit of the GPU" in message


def test_begin_load_refuses_dense_quant_on_a_non_pipeline_video_kind(fake_runtime, monkeypatch):
    # The DiT quant only exists on the full-pipeline path, so an explicit scheme on a GGUF pick was
    # accepted and then ignored outright. Refused before the load starts (the route's 409).
    backend = VideoBackend()
    with pytest.raises(RuntimeError) as excinfo:
        backend.begin_load(
            "unsloth/LTX-2.3-GGUF",
            gguf_filename = "ltx-2.3-Q4_K_M.gguf",
            model_kind = "gguf",
            transformer_quant = "fp8",
        )
    assert "full-pipeline loads only" in str(excinfo.value)


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
    # A GGUF/single-file checkpoint replaces the DiT: the base transformer WEIGHTS never pull.
    info = types.SimpleNamespace(siblings = _LTX2_SIBLINGS)
    names = [n for n, _ in VideoBackend._base_download_files(info, "gguf")]
    transformer = [n for n in names if n.startswith("transformer/")]
    # config.json is the one exception, and it is not an oversight: from_single_file resolves
    # config = <repo id> through the Hub, so an API load that promised to download nothing needs
    # this ~1 KB file staged, and the locality gate needs to count it. Everything else under
    # transformer/ is supplied by the checkpoint itself.
    assert transformer == ["transformer/config.json"]
    assert "text_encoder/model-00001-of-00002.safetensors" in names


_H3_SIBLINGS = [
    _sibling("model_index.json", 10),
    _sibling("modular_model_index.json", 10),
    _sibling("transformer/config.json", 1),
    _sibling("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 33),
    _sibling("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 33),
    _sibling("transformer_ref/config.json", 1),
    _sibling("transformer_ref/diffusion_pytorch_model-00001-of-00002.safetensors", 33),
    _sibling("transformer_ref/diffusion_pytorch_model-00002-of-00002.safetensors", 33),
    _sibling("text_encoder/model-00001-of-00001.safetensors", 15),
    _sibling("tokenizer/chat_template.jinja", 1),
    _sibling("vae/diffusion_pytorch_model.safetensors", 3),
    _sibling("audio_vae/diffusion_pytorch_model.safetensors", 1),
    _sibling("scheduler/scheduler_config.json", 1),
    _sibling("audio_scheduler/scheduler_config.json", 1),
    _sibling("processor/preprocessor_config.json", 1),
]


def test_base_download_files_stages_the_h3_partition_the_load_will_open():
    """H3 ships two denoisers in separate subfolders, 66.28 GB each, and a load opens one.

    ref2va reads transformer_ref/, which the scoped list left out entirely, so a reference load
    staged the wrong 66.28 GB and then pulled the right ones inline, outside the download panel.
    Both partitions must never be staged at once either: that is the whole 132 GB.
    """
    info = types.SimpleNamespace(siblings = _H3_SIBLINGS)

    keyframes = [n for n, _ in VideoBackend._base_download_files(info, "pipeline")]
    assert "transformer/diffusion_pytorch_model-00001-of-00002.safetensors" in keyframes
    assert not any(n.startswith("transformer_ref/") for n in keyframes)

    references = [
        n for n, _ in VideoBackend._base_download_files(info, "pipeline", h3_task = "ref2va")
    ]
    assert "transformer_ref/diffusion_pytorch_model-00001-of-00002.safetensors" in references
    assert "transformer_ref/diffusion_pytorch_model-00002-of-00002.safetensors" in references
    assert not any(n.startswith("transformer/") for n in references)
    # Everything shared still comes along on both.
    for shared in (
        "model_index.json",
        "text_encoder/model-00001-of-00001.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ):
        assert shared in keyframes and shared in references


def test_base_download_files_skips_the_partition_the_prequant_checkpoint_replaces():
    """``skip_transformer_weights`` has to drop the shards of the partition THIS load opens.

    The skip named the literal ``transformer/``, which a reference load never stages in the first
    place, so a ref2va pre-quantized pick dropped nothing: the plan carried the full 66.28 GB of
    ``transformer_ref/`` that the checkpoint exists to replace, and the disk preflight sized a
    download the load never opens. Both partitions' configs must survive their own skip, since
    the pre-quant loader meta-inits the DiT from one of them.
    """
    info = types.SimpleNamespace(siblings = _H3_SIBLINGS)

    keyframes = dict(
        VideoBackend._base_download_files(info, "pipeline", skip_transformer_weights = True)
    )
    assert not any(n.startswith("transformer/diffusion_pytorch_model") for n in keyframes)
    assert "transformer/config.json" in keyframes

    references = dict(
        VideoBackend._base_download_files(
            info, "pipeline", skip_transformer_weights = True, h3_task = "ref2va"
        )
    )
    assert not any(n.startswith("transformer_ref/diffusion_pytorch_model") for n in references)
    assert "transformer_ref/config.json" in references
    # And the other partition is still absent entirely, so the skip cannot have swapped which one
    # is staged instead of dropping its weights.
    assert not any(n.startswith("transformer/") for n in references)
    # 66 of the 68 units in the reference plan were the dense denoiser; without the fix the totals
    # are identical with and without the skip.
    assert sum(references.values()) < sum(
        size for _n, size in VideoBackend._base_download_files(info, "pipeline", h3_task = "ref2va")
    )


def _h3_pipeline_load_is_attemptable(fam) -> bool:
    """Whether validate_load_request can even reach a pick's own checks for an H3 pipeline here.

    Two of its refusals are about the machine, not the request: Metal cannot place the modular
    workflow at all (torch.mps exposes no mem_get_info for the auto CPU offload), and a diffusers
    without the bundled revision has no transformer class to build. Both raise the same ValueError
    a genuine refusal does, so a caller that reads any ValueError as "this pick was rejected"
    reports a regression on hosts where the pick was never in question.

    No diffusers at all is the third such host, and it is a supported one: studio.txt does not
    install diffusers (it arrives with the torch-bound ML stack), and the native sd.cpp engine
    serves H3 without it. assert_pipeline_class_available answers only "is the installed
    diffusers new enough", so under its default non-strict mode an unimportable one returns
    rather than raising -- which means it cannot be the guard for this, and the probe below has
    to carry its own."""
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.diffusion_families import assert_pipeline_class_available

    if resolve_diffusion_device_target().device == "mps":
        return False
    try:
        assert_pipeline_class_available(fam.pipeline_class, fam.name)
    except Exception:  # noqa: BLE001 -- an unavailable pipeline class is a host fact, not a verdict
        return False
    if fam.modular_workflow:
        try:
            import diffusers

            # hasattr, not the import, is what pulls in the lazy submodule, so a partially
            # installed diffusers raises here rather than at the import statement.
            return hasattr(diffusers, fam.transformer_class)
        except Exception:  # noqa: BLE001 -- no importable diffusers is a host fact too
            return False
    return True


def test_the_h3_attemptability_probe_survives_a_host_without_diffusers(monkeypatch):
    """The probe exists to turn host limitations into "not attemptable" instead of a red test, so
    it must not itself raise on the most ordinary limitation of all. studio.txt installs no
    diffusers, and assert_pipeline_class_available does NOT stand in for the check: non-strict is
    its default and an unimportable diffusers makes it return, not raise, so control reaches the
    modular-workflow probe below it. Unguarded, that probe raised ModuleNotFoundError straight out
    of the helper and failed the caller before its own `except Exception` could see it."""
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    # H3 is modular, so the probe really does reach the import this guards.
    assert fam.modular_workflow

    original_import = builtins.__import__

    def _no_diffusers_import(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers."):
            raise ModuleNotFoundError(f"No module named '{name}'", name = name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_diffusers_import)
    assert _h3_pipeline_load_is_attemptable(fam) is False


def test_a_quantized_reference_load_resolves_the_reference_denoiser():
    # This pairing used to be refused outright: the only hosted checkpoints were fl2va denoisers,
    # and one seeded into the reference workflow would have installed cleanly, passed every
    # metadata check and generated from the keyframe partition. Now that a ref2va artifact exists
    # for both schemes the refusal must be gone -- and must resolve the REFERENCE file, since
    # picking the keyframe one is the exact failure the refusal was standing in for.
    from core.inference.diffusion_prequant import resolve_prequant_source

    backend = VideoBackend()
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    # The resolution below is a registry lookup and holds on every host; only the refusal check
    # needs a host that can attempt the load at all.
    check_refusal = _h3_pipeline_load_is_attemptable(fam)
    for scheme, expected in (
        ("int8", "MiniMax-H3-Ref2VA-INT8-ConvRot.pt"),
        ("fp8", "MiniMax-H3-Ref2VA-FP8.pt"),
    ):
        if check_refusal:
            try:
                backend.validate_load_request(
                    "MiniMaxAI/MiniMax-H3",
                    family_override = "minimax-h3",
                    model_kind = "pipeline",
                    transformer_quant = scheme,
                    h3_task = "ref2va",
                )
            except ValueError as exc:  # pragma: no cover - only on a regression
                pytest.fail(f"ref2va {scheme} should be loadable but was refused: {exc}")
            except Exception:
                # Anything past the quant check (the diffusers probe) is not this test's business.
                pass
        source = resolve_prequant_source(fam, scheme, task = "ref2va")
        assert source.filename == expected


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
        "utils.models.gguf_metadata.read_gguf_architecture",
        lambda p: "ltxv",
    )
    fam = vid._detect_load_family(str(d), "model.gguf", None)
    assert fam is not None and fam.name == "ltx-2"

    # A video arch with no backend family (wan) stays None, so the loader 400s as before.
    monkeypatch.setattr(
        "utils.models.gguf_metadata.read_gguf_architecture",
        lambda p: "wan",
    )
    assert vid._detect_load_family(str(d), "model.gguf", None) is None

    # An explicit family_override skips the arch read entirely (worker parity).
    monkeypatch.setattr(
        "utils.models.gguf_metadata.read_gguf_architecture",
        lambda p: "ltxv",
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
    # These tests describe their cache state explicitly; never let a developer's real Unsloth
    # cache make an entry disappear from an otherwise hermetic plan.
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(lambda repo_id, filename, revision = None, expected_size = None, **kwargs: False),
    )


def _plan_cache(monkeypatch, cached):
    """Force the plan's cache verdict for every file."""
    from core.inference.diffusion import DiffusionBackend
    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(
            lambda repo_id, filename, revision = None, expected_size = None, **kwargs: cached(filename)
        ),
    )


def test_download_plan_omits_cached_video_files_but_keeps_the_footprint(monkeypatch):
    # The Video page plans every hub pick, so an unfiltered plan would re-stage a model that is
    # already on disk. required_bytes stays the full footprint either way.
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )
    _plan_cache(monkeypatch, lambda name: name.endswith(".gguf"))

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    staged = {f for e in plan["entries"] for f in e["files"]}
    assert "ltx-2.3-22b-distilled.gguf" in staged, "the scoped claim stays stable as the repo warms"
    assert not any(
        e["checkpoint"] for e in plan["entries"] if e["repo_id"] == "unsloth/LTX-2.3-GGUF"
    )
    assert staged, "its uncached companions still have to be fetched"
    assert plan["checkpoint_bytes"] > 0
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])
    assert plan["required_bytes"] == plan["total_bytes"] + plan["checkpoint_bytes"]


def test_download_plan_is_empty_when_the_whole_video_pick_is_cached(monkeypatch):
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )
    _plan_cache(monkeypatch, lambda name: True)

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    assert plan["entries"] == [] and plan["total_bytes"] == 0
    # Nothing to fetch, but the load still needs every one of those bytes on disk.
    assert plan["required_bytes"] > 0


def test_download_plan_restages_a_video_file_shadowed_in_the_live_cache(monkeypatch):
    # The live cache holds a stale copy under the right name and the import-time cache the good
    # one. reuse_other_cache_root switches roots only when the live lookup resolves nothing, so
    # the stale entry shadows the good copy: accepting "cached in either root" drops the file from
    # the plan and the load then reads the stale file or refetches it inline.
    #
    # Every OTHER file here lives wholly in the import-time root, so the split branch cannot stage
    # anything on its own: only recognising the shadow makes this repo straddle the two roots.
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )
    from core.inference.diffusion import DiffusionBackend

    shadowed = "vae/ltx-2.3-22b-distilled_video_vae.safetensors"

    def _cached(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        roots = None,
    ):
        if roots != ("live",):
            return True  # every file has a good copy in the import-time root
        # The live root holds exactly one file, under the right name and with the wrong bytes.
        return filename == shadowed and expected_size is None

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(_cached))
    monkeypatch.setattr("core.inference.video.hub_cache_dir", lambda: "live")

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    staged = {f for e in plan["entries"] for f in e["files"]}
    assert shadowed in staged, "a shadowed file must be restaged, not trusted"
    # The shadowed file has to land in the live root, so the rest of that repo now straddles both
    # and comes with it. The base repo is wholly in the other root and stays untouched.
    assert staged == {s.rfilename for s in _LTX23_REPO_SIBLINGS} - {
        "vae/ltx-2.3-22b-dev_video_vae.safetensors"
    }


def test_download_plan_restages_a_video_base_split_across_cache_roots(monkeypatch):
    # Every file resolves in ONE of the two roots, so a per-file check finds nothing to do. But a
    # base straddling both roots cannot be handed to from_pretrained as a snapshot:
    # _predownload_base returns nothing and the assembly is pinned to hub_cache_dir(), so the
    # other-root subset is refetched inline, or the load fails offline.
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )
    from core.inference.diffusion import DiffusionBackend

    live_root = "live"
    elsewhere = {
        "vae/ltx-2.3-22b-distilled_video_vae.safetensors",
        "vae/ltx-2.3-22b-distilled_audio_vae.safetensors",
    }

    def _cached(
        repo_id,
        filename,
        revision = None,
        expected_size = None,
        roots = None,
    ):
        """Half the repo in the live root, half in the other one, nothing missing."""
        if roots == (live_root,):
            return filename not in elsewhere
        return True

    monkeypatch.setattr(DiffusionBackend, "_hub_file_is_cached", staticmethod(_cached))
    monkeypatch.setattr("core.inference.video.hub_cache_dir", lambda: live_root)

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    entry = next(e for e in plan["entries"] if e["repo_id"] == "unsloth/LTX-2.3-GGUF")
    expected_scope = {s.rfilename for s in _LTX23_REPO_SIBLINGS} - {
        "vae/ltx-2.3-22b-dev_video_vae.safetensors"
    }
    assert set(entry["files"]) == expected_scope
    sizes = {s.rfilename: s.size for s in _LTX23_REPO_SIBLINGS}
    assert entry["bytes"] == sum(sizes[name] for name in elsewhere)


def test_download_plan_keeps_a_video_base_that_lives_wholly_in_the_other_root(monkeypatch):
    # Not a split: reuse_other_cache_root resolves the whole repo from the other root, so staging
    # any of it would re-download a model that is entirely on disk.
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        DiffusionBackend,
        "_hub_file_is_cached",
        staticmethod(
            lambda repo_id, filename, revision = None, expected_size = None, roots = None: (
                roots != ("live",)
            )
        ),
    )
    monkeypatch.setattr("core.inference.video.hub_cache_dir", lambda: "live")

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    assert plan["entries"] == [] and plan["required_bytes"] > 0


def test_a_companion_only_entry_is_not_labelled_the_checkpoint(monkeypatch):
    # The 2.3 checkpoint and its extras share one repo. With the GGUF already cached and the
    # extras missing, the stable download scope still names the checkpoint for job adoption,
    # but only the companion files contribute bytes. Labelling that work as the model file
    # would misdescribe what the panel is downloading.
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )
    _plan_cache(monkeypatch, lambda name: name.endswith(".gguf"))

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    entry = next(e for e in plan["entries"] if e["repo_id"] == "unsloth/LTX-2.3-GGUF")
    assert "ltx-2.3-22b-distilled.gguf" in entry["files"]
    assert entry["bytes"] == 2_400_000_000 + 200_000_000 + 900_000_000
    assert entry["checkpoint"] is False, "companion files only, so not the model file"


def test_the_checkpoint_entry_is_labelled_when_its_file_is_staged(monkeypatch):
    # The other half of the same rule: nothing cached, so the GGUF itself is in the entry.
    _plan_api(
        monkeypatch,
        {
            "unsloth/LTX-2.3-GGUF": _LTX23_REPO_SIBLINGS,
            "Lightricks/LTX-2": _LTX_BASE_SIBLINGS,
        },
    )
    _plan_cache(monkeypatch, lambda name: False)

    plan = VideoBackend().download_plan(
        "unsloth/LTX-2.3-GGUF",
        gguf_filename = "ltx-2.3-22b-distilled.gguf",
        family_override = "ltx-2",
    )

    entry = next(e for e in plan["entries"] if e["repo_id"] == "unsloth/LTX-2.3-GGUF")
    assert "ltx-2.3-22b-distilled.gguf" in entry["files"]
    assert entry["checkpoint"] is True


def test_h3_native_download_plan_stages_the_complete_runtime(monkeypatch):
    # The mirror carries the Qwen3-VL encoder quants alongside the denoisers AND the VAEs, so a
    # single repo covers the whole native runtime and the plan names no other owner.
    _plan_api(
        monkeypatch,
        {
            "unsloth/MiniMax-H3-GGUF": [
                _PlanSibling("minimax_h3_fl2va-Q4_K_M.gguf", 19),
                _PlanSibling("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", 18),
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
    # Every owner the plan names, not just the one we went looking for: staging a fresh install
    # from anyone else is the regression this test exists to catch.
    assert list(by_repo) == ["unsloth/MiniMax-H3-GGUF"]
    assert by_repo["unsloth/MiniMax-H3-GGUF"]["files"] == [
        "minimax_h3_fl2va-Q4_K_M.gguf",
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ]
    assert plan["total_bytes"] == 43

    assert plan["required_bytes"] == 43
    assert plan["checkpoint_bytes"] == 19
    assert by_repo["unsloth/MiniMax-H3-GGUF"]["checkpoint"] is True

    _plan_cache(monkeypatch, lambda name: name == "minimax_h3_fl2va-Q4_K_M.gguf")
    warming = VideoBackend().download_plan(
        "unsloth/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        family_override = "minimax-h3",
        model_kind = "gguf",
    )
    warming_entry = next(e for e in warming["entries"] if e["repo_id"] == "unsloth/MiniMax-H3-GGUF")
    assert warming_entry["files"] == by_repo["unsloth/MiniMax-H3-GGUF"]["files"]
    assert warming_entry["checkpoint"] is False

    _plan_cache(monkeypatch, lambda _name: True)
    cached = VideoBackend().download_plan(
        "unsloth/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        family_override = "minimax-h3",
        model_kind = "gguf",
    )
    assert cached == {
        "entries": [],
        "total_bytes": 0,
        "required_bytes": 43,
        "checkpoint_bytes": 19,
    }


def test_h3_native_uses_the_local_bundles_own_text_encoder(monkeypatch, tmp_path):
    """A local clone of the H3 GGUF bundle ships the encoder beside the denoisers.

    Hardcoding the Hub repo for the encoder re-fetches multiple GB that are already on disk next
    to the picked checkpoint, and fails outright with no network. The denoiser was resolved
    locally and the encoder was not, from the same directory.
    """
    local = tmp_path / "MiniMax-H3-GGUF"
    local.mkdir()
    (local / "minimax_h3_fl2va-Q4_K_M.gguf").write_bytes(b"x")
    (local / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf").write_bytes(b"x")

    assert VideoBackend._h3_text_encoder_repo(
        str(local), "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    ) == str(local)
    # A clone WITHOUT the encoder still falls back to the hosted copy, so nothing regresses.
    bare = tmp_path / "bare"
    bare.mkdir()
    assert (
        VideoBackend._h3_text_encoder_repo(str(bare), "qwen3vl_32b_minimax_h3-Q4_K_M.gguf")
        == "unsloth/MiniMax-H3-GGUF"
    )

    # And the plan stages only the VAEs: both halves of the local bundle are already on disk.
    _plan_api(
        monkeypatch,
        {
            # One repo now: the VAEs were mirrored in beside the denoisers, so the native pick
            # no longer reaches a second, community-owned repo for them.
            "unsloth/MiniMax-H3-GGUF": [
                _PlanSibling("minimax_h3_fl2va-Q4_K_M.gguf", 19),
                _PlanSibling("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", 18),
                _PlanSibling("vae/minimax_h3_video_vae_fp16.safetensors", 5),
                _PlanSibling("vae/minimax_h3_audio_vae_fp32.safetensors", 1),
            ],
        },
    )
    plan = VideoBackend._h3_native_download_plan(
        str(local), "minimax_h3_fl2va-Q4_K_M.gguf", hf_token = None
    )
    # The VAEs are the only staged entry, and they now come from our own mirror rather than the
    # community repack the components used to be served from.
    assert [entry["repo_id"] for entry in plan["entries"]] == ["unsloth/MiniMax-H3-GGUF"]
    assert plan["total_bytes"] == 6


_H3_BASE_SIBLINGS = [
    _PlanSibling("transformer/config.json", 1),
    _PlanSibling("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 40),
    _PlanSibling("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 26),
    _PlanSibling("scheduler/scheduler_config.json", 1),
    _PlanSibling("vae/diffusion_pytorch_model.safetensors", 11),
]

_H3_PREQUANT_SIBLINGS = [
    _PlanSibling("MiniMax-H3-INT8.pt", 20),
    _PlanSibling("MiniMax-H3-FP8.pt", 20),
]


def test_download_plan_stages_the_prequant_denoiser_it_drops_the_dense_shards_for(monkeypatch):
    # The dense DiT shards leave the base entry as soon as a hosted pre-quantized checkpoint
    # covers them, so the checkpoint has to arrive in their place. Without it the plan skipped
    # 66 GB and added nothing: the byte total under-reported the stage by the size of the
    # artifact, the disk preflight cleared a volume that could not hold it, and an offline stage
    # finished without the one file the load opens.
    _cuda_bf16_target(monkeypatch)
    _plan_api(
        monkeypatch,
        {
            "MiniMaxAI/MiniMax-H3": _H3_BASE_SIBLINGS,
            "unsloth/MiniMax-H3-FP8": _H3_PREQUANT_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "MiniMaxAI/MiniMax-H3",
        family_override = "minimax-h3",
        model_kind = "pipeline",
        transformer_quant = "int8",
    )

    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    # int8 and fp8 share one repo, so the SCHEME picks the file, not the repo.
    assert by_repo["unsloth/MiniMax-H3-FP8"]["files"] == ["MiniMax-H3-INT8.pt"]
    base = by_repo["MiniMaxAI/MiniMax-H3"]
    assert not any(f.startswith("transformer/diffusion_pytorch_model") for f in base["files"])
    # The config still goes, exactly as on the pre-cast encoder path: the class is read from it.
    assert "transformer/config.json" in base["files"]
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_download_plan_keeps_the_dense_denoiser_when_the_prequant_repo_is_missing(monkeypatch):
    # An unpublished / gated / renamed checkpoint must neither strand the load without a denoiser
    # nor sink the whole plan: the dense shards stay and no phantom entry is invented.
    _cuda_bf16_target(monkeypatch)

    class _Api:
        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            if repo_id == "unsloth/MiniMax-H3-FP8":
                raise RuntimeError("404 gated")
            return _PlanInfo(_H3_BASE_SIBLINGS)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api())

    plan = VideoBackend().download_plan(
        "MiniMaxAI/MiniMax-H3",
        family_override = "minimax-h3",
        model_kind = "pipeline",
        transformer_quant = "int8",
    )

    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert "unsloth/MiniMax-H3-FP8" not in by_repo
    assert any(
        f.startswith("transformer/diffusion_pytorch_model")
        for f in by_repo["MiniMaxAI/MiniMax-H3"]["files"]
    )
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


_H3_REF_BASE_SIBLINGS = _H3_BASE_SIBLINGS + [
    # What marks the repo as the modular workflow, and so what makes the partition split real.
    _PlanSibling("modular_model_index.json", 1),
    _PlanSibling("transformer_ref/config.json", 1),
    _PlanSibling("transformer_ref/diffusion_pytorch_model-00001-of-00002.safetensors", 40),
    _PlanSibling("transformer_ref/diffusion_pytorch_model-00002-of-00002.safetensors", 26),
]


def test_download_plan_keeps_the_dense_reference_shards_when_its_artifact_is_absent(monkeypatch):
    # A task-specific row gets NO filename fallback, so a Ref2VA checkpoint that is renamed or not
    # yet published resolves to nothing while the repo itself reads fine. The registry still says
    # the scheme is covered, and dropping transformer_ref/ on that word alone leaves a plan with
    # neither denoiser: the disk preflight under-reports by 66 GB and an offline stage finishes
    # with nothing for the documented bf16 fallback to open.
    _cuda_bf16_target(monkeypatch)
    _plan_api(
        monkeypatch,
        {
            "MiniMaxAI/MiniMax-H3": _H3_REF_BASE_SIBLINGS,
            # Only the keyframe artifacts; the reference ones are missing.
            "unsloth/MiniMax-H3-FP8": _H3_PREQUANT_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "MiniMaxAI/MiniMax-H3",
        family_override = "minimax-h3",
        model_kind = "pipeline",
        transformer_quant = "fp8",
        h3_task = "ref2va",
    )

    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    # No entry may be invented for a file the repo does not have.
    assert "unsloth/MiniMax-H3-FP8" not in by_repo
    base = by_repo["MiniMaxAI/MiniMax-H3"]["files"]
    assert "transformer_ref/diffusion_pytorch_model-00001-of-00002.safetensors" in base
    assert "transformer_ref/diffusion_pytorch_model-00002-of-00002.safetensors" in base
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_download_plan_still_drops_the_reference_shards_its_artifact_replaces(monkeypatch):
    # The other half of the gate: once the Ref2VA artifact really resolves, the dense reference
    # shards leave the base entry and the checkpoint is staged in their place.
    _cuda_bf16_target(monkeypatch)
    _plan_api(
        monkeypatch,
        {
            "MiniMaxAI/MiniMax-H3": _H3_REF_BASE_SIBLINGS,
            "unsloth/MiniMax-H3-FP8": _H3_PREQUANT_SIBLINGS
            + [_PlanSibling("MiniMax-H3-Ref2VA-FP8.pt", 20)],
        },
    )

    plan = VideoBackend().download_plan(
        "MiniMaxAI/MiniMax-H3",
        family_override = "minimax-h3",
        model_kind = "pipeline",
        transformer_quant = "fp8",
        h3_task = "ref2va",
    )

    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert by_repo["unsloth/MiniMax-H3-FP8"]["files"] == ["MiniMax-H3-Ref2VA-FP8.pt"]
    base = by_repo["MiniMaxAI/MiniMax-H3"]["files"]
    assert not any(f.startswith("transformer_ref/diffusion_pytorch_model") for f in base)
    assert "transformer_ref/config.json" in base
    assert plan["total_bytes"] == sum(e["bytes"] for e in plan["entries"])


def test_the_verified_probe_keeps_the_dense_shards_when_the_artifact_is_absent(monkeypatch):
    # The load path drops 66 GB from the pull and never re-checks, so the registry's word is not
    # enough: only an artifact that really resolves on the Hub earns that skip. Same rule the
    # conditioner already follows through _h3_te_quant_scheme_verified.
    from core.inference.video import _detect_load_family as _fam

    fam = _fam("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    backend = VideoBackend()

    class _Api:
        def __init__(self, names):
            self._names = names

        def model_info(
            self,
            repo_id,
            files_metadata = False,
            token = None,
        ):
            return _PlanInfo([_PlanSibling(n, 20) for n in self._names])

    def _use(names):
        monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Api(names))

    # The keyframe artifacts exist, the reference one does not.
    _use(["MiniMax-H3-FP8.pt", "MiniMax-H3-INT8-ConvRot.pt"])
    assert (
        backend._denoiser_prequant_verified(fam, "fp8", "MiniMaxAI/MiniMax-H3", "ref2va", None)
        is False
    )
    assert (
        backend._denoiser_prequant_verified(fam, "fp8", "MiniMaxAI/MiniMax-H3", "fl2va", None)
        is True
    )

    # Once the reference artifact is published the skip is earned again.
    _use(["MiniMax-H3-FP8.pt", "MiniMax-H3-Ref2VA-FP8.pt"])
    assert (
        backend._denoiser_prequant_verified(fam, "fp8", "MiniMaxAI/MiniMax-H3", "ref2va", None)
        is True
    )

    # An unreadable repo is a refusal, not a pass: the dense shards stay.
    class _Boom:
        def model_info(self, *a, **k):
            raise RuntimeError("404 gated")

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _Boom())
    assert (
        backend._denoiser_prequant_verified(fam, "fp8", "MiniMaxAI/MiniMax-H3", "ref2va", None)
        is False
    )
    # And bf16 never asks in the first place.
    assert (
        backend._denoiser_prequant_verified(fam, None, "MiniMaxAI/MiniMax-H3", "ref2va", None)
        is False
    )


def test_the_load_path_gates_its_dense_skip_on_the_verified_probe():
    """``_run_load`` must ask the VERIFIED probe, not the registry-only one.

    This is the flag that removes the dense denoiser from the actual pull, so a registry-only
    answer stages neither denoiser and leaves the loader's documented bf16 fallback with nothing
    to open offline.
    """
    import ast
    import inspect
    import textwrap

    from core.inference.video import VideoBackend as _VB

    tree = ast.parse(textwrap.dedent(inspect.getsource(_VB._run_load)))
    assigned = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "skip_transformer_weights" for t in n.targets)
    ]
    assert assigned, "skip_transformer_weights is no longer assigned in _run_load"
    called = {
        n.func.attr
        for a in assigned
        for n in ast.walk(a)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_denoiser_prequant_verified" in called
    assert "_denoiser_prequant_covered" not in called


def test_a_named_dit_view_presents_the_reference_denoiser_as_the_transformer():
    # The speed and attention helpers enumerate transformer / transformer_2 /
    # unconditional_transformer only. H3 keeps one denoiser per partition, so a reference load's
    # DiT is invisible to both without this view.
    from core.inference.diffusion_attention import _attention_dits
    from core.inference.diffusion_speed import _denoiser_dits
    from core.inference.video import _denoiser_view

    ref = object()
    pipe = types.SimpleNamespace(transformer_ref = ref, vae = "vae")

    assert _denoiser_dits(pipe) == [] and _attention_dits(pipe) == []

    view = _denoiser_view(pipe, "transformer_ref")
    assert view.transformer is ref
    assert _denoiser_dits(view) == [ref]
    assert _attention_dits(view) == [ref]
    # Everything else reads through, and a helper's reassignment lands on the real partition.
    assert view.vae == "vae"
    replacement = object()
    view.transformer = replacement
    assert pipe.transformer_ref is replacement
    assert not hasattr(pipe, "transformer")

    # A keyframe load is handed the pipe itself, so its behaviour is byte-for-byte unchanged.
    keyframe = types.SimpleNamespace(transformer = ref)
    assert _denoiser_view(keyframe, "transformer") is keyframe


def test_the_h3_loader_optimises_the_partition_it_denoises_with():
    """``apply_attention_backend`` / ``apply_speed_optims`` must see this workflow's denoiser.

    The pre-quantized reference pin keeps the profile out of the eager downgrade, so handing these
    the bare pipe leaves the reference DiT native and uncompiled while the resolved record still
    reports the requested profile.
    """
    import ast
    import inspect
    import textwrap

    from core.inference.video import VideoBackend as _VB

    tree = ast.parse(textwrap.dedent(inspect.getsource(_VB._load_h3_modular_pipeline)))
    for helper in ("apply_attention_backend", "apply_speed_optims"):
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == helper
        ]
        assert len(calls) == 1, f"{helper} is called {len(calls)} times"
        first = calls[0].args[0]
        assert (
            getattr(first, "id", None) == "speed_view"
        ), f"{helper} is handed {ast.dump(first)}, not the denoiser view"


def test_download_plan_adds_no_prequant_entry_when_bfloat16_is_pinned(monkeypatch):
    # transformer_quant="none" keeps the dense shards, so there is nothing to replace and no
    # third repo to stage. An UNSET request resolves to int8 now and stages the hosted
    # checkpoint instead, which is a different case; this one is the explicit opt-out.
    _cuda_bf16_target(monkeypatch)
    _plan_api(
        monkeypatch,
        {
            "MiniMaxAI/MiniMax-H3": _H3_BASE_SIBLINGS,
            "unsloth/MiniMax-H3-FP8": _H3_PREQUANT_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        "MiniMaxAI/MiniMax-H3",
        family_override = "minimax-h3",
        model_kind = "pipeline",
        transformer_quant = "none",
    )

    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert "unsloth/MiniMax-H3-FP8" not in by_repo
    assert any(
        f.startswith("transformer/diffusion_pytorch_model")
        for f in by_repo["MiniMaxAI/MiniMax-H3"]["files"]
    )


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


def test_h3_native_load_claims_the_companion_repos_before_the_preflight(monkeypatch, tmp_path):
    # asset_repos stops the delete-cached guard dropping the H3 companion repos mid-load, and the
    # preflight can spend minutes installing the sd-cli prebuilt. A delete admitted in that window
    # is not revoked by claiming the repos later.
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend, sd_cpp_engine
    from core.inference.video_minimax_h3 import (
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        H3_LEGACY_COMPONENT_REPO,
    )

    class _Api:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, *_args, **_kwargs):
            return _PlanInfo([])

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "cpu", device = "cpu", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)

    seen: list[tuple[str, ...]] = []

    def _ensure(**_kwargs):
        # What the delete-cached guard would see while the install runs.
        seen.append(backend._loading.asset_repos)
        return "/existing/sd-cli"

    monkeypatch.setattr(sd_cpp_backend, "ensure_h3_sd_cpp_binary", _ensure)
    monkeypatch.setattr(sd_cpp_backend, "sd_cpp_binary_vets_for_h3", lambda _b: True)

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    def _download(_repo, wanted, *_args, **_kwargs):
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

    backend = VideoBackend()
    fam = _detect_load_family("unsloth/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    backend._loading = video_mod._VideoLoadingState(
        repo_id = "unsloth/MiniMax-H3-GGUF", base_repo = fam.base_repo
    )
    backend._load_token = 11

    backend._run_load_h3_native(
        fam = fam,
        token = 11,
        cancel_event = threading.Event(),
        repo_id = "unsloth/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
    )

    # Every id a component source can resolve to, not one resolved answer: the two VAEs are
    # resolved independently, so an interrupted pre-move pull can leave one on the repack and the
    # other on the mirror, and a claim naming either alone leaves the other deletable mid-load.
    # Same triple begin_load publishes.
    assert seen == [(H3_GGUF_REPO, H3_COMPONENT_REPO, H3_LEGACY_COMPONENT_REPO)]


def test_begin_load_publishes_the_h3_companion_claim_with_the_loading_state(
    fake_runtime, monkeypatch
):
    # begin_load returns as soon as the worker thread is scheduled, so a claim made in the worker
    # leaves a window where the guard sees only repo_id and base_repo and admits a delete of a
    # companion repo. Admitting it is the irreversible part, so the claim has to be published in
    # the same locked section as _loading.
    import threading
    from types import SimpleNamespace

    from core.inference.video_minimax_h3 import (
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        H3_LEGACY_COMPONENT_REPO,
    )

    backend = VideoBackend()
    # Never started: the window under test is before the load thread is scheduled.
    monkeypatch.setattr(
        threading, "Thread", lambda *a, **k: SimpleNamespace(start = lambda: None, daemon = True)
    )

    backend.begin_load(
        "unsloth/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        family_override = "minimax-h3",
        model_kind = "gguf",
    )

    claimed = backend.loading_repo_ids()
    assert H3_GGUF_REPO in claimed
    assert H3_COMPONENT_REPO in claimed


def test_begin_load_claims_no_companion_repos_for_a_non_h3_family(fake_runtime, monkeypatch):
    # H3-native only: naming the companions on a pipeline load would block deletes of repos it
    # never reads.
    import threading
    from types import SimpleNamespace

    from core.inference.video_minimax_h3 import (
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        H3_LEGACY_COMPONENT_REPO,
    )

    backend = VideoBackend()
    monkeypatch.setattr(
        threading, "Thread", lambda *a, **k: SimpleNamespace(start = lambda: None, daemon = True)
    )

    backend.begin_load("Wan-AI/Wan2.2-TI2V-5B-Diffusers", family_override = "wan2.2-ti2v-5b")

    claimed = backend.loading_repo_ids()
    assert H3_GGUF_REPO not in claimed
    assert H3_COMPONENT_REPO not in claimed


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


def test_h3_native_cpu_fallback_releases_the_video_gpu_claim(monkeypatch, tmp_path):
    """The accelerator binary is missing, so the runtime commits to the CPU build and holds no
    VRAM; /video/load's VIDEO claim (taken off the non-CPU device target) must not survive it,
    else the next chat/image acquire evicts a model that is not on the GPU."""
    from core.inference import gpu_arbiter
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
        lambda: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: "/existing/sd-cli" if accelerator == "cpu" else None,
    )

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    def _download(_repo, wanted, *_args, **_kwargs):
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)

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

    assert backend._state is not None
    assert backend._state.device == "cpu"
    assert gpu_arbiter.current_owner() is None


def test_h3_native_accelerator_load_keeps_the_video_gpu_claim(monkeypatch, tmp_path):
    """The counterpart: when the accelerator binary IS there the runtime really uses the GPU,
    so the VIDEO claim has to stay."""
    from core.inference import gpu_arbiter
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
        lambda: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: "/existing/sd-cli",
    )

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    def _download(_repo, wanted, *_args, **_kwargs):
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)

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

    assert backend._state is not None
    assert backend._state.device == "cuda"
    assert gpu_arbiter.current_owner() == gpu_arbiter.VIDEO


def test_the_load_time_accelerator_probe_runs_under_the_reader_claim(monkeypatch, tmp_path):
    """--list-devices SPAWNS the managed sd-cli, so it needs the same claim the run takes.

    Unclaimed, an install started by another in-process load sees no reader and extracts over the
    executing binary: on Windows that fails on the locked file, on Linux it can leave the
    replacement half-written. The later claimed recheck compares answers; it cannot undo damage
    this first probe already allowed.
    """
    from core.inference import gpu_arbiter
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend, sd_cpp_engine

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    managed = root / "sd-bin" / "sd-cli"
    managed.write_bytes(b"managed")

    class _Api:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, *_args, **_kwargs):
            return _PlanInfo([])

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: str(managed),
    )

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    def _download(_repo, wanted, *_args, **_kwargs):
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)

    held: list = []

    def _watching_probe(binary):
        held.append(sd_cpp_backend._tree_readers)
        # An install decided at this instant must stand down rather than replace what is running.
        with sd_cpp_backend._tree_claimed_for_install() as claimed:
            held.append(claimed)
        return True

    # video.py imports this inside the load function, so the name must be replaced at its source
    # module; patching the video module misses it entirely and the test would pass vacuously.
    monkeypatch.setattr(sd_cpp_backend, "sd_cpp_lists_accelerator_device", _watching_probe)

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

    # First two entries are the load-time probe; the claimed recheck later in the load adds its
    # own pair, so assert on the opening ones rather than the whole list.
    assert held[:2] == [1, False], f"probe ran unclaimed: {held}"
    assert sd_cpp_backend._tree_readers == 0, "the claim must be released after the probe"


def _load_h3_native_offload(
    monkeypatch,
    tmp_path,
    *,
    help_text,
    accelerator = True,
    memory_mode = None,
):
    """Run the native H3 load against a stubbed sd-cli and hand back its committed offload flags.

    ``help_text`` is what the binary answers ``--help`` with, which is the only thing the graph-cut
    gate reads. ``accelerator = False`` reproduces the Linux CUDA host with only the CPU prebuilt,
    where the load commits to ``native_device = "cpu"``.
    """
    from core.inference import gpu_arbiter
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
        lambda: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: "/existing/sd-cli",
    )
    # One probe helper serves both --list-devices and --help, so the ggml device line rides along or the CPU case is unreachable.
    devices = "CUDA0\tNVIDIA GeForce RTX 4070 Ti\n" if accelerator else "CPU\tAMD Ryzen 9\n"
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", lambda *_a: devices + help_text)

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    def _download(_repo, wanted, *_args, **_kwargs):
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)

    backend = VideoBackend()
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    backend._run_load_h3_native(
        fam = fam,
        token = None,
        cancel_event = threading.Event(),
        repo_id = "leejet/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        memory_mode = memory_mode,
    )
    assert backend._state is not None
    return backend._state, list(backend._state.pipe.offload_flags)


# Both fixtures carry the project banner the identity gate reads and the H3 marker
# ensure_h3_sd_cpp_binary gates on, and differ only in the graph-cut options.
_H3_HELP = (
    "stable-diffusion.cpp version master-813\n"
    "  --ref-video           MiniMax-H3 Ref2VA reference video frame directory\n"
)
_GRAPH_CUT_HELP = (
    _H3_HELP + "  --max-vram <string>   VRAM budget\n  --stream-layers       prefetch\n"
)
_NO_GRAPH_CUT_HELP = _H3_HELP + "  --offload-to-cpu      place the weights in RAM\n"


def test_h3_native_emits_the_graph_cut_flags_on_an_accelerator(monkeypatch, tmp_path):
    """H3's modules are individually larger than the cards it is offered on, so --offload-to-cpu
    alone still allocates each one whole and cudaMallocs before any tensor is resident. The graph
    cut is what makes the checkpoint renderable, and auto -- not low_vram -- is the default mode
    that has to carry it."""
    state, offload = _load_h3_native_offload(monkeypatch, tmp_path, help_text = _GRAPH_CUT_HELP)
    assert state.device == "cuda"
    # auto offloads, which is what makes --stream-layers take effect.
    assert "--offload-to-cpu" in offload
    assert offload[-3:] == ["--max-vram", "-1", "--stream-layers"]
    # A negative budget auto-detects per device. A fixed number would misjudge the other card.
    assert "0" not in offload


def test_h3_native_drops_stream_layers_without_cpu_offload(monkeypatch, tmp_path):
    """fast keeps the params resident on the device, and upstream only honours --stream-layers when
    the diffusion params backend is CPU: without --offload-to-cpu it warns and ignores the flag.
    The budget still segments on its own, so --max-vram stays."""
    _state, offload = _load_h3_native_offload(
        monkeypatch, tmp_path, help_text = _GRAPH_CUT_HELP, memory_mode = "fast"
    )
    assert "--offload-to-cpu" not in offload
    assert offload[-2:] == ["--max-vram", "-1"]
    assert "--stream-layers" not in offload


def test_h3_native_skips_the_graph_cut_flags_on_an_older_build(monkeypatch, tmp_path):
    """sd-cli exits non-zero on an option it does not know, so emitting these unconditionally
    would break every generation on a build that predates the executor."""
    _state, offload = _load_h3_native_offload(monkeypatch, tmp_path, help_text = _NO_GRAPH_CUT_HELP)
    assert "--max-vram" not in offload
    assert "--stream-layers" not in offload


def test_h3_native_skips_the_graph_cut_flags_on_cpu(monkeypatch, tmp_path):
    """The cut splits a module to fit a device budget; the CPU backend allocates from system RAM,
    so there is nothing to size against."""
    state, offload = _load_h3_native_offload(
        monkeypatch, tmp_path, help_text = _GRAPH_CUT_HELP, accelerator = False
    )
    assert state.device == "cpu"
    assert "--max-vram" not in offload


def test_h3_native_reused_cpu_binary_still_commits_to_cpu(monkeypatch, tmp_path):
    """The second load on a Linux CUDA host. The first one installed the CPU prebuilt (upstream
    publishes no Linux CUDA asset for the pinned tag), and from then on ensure_sd_cpp_binary finds
    that binary and returns it whatever accelerator it is asked for -- so the fallback below it was
    skipped, native_device stayed "cuda", and Unsloth kept the VIDEO claim and applied GPU offload
    policy while sd-cli ran wholly on the CPU. This is the common path, not an edge case."""
    from core.inference import gpu_arbiter
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
        lambda: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    # The reuse itself: the already-installed binary comes back for every accelerator.
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: "/existing/sd-cli",
    )

    def _probe(_binary, *args):
        if args == ("--list-devices",):
            # What the CPU prebuilt answers: the one ggml device it was built with.
            return "CPU\tIntel(R) Xeon(R) Platinum 8559C\n"
        # The banner too: it is what identifies the binary AS stable-diffusion.cpp, and the gate
        # asks that before it asks whether the build carries H3.
        return (
            "stable-diffusion.cpp version unknown, commit unknown\n"
            "  --ref-video   MiniMax-H3 Ref2VA reference video frame directory at 24 fps\n"
        )

    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    def _download(_repo, wanted, *_args, **_kwargs):
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)
    monkeypatch.setattr(gpu_arbiter, "_owner", gpu_arbiter.VIDEO)

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

    assert backend._state is not None
    assert backend._state.device == "cpu"
    assert backend._state.offload_policy == "none"
    # Reaching "cpu" is what lets the existing release_if drop the claim on this path too.
    assert gpu_arbiter.current_owner() is None


def _h3_managed_cpu_fallback_load(monkeypatch, tmp_path, *, swap_on_fallback):
    """An H3 native load on a CUDA target whose first probe drops it to the CPU build.

    The binary is a MANAGED one, i.e. one an install may replace at the same path.
    ``swap_on_fallback`` makes that install land the moment the fallback's ensure returns, which
    is the window the second probe used to be sampled in."""
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend, sd_cpp_engine

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    managed = root / "sd-bin" / "sd-cli"
    managed.write_bytes(b"the cpu build the fallback asked for")

    class _Api:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, *_args, **_kwargs):
            return _PlanInfo([])

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "cuda", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)

    swapped: list = []

    def _ensure(*, allow_install, accelerator):
        if accelerator == "cpu" and swap_on_fallback:
            # The CPU bundle this call asked for is what came back; the install that replaces it
            # with a GPU build lands immediately afterwards, before anything probes it again.
            swapped.append(True)
        return str(managed)

    monkeypatch.setattr(sd_cpp_backend, "ensure_sd_cpp_binary", _ensure)

    def _probe(_binary, *args):
        if args == ("--list-devices",):
            if swapped:
                return "CUDA0\tNVIDIA H100 PCIe\nCPU\tIntel(R) Xeon(R) Platinum 8559C\n"
            return "CPU\tIntel(R) Xeon(R) Platinum 8559C\n"
        # The banner too: it is what identifies the binary AS stable-diffusion.cpp, and the gate
        # asks that before it asks whether the build carries H3.
        return (
            "stable-diffusion.cpp version unknown, commit unknown\n"
            "  --ref-video   MiniMax-H3 Ref2VA reference video frame directory at 24 fps\n"
        )

    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
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

    def _run():
        backend._run_load_h3_native(
            fam = fam,
            token = None,
            cancel_event = threading.Event(),
            repo_id = "leejet/MiniMax-H3-GGUF",
            gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        )

    return backend, _run


def test_h3_native_cpu_fallback_refuses_a_gpu_build_that_replaced_it(monkeypatch, tmp_path):
    """The CPU fallback's baseline is the DECISION, not a second probe. Sampling the binary again
    after the fallback's ensure records whatever an install has just put at that path, so the
    re-check under the reader claim compared a CUDA executable against itself and passed -- with
    native_device already forced to "cpu" and CPU resource accounting committed around a build
    that runs on VRAM nothing accounted for."""
    backend, run = _h3_managed_cpu_fallback_load(monkeypatch, tmp_path, swap_on_fallback = True)
    with pytest.raises(RuntimeError, match = "different accelerator"):
        run()
    assert backend._state is None


def test_h3_native_cpu_fallback_still_commits_to_a_managed_cpu_build(monkeypatch, tmp_path):
    """The control: nothing replaced the binary, so the fallback has to load exactly as before.
    A CPU fallback refused for the accelerator it deliberately chose would take H3 away from every
    GPU host with no matching prebuilt, which is the common case this fallback exists for."""
    backend, run = _h3_managed_cpu_fallback_load(monkeypatch, tmp_path, swap_on_fallback = False)
    run()
    assert backend._state is not None
    assert backend._state.device == "cpu"
    assert backend._state.offload_policy == "none"


def test_h3_native_load_publishes_the_companion_repos_while_downloading(monkeypatch, tmp_path):
    """The in-flight twin of loaded_repo_ids(). An H3 load downloads from the GGUF and component
    companion repos as well as repo_id, but loading_repo_ids() reported only repo_id and base_repo,
    so the cached-model delete guard allowed deleting Comfy-Org/MiniMax-H3 (and, when loading from
    another mirror as here, the GGUF companion) out from under the running download."""
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend, sd_cpp_engine
    from core.inference.video_minimax_h3 import (
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        H3_LEGACY_COMPONENT_REPO,
    )

    class _Api:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, *_args, **_kwargs):
            return _PlanInfo([])

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "cpu", device = "cpu", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: "/existing/sd-cli",
    )

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    backend = VideoBackend()
    backend._load_token = 7
    backend._loading = video_mod._VideoLoadingState(repo_id = "leejet/MiniMax-H3-GGUF", base_repo = "")
    guarded: list[tuple[str, ...]] = []

    def _download(_repo, wanted, *_args, **_kwargs):
        # Sampled per file, so the guard has to be armed BEFORE any of the bytes land.
        guarded.append(tuple(backend.loading_repo_ids()))
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    backend._run_load_h3_native(
        fam = fam,
        token = 7,
        cancel_event = threading.Event(),
        repo_id = "leejet/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
    )

    assert len(guarded) == 4
    for ids in guarded:
        assert "leejet/MiniMax-H3-GGUF" in ids
        assert fam.base_repo in ids
        assert H3_GGUF_REPO in ids
        assert H3_COMPONENT_REPO in ids


def test_h3_native_load_refuses_a_binary_that_predates_h3(monkeypatch, tmp_path):
    """ensure_sd_cpp_binary probes runnability only, so an sd.cpp build older than H3 support is
    handed back and clears the version() gate: the load reported ready and the first generation
    failed, i.e. AFTER the multi-tens-of-GB bundle had downloaded. Gate on the capability.

    And gate BEFORE the download: the point of refusing early is lost if the four-file bundle has
    already been fetched by the time the gate runs, which is what the ordering here asserts."""
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
        lambda: types.SimpleNamespace(backend = "cpu", device = "cpu", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: "/usr/local/bin/sd-cli",
    )
    # The user's own build: not ours to delete, so the load must fail rather than reinstall.
    monkeypatch.setattr(sd_cpp_backend, "is_managed_binary", lambda _b: False)
    # A pre-H3 --help. vid_gen and --audio-vae are both in it (they predate H3, they came with
    # LTX-2), which is why the H3-only options are what the gate looks for.
    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda *_args: (
            # The banner is what identifies it AS stable-diffusion.cpp. Without it the gate would
            # (correctly) report a different failure: not an old build, not the project at all.
            "stable-diffusion.cpp version unknown, commit unknown\n"
            "  -M, --mode                    run mode, one of [img_gen, vid_gen, upscale]\n"
            "  --audio-vae <string>          path to standalone LTX audio vae model\n"
        ),
    )

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            # The old build runs perfectly well, which is exactly why this gate cannot catch it.
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    downloads: list[str] = []

    def _download(_repo, wanted, *_args, **_kwargs):
        downloads.append(wanted)
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

    backend = VideoBackend()
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    with pytest.raises(RuntimeError, match = "does not advertise MiniMax-H3"):
        backend._run_load_h3_native(
            fam = fam,
            token = None,
            cancel_event = threading.Event(),
            repo_id = "leejet/MiniMax-H3-GGUF",
            gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        )

    assert backend._state is None
    # Not one byte of the bundle: the binary is vetted before the four files are resolved.
    assert downloads == []


def test_h3_native_load_refuses_a_missing_binary_before_downloading(monkeypatch, tmp_path):
    """ensure_h3_sd_cpp_binary returns None whenever it cannot produce a binary -- auto-install off,
    unsupported platform, no network, or a stale managed copy something else is running out of. The
    only `not binary` check used to sit after the download loop, so every one of those cases still
    fetched the four-file bundle first."""
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend

    class _Api:
        def __init__(self, **_kwargs):
            pass

        def model_info(self, *_args, **_kwargs):
            return _PlanInfo([])

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "cpu", device = "cpu", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: False)
    monkeypatch.setattr(sd_cpp_backend, "ensure_h3_sd_cpp_binary", lambda **_kwargs: None)

    downloads: list[str] = []

    def _download(_repo, wanted, *_args, **_kwargs):
        downloads.append(wanted)
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

    backend = VideoBackend()
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    with pytest.raises(RuntimeError, match = "could not be installed or started"):
        backend._run_load_h3_native(
            fam = fam,
            token = None,
            cancel_event = threading.Event(),
            repo_id = "leejet/MiniMax-H3-GGUF",
            gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        )

    assert backend._state is None
    assert downloads == []


def test_h3_native_load_checks_cancellation_before_the_binary_preflight(monkeypatch):
    """The preflight may install the sd-cli prebuilt and takes no cancel_event, so a load cancelled
    before this thread got going must not pay for it. The download loop used to be the first check;
    moving the preflight above it put an uncancellable install in front of that."""
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend

    ensures: list[str] = []
    monkeypatch.setattr(
        video_mod,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "cpu", device = "cpu", dtype = None),
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_h3_sd_cpp_binary",
        lambda **_kwargs: (ensures.append("ensure"), "/usr/local/bin/sd-cli")[1],
    )

    cancelled = threading.Event()
    cancelled.set()
    backend = VideoBackend()
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend._run_load_h3_native(
            fam = fam,
            token = None,
            cancel_event = cancelled,
            repo_id = "leejet/MiniMax-H3-GGUF",
            gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        )
    assert ensures == []


def test_h3_native_load_refuses_a_binary_that_is_not_sd_cpp_before_downloading(
    monkeypatch, tmp_path
):
    """#8507: an unrelated executable named `sd` was reported as an sd.cpp build predating H3.

    Discovery now skips it, so reaching the gate takes a deliberate SD_CLI_PATH override -- and
    when it is reached the message says what is actually wrong, still before any download."""
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
        lambda: types.SimpleNamespace(backend = "cpu", device = "cpu", dtype = None),
    )
    monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        sd_cpp_backend,
        "ensure_sd_cpp_binary",
        lambda *, allow_install, accelerator: "/usr/bin/sd",
    )
    monkeypatch.setattr(sd_cpp_backend, "is_managed_binary", lambda _b: False)
    # Debian/Ubuntu's find-and-replace `sd`, which is what /usr/bin/sd is on the reporter's box.
    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda *_args: "sd 1.0.0\nFind & replace CLI\n\nUSAGE:\n    sd <find> <replace-with>\n",
    )

    class _Engine:
        def __init__(self, binary):
            self.binary = binary

        def version(self):
            return "stub-version"

    monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

    downloads: list[str] = []

    def _download(_repo, wanted, *_args, **_kwargs):
        downloads.append(wanted)
        path = tmp_path / Path(wanted).name
        path.write_bytes(b"x")
        return str(path)

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

    backend = VideoBackend()
    fam = _detect_load_family("leejet/MiniMax-H3-GGUF", None, "minimax-h3")
    assert fam is not None
    with pytest.raises(RuntimeError, match = "is not stable-diffusion.cpp"):
        backend._run_load_h3_native(
            fam = fam,
            token = None,
            cancel_event = threading.Event(),
            repo_id = "leejet/MiniMax-H3-GGUF",
            gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
        )

    assert backend._state is None
    assert downloads == []


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


def test_h3_native_loaded_repo_ids_cover_the_companion_repos():
    # The delete guard reads repo_id + base_repo, but the native runtime keeps paths into the GGUF
    # mirror (Qwen encoder) and the component repo (both VAEs) and re-reads them every generation,
    # so deleting either On Device while H3 is loaded must be refused.
    from core.inference.video import _VideoLoadState
    from core.inference.video_minimax_h3 import (
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        H3_LEGACY_COMPONENT_REPO,
    )

    backend = VideoBackend()
    assert backend.loaded_repo_ids() == ()
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
    # Both component ids: the components moved to our own mirror, but an install whose cache
    # predates that move still reads them from the repack under the old id, and the delete guard
    # has to cover whichever one this load is actually holding.
    assert backend.loaded_repo_ids() == (
        "leejet/MiniMax-H3-GGUF",
        H3_GGUF_REPO,
        H3_COMPONENT_REPO,
        H3_LEGACY_COMPONENT_REPO,
    )
    # A diffusers load holds its weights in memory, and base_repo already names its repo.
    object.__setattr__(backend._state, "engine", "diffusers")
    assert backend.loaded_repo_ids() == ()


def test_h3_modular_load_forwards_the_hub_token_to_the_component_loads(fake_runtime):
    # The index load and the component from_pretrained calls are separate hub trips; the token has
    # to reach both, or gated/private components load anonymously (and only warn on failure).
    pipe = _load_h3_modular(VideoBackend(), hf_token = "hf_secret")
    assert _FakeModularPipeline.last["token"] == "hf_secret"
    assert pipe.load_kwargs["token"] == "hf_secret"
    # Without a configured token nothing extra is passed, so the hub's own resolution still applies.
    pipe = _load_h3_modular(VideoBackend())
    assert "token" not in pipe.load_kwargs


def test_h3_modular_load_pins_the_component_loads_to_the_studio_cache(fake_runtime):
    """The component from_pretrained calls need the same cache_dir the index load got.

    load_components forwards its extra kwargs through ComponentSpec.load into each component's
    from_pretrained. Without cache_dir those ~145 GB of Hub-pinned components resolve against the
    HF_HUB_CACHE snapshot taken at import time, while the scoped pre-download stages into the
    cache folder Unsloth currently points at, and the two really can differ (it is a live setting).
    """
    from core.inference.video import hub_cache_dir

    pipe = _load_h3_modular(VideoBackend())
    assert _FakeModularPipeline.last["cache_dir"] == hub_cache_dir()
    assert pipe.load_kwargs["cache_dir"] == hub_cache_dir()


def test_h3_modular_load_refuses_a_text_encoder_quant_it_cannot_honour(fake_runtime):
    """An explicit text_encoder_quant the modular path cannot honour must RAISE.

    This started life asserting the weaker contract: record the request so a decline reads as
    FELL_BACK rather than vanishing into a backend-owned row stamped requested=null / APPLIED.
    #8283 went further and made the modular encoder path fail closed, matching what the
    conventional path already did, so an unhonourable explicit request never reaches a load at
    all. That subsumes the original concern -- a request cannot be silently dropped from a run
    that does not happen -- so this pins the refusal, and the reason, instead.
    """
    backend = VideoBackend()
    diffusers = sys.modules["diffusers"]
    diffusers.ComponentsManager = _FakeComponentsManager
    diffusers.ModularPipeline = _FakeModularPipeline
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    with pytest.raises(RuntimeError, match = "text_encoder_quant='fp8' could not be used"):
        backend._load_h3_modular_pipeline(
            diffusers = diffusers,
            torch = sys.modules["torch"],
            fam = fam,
            repo_id = "MiniMaxAI/MiniMax-H3",
            base = fam.base_repo,
            kind = "pipeline",
            dtype = sys.modules["torch"].bfloat16,
            device = "cpu",
            hf_token = None,
            memory_mode = None,
            text_encoder_quant = "fp8",
            _load_token = None,
            _base_local_dir = None,
        )


def test_h3_modular_generation_ticks_and_cancels_through_the_scheduler(fake_runtime):
    backend = VideoBackend()
    pipe = _load_h3_modular(backend)
    steps_seen: list = []
    pipe.scheduler.on_step = lambda n: steps_seen.append(backend._gen.get("step"))
    result = backend.generate(prompt = "a fox", steps = 4)
    assert "callback_on_step_end" not in pipe.last_kwargs
    # One wrapped tick per denoise step, then the original method back in place.
    assert steps_seen == [1, 2, 3, 4]
    assert pipe.scheduler.step.__func__ is _FakeH3Scheduler.step
    assert result["num_frames"] == 124 and result["has_audio"] is False

    backend = VideoBackend()
    pipe = _load_h3_modular(backend)
    # The cancel lands during the FIRST step; the next wrapped call must unwind the modular denoise
    # instead of leaving Cancel to take effect only after the whole multi-minute run returns.
    pipe.scheduler.on_step = lambda n: backend.cancel_generate() if n == 1 else None
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(prompt = "a fox", steps = 4)
    assert pipe.scheduler.calls == 1
    assert pipe.scheduler.step.__func__ is _FakeH3Scheduler.step


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


def test_download_plan_keeps_dense_encoder_for_a_custom_family_pipeline(monkeypatch):
    _cuda_bf16_target(monkeypatch)
    custom = "someone/custom-ltx-2"
    _plan_api(
        monkeypatch,
        {
            custom: _LTX_BASE_SIBLINGS,
            "unsloth/LTX-2-FP8": _LTX2_FP8_SIBLINGS,
        },
    )

    plan = VideoBackend().download_plan(
        custom,
        model_kind = "pipeline",
        family_override = "ltx-2",
        text_encoder_quant = "fp8",
    )

    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    assert "unsloth/LTX-2-FP8" not in by_repo
    assert any(name.startswith("text_encoder/") for name in by_repo[custom]["files"])


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
        **kwargs,
    ):
        raise OSError("404")

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _boom)
    assert backend._fetch_te_prequant({"text_encoder": source}, None) == ()

    monkeypatch.setattr(
        "utils.hf_xet_fallback.hf_hub_download_with_xet_fallback",
        lambda repo, filename, token, cancel_event = None, **kwargs: "/tmp/precast.pt",
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
    """Only a released DENOISER partition is a valid pick, and the mirror ships more than that.

    The Qwen3-VL encoder quants live in the same repo as the denoisers, so the picker lists both
    and a user can name either. Loading the encoder as the transformer would waste a ~12 GB
    download and fail deep inside sd-cli rather than at the boundary. Both fl2va and ref2va are
    valid picks -- which one is picked IS the task -- and each names its own task.

    The accept cases include the dynamic rung names specifically: the guard is a prefix/suffix
    check, and `-UD-Q2_K_XL` is a shape it had never seen when it was written.
    """
    from core.inference.video_minimax_h3 import (
        H3_TASK_KEYFRAMES,
        H3_TASK_REFERENCES,
        h3_transformer_task,
        validate_h3_transformer_filename,
    )

    for good, task in (
        ("minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf", H3_TASK_KEYFRAMES),
        ("minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf", H3_TASK_KEYFRAMES),
        ("minimax_h3_fl2va_pruned-Q4_K.gguf", H3_TASK_KEYFRAMES),
        ("minimax_h3_fl2va-Q4_K_M.gguf", H3_TASK_KEYFRAMES),
        ("minimax_h3_ref2va_pruned-Q2_K_M.gguf", H3_TASK_REFERENCES),
        ("minimax_h3_ref2va-Q4_K_M.gguf", H3_TASK_REFERENCES),
    ):
        validate_h3_transformer_filename(good)
        assert h3_transformer_task(good) == task

    for bad in (
        "qwen3vl_32b_minimax_h3-Q2_K_M.gguf",
        "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_fl2va_pruned_bf16.safetensors",
    ):
        with pytest.raises(ValueError):
            validate_h3_transformer_filename(bad)


def _no_legacy_cache(monkeypatch):
    """Force "nothing is cached under the old repo ids", i.e. exactly a fresh install.

    The source resolvers stat the real HF cache, so a developer machine or a CI runner with a
    persistent cache that still holds the repack would otherwise answer with the legacy id and
    make a fresh-install guard fail on correct code.
    """
    from core.inference import diffusion_families
    monkeypatch.setattr(
        diffusion_families, "_upstream_is_cached", lambda *a, **k: False, raising = True
    )


def test_every_h3_asset_comes_from_an_unsloth_repo(monkeypatch):
    """No H3 download may reach a community repack on a fresh install.

    The narrower `test_the_h3_native_repo_matches_the_family_gguf_repo` above only pins the
    transformer and text encoder. The VAEs and the quantized conditioner were served from
    Comfy-Org/MiniMax-H3 for exactly that reason: nothing asserted over them, so the divergence
    passed CI. Assert over EVERY repo the H3 paths name instead of a chosen few.

    The legacy ids are deliberately exempt: they are never fetched from on a fresh install, only
    reused when a pre-existing cache already holds those bytes. Which is why the cache is forced
    empty here -- `h3_native_hub_files` and `h3_te_quant_source` consult it, so on a machine that
    still holds the repack this guard would otherwise report the exemption as an offender.
    """
    from core.inference.video_minimax_h3 import (
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        h3_native_hub_files,
    )
    from core.inference.video_minimax_h3_te import H3_TE_QUANT_REPO, h3_te_quant_source

    _no_legacy_cache(monkeypatch)
    named = {H3_GGUF_REPO, H3_COMPONENT_REPO, H3_TE_QUANT_REPO, h3_te_quant_source("int8")}
    named.update(repo for repo, _ in h3_native_hub_files("minimax_h3_fl2va_pruned-Q2_K.gguf"))
    offenders = sorted(r for r in named if not r.startswith("unsloth/"))
    assert not offenders, f"H3 would download from a repo we do not control: {offenders}"


def test_the_h3_legacy_ids_are_the_ones_the_shared_table_names():
    """The constants the delete-cached claims read must be the table's own answer.

    The claims name `H3_LEGACY_*` directly, while the source resolvers go through
    `_SD_CPP_LEGACY_SOURCES`. If those two ever disagree the claim protects one repo while the
    load reads another, which is the exact mid-load deletion the claim exists to stop.
    """
    from core.inference.diffusion_families import legacy_source_repo
    from core.inference.video_minimax_h3 import H3_COMPONENT_REPO, H3_LEGACY_COMPONENT_REPO
    from core.inference.video_minimax_h3_te import H3_LEGACY_TE_QUANT_REPO, H3_TE_QUANT_REPO

    assert legacy_source_repo(H3_COMPONENT_REPO) == H3_LEGACY_COMPONENT_REPO
    assert legacy_source_repo(H3_TE_QUANT_REPO) == H3_LEGACY_TE_QUANT_REPO


def test_the_h3_components_fall_back_to_a_cache_that_predates_the_move(monkeypatch):
    """An install holding the old repack's bytes must not re-download them.

    The HF cache is keyed by repo id, so repointing the constant alone re-fetches ~5.8 GB on
    upgrade and fails outright offline. The bytes are identical either way.
    """
    from core.inference import diffusion_families
    from core.inference import video_minimax_h3 as h3

    monkeypatch.setattr(diffusion_families, "_upstream_is_cached", lambda *a, **k: True)
    assert h3.h3_component_source() == h3.H3_LEGACY_COMPONENT_REPO
    files = dict(h3_files := h3.h3_native_hub_files("minimax_h3_fl2va_pruned-Q2_K.gguf"))
    assert files[h3.H3_LEGACY_COMPONENT_REPO] in (h3.H3_VIDEO_VAE, h3.H3_AUDIO_VAE)
    assert len(h3_files) == 4

    monkeypatch.setattr(diffusion_families, "_upstream_is_cached", lambda *a, **k: False)
    assert h3.h3_component_source() == h3.H3_COMPONENT_REPO


def test_the_h3_component_probe_counts_the_other_cache_root(monkeypatch):
    """A repack left behind by a cache-folder change still counts.

    The native fetch passes `reuse_other_cache_root`, so bytes under huggingface_hub's
    import-time root really are reusable -- but only the OLD repo id can reach them. A live-root
    only probe (`cache_holds_files`) calls them absent, picks the mirror and re-pulls ~5.8 GB;
    offline it fails outright, which is the whole failure this fallback exists to prevent.
    """
    from core.inference import diffusion_families
    from core.inference import video_minimax_h3 as h3

    seen: list[dict] = []

    def _probe(
        repo_id,
        files = None,
        *,
        other_root = False,
    ):
        seen.append({"repo": repo_id, "files": tuple(files or ()), "other_root": other_root})
        return other_root  # cached ONLY in the root the live-root probe cannot see

    monkeypatch.setattr(diffusion_families, "_upstream_is_cached", _probe)
    assert h3.h3_component_source() == h3.H3_LEGACY_COMPONENT_REPO
    assert seen and seen[0]["other_root"] is True
    assert seen[0]["repo"] == h3.H3_LEGACY_COMPONENT_REPO
    assert set(seen[0]["files"]) == {h3.H3_VIDEO_VAE, h3.H3_AUDIO_VAE}


def test_the_h3_conditioner_falls_back_to_a_cache_that_predates_the_move(monkeypatch):
    """The 27 GB int8 conditioner needs the same fallback the VAEs got, for a worse failure.

    `H3_TE_QUANT_REPO` used to alias the repack, so an install that pulled the artifact before the
    move holds it under the old id. Re-pointing the constant alone re-downloads 27 GB online, and
    offline leaves the pipeline with no encoder at all: the load that asks for this artifact has
    already dropped the dense encoder shards from its pull, so there is nothing to fall back to.
    """
    from core.inference import diffusion_families
    from core.inference import video_minimax_h3_te as te

    asked: list[tuple] = []

    def _probe(
        repo_id,
        files = None,
        *,
        other_root = False,
    ):
        asked.append((repo_id, tuple(files or ()), other_root))
        return True

    monkeypatch.setattr(diffusion_families, "_upstream_is_cached", _probe)
    assert te.h3_te_quant_source("int8") == te.H3_LEGACY_TE_QUANT_REPO
    assert asked == [
        (te.H3_LEGACY_TE_QUANT_REPO, (te.H3_TE_QUANT_FILES["int8"],), True)
    ], "the conditioner must be probed by its own filename, in both cache roots"

    monkeypatch.setattr(diffusion_families, "_upstream_is_cached", lambda *a, **k: False)
    assert te.h3_te_quant_source("int8") == te.H3_TE_QUANT_REPO
    # A scheme with no hosted artifact has nothing to probe and never leaves the mirror.
    assert te.h3_te_quant_source(None) == te.H3_TE_QUANT_REPO
    assert te.h3_te_quant_source("nvfp4") == te.H3_TE_QUANT_REPO


def test_the_native_h3_vaes_come_from_a_local_bundle_when_it_has_them(tmp_path):
    """A local clone of the mirror is self-contained, so nothing may go to the Hub.

    The mirror now ships the VAEs beside the denoisers. The transformer and the Qwen encoder
    already resolved from a local pick; the VAEs did not, so the plan still staged them from the
    network and the load failed offline with every required file sitting on disk.
    """
    from core.inference.video import VideoBackend
    from core.inference.video_minimax_h3 import (
        H3_AUDIO_VAE,
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        H3_VIDEO_VAE,
        h3_text_encoder_filename,
    )

    gguf = "minimax_h3_fl2va_pruned-Q2_K.gguf"
    qwen = h3_text_encoder_filename(gguf)
    bundle = tmp_path / "MiniMax-H3-GGUF"
    (bundle / "vae").mkdir(parents = True)
    for name in (gguf, qwen):
        (bundle / name).write_bytes(b"x")
    for name in (H3_VIDEO_VAE, H3_AUDIO_VAE):
        (bundle / name).write_bytes(b"x")

    requests = VideoBackend._h3_native_requests(str(bundle), gguf, qwen)
    assert [repo for repo, _ in requests] == [str(bundle)] * 4
    assert [name for _, name in requests] == [gguf, qwen, H3_VIDEO_VAE, H3_AUDIO_VAE]

    # A bundle missing the audio VAE keeps that ONE file on the mirror, not the whole set.
    (bundle / H3_AUDIO_VAE).unlink()
    assert VideoBackend._h3_native_requests(str(bundle), gguf, qwen) == (
        (str(bundle), gguf),
        (str(bundle), qwen),
        (str(bundle), H3_VIDEO_VAE),
        (H3_COMPONENT_REPO, H3_AUDIO_VAE),
    )

    # And a Hub pick is untouched: every component still comes from a repo id.
    assert VideoBackend._h3_native_requests(H3_GGUF_REPO, gguf, qwen) == (
        (H3_GGUF_REPO, gguf),
        (H3_GGUF_REPO, qwen),
        (H3_COMPONENT_REPO, H3_VIDEO_VAE),
        (H3_COMPONENT_REPO, H3_AUDIO_VAE),
    )


def _legacy_cache_holding(monkeypatch, names):
    """Pretend the repack's cache holds exactly ``names``, and nothing else does."""
    from core.inference import diffusion_families

    def _probe(
        repo_id,
        files = None,
        *,
        other_root = False,
    ):
        return bool(files) and set(files) <= set(names)

    monkeypatch.setattr(diffusion_families, "_upstream_is_cached", _probe)


def test_a_pre_move_cache_holding_one_vae_still_gets_reused_for_that_one(monkeypatch, tmp_path):
    """The two VAEs are fetched one at a time, so the source is decided one at a time.

    A pre-move pull interrupted between them leaves the 5.2 GB video VAE under the old id and
    nothing else. Deciding the pair together calls the old id useless and re-downloads the file
    already on disk, or fails offline.
    """
    from core.inference.video import VideoBackend
    from core.inference.video_minimax_h3 import (
        H3_AUDIO_VAE,
        H3_COMPONENT_REPO,
        H3_GGUF_REPO,
        H3_LEGACY_COMPONENT_REPO,
        H3_VIDEO_VAE,
        h3_component_source,
        h3_native_hub_files,
    )

    gguf = "minimax_h3_fl2va_pruned-Q2_K.gguf"
    qwen = "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"
    _legacy_cache_holding(monkeypatch, {H3_VIDEO_VAE})

    assert h3_component_source(H3_VIDEO_VAE) == H3_LEGACY_COMPONENT_REPO
    assert h3_component_source(H3_AUDIO_VAE) == H3_COMPONENT_REPO
    assert VideoBackend._h3_native_requests(H3_GGUF_REPO, gguf, qwen) == (
        (H3_GGUF_REPO, gguf),
        (H3_GGUF_REPO, qwen),
        (H3_LEGACY_COMPONENT_REPO, H3_VIDEO_VAE),
        (H3_COMPONENT_REPO, H3_AUDIO_VAE),
    )
    assert dict(h3_native_hub_files(gguf))[H3_LEGACY_COMPONENT_REPO] == H3_VIDEO_VAE

    # And with a local bundle carrying the other one, nothing is left for the network at all.
    bundle = tmp_path / "MiniMax-H3-GGUF"
    (bundle / "vae").mkdir(parents = True)
    for name in (gguf, qwen, H3_AUDIO_VAE):
        (bundle / name).write_bytes(b"x")
    assert VideoBackend._h3_native_requests(str(bundle), gguf, qwen) == (
        (str(bundle), gguf),
        (str(bundle), qwen),
        (H3_LEGACY_COMPONENT_REPO, H3_VIDEO_VAE),
        (str(bundle), H3_AUDIO_VAE),
    )


def test_the_plan_sizes_a_cached_repack_from_the_repo_we_control(monkeypatch):
    """A repack that is gone must not fail a plan for a load its own cache still satisfies.

    `model_info` on the repack raises once it is renamed or taken down, which is the failure the
    move exists to survive, and one raising call fails the WHOLE native plan: `plan_failed` makes
    every locality-dependent caller (media auto-switch) refuse a load that would have worked. The
    two copies are byte identical, so the size comes from the mirror while the entry keeps the id
    the bytes are read from.
    """
    from core.inference.video_minimax_h3 import (
        H3_AUDIO_VAE,
        H3_LEGACY_COMPONENT_REPO,
        H3_VIDEO_VAE,
    )

    _legacy_cache_holding(monkeypatch, {H3_VIDEO_VAE, H3_AUDIO_VAE})
    # Only the mirror answers: asking the repack raises KeyError, exactly as a taken-down repo
    # would raise from the Hub.
    _plan_api(
        monkeypatch,
        {
            "unsloth/MiniMax-H3-GGUF": [
                _PlanSibling("minimax_h3_fl2va-Q4_K_M.gguf", 19),
                _PlanSibling("qwen3vl_32b_minimax_h3-Q4_K_M.gguf", 18),
                _PlanSibling(H3_VIDEO_VAE, 5),
                _PlanSibling(H3_AUDIO_VAE, 1),
            ],
        },
    )

    plan = VideoBackend._h3_native_download_plan(
        "unsloth/MiniMax-H3-GGUF", "minimax_h3_fl2va-Q4_K_M.gguf", hf_token = None
    )
    assert not plan.get("plan_failed")
    by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
    # The VAEs are staged under the id they will be read from, at the mirror's sizes.
    assert sorted(by_repo[H3_LEGACY_COMPONENT_REPO]["files"]) == sorted(
        [H3_VIDEO_VAE, H3_AUDIO_VAE]
    )
    assert by_repo[H3_LEGACY_COMPONENT_REPO]["bytes"] == 6
    assert plan["required_bytes"] == 43


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


def test_every_video_fetch_resolves_both_cache_roots():
    """The plan probe accepts a file cached under EITHER root (Unsloth's cache folder is a
    setting, so a pre-move download sits under huggingface_hub's import-time root) and stages
    neither. So every fetch on the load path has to resolve both roots as well, or the file the
    planner skipped is re-pulled inside the load, outside the manager's progress, cancel and disk
    preflight -- and fails outright offline. The diffusion and sd.cpp fetches already opt in."""
    # Every module on the video load path, not just video.py: the LTX-2.3 extras loader lives in
    # video_ltx2.py and shipped without the opt-in while the plan already skipped its files.
    root = Path(__file__).resolve().parents[1] / "core/inference"
    for name in ("video.py", "video_ltx2.py"):
        src = (root / name).read_text(encoding = "utf-8")
        # The bare `from ... import name` has no trailing paren, so this counts call sites only.
        calls = src.count("hf_hub_download_with_xet_fallback(")
        if calls == 0:
            continue
        optins = src.count("reuse_other_cache_root = True")
        assert optins == calls, (
            f"{name}: {calls - optins} of {calls} video fetches resolve only the active cache "
            "root, so the planner's both-roots probe can drop a file the load cannot then find"
        )


# ── unified-memory oversize refusal, at the video load seam ───────────────────


def _unified_snapshot(total_gib):
    """Stand in for a Mac's memory snapshot: unified pool, 80% of RAM free."""
    from core.inference.diffusion_memory import DeviceMemory

    total = total_gib * 1024
    return lambda target: DeviceMemory("mps", "mps", "unified_memory", int(total * 0.80), total)


def test_unified_memory_refuses_an_oversized_video_load(fake_runtime, monkeypatch):
    """A 16 GiB Mac loading LTX-2 (about 65 GiB of weights): the planner has no offload tier to
    fall back to on unified memory and PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 removes the
    allocator's limit, so without this refusal the OS kills Unsloth with no Python exception.
    _run_load stringifies this onto load_progress, so the text is what the UI toasts."""
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "settled_snapshot_device_memory", _unified_snapshot(16))

    backend = VideoBackend()
    with pytest.raises(RuntimeError) as excinfo:
        backend.load_pipeline("Lightricks/LTX-2", model_kind = "pipeline")
    message = str(excinfo.value)
    assert "ltx-2" in message  # names the family
    assert "unified memory" in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD=1" in message
    # Refused BEFORE the pipeline was built: nothing was allocated and nothing is loaded.
    assert backend.status()["loaded"] is False


def test_unified_memory_allows_a_video_load_that_fits(fake_runtime, monkeypatch):
    # The same family on a 128 GiB Mac fits, so the refusal must stay out of the way.
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "settled_snapshot_device_memory", _unified_snapshot(128))

    backend = VideoBackend()
    status = backend.load_pipeline("Lightricks/LTX-2", model_kind = "pipeline")
    assert status["loaded"] is True


def test_unified_memory_refusal_is_overridable_at_the_video_load_seam(fake_runtime, monkeypatch):
    import core.inference.video as video_mod
    from core.inference.diffusion_memory import UNIFIED_OVERSIZE_ENV

    monkeypatch.setattr(video_mod, "settled_snapshot_device_memory", _unified_snapshot(16))
    monkeypatch.setenv(UNIFIED_OVERSIZE_ENV, "1")

    backend = VideoBackend()
    assert backend.load_pipeline("Lightricks/LTX-2", model_kind = "pipeline")["loaded"] is True


def test_discrete_vram_video_load_is_unaffected_by_the_refusal(fake_runtime, monkeypatch):
    """The same impossible-looking numbers on a discrete card still load: offload streams the
    weights from host RAM, so refusing there would break a path that works today. (The fake
    runtime resolves a CPU target, so the policy itself is not meaningful here; what this pins
    is that a discrete-VRAM snapshot never reaches the refusal.)"""
    import core.inference.video as video_mod
    from core.inference.diffusion_memory import DeviceMemory

    monkeypatch.setattr(
        video_mod,
        "settled_snapshot_device_memory",
        lambda target: DeviceMemory("cuda", "cuda", "discrete_vram", 13_107, 16_384),
    )

    backend = VideoBackend()
    assert backend.load_pipeline("Lightricks/LTX-2", model_kind = "pipeline")["loaded"] is True


def test_the_mps_allocator_cache_is_released_before_the_budget_is_read(monkeypatch):
    """Dropping a pipeline leaves its buffers RESERVED in torch's MPS caching allocator. The
    budget is a system-memory reading, which counts those bytes as used, so without emptying the
    cache first a model swap that fits is refused. torch.mps.empty_cache "releases all unoccupied
    cached memory currently held by the caching allocator"."""
    import sys
    import types

    import core.inference.diffusion_memory as dm

    calls: list = []
    fake_torch = types.SimpleNamespace(
        mps = types.SimpleNamespace(empty_cache = lambda: calls.append("mps"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        dm,
        "snapshot_device_memory",
        lambda t: dm.DeviceMemory("mps", "mps", "unified_memory", 1, 2),
    )

    dm.settled_snapshot_device_memory(types.SimpleNamespace(device = "mps", backend = "mps"))
    assert calls == ["mps"], "the MPS allocator must be emptied before the reading is taken"


def _fake_h3_vae():
    """A stand-in with the shapes that matter: an encoder half, a decoder with both
    autocast-eligible and float32-only parameters, and a post_quant_conv."""
    import torch

    class _VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Conv3d(4, 8, 1))
            self.quant_conv = torch.nn.Conv3d(8, 8, 1)
            self.decoder = torch.nn.Module()
            self.decoder.proj = torch.nn.Linear(8, 16)
            self.decoder.conv = torch.nn.Conv3d(16, 4, 1)
            self.decoder.norm = torch.nn.LayerNorm(16)
            # Read directly rather than through an autocast-eligible op.
            self.decoder.register_parameter(
                "scale1", torch.nn.Parameter(torch.ones(16, dtype = torch.float32))
            )
            self.post_quant_conv = torch.nn.Conv3d(4, 4, 1)

    return _VAE().to(torch.float32)


def test_h3_vae_trim_drops_the_encoder_only_for_a_workflow_that_never_encodes():
    import torch

    from core.inference.video_minimax_h3 import trim_h3_video_vae

    vae = _fake_h3_vae()
    report = trim_h3_video_vae(vae, workflow = "t2va")
    assert vae.encoder is None and vae.quant_conv is None
    assert report["encoder_freed"] > 0

    # An image-conditioned workflow encodes, so the same call must keep that half.
    other = _fake_h3_vae()
    other_report = trim_h3_video_vae(other, workflow = "ref2va")
    assert other.encoder is not None and other.quant_conv is not None
    assert other_report["encoder_freed"] == 0
    # The decoder pre-cast is workflow-independent and still happens.
    assert other_report["decoder_freed"] > 0
    assert other.decoder.proj.weight.dtype is torch.float16


def test_h3_vae_trim_precasts_only_what_autocast_would_cast():
    import torch

    from core.inference.video_minimax_h3 import trim_h3_video_vae

    vae = _fake_h3_vae()
    trim_h3_video_vae(vae, workflow = "t2va")

    # Linear and Conv weights and biases are what torch.autocast casts on entry.
    assert vae.decoder.proj.weight.dtype is torch.float16
    assert vae.decoder.proj.bias.dtype is torch.float16
    assert vae.decoder.conv.weight.dtype is torch.float16
    assert vae.post_quant_conv.weight.dtype is torch.float16
    # A norm gain is on autocast's float32 promote list and a bare parameter is read
    # directly, so halving either would change the arithmetic rather than preserve it.
    assert vae.decoder.norm.weight.dtype is torch.float32
    assert vae.decoder.scale1.dtype is torch.float32


def test_h3_vae_trim_leaves_the_decode_arithmetic_bit_identical():
    # The whole justification for pre-casting is that autocast performs exactly this cast
    # anyway, so x.to(f16).to(f16) == x.to(f16). Check the claim on a real matmul rather
    # than trusting the reasoning.
    import torch

    from core.inference.video_minimax_h3 import trim_h3_video_vae

    torch.manual_seed(0)
    reference = _fake_h3_vae()
    trimmed = _fake_h3_vae()
    trimmed.load_state_dict(reference.state_dict())
    trim_h3_video_vae(trimmed, workflow = "t2va")

    x = torch.randn(3, 8, dtype = torch.float16)
    # What autocast does with the float32 original, against the pre-cast weight.
    autocast_style = torch.nn.functional.linear(
        x,
        reference.decoder.proj.weight.to(torch.float16),
        reference.decoder.proj.bias.to(torch.float16),
    )
    precast = trimmed.decoder.proj(x)
    assert torch.equal(autocast_style, precast)


def test_h3_vae_trim_survives_a_renamed_attribute():
    # A diffusers release that moves these attributes should cost the saving, not the render.
    import torch

    from core.inference.video_minimax_h3 import trim_h3_video_vae

    assert trim_h3_video_vae(None, workflow = "t2va") == {"encoder_freed": 0, "decoder_freed": 0}
    bare = torch.nn.Module()
    assert trim_h3_video_vae(bare, workflow = "t2va")["decoder_freed"] == 0


# ── MiniMax-H3 keyframes (image-to-video) ────────────────────────────────────


def test_h3_canvas_follows_the_released_checkpoint_rule():
    from core.inference.video_minimax_h3 import h3_canvas_for_aspect

    # The area cap brings 16:9 to the checkpoint's 1344x768 default.
    assert h3_canvas_for_aspect(1920, 1080) == (1344, 768)
    assert h3_canvas_for_aspect(1080, 1920) == (768, 1344)
    assert h3_canvas_for_aspect(1000, 1000) == (768, 768)
    # 4:3 stays under the cap, so the short edge is honoured exactly.
    assert h3_canvas_for_aspect(1024, 768) == (1024, 768)
    # Only the ratio matters, so a thumbnail and a 4K frame of the same shape agree.
    assert h3_canvas_for_aspect(160, 90) == h3_canvas_for_aspect(3840, 2160)
    # Outside the trained 1:4 - 4:1 band, refuse rather than generate off-distribution.
    with pytest.raises(ValueError, match = "aspect ratios"):
        h3_canvas_for_aspect(2000, 400)
    with pytest.raises(ValueError, match = "aspect ratio"):
        h3_canvas_for_aspect(0, 100)


def test_h3_keyframe_fit_stretches_the_first_and_crops_the_last():
    Image = pytest.importorskip("PIL.Image")
    from core.inference.video_minimax_h3 import (
        H3_ANCHOR_FIRST,
        H3_ANCHOR_LAST,
        fit_h3_keyframe,
    )

    # A 2:1 source on a square canvas distinguishes stretch from center crop.
    source = Image.new("RGB", (200, 100), (0, 0, 0))
    for x in range(200):
        for y in range(100):
            source.putpixel((x, y), (255, 0, 0) if x < 20 or x >= 180 else (0, 0, 255))

    first = fit_h3_keyframe(source, 128, 128, anchor = H3_ANCHOR_FIRST)
    assert first.size == (128, 128)
    assert first.getpixel((2, 64))[0] > 200  # the red margin was squeezed in, not cut

    last = fit_h3_keyframe(source, 128, 128, anchor = H3_ANCHOR_LAST)
    assert last.size == (128, 128)
    assert last.getpixel((2, 64))[2] > 200  # the red margin fell outside the centre crop

    with pytest.raises(ValueError, match = "anchor"):
        fit_h3_keyframe(source, 64, 64, anchor = "middle")


def _h3_native_backend(
    monkeypatch,
    calls,
    binary = None,
):
    """A backend with an H3 sd.cpp state whose engine records the params it was handed.

    ``binary`` stands in for what _run_load_h3_native resolves and vets: the engine gets the path
    and the runtime records that file's identity, exactly as the real load does."""
    from core.inference.video import _VideoLoadState
    from core.inference.video_minimax_h3 import MiniMaxH3NativeRuntime

    class _Engine:
        def generate_video(self, files, params, **kwargs):
            from PIL import Image

            # The staged PNGs live in a scratch dir the run deletes, so measure them here.
            def staged_size(path):
                if path is None:
                    return None
                with Image.open(path) as image:
                    return image.size

            calls.append(
                {
                    "files": files,
                    "params": params,
                    "init_size": staged_size(params.init_img),
                    "end_size": staged_size(params.end_img),
                    **kwargs,
                }
            )
            return Path("/tmp/does-not-exist.webm")

    backend = VideoBackend()
    fam = _detect_load_family("unsloth/MiniMax-H3-GGUF", None, "minimax-h3")
    from core.inference.video import _sd_cli_identity

    engine = _Engine()
    if binary is not None:
        engine.binary = str(binary)
    backend._state = _VideoLoadState(
        pipe = MiniMaxH3NativeRuntime(
            engine = engine,
            files = object(),
            offload_flags = (),
            binary_identity = _sd_cli_identity(str(binary)) if binary is not None else None,
        ),
        family = fam,
        repo_id = "unsloth/MiniMax-H3-GGUF",
        base_repo = fam.base_repo,
        device = "cuda",
        dtype = "Q4_K_M",
        kind = "gguf",
        engine = "sd_cpp",
        h3_task = "fl2va",
        gguf_filename = "minimax_h3_fl2va-Q4_K_M.gguf",
    )
    # Imported inside _generate_h3_native, so patched at their own module.
    from core.inference import video_minimax_h3 as h3_mod

    monkeypatch.setattr(h3_mod, "inspect_video", lambda path: (1344, 768, 124, True))
    monkeypatch.setattr(h3_mod, "transcode_video_to_mp4", lambda path, fps: b"MP4")
    return backend


def _data_url(width, height):
    import base64
    import io

    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_h3_native_generate_stages_both_keyframes_on_the_canvas(monkeypatch):
    pytest.importorskip("PIL.Image")
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    result = backend.generate(
        prompt = "a fox runs through snow",
        width = 960,
        height = 544,
        first_frame = _data_url(1920, 1080),
        last_frame = _data_url(400, 400),
    )

    assert calls[0]["params"].init_img and calls[0]["params"].end_img
    # Both staged images already use the requested canvas.
    assert calls[0]["init_size"] == (960, 544)
    assert calls[0]["end_size"] == (960, 544)
    assert result["conditioning"] == "fl2va"


def test_h3_native_generate_records_the_build_it_ran_on(monkeypatch):
    """_run_generate persists these with result.get(...), so a field the native path omits lands
    in the gallery sidecar as null. Clips generated from different GGUF quantizations of the same
    repo then cannot be told apart or reproduced from their saved recipe, unlike the diffusers
    path, which records the same set off the engaged state."""
    pytest.importorskip("PIL.Image")
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    result = backend.generate(prompt = "a fox runs through snow", width = 960, height = 544)

    state = backend._state
    assert result["model_kind"] == state.kind == "gguf"
    assert result["gguf_filename"] == state.gguf_filename
    assert result["memory_mode"] == state.memory_mode
    assert result["offload_policy"] == state.offload_policy
    assert result["transformer_quant"] == state.transformer_quant
    assert result["text_encoder_quant"] == state.text_encoder_quant


def test_h3_native_generate_names_each_keyframe_combination(monkeypatch):
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)
    image = _data_url(640, 360)

    assert backend.generate(prompt = "p", width = 960, height = 544)["conditioning"] == "t2va"
    assert calls[-1]["params"].init_img is None and calls[-1]["params"].end_img is None

    assert (
        backend.generate(prompt = "p", width = 960, height = 544, first_frame = image)["conditioning"]
        == "i2va"
    )
    assert calls[-1]["params"].end_img is None

    assert (
        backend.generate(prompt = "p", width = 960, height = 544, last_frame = image)["conditioning"]
        == "l2va"
    )
    assert calls[-1]["params"].init_img is None


def test_h3_omitted_size_takes_the_canvas_from_the_keyframe(monkeypatch):
    pytest.importorskip("PIL.Image")
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    backend.generate(prompt = "p", first_frame = _data_url(1920, 1080))
    assert (calls[-1]["params"].width, calls[-1]["params"].height) == (1344, 768)

    # With no first frame the last one drives it, matching the Diffusers blocks.
    backend.generate(prompt = "p", last_frame = _data_url(1080, 1920))
    assert (calls[-1]["params"].width, calls[-1]["params"].height) == (768, 1344)
    assert calls[-1]["end_size"] == (768, 1344)

    # Without a keyframe there is nothing to match, so the family's first preset still wins.
    backend.generate(prompt = "p")
    assert (calls[-1]["params"].width, calls[-1]["params"].height) == (1344, 768)


def test_h3_native_clip_records_the_build_it_came_off(monkeypatch):
    """A native GGUF clip must carry the same build record as the diffusers twin.

    _run_generate copies model_kind / gguf_filename / transformer_quant / text_encoder_quant /
    memory_mode / offload_policy out of the result into the saved sidecar, so if the native return
    dict omits them every native clip saves a blank recipe and the gallery shows nothing.
    """
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)
    import dataclasses

    backend._state = dataclasses.replace(
        backend._state,
        transformer_quant = "fp8",
        text_encoder_quant = "int8",
        memory_mode = "low",
        offload_policy = "model",
    )

    result = backend.generate(prompt = "p", width = 960, height = 544)

    assert result["model_kind"] == "gguf"
    assert result["gguf_filename"] == "minimax_h3_fl2va-Q4_K_M.gguf"
    assert result["transformer_quant"] == "fp8"
    assert result["text_encoder_quant"] == "int8"
    assert result["memory_mode"] == "low"
    assert result["offload_policy"] == "model"


def test_a_cfg_free_family_records_the_guidance_it_actually_ran(monkeypatch):
    """H3 has no CFG, so a requested guidance must not reach the recipe.

    The native path pins cfg_scale to 1.0 and the diffusers path passes no guidance kwarg at all,
    so recording the caller's number would label the clip with a scale that never ran.
    """
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    result = backend.generate(prompt = "p", width = 960, height = 544, guidance = 7.0)

    assert result["guidance"] == 1.0
    assert calls[-1]["params"].cfg_scale == 1.0


def test_keyframes_are_refused_by_a_family_that_has_none(fake_runtime, tmp_path):
    # Silently dropping the image would render a text-only clip with nothing to do with it.
    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    with pytest.raises(ValueError, match = "no first or last frame"):
        backend.begin_generate(prompt = "a sloth", first_frame = _data_url(64, 64))
    with pytest.raises(ValueError, match = "no first or last frame"):
        backend.generate(prompt = "a sloth", last_frame = _data_url(64, 64))
    # The refusal must not leave the busy flag set, or every later generation 409s.
    assert backend.generate_progress()["active"] is False


def test_begin_generate_rejects_an_undecodable_keyframe(monkeypatch):
    # begin_generate returns before the job runs, so a bad image has to fail on the POST.
    backend = _h3_native_backend(monkeypatch, [])
    with pytest.raises(ValueError):
        backend.begin_generate(prompt = "p", first_frame = "data:image/png;base64,notanimage")
    assert backend.generate_progress()["active"] is False


def test_h3_modular_load_restricts_the_components_not_the_blocks(monkeypatch, tmp_path):
    # Restrict components without pruning t2va/fl2va routing blocks.
    import types

    from core.inference.video import VideoBackend

    seen: dict = {}

    class _FakeModularPipeline:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            seen["from_pretrained"] = {"repo": repo, **kwargs}
            return cls()

        def load_components(self, **kwargs):
            seen["load_components"] = kwargs

        def to(self, device):
            return self

    diffusers = types.SimpleNamespace(
        ComponentsManager = lambda: types.SimpleNamespace(
            enable_auto_cpu_offload = lambda **kwargs: seen.setdefault("offload", kwargs)
        ),
        ModularPipeline = _FakeModularPipeline,
    )
    torch = types.SimpleNamespace(bfloat16 = "bf16")
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")

    backend = VideoBackend()
    status = backend._load_h3_modular_pipeline(
        # Precision is not what this covers, and an unset request resolves to int8 now,
        # which sends the load down the hosted-checkpoint path instead of this one.
        transformer_quant = "none",
        diffusers = diffusers,
        torch = torch,
        fam = fam,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base = fam.base_repo,
        kind = "pipeline",
        dtype = torch.bfloat16,
        device = "cuda",
        hf_token = None,
        memory_mode = None,
        _load_token = None,
        _base_local_dir = None,
    )

    assert "workflow" not in seen["from_pretrained"]
    assert seen["load_components"]["workflow"] == "fl2va"
    assert status["supports_keyframes"] is True
    assert status["defaults"]["canvas_short_edge"] == 768


def test_h3_modular_load_pins_a_hosted_prequant_denoiser_out_of_the_offload_rotation(monkeypatch):
    """A pre-quantized denoiser must be placed at LOAD time, not per forward.

    ComponentsManager moves each component onto the accelerator inside its own pre_forward, and a
    torchao-quantized module does not survive that mid-block move: the denoise loop died on step 1
    with "Attempted to set the storage of a tensor on device cuda:0 to a storage on different
    device cpu". So the loader has to pin it once the seeding actually engaged, and leave the
    released bfloat16 path exactly as it was.
    """
    import types

    from core.inference.video import VideoBackend

    calls: dict = {}

    class _FakeModularPipeline:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            return cls()

        def load_components(self, **kwargs):
            pass

        def update_components(self, **kwargs):
            self.transformer = kwargs.get("transformer")

        def to(self, device):
            return self

    manager = types.SimpleNamespace(enable_auto_cpu_offload = lambda **kwargs: None)
    diffusers = types.SimpleNamespace(
        ComponentsManager = lambda: manager,
        ModularPipeline = _FakeModularPipeline,
        MiniMaxH3Transformer3DModel = object,
    )
    torch = types.SimpleNamespace(bfloat16 = "bf16")
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")

    seeded = object()
    import core.inference.diffusion_prequant as prequant_module

    monkeypatch.setattr(
        prequant_module,
        "load_prequantized_transformer",
        lambda *args, **kwargs: seeded,
    )
    monkeypatch.setattr(
        prequant_module,
        "pin_prequantized_module",
        lambda mgr, module, device, **kwargs: (
            calls.setdefault("pinned", {"manager": mgr, "module": module, "device": device})
            is not None
        ),
    )

    def load(scheme):
        calls.clear()
        return VideoBackend()._load_h3_modular_pipeline(
            diffusers = diffusers,
            torch = torch,
            fam = fam,
            repo_id = "MiniMaxAI/MiniMax-H3",
            base = fam.base_repo,
            kind = "pipeline",
            dtype = torch.bfloat16,
            device = "cuda",
            hf_token = None,
            memory_mode = None,
            transformer_quant = scheme,
            _load_token = None,
            _base_local_dir = None,
        )

    status = load("fp8")
    assert status["transformer_quant"] == "fp8"
    assert calls["pinned"]["module"] is seeded
    assert calls["pinned"]["manager"] is manager
    assert calls["pinned"]["device"] == "cuda"

    # No hosted checkpoint engaged -> released bfloat16 components, which the manager moves
    # fine. Asked for EXPLICITLY: an unset request resolves to int8 now.
    status = load("none")
    assert status["transformer_quant"] is None
    assert "pinned" not in calls


def test_h3_modular_load_seeds_the_partition_its_workflow_denoises_against(monkeypatch):
    """One repo, two partitions, two component names.

    ref2va's denoise step reads ``transformer_ref``; fl2va (and text-only through it) reads
    ``transformer``. Seeding the wrong attribute is silent in both directions: the block finds no
    denoiser where it looks, and ``load_components`` then fetches the dense 66.28 GB partition the
    seed existed to replace. The offload pin has to follow the same name, or a pre-quantized
    denoiser stays in the rotation and dies mid-block on its first move.
    """
    import types

    from core.inference.video import VideoBackend

    calls: dict = {}

    class _FakeModularPipeline:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            return cls()

        def load_components(self, **kwargs):
            calls["workflow"] = kwargs.get("workflow")

        def update_components(self, **kwargs):
            calls.setdefault("seeded", {}).update(kwargs)
            for name, value in kwargs.items():
                setattr(self, name, value)

        def to(self, device):
            return self

    manager = types.SimpleNamespace(enable_auto_cpu_offload = lambda **kwargs: None)
    diffusers = types.SimpleNamespace(
        ComponentsManager = lambda: manager,
        ModularPipeline = _FakeModularPipeline,
        MiniMaxH3Transformer3DModel = object,
    )
    torch = types.SimpleNamespace(bfloat16 = "bf16")
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")

    seeded = object()
    import core.inference.diffusion_prequant as prequant_module

    monkeypatch.setattr(
        prequant_module,
        "load_prequantized_transformer",
        lambda *args, **kwargs: calls.setdefault("loader_kwargs", kwargs) and seeded or seeded,
    )
    monkeypatch.setattr(
        prequant_module,
        "pin_prequantized_module",
        lambda mgr, module, device, **kwargs: calls.setdefault("pinned", module) is not None,
    )

    def load(task):
        calls.clear()
        return VideoBackend()._load_h3_modular_pipeline(
            diffusers = diffusers,
            torch = torch,
            fam = fam,
            repo_id = "MiniMaxAI/MiniMax-H3",
            base = fam.base_repo,
            kind = "pipeline",
            dtype = torch.bfloat16,
            device = "cuda",
            hf_token = None,
            memory_mode = None,
            transformer_quant = "int8",
            h3_task = task,
            _load_token = None,
            _base_local_dir = None,
        )

    status = load("ref2va")
    assert status["transformer_quant"] == "int8"
    assert calls["workflow"] == "ref2va"
    assert calls["seeded"] == {"transformer_ref": seeded}
    assert calls["pinned"] is seeded
    # The config comes from the same partition, because that is the only one the scoped download
    # stages for this task.
    assert calls["loader_kwargs"]["config_subfolder"] == "transformer_ref"

    status = load("fl2va")
    assert status["transformer_quant"] == "int8"
    assert calls["seeded"] == {"transformer": seeded}
    assert calls["loader_kwargs"]["config_subfolder"] == "transformer"


def test_denoiser_prequant_coverage_is_asked_per_partition():
    # The dense-shard skip is only safe when a checkpoint really covers THIS task. A scheme whose
    # only hosted artifact belongs to the other partition covers nothing, and answering yes would
    # drop the very shards the load then has to open.
    from core.inference.video_families import VideoFamily

    fam = VideoFamily(
        name = "partitioned",
        pipeline_class = "P",
        transformer_class = "T",
        base_repo = "org/partitioned",
        modular_workflow = "fl2va",
        prequant_repos = (("int8", "unsloth/Test-FP8"), ("fp8", "unsloth/Test-FP8")),
        prequant_filenames = (("fp8", "ref2va", "Test-Ref2VA-FP8.pt"),),
        prequant_partition_tasks = ("ref2va",),
    )
    assert VideoBackend._denoiser_prequant_covered(fam, "int8", None, "fl2va") is True
    assert VideoBackend._denoiser_prequant_covered(fam, "int8", None, "ref2va") is False
    assert VideoBackend._denoiser_prequant_covered(fam, "fp8", None, "ref2va") is True
    # No task named at all keeps the historical per-scheme answer.
    assert VideoBackend._denoiser_prequant_covered(fam, "int8", None) is True


def test_h3_native_progress_reads_only_the_denoise_bar(monkeypatch):
    # Replay real log shapes to isolate the denoise progress bar.
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)
    transcript = [
        "[INFO ] stable-diffusion.cpp:4365 - sampling using Euler method",
        "[INFO ] stable-diffusion.cpp:6107 - MiniMax-H3 I2VA",
        # keyframe VAE encode: same shape, its own loop
        "  |==============>       | 3/6 - 9.17it/s",
        "[INFO ] stable-diffusion.cpp:6899 - generate_video 640x384x124",
        # the denoiser's own weights streaming in, rated in GB/s
        "  |======================| 532/532 - 3.81GB/s",
        "  |=====>                | 1/4 - 1.63it/s",
        "  |==========>           | 2/4 - 1.70it/s",
        "[INFO ] stable-diffusion.cpp:6997 - sampling completed, taking 13.20s",
        # VAE decode tiles, after the denoise window closed
        "  |======================| 6/6 - 9.17it/s",
    ]

    class _ReplayEngine:
        def generate_video(self, files, params, *, on_log, **kwargs):
            seen = []
            for line in transcript:
                on_log(line)
                seen.append((backend._gen["step"], backend._gen["total"], backend._gen["phase"]))
            calls.append(seen)
            return Path("/tmp/does-not-exist.webm")

    from core.inference.video_minimax_h3 import MiniMaxH3NativeRuntime

    # Install the replay runtime into the frozen load state.
    object.__setattr__(
        backend._state,
        "pipe",
        MiniMaxH3NativeRuntime(engine = _ReplayEngine(), files = object(), offload_flags = ()),
    )
    backend.generate(prompt = "p", width = 640, height = 384, steps = 4)

    seen = calls[-1]
    # Only bars inside the denoise window update progress.
    assert seen[2] == (0, 4, "denoise")
    assert seen[4] == (0, 4, "denoise")
    assert seen[5] == (1, 4, "denoise")
    assert seen[6] == (2, 4, "denoise")
    assert seen[7] == (2, 4, "decode")
    assert seen[8] == (2, 4, "decode")


def test_h3_native_progress_survives_the_real_in_place_redraws(monkeypatch):
    """The two halves of the frozen-bar defect have to be fixed together.

    ``sd_cpp_engine`` splits sd-cli's in-place redraws (leading ``\\r``, no newline until the last
    step, closed by ``\\033[K``) into records, and the backend's bar pattern reads them. Fix only
    the reader and the bar still never matches; fix only the pattern and the reader never hands a
    record over while sampling is running. This drives the REAL byte stream through both.
    """
    from core.inference.sd_cpp_engine import iter_sd_cpp_records

    redraw = "\r  |=====>                | {}/{} - 1.63it/s\x1b[K"
    # One flush per element, as the child actually writes them: the bar for step N is its own
    # write with no newline, and the carriage return that would terminate it does not arrive
    # until step N+1. Reading in one big chunk would hide exactly the lateness under test.
    flushes = [
        "[INFO ] stable-diffusion.cpp:6899 - generate_video 640x384x124\n",
        redraw.format(1, 4),
        redraw.format(2, 4),
        redraw.format(3, 4),
        redraw.format(4, 4) + "\n",
        "[INFO ] stable-diffusion.cpp:6997 - sampling completed, taking 13.20s\n",
    ]

    class _Raw:
        def __init__(self, owner) -> None:
            self._owner = owner

        def read1(self, _n: int) -> bytes:
            if not self._owner.pending:
                return b""
            self._owner.reads += 1
            return self._owner.pending.pop(0).encode()

    class _PipeStream:
        def __init__(self, payload: list[str]) -> None:
            self.pending = list(payload)
            self.reads = 0
            self.buffer = _Raw(self)

        def __iter__(self):
            raise AssertionError("a redraw carries no newline, so line iteration would block")

    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    class _ReplayEngine:
        def generate_video(self, files, params, *, on_log, **kwargs):
            seen = []
            stream = _PipeStream(flushes)
            for record in iter_sd_cpp_records(stream):
                on_log(record)
                seen.append((stream.reads, backend._gen["step"], backend._gen["phase"]))
            calls.append(seen)
            return Path("/tmp/does-not-exist.webm")

    from core.inference.video_minimax_h3 import MiniMaxH3NativeRuntime

    object.__setattr__(
        backend._state,
        "pipe",
        MiniMaxH3NativeRuntime(engine = _ReplayEngine(), files = object(), offload_flags = ()),
    )
    backend.generate(prompt = "p", width = 640, height = 384, steps = 4)

    seen = calls[-1]
    # Every sampling step is observed WHILE sampling, in order, not all at once at the end.
    # A redraw yields two records (the empty run before its leading CR, then the bar), so
    # collapse repeats: what matters is that 1, 2, 3 and 4 each arrive, and in that order.
    progression: list[int] = []
    for _reads, step, phase in seen:
        if phase == "denoise" and (not progression or progression[-1] != step):
            progression.append(step)
    assert progression == [0, 1, 2, 3, 4]
    # Step N must be visible after the flush that carried it (read N + 1, the opening log line
    # being read 1), not one redraw later. Being "eventually right" is the bug, not the fix.
    first_read_for = {}
    for reads, step, phase in seen:
        if phase == "denoise" and step:
            first_read_for.setdefault(step, reads)
    assert first_read_for == {1: 2, 2: 3, 3: 4, 4: 5}
    # And the run does not end parked on a full denoise bar.
    assert seen[-1][2] == "decode"


# ── MiniMax-H3 references (Ref2VA) ───────────────────────────────────────────


def _h3_ref_backend(monkeypatch, calls):
    """The native backend with the Ref2VA partition resident instead of FL2VA."""
    backend = _h3_native_backend(monkeypatch, calls)
    object.__setattr__(backend._state, "h3_task", "ref2va")
    object.__setattr__(backend._state, "gguf_filename", "minimax_h3_ref2va_pruned-Q4_K_M.gguf")
    return backend


def _reference_video_data_url(
    seconds = 3.0,
    fps = 24,
    size = (160, 96),
    with_audio = True,
    audio_seconds = None,
    audio_start_seconds = 0.0,
):
    """A real encoded MP4, so the decode path is exercised rather than stubbed."""
    import base64
    import io

    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")

    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as out:
        video = out.add_stream("libx264", rate = fps)
        video.width, video.height = size
        video.pix_fmt = "yuv420p"
        audio = None
        if with_audio:
            audio = out.add_stream("aac", rate = 44_100)
            audio.layout = "stereo"
        for index in range(int(seconds * fps)):
            frame = av.VideoFrame.from_ndarray(
                np.full((size[1], size[0], 3), index % 255, dtype = np.uint8), format = "rgb24"
            )
            for packet in video.encode(frame):
                out.mux(packet)
        if audio is not None:
            written = 0
            total = int((seconds if audio_seconds is None else audio_seconds) * 44_100)
            pts = int(round(audio_start_seconds * 44_100))
            while written < total:
                count = min(1024, total - written)
                samples = np.zeros((1, count * 2), dtype = np.int16)
                frame = av.AudioFrame.from_ndarray(samples, format = "s16", layout = "stereo")
                frame.sample_rate = 44_100
                # Placing the track later on the shared timeline, as an offset soundtrack does.
                frame.pts = pts
                for packet in audio.encode(frame):
                    out.mux(packet)
                written += count
                pts += count
            for packet in audio.encode():
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()


def _vfr_reference_video_data_url():
    """A four-second VFR clip whose media timeline begins at five seconds."""
    import base64
    import io
    from fractions import Fraction

    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")

    timestamps = []
    elapsed = 0.0
    while elapsed < 4.0 - 1e-6:
        timestamps.append(elapsed)
        elapsed += 0.1 if elapsed < 2.0 - 1e-6 else 1 / 30

    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as out:
        video = out.add_stream("libx264", rate = 30)
        video.width, video.height = (160, 96)
        video.pix_fmt = "yuv420p"
        video.time_base = Fraction(1, 1000)
        video.codec_context.gop_size = 12
        for elapsed in timestamps:
            value = min(250, int(round(elapsed * 50)))
            frame = av.VideoFrame.from_ndarray(
                np.full((96, 160, 3), value, dtype = np.uint8), format = "rgb24"
            )
            frame.pts = 5_000 + int(round(elapsed * 1_000))
            frame.time_base = Fraction(1, 1000)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()


def _offset_audio_reference_video_data_url():
    """A four-second video at t=5 whose three-second soundtrack begins at t=6."""
    import base64
    import io
    from fractions import Fraction

    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")

    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "matroska") as out:
        video = out.add_stream("ffv1", rate = 24)
        video.width, video.height = (64, 64)
        video.pix_fmt = "yuv444p"
        video.time_base = Fraction(1, 1000)
        audio = out.add_stream("pcm_s16le", rate = 44_100)
        audio.layout = "stereo"
        audio.time_base = Fraction(1, 44_100)
        for index in range(4 * 24):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((64, 64, 3), dtype = np.uint8), format = "rgb24"
            )
            frame.pts = 5_000 + round(index * 1_000 / 24)
            frame.time_base = Fraction(1, 1000)
            for packet in video.encode(frame):
                out.mux(packet)
        written = 0
        while written < 3 * 44_100:
            count = min(1024, 3 * 44_100 - written)
            samples = np.full((1, count * 2), 10_000, dtype = np.int16)
            frame = av.AudioFrame.from_ndarray(samples, format = "s16", layout = "stereo")
            frame.sample_rate = 44_100
            frame.pts = 6 * 44_100 + written
            frame.time_base = Fraction(1, 44_100)
            for packet in audio.encode(frame):
                out.mux(packet)
            written += count
        for packet in audio.encode():
            out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
    return "data:video/x-matroska;base64," + base64.b64encode(buf.getvalue()).decode()


def test_h3_reference_video_decodes_onto_the_models_own_clock():
    pytest.importorskip("av")
    from core.inference.video_minimax_h3 import decode_h3_reference_video

    import base64

    blob = base64.b64decode(_reference_video_data_url(seconds = 3.0, fps = 30).split(",", 1)[1])
    frames, waveform, sample_rate = decode_h3_reference_video(blob)
    # 3 seconds at 24 fps whatever the container's own rate was.
    assert len(frames) == 72
    assert waveform is not None and sample_rate == 44_100
    # And onto the canvas its own aspect ratio resolves to, never upscaled past the source.
    assert frames[0].size == (160, 96)


def test_h3_reference_video_refuses_a_clip_below_the_trained_window():
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(_reference_video_data_url(seconds = 0.5).split(",", 1)[1])
    with pytest.raises(ValueError, match = "2 to 15 seconds"):
        decode_h3_reference_video(blob)


def test_h3_reference_video_refuses_instead_of_silently_truncating_a_long_clip():
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(seconds = 15.1, fps = 24, with_audio = True).split(",", 1)[1]
    )
    with pytest.raises(ValueError, match = "longer than 15s"):
        decode_h3_reference_video(blob)


def test_h3_reference_video_decodes_an_explicit_trim_with_matching_audio():
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(_reference_video_data_url(seconds = 20.0, fps = 24).split(",", 1)[1])
    frames, waveform, sample_rate = decode_h3_reference_video(
        blob,
        trim_start_seconds = 5.0,
        trim_end_seconds = 15.0,
    )

    assert len(frames) == 240
    assert sample_rate == 44_100
    assert waveform is not None and waveform.shape[0] == 10 * sample_rate


def test_h3_reference_video_trim_uses_vfr_pts_relative_to_a_nonzero_stream_start():
    pytest.importorskip("av")
    import base64
    import numpy as np

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(_vfr_reference_video_data_url().split(",", 1)[1])
    frames, waveform, sample_rate = decode_h3_reference_video(
        blob,
        trim_start_seconds = 0.5,
        trim_end_seconds = 2.5,
    )

    assert len(frames) == 48
    assert waveform is None and sample_rate is None
    # Ordinal selection would start near value 50 instead of 25.
    assert float(np.asarray(frames[0]).mean()) == pytest.approx(25, abs = 8)
    assert float(np.asarray(frames[-1]).mean()) > 110


def test_h3_reference_video_trim_preserves_an_embedded_audio_timeline_offset():
    pytest.importorskip("av")
    import base64
    import numpy as np

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(_offset_audio_reference_video_data_url().split(",", 1)[1])
    frames, waveform, sample_rate = decode_h3_reference_video(
        blob,
        trim_start_seconds = 0.5,
        trim_end_seconds = 2.5,
    )

    assert len(frames) == 48
    assert sample_rate == 44_100
    assert waveform is not None and waveform.shape == (2 * sample_rate, 2)
    assert np.max(np.abs(waveform[: sample_rate // 2])) == 0
    assert np.mean(np.abs(waveform[sample_rate // 2 :])) > 0.25


@pytest.mark.parametrize("duration", [2.0, 15.0])
def test_h3_reference_video_trim_keeps_exact_boundary_frame_and_sample_counts(duration):
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(_reference_video_data_url(seconds = duration, fps = 24).split(",", 1)[1])
    frames, waveform, sample_rate = decode_h3_reference_video(
        blob,
        trim_start_seconds = 0.0,
        trim_end_seconds = duration,
    )

    assert len(frames) == round(duration * 24)
    assert sample_rate == 44_100
    assert waveform is not None and waveform.shape[0] == round(duration * sample_rate)


@pytest.mark.parametrize("duration", [2.0, 15.0])
def test_h3_reference_video_without_trim_discards_aac_padding(duration):
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(_reference_video_data_url(seconds = duration, fps = 24).split(",", 1)[1])
    frames, waveform, sample_rate = decode_h3_reference_video(blob)

    assert len(frames) == round(duration * 24)
    assert sample_rate == 44_100
    assert waveform is not None and waveform.shape[0] == round(duration * sample_rate)


@pytest.mark.parametrize("audio_seconds", [15.05, 15.1, 16.0])
def test_h3_reference_video_without_trim_clamps_audio_past_the_video(audio_seconds):
    """A soundtrack longer than its video is clamped to it, not refused.

    Longer tracks decoded fine before trimming existed, and a codec frame of overshoot is
    ordinary padding, so refusing one rejects media that already worked.
    """
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(seconds = 15.0, fps = 24, audio_seconds = audio_seconds).split(",", 1)[
            1
        ]
    )
    frames, waveform, sample_rate = decode_h3_reference_video(blob)

    assert len(frames) == round(15.0 * 24)
    assert waveform is not None
    assert waveform.shape[0] == round(15.0 * sample_rate)


def test_h3_reference_video_trim_seeks_and_stops_both_decoders(monkeypatch):
    av = pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(_reference_video_data_url(seconds = 20.0, fps = 24).split(",", 1)[1])
    real_open = av.open
    opened = []

    class _CountingContainer:
        def __init__(self, *args, **kwargs):
            self.inner = real_open(*args, **kwargs)
            self.seek_types = []
            self.decoded = {"video": 0, "audio": 0}
            opened.append(self)

        @property
        def streams(self):
            return self.inner.streams

        def seek(self, *args, **kwargs):
            self.seek_types.append(kwargs["stream"].type)
            return self.inner.seek(*args, **kwargs)

        def decode(self, *args, **kwargs):
            media_type = "video" if "video" in kwargs else "audio"
            for frame in self.inner.decode(*args, **kwargs):
                self.decoded[media_type] += 1
                yield frame

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

    monkeypatch.setattr(av, "open", _CountingContainer)
    frames, waveform, sample_rate = decode_h3_reference_video(
        blob,
        trim_start_seconds = 15.0,
        trim_end_seconds = 17.0,
    )

    assert len(frames) == 48
    assert waveform is not None and waveform.shape[0] == 2 * sample_rate
    assert any("video" in item.seek_types for item in opened)
    assert any("audio" in item.seek_types for item in opened)
    assert sum(item.decoded["video"] for item in opened) < 200
    assert sum(item.decoded["audio"] for item in opened) < 120


def test_h3_reference_video_trim_outlasting_a_short_soundtrack_stays_silent():
    """A trim the embedded audio does not cover keeps the video and pads the rest silent.

    A video carrying no audio at all is accepted, so one whose track merely ends early must
    be too; the untrimmed path clamps the same mismatch rather than refusing it.
    """
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(seconds = 10.0, fps = 24, audio_seconds = 6.0).split(",", 1)[1]
    )
    frames, waveform, sample_rate = decode_h3_reference_video(
        blob, trim_start_seconds = 2.0, trim_end_seconds = 8.0
    )

    assert len(frames) == round(6.0 * 24)
    assert waveform is not None
    assert waveform.shape[0] == round(6.0 * sample_rate)
    # Audio runs out 4s into the interval; everything past that is silence, not a refusal.
    assert abs(waveform[: round(4.0 * sample_rate)]).max() >= 0.0
    assert abs(waveform[round(4.0 * sample_rate) + sample_rate // 10 :]).max() == 0.0


def test_h3_reference_video_trim_entirely_past_the_soundtrack_drops_the_audio():
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(seconds = 12.0, fps = 24, audio_seconds = 3.0).split(",", 1)[1]
    )
    frames, waveform, _ = decode_h3_reference_video(
        blob, trim_start_seconds = 6.0, trim_end_seconds = 10.0
    )

    assert len(frames) == round(4.0 * 24)
    assert waveform is None


def test_h3_reference_video_trim_before_an_offset_soundtrack_drops_the_audio():
    """A track starting after the interval is absent from it, not silent within it.

    The mirror of the test above, which ends its soundtrack before the trim. Neither leaves a
    sample inside the interval, so both must report no soundtrack: a silent waveform would be
    a fabricated track, and would hide the gap from stage_h3_references' positional pairing.
    """
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(
            seconds = 15.0, fps = 24, audio_seconds = 5.0, audio_start_seconds = 10.0
        ).split(",", 1)[1]
    )
    frames, waveform, _ = decode_h3_reference_video(
        blob, trim_start_seconds = 0.0, trim_end_seconds = 5.0
    )

    assert len(frames) == round(5.0 * 24)
    assert waveform is None


def test_h3_reference_video_trim_reaching_an_offset_soundtrack_keeps_the_audio():
    """The same offset track is still decoded by a trim that does overlap it."""
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(
            seconds = 15.0, fps = 24, audio_seconds = 5.0, audio_start_seconds = 10.0
        ).split(",", 1)[1]
    )
    frames, waveform, sample_rate = decode_h3_reference_video(
        blob, trim_start_seconds = 9.0, trim_end_seconds = 13.0
    )

    assert len(frames) == round(4.0 * 24)
    assert waveform is not None
    assert waveform.shape[0] == round(4.0 * sample_rate)


def test_h3_reference_video_trim_tolerates_a_container_longer_than_its_video():
    """A trim taken from the container duration may reach just past the video track.

    A container reports its longest track, so a file whose audio outruns its video reads as
    longer than it can show, and a browser hands Unsloth that duration. The last frame is held
    across the shortfall instead of refusing a clip that decodes fine untrimmed.
    """
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(seconds = 14.9, fps = 24, audio_seconds = 15.2).split(",", 1)[1]
    )
    frames, _, _ = decode_h3_reference_video(blob, trim_start_seconds = 0.0, trim_end_seconds = 15.0)
    assert len(frames) == round(15.0 * 24)


def test_h3_reference_video_trim_still_refuses_a_real_overshoot():
    """The slack covers container metadata, not a range that genuinely is not there."""
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(seconds = 6.0, fps = 24, with_audio = False).split(",", 1)[1]
    )
    with pytest.raises(ValueError, match = "after the source video"):
        decode_h3_reference_video(blob, trim_start_seconds = 2.0, trim_end_seconds = 10.0)


def test_h3_reference_video_trim_must_be_complete_bounded_and_inside_the_source():
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_video

    blob = base64.b64decode(
        _reference_video_data_url(seconds = 3.0, with_audio = False).split(",", 1)[1]
    )
    with pytest.raises(ValueError, match = "provided together"):
        decode_h3_reference_video(blob, trim_start_seconds = 0.0)
    with pytest.raises(ValueError, match = "2 to 15 seconds"):
        decode_h3_reference_video(blob, trim_start_seconds = 0.0, trim_end_seconds = 1.0)
    with pytest.raises(ValueError, match = "after the source video"):
        decode_h3_reference_video(blob, trim_start_seconds = 2.0, trim_end_seconds = 5.0)


def _frame_index_video(
    seconds = 10.0,
    fps = 24,
    width = 256,
    height = 64,
    bar = 4,
):
    """An MP4 whose every frame encodes its own index as the position of a white bar.

    A flat grey level per frame would not survive the round trip: limited-range YUV folds
    0..255 into 16..235, so neighbouring indices collide. A bar position does.
    """
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    import io

    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as out:
        video = out.add_stream("libx264", rate = fps)
        video.width, video.height = width, height
        video.pix_fmt = "yuv420p"
        # Every frame a keyframe, so seeks land exactly and the two decoders differ only in
        # how they select, never in what they can reach.
        video.options = {"g": "1", "crf": "0", "tune": "zerolatency"}
        for index in range(int(seconds * fps)):
            plane = np.zeros((height, width, 3), dtype = np.uint8)
            plane[:, index : index + bar] = 255
            for packet in video.encode(av.VideoFrame.from_ndarray(plane, format = "rgb24")):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
    return buf.getvalue()


def _decoded_frame_index(image, width = 256):
    """Recover the source frame index from the bar's left edge."""
    np = pytest.importorskip("numpy")

    plane = np.asarray(image.convert("L"), dtype = "float64")
    columns = plane.mean(axis = 0)
    lit = np.flatnonzero(columns > columns.max() / 2)
    assert lit.size, "no bar found in the decoded frame"
    return int(round(lit[0] * width / plane.shape[1]))


def test_h3_reference_video_ordinal_fallback_selects_the_same_frames_as_timestamps():
    """The fallback for streams whose frames carry no presentation timestamps.

    Nothing else reaches this path. A fractional start separates the two rules: the frame on
    screen at t is floor(t*fps), and ceiling it drifts the selection forward by one.
    """
    av = pytest.importorskip("av")

    from core.inference.video_minimax_h3 import (
        _decode_h3_video_trim_by_ordinal,
        _decode_h3_video_trim_by_timestamp,
    )

    blob = _frame_index_video()

    # A start that lands on a frame boundary, and one that falls between two.
    for start, end in ((2.0, 8.0), (2.02, 8.02)):
        ordinal, _ = _decode_h3_video_trim_by_ordinal(blob, av, (start, end))
        timestamp, _ = _decode_h3_video_trim_by_timestamp(blob, av, (start, end))
        assert len(ordinal) == len(timestamp) == round((end - start) * 24)
        assert _decoded_frame_index(ordinal[0]) == _decoded_frame_index(timestamp[0])

    ordinal, _ = _decode_h3_video_trim_by_ordinal(blob, av, (2.02, 8.02))
    assert _decoded_frame_index(ordinal[0]) == 48


def test_h3_reference_video_falls_back_when_timestamps_are_missing(monkeypatch):
    """Prove the dispatcher reaches the ordinal decoder at all, rather than raising."""
    av = pytest.importorskip("av")

    import core.inference.video_minimax_h3 as h3

    blob = _frame_index_video()
    calls = []
    original = h3._decode_h3_video_trim_by_ordinal
    monkeypatch.setattr(
        h3,
        "_decode_h3_video_trim_by_ordinal",
        lambda *args: (calls.append(1), original(*args))[1],
    )

    real_open = av.open

    class _PtsLess:
        """PyAV container attributes are read-only, so the generator is replaced from outside."""

        def __init__(self, container):
            self._container = container

        def decode(self, *args, **kwargs):
            for frame in self._container.decode(*args, **kwargs):
                frame.pts = None
                yield frame

        def __getattr__(self, name):
            return getattr(self._container, name)

        def __enter__(self):
            self._container.__enter__()
            return self

        def __exit__(self, *exc):
            return self._container.__exit__(*exc)

    monkeypatch.setattr(av, "open", lambda *a, **k: _PtsLess(real_open(*a, **k)))

    frames, _, _ = h3.decode_h3_reference_video(
        blob, trim_start_seconds = 2.0, trim_end_seconds = 8.0, decode_audio = False
    )
    assert calls == [1]
    assert len(frames) == round(6.0 * 24)


def test_h3_replacement_audio_bypasses_a_short_embedded_soundtrack():
    pytest.importorskip("av")
    from core.inference.video import VideoBackend
    from core.inference.video_families import detect_video_family

    references = VideoBackend._resolve_references(
        detect_video_family("MiniMaxAI/MiniMax-H3", None),
        "ref2va",
        "diffusers",
        None,
        [
            {
                "video": _reference_video_data_url(
                    seconds = 5.0, fps = 24, with_audio = True, audio_seconds = 1.0
                ),
                "audio": _data_url_wav(seconds = 5.0, rate = 32_000),
                "trim_start_seconds": 2.0,
                "trim_end_seconds": 4.0,
            }
        ],
        None,
        "match",
        960,
        544,
    )

    frames, waveform, sample_rate = references.videos[0]
    assert len(frames) == 48
    assert sample_rate == 32_000
    assert waveform.shape == (64_000, 2)


def test_h3_replacement_audio_starts_at_its_own_zero():
    """A replacement soundtrack is an independent file, not a second cut of the video.

    Its timeline has no relation to the video's, and the picker offers no way to offset it, so
    it plays the clip from its own start. The video's coordinates dropped its first trim_start
    seconds, and refused it when it was shorter than that.
    """
    pytest.importorskip("av")
    import numpy as np

    from core.inference.video import VideoBackend
    from core.inference.video_families import detect_video_family

    # Signal only in the first second, so the decoded offset is recoverable.
    references = VideoBackend._resolve_references(
        detect_video_family("MiniMaxAI/MiniMax-H3", None),
        "ref2va",
        "diffusers",
        None,
        [
            {
                "video": _reference_video_data_url(seconds = 12.0, fps = 24, with_audio = False),
                "audio": _data_url_wav(seconds = 3.0, rate = 32_000, silent_after = 1.0),
                "trim_start_seconds": 5.0,
                "trim_end_seconds": 8.0,
            }
        ],
        None,
        "match",
        960,
        544,
    )

    _, waveform, sample_rate = references.videos[0]
    assert sample_rate == 32_000
    # The clip's length, taken from the replacement's start rather than from its 5s mark.
    assert waveform.shape == (96_000, 2)
    energy = np.abs(waveform).max(axis = 1)
    assert energy[: sample_rate // 2].max() > 0.01
    assert energy[2 * sample_rate :].max() <= 0.01


def test_h3_replacement_audio_shorter_than_the_trim_start_is_kept():
    """The same file, shorter than where the video's trim begins, is padded not refused."""
    pytest.importorskip("av")

    from core.inference.video import VideoBackend
    from core.inference.video_families import detect_video_family

    references = VideoBackend._resolve_references(
        detect_video_family("MiniMaxAI/MiniMax-H3", None),
        "ref2va",
        "diffusers",
        None,
        [
            {
                "video": _reference_video_data_url(seconds = 20.0, fps = 24, with_audio = False),
                "audio": _data_url_wav(seconds = 2.0, rate = 32_000),
                "trim_start_seconds": 10.0,
                "trim_end_seconds": 14.0,
            }
        ],
        None,
        "match",
        960,
        544,
    )

    _, waveform, sample_rate = references.videos[0]
    assert waveform.shape == (4 * 32_000, 2)
    assert sample_rate == 32_000


def test_h3_begin_generate_reuses_preflight_resolved_references(monkeypatch):
    import core.inference.video as video_mod

    backend = _h3_ref_backend(monkeypatch, [])
    expected = object()
    resolve_calls = []
    captured = {}

    def _resolve(*args, **kwargs):
        resolve_calls.append((args, kwargs))
        return expected

    class _DeferredThread:
        def __init__(self, *, target, kwargs, daemon):
            captured.update(target = target, kwargs = kwargs, daemon = daemon)

        def start(self):
            return None

    monkeypatch.setattr(backend, "_resolve_references", _resolve)
    monkeypatch.setattr(backend, "_state_device_target", lambda _state: None)
    monkeypatch.setattr(video_mod.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(backend, "_generate_h3_native", lambda **kwargs: kwargs["references"])

    backend.begin_generate(prompt = "p")
    assert len(resolve_calls) == 1

    original_state = backend._state
    backend._state = dataclasses.replace(original_state)
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(**captured["kwargs"])
    backend._state = original_state

    assert backend.generate(**captured["kwargs"]) is expected
    assert len(resolve_calls) == 1

    # Direct callers still resolve their own raw inputs.
    assert backend.generate(prompt = "p") is expected
    assert len(resolve_calls) == 2


def test_a_v1_videos_job_id_stays_out_of_the_replayable_worker_kwargs(monkeypatch):
    """begin_generate's worker kwargs must stay a valid generate() call.

    The /v1/videos job id is worker bookkeeping, not a generation input, so it rides on the
    thread target beside job_token. Putting it in kwargs made every caller that replays them
    -- the H3 preflight reuse path does -- raise TypeError on an unexpected keyword."""
    import core.inference.video as video_mod

    backend = _h3_ref_backend(monkeypatch, [])
    expected = object()
    captured = {}

    class _DeferredThread:
        def __init__(self, *, target, kwargs, daemon):
            captured.update(target = target, kwargs = kwargs, daemon = daemon)

        def start(self):
            return None

    monkeypatch.setattr(backend, "_resolve_references", lambda *a, **k: expected)
    monkeypatch.setattr(backend, "_state_device_target", lambda _state: None)
    monkeypatch.setattr(video_mod.threading, "Thread", _DeferredThread)
    monkeypatch.setattr(backend, "_generate_h3_native", lambda **kwargs: kwargs["references"])

    backend.begin_generate(prompt = "p", video_id = "video_abc123")

    assert "video_id" not in captured["kwargs"]
    assert captured["target"].keywords["video_id"] == "video_abc123"
    # The replay the H3 preflight path performs must still be accepted.
    assert backend.generate(**captured["kwargs"]) is expected


def test_h3_native_refuses_a_later_video_soundtrack_after_a_silent_video(monkeypatch):
    pytest.importorskip("av")
    backend = _h3_ref_backend(monkeypatch, [])

    with pytest.raises(ValueError, match = "Video 2 has audio after a silent earlier video"):
        backend.begin_generate(
            prompt = "match <Video 1> and <Video 2>",
            reference_videos = [
                {"video": _reference_video_data_url(with_audio = False)},
                {"video": _reference_video_data_url(with_audio = True)},
            ],
        )


def test_h3_reference_images_are_fitted_to_the_requested_policy():
    Image = pytest.importorskip("PIL.Image")
    from core.inference.video_minimax_h3 import fit_h3_reference_image

    source = Image.new("RGB", (6000, 3000), (0, 0, 0))
    # "match" brings the reference down to the generation's pixel area, aspect kept.
    matched = fit_h3_reference_image(source, width = 960, height = 544, policy = "match")
    assert matched.size[0] / matched.size[1] == pytest.approx(2.0, abs = 0.05)
    assert matched.size[0] * matched.size[1] <= 960 * 544 * 1.1
    # "max" targets the reference pipeline's own 2048px short edge instead.
    biggest = fit_h3_reference_image(source, width = 960, height = 544, policy = "max")
    assert min(biggest.size) == 2048 and max(biggest.size) == 4096
    # Neither ever upscales a source already under its target; it only snaps to 32.
    small = Image.new("RGB", (128, 128), (0, 0, 0))
    assert fit_h3_reference_image(small, width = 1344, height = 768, policy = "max").size == (128, 128)
    with pytest.raises(ValueError, match = "policy"):
        fit_h3_reference_image(source, width = 960, height = 544, policy = "huge")


def test_h3_reference_image_fits_a_phone_photo_before_the_generic_source_limit():
    pytest.importorskip("PIL.Image")
    from core.inference.diffusion import decode_b64_image
    from core.inference.video import VideoBackend
    from core.inference.video_families import detect_video_family

    source = _data_url(5712, 4284)
    with pytest.raises(ValueError, match = "maximum is 4096px per side"):
        decode_b64_image(source)

    references = VideoBackend._resolve_references(
        detect_video_family("MiniMaxAI/MiniMax-H3", None),
        "ref2va",
        "diffusers",
        [source],
        None,
        None,
        "match",
        960,
        544,
    )

    assert len(references.images) == 1
    fitted = references.images[0]
    assert fitted.size[0] * fitted.size[1] <= 960 * 544 * 1.1
    assert fitted.size[0] / fitted.size[1] == pytest.approx(5712 / 4284, abs = 0.05)

    largest = VideoBackend._resolve_references(
        detect_video_family("MiniMaxAI/MiniMax-H3", None),
        "ref2va",
        "diffusers",
        [source],
        None,
        None,
        "max",
        960,
        544,
    ).images[0]
    assert min(largest.size) == 2048


def test_h3_reference_image_source_policy_rejects_excessive_area_before_loading(monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    from core.inference.diffusion import decode_b64_image
    from core.inference.video_minimax_h3 import (
        H3_REF_IMAGE_SOURCE_MAX_PIXELS,
        H3_REF_IMAGE_SOURCE_MAX_SIDE,
    )

    loaded = False

    def tracked_load():
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(
        Image,
        "open",
        lambda _stream: types.SimpleNamespace(size = (8000, 5000), load = tracked_load),
    )
    with pytest.raises(ValueError, match = "source pixels"):
        decode_b64_image(
            "eA==",
            max_side = H3_REF_IMAGE_SOURCE_MAX_SIDE,
            max_pixels = H3_REF_IMAGE_SOURCE_MAX_PIXELS,
        )
    assert loaded is False


def test_h3_native_generate_stages_every_reference_kind(monkeypatch, tmp_path):
    pytest.importorskip("av")
    calls: list = []
    backend = _h3_ref_backend(monkeypatch, calls)

    staged: dict = {}

    class _Engine:
        def generate_video(self, files, params, **kwargs):
            # The scratch dir is deleted with the run, so read it here.
            staged["images"] = [Path(p).name for p in params.ref_images]
            staged["video_frames"] = sorted(Path(params.ref_videos[0]).iterdir())
            staged["video_audios"] = [Path(p).name for p in params.ref_video_audios]
            staged["audios"] = [Path(p).name for p in params.ref_audios]
            staged["wav_ok"] = all(
                Path(p).read_bytes()[:4] == b"RIFF"
                for p in params.ref_video_audios + params.ref_audios
            )
            staged["keyframes"] = (params.init_img, params.end_img)
            return Path("/tmp/does-not-exist.webm")

    from core.inference.video_minimax_h3 import MiniMaxH3NativeRuntime

    object.__setattr__(
        backend._state,
        "pipe",
        MiniMaxH3NativeRuntime(engine = _Engine(), files = object(), offload_flags = ()),
    )

    result = backend.generate(
        prompt = "the cat from <Picture 1> surfing, matching <Video 1>",
        width = 960,
        height = 544,
        reference_images = [_data_url(1200, 800), _data_url(640, 640)],
        reference_videos = [{"video": _reference_video_data_url()}],
        reference_audios = [_data_url_wav()],
    )

    assert staged["images"] == ["ref-image-00.png", "ref-image-01.png"]
    # A reference video is a DIRECTORY of frames sd-cli reads lexicographically, at 24 fps.
    assert len(staged["video_frames"]) == 72
    assert [p.name for p in staged["video_frames"][:2]] == ["00000.png", "00001.png"]
    assert staged["video_audios"] == ["ref-video-audio-00.wav"]
    assert staged["audios"] == ["ref-audio-00.wav"]
    assert staged["wav_ok"]
    assert staged["keyframes"] == (None, None)
    assert result["conditioning"] == "ref2va"


def _data_url_wav(
    seconds = 1.0,
    rate = 32_000,
    silent_after = None,
):
    """A 440Hz tone; `silent_after` mutes the tail so a decoded offset is recoverable."""
    import base64
    import io
    import math
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        loud = int(seconds * rate) if silent_after is None else int(silent_after * rate)
        for i in range(int(seconds * rate)):
            value = int(8000 * math.sin(2 * math.pi * 440 * i / rate)) if i < loud else 0
            frames += int(value).to_bytes(2, "little", signed = True) * 2
        handle.writeframes(bytes(frames))
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def test_h3_reference_audio_decodes_inside_the_trained_window():
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_audio

    blob = base64.b64decode(_data_url_wav(seconds = 2.0).split(",", 1)[1])
    waveform, rate = decode_h3_reference_audio(blob)
    assert rate == 32_000
    assert waveform.shape == (64_000, 2)


def test_h3_reference_audio_refuses_a_clip_past_the_window_while_decoding():
    # The encoded size does not bound the decoded size: the route's 32 MiB limit admits over half
    # an hour of compressed audio, which lands as ~1.9 GB of float32 and doubles again in the
    # concatenate, three times over per request. The refusal has to happen DURING the decode, not
    # after it, or the allocation the bound exists to prevent has already happened.
    pytest.importorskip("av")
    import base64

    from core.inference.video_minimax_h3 import decode_h3_reference_audio

    blob = base64.b64decode(_data_url_wav(seconds = 15.5).split(",", 1)[1])
    with pytest.raises(ValueError, match = "up to 15 seconds"):
        decode_h3_reference_audio(blob)


def test_h3_partitions_refuse_each_others_conditioning(monkeypatch):
    # The resident partition decides which conditioning is valid.
    keyframe_backend = _h3_native_backend(monkeypatch, [])
    with pytest.raises(ValueError, match = "Load a minimax_h3_ref2va checkpoint"):
        keyframe_backend.generate(prompt = "p", reference_images = [_data_url(64, 64)])

    reference_backend = _h3_ref_backend(monkeypatch, [])
    with pytest.raises(ValueError, match = "Load a minimax_h3_fl2va checkpoint"):
        reference_backend.generate(prompt = "p", first_frame = _data_url(64, 64))


def test_h3_native_refuses_the_max_reference_policy(monkeypatch):
    # sd.cpp cannot preserve the 2048px "max" policy.
    backend = _h3_ref_backend(monkeypatch, [])
    with pytest.raises(ValueError, match = "needs the Diffusers engine"):
        backend.generate(
            prompt = "p", reference_images = [_data_url(64, 64)], reference_image_size = "max"
        )


def test_h3_status_reports_the_resident_partition(monkeypatch):
    keyframe_backend = _h3_native_backend(monkeypatch, [])
    status = keyframe_backend.status()
    assert status["h3_task"] == "fl2va"
    assert status["supports_keyframes"] is True and status["supports_references"] is False

    reference_backend = _h3_ref_backend(monkeypatch, [])
    status = reference_backend.status()
    assert status["h3_task"] == "ref2va"
    assert status["supports_keyframes"] is False and status["supports_references"] is True


def test_h3_load_refuses_a_task_that_contradicts_the_picked_checkpoint():
    backend = VideoBackend()
    backend.validate_load_request(
        "unsloth/MiniMax-H3-GGUF",
        gguf_filename = "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
        family_override = "minimax-h3",
        h3_task = "ref2va",
    )
    with pytest.raises(ValueError, match = "is the ref2va partition"):
        backend.validate_load_request(
            "unsloth/MiniMax-H3-GGUF",
            gguf_filename = "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
            family_override = "minimax-h3",
            h3_task = "fl2va",
        )


def test_h3_modular_load_brings_up_the_requested_partition(monkeypatch):
    import types

    seen: dict = {}

    class _FakeModularPipeline:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            return cls()

        def load_components(self, **kwargs):
            seen["workflow"] = kwargs.get("workflow")

        def to(self, device):
            return self

    diffusers = types.SimpleNamespace(
        ComponentsManager = lambda: types.SimpleNamespace(
            enable_auto_cpu_offload = lambda **kwargs: None
        ),
        ModularPipeline = _FakeModularPipeline,
    )
    torch = types.SimpleNamespace(bfloat16 = "bf16")
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")

    backend = VideoBackend()
    status = backend._load_h3_modular_pipeline(
        # Precision is not what this covers, and an unset request resolves to int8 now,
        # which sends the load down the hosted-checkpoint path instead of this one.
        transformer_quant = "none",
        diffusers = diffusers,
        torch = torch,
        fam = fam,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base = fam.base_repo,
        kind = "pipeline",
        dtype = torch.bfloat16,
        device = "cuda",
        hf_token = None,
        memory_mode = None,
        h3_task = "ref2va",
    )
    assert seen["workflow"] == "ref2va"
    assert status["supports_references"] is True and status["supports_keyframes"] is False


def test_h3_flow_shift_defaults_to_the_released_schedule(monkeypatch):
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    result = backend.generate(prompt = "p", width = 640, height = 384)
    assert calls[-1]["params"].flow_shift == 12.0
    assert (result["flow_shift"], result["audio_flow_shift"]) == (12.0, 3.0)

    backend.generate(prompt = "p", width = 640, height = 384, flow_shift = 6.5)
    assert calls[-1]["params"].flow_shift == 6.5

    # sd.cpp accepts only the released audio shift it already applies.
    backend.generate(prompt = "p", width = 640, height = 384, audio_flow_shift = 3.0)
    with pytest.raises(ValueError, match = "Diffusers engine"):
        backend.generate(prompt = "p", width = 640, height = 384, audio_flow_shift = 5.0)

    status = backend.status()
    assert status["defaults"]["flow_shift"] == 12.0
    assert status["defaults"]["audio_flow_shift"] == 3.0
    assert status["defaults"]["supports_audio_flow_shift"] is False


def test_schedule_shifts_are_refused_by_families_that_do_not_expose_them():
    fam = _detect_load_family("Lightricks/LTX-2", None, None)

    with pytest.raises(ValueError, match = "does not expose a video flow_shift"):
        VideoBackend._resolve_flow_shifts(fam, "diffusers", 5.0, None)
    with pytest.raises(ValueError, match = "does not expose an audio_flow_shift"):
        VideoBackend._resolve_flow_shifts(fam, "diffusers", None, 3.0)


def test_h3_modular_generate_sets_both_schedule_shifts(fake_runtime, monkeypatch):
    # Scheduler components must receive both shifts on every run.
    import contextlib
    import types

    from core.inference.video import _VideoLoadState

    shifts: dict = {}

    class _Scheduler:
        def __init__(self, name):
            self._name = name

        def set_shift(self, value):
            shifts[self._name] = value

        def step(self, *args, **kwargs):
            return None

    class _ModularPipe:
        scheduler = _Scheduler("video")
        audio_scheduler = _Scheduler("audio")

        def __call__(self, **kwargs):
            shifts["called"] = True
            return {"videos": [[object()]], "audio": None, "sampling_rate": None}

    backend = VideoBackend()
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    backend._state = _VideoLoadState(
        pipe = _ModularPipe(),
        family = fam,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base_repo = fam.base_repo,
        device = "cpu",
        dtype = "bfloat16",
        kind = "pipeline",
        engine = "diffusers",
        h3_task = "fl2va",
    )
    monkeypatch.setattr(VideoBackend, "_encode_mp4", staticmethod(lambda *a, **k: b"MP4"))
    monkeypatch.setattr("core.inference.video.contextlib", contextlib)

    result = backend.generate(prompt = "p", width = 960, height = 544, steps = 2)
    assert shifts["video"] == 12.0 and shifts["audio"] == 3.0

    backend.generate(
        prompt = "p",
        width = 960,
        height = 544,
        steps = 2,
        flow_shift = 7.0,
        audio_flow_shift = 4.0,
    )
    assert shifts["video"] == 7.0 and shifts["audio"] == 4.0
    assert result["flow_shift"] == 12.0


def test_h3_begin_generate_refuses_an_unhonourable_audio_shift(monkeypatch):
    # Reject unsupported shifts before the asynchronous job starts.
    backend = _h3_native_backend(monkeypatch, [])
    with pytest.raises(ValueError, match = "Diffusers engine"):
        backend.begin_generate(prompt = "p", audio_flow_shift = 6.0)
    assert backend.generate_progress()["active"] is False
    # The released value is not a request to change anything, so it must still start.
    backend.begin_generate(prompt = "p", audio_flow_shift = 3.0)


def test_h3_ref2va_partition_refuses_a_reference_less_request(monkeypatch):
    """A Ref2VA load fetched only `transformer_ref`, and only the reference branch reads it.

    A text-only request routes to the t2va branch, finds `transformer` unloaded and dies inside
    the Diffusers blocks with "'NoneType' object has no attribute 'forward'" -- caught running
    the real BF16 weights. Refused at the boundary instead, on both engines, so the rule does
    not depend on which one is active.
    """
    backend = _h3_ref_backend(monkeypatch, [])
    with pytest.raises(ValueError, match = "Add at least one reference"):
        backend.generate(prompt = "a fox in snow", width = 640, height = 384)
    with pytest.raises(ValueError, match = "Add at least one reference"):
        backend.begin_generate(prompt = "a fox in snow")
    assert backend.generate_progress()["active"] is False
    # The FL2VA partition is where text-only belongs, and it still takes it.
    keyframe_backend = _h3_native_backend(monkeypatch, [])
    assert (
        keyframe_backend.generate(prompt = "a fox in snow", width = 640, height = 384)["conditioning"]
        == "t2va"
    )


def test_h3_vae_trim_keeps_the_encoder_for_the_workflows_that_encode():
    """The encoder drop is gated on t2va, and neither workflow Unsloth loads is text-only.

    fl2va encodes its keyframes and ref2va its references, both through this VAE, so dropping the
    encoder half would break image conditioning outright. The decoder pre-cast -- the larger of
    the two savings -- is unconditional and must still happen for either.
    """
    torch = pytest.importorskip("torch")
    from core.inference.video_minimax_h3 import trim_h3_video_vae

    def _vae():
        vae = types.SimpleNamespace()
        vae.encoder = torch.nn.Conv2d(3, 8, 3)
        vae.quant_conv = torch.nn.Conv2d(8, 8, 1)
        vae.decoder = torch.nn.Conv2d(8, 3, 3)
        vae.post_quant_conv = torch.nn.Conv2d(8, 8, 1)
        return vae

    for workflow in ("fl2va", "ref2va"):
        vae = _vae()
        report = trim_h3_video_vae(vae, workflow = workflow)
        assert vae.encoder is not None, workflow
        assert vae.quant_conv is not None, workflow
        assert report["encoder_freed"] == 0, workflow
        # The decoder is still pre-cast to float16, which is where the bulk of the saving is.
        assert report["decoder_freed"] > 0, workflow
        assert vae.decoder.weight.dtype is torch.float16, workflow

    # The gate itself still works for a text-only load, so the saving is not lost, only unused.
    vae = _vae()
    assert trim_h3_video_vae(vae, workflow = "t2va")["encoder_freed"] > 0
    assert vae.encoder is None


# ── a failed video must say what it was rendering (#8225) ────────────────────────


def _failing_pipe_call(exc):
    # The explicit signature matters: generate() inspects it, and a bare **kwargs would advertise
    # no callback_on_step_end and route the call down the scheduler-wrapping path instead.
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
        raise exc

    return _boom


def _capture_generate_failures(monkeypatch):
    """Collect the ``video.generate_failed_request`` lines.

    The module logger is structlog, which caplog does not see, so wrap it and pass everything
    else straight through to the real one."""
    import core.inference.video as video_mod

    lines: list = []
    real = video_mod.logger

    class _Recorder:
        def error(self, fmt, *a, **k):
            message = fmt % a if a else fmt
            if "generate_failed_request" in message:
                lines.append(message)
            return real.error(fmt, *a, **k)

        def __getattr__(self, name):
            return getattr(real, name)

    monkeypatch.setattr(video_mod, "logger", _Recorder())
    return lines


def test_a_failed_generation_logs_the_resolved_request(fake_runtime, tmp_path, monkeypatch):
    # The reported bug: the entire server-side record of the OOM in #8225 was the exception string,
    # so the resolution and frame count behind a 66.54 GiB allocation had to be recovered by
    # dividing it by the head count. Log the request that produced it.
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(
        type(backend._state.pipe),
        "__call__",
        _failing_pipe_call(RuntimeError("CUDA out of memory. Tried to allocate 66.54 GiB.")),
    )
    monkeypatch.setattr(video_mod, "sdpa_math_only", lambda target: False)
    records = _capture_generate_failures(monkeypatch)

    with pytest.raises(RuntimeError):
        backend.generate(prompt = "a sloth surfing", width = 1000, height = 700, num_frames = 120)

    assert len(records) == 1
    line = records[0]
    # The SNAPPED shape, which is what actually ran, not the caller's raw request.
    assert "width=992" in line and "height=672" in line and "frames=113" in line
    assert "steps=" in line and "seed=" in line and "family=ltx-2" in line
    assert "device=" in line and "offload=" in line


def test_a_rejected_request_is_not_recorded_as_a_server_failure(
    fake_runtime, tmp_path, monkeypatch
):
    # _run_generate maps every ValueError from the pipeline to client input feedback and gives it
    # no video.generate_failed record, so the request-shape log must not turn the same rejection
    # into an ERROR entry: a user asking for a shape the pipeline refuses is not a server failure.
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(
        type(backend._state.pipe),
        "__call__",
        _failing_pipe_call(ValueError("`height` and `width` have to be divisible by 32")),
    )
    records = _capture_generate_failures(monkeypatch)

    with pytest.raises(ValueError):
        backend.generate(prompt = "a sloth surfing", width = 1000, height = 700, num_frames = 120)

    assert records == []


def test_the_failure_log_carries_no_prompt_text(fake_runtime, tmp_path, monkeypatch):
    # A length distinguishes an empty prompt from a long one, which is all a bug report needs; the
    # server log is not the place for user content.
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(
        type(backend._state.pipe), "__call__", _failing_pipe_call(RuntimeError("kaboom"))
    )
    monkeypatch.setattr(video_mod, "sdpa_math_only", lambda target: False)
    records = _capture_generate_failures(monkeypatch)

    with pytest.raises(RuntimeError):
        backend.generate(prompt = "a secret about oranges", negative_prompt = "blurry")

    line = records[0]
    assert "oranges" not in line and "blurry" not in line
    assert "prompt_chars=22" in line and "negative_prompt_chars=6" in line


def test_an_oom_on_a_math_only_device_names_the_quadratic_cost(fake_runtime, tmp_path, monkeypatch):
    # The whole point of #8225: the model was not too big, the score matrix was. When the device
    # has no fused kernel, an OOM should arrive with that diagnosis attached.
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(
        type(backend._state.pipe),
        "__call__",
        _failing_pipe_call(RuntimeError("CUDA out of memory. Tried to allocate 66.54 GiB.")),
    )
    monkeypatch.setattr(video_mod, "sdpa_math_only", lambda target: True)
    records = _capture_generate_failures(monkeypatch)

    with pytest.raises(RuntimeError):
        backend.generate(prompt = "a clip")

    assert len(records) == 2
    assert "score matrix" in records[1] and "SQUARE" in records[1]


def test_a_non_oom_failure_does_not_blame_attention(fake_runtime, tmp_path, monkeypatch):
    # Math-only is a real condition, but it did not cause a scheduler bug. Only allocator failures
    # get the diagnosis, or it becomes noise on every unrelated crash.
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(
        type(backend._state.pipe),
        "__call__",
        _failing_pipe_call(RuntimeError("scheduler produced NaN sigmas")),
    )
    monkeypatch.setattr(video_mod, "sdpa_math_only", lambda target: True)
    records = _capture_generate_failures(monkeypatch)

    with pytest.raises(RuntimeError):
        backend.generate(prompt = "a clip")

    assert len(records) == 1
    assert "score matrix" not in records[0]


def test_a_cancel_is_not_logged_as_a_failure(fake_runtime, tmp_path, monkeypatch):
    # Cancels unwind through the same handler. A user pressing stop is an outcome, not a bug
    # report, and logging it would bury the real ones.
    from core.inference.video_families import VIDEO_CANCELLED_MSG

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    records = _capture_generate_failures(monkeypatch)
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(prompt = "a clip", cancel_event = cancel)

    assert records == []


def test_a_failure_before_the_request_resolves_logs_nothing(fake_runtime, tmp_path, monkeypatch):
    # There is nothing truthful to report until the shape is snapped, and a half-bound record
    # would be worse than none.
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    records = _capture_generate_failures(monkeypatch)

    def _broken(fam, w, h):
        raise RuntimeError("resolution table is broken")

    monkeypatch.setattr(video_mod, "snap_video_size", _broken)

    with pytest.raises(RuntimeError, match = "resolution table"):
        backend.generate(prompt = "a clip")

    assert records == []


def test_a_broken_diagnostic_never_replaces_the_real_failure(fake_runtime, tmp_path, monkeypatch):
    # If the probe or the formatting raises, the caller must still see the original exception.
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(
        type(backend._state.pipe),
        "__call__",
        _failing_pipe_call(RuntimeError("CUDA out of memory. Tried to allocate 66.54 GiB.")),
    )

    def _boom(target):
        raise RuntimeError("the probe itself is broken")

    monkeypatch.setattr(video_mod, "sdpa_math_only", _boom)

    with pytest.raises(RuntimeError, match = "66.54 GiB"):
        backend.generate(prompt = "a clip")


@pytest.mark.parametrize(
    "exc, expected",
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 66.54 GiB."), True),
        (RuntimeError("HIP out of memory"), True),
        (RuntimeError("Tried to allocate 2.00 GiB"), True),
        (RuntimeError("scheduler produced NaN sigmas"), False),
        (ValueError("bad prompt"), False),
    ],
)
def test_out_of_memory_is_recognised_by_text_not_only_by_class(exc, expected):
    # torch raises torch.OutOfMemoryError on CUDA but a plain RuntimeError on some backends, so
    # an isinstance check alone would miss exactly the reports this is for.
    import core.inference.video as video_mod
    assert video_mod._is_out_of_memory(exc) is expected


def test_the_oom_diagnosis_is_skipped_when_an_external_backend_ran(
    fake_runtime, tmp_path, monkeypatch
):
    """AITER / xFormers replace torch's own dispatch, so probing native SDPA answers about code the
    generation never executed. On a ROCm card with AITER engaged that turned every OOM into a
    confident false statement that attention ran on the quadratic math backend."""
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    backend._state = replace(backend._state, attention_backend = "aiter")
    monkeypatch.setattr(
        type(backend._state.pipe),
        "__call__",
        _failing_pipe_call(RuntimeError("HIP out of memory. Tried to allocate 66.54 GiB.")),
    )
    probed: list = []

    def _never(target):
        probed.append(target)
        return True

    monkeypatch.setattr(video_mod, "sdpa_math_only", _never)
    records = _capture_generate_failures(monkeypatch)

    with pytest.raises(RuntimeError):
        backend.generate(prompt = "a clip")

    assert probed == [], "an engaged external backend must not be diagnosed as native SDPA"
    assert not any(video_mod.SDPA_MATH_ONLY_MESSAGE in line for line in records)
    # The request record itself is still logged; only the diagnosis is withheld.
    assert any("family=ltx-2" in line for line in records)


def test_the_oom_probe_uses_the_dtype_the_run_actually_used(fake_runtime, tmp_path, monkeypatch):
    """Video families promote a resolved fp16 to fp32, and the fused SDPA kernels are
    half-precision only. A dtypeless probe target defaults to fp16, so it would answer for a
    precision the run never used."""
    import torch

    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    backend._state = replace(backend._state, dtype = "float32", attention_backend = None)
    monkeypatch.setattr(
        type(backend._state.pipe),
        "__call__",
        _failing_pipe_call(RuntimeError("CUDA out of memory. Tried to allocate 66.54 GiB.")),
    )
    seen: list = []
    monkeypatch.setattr(video_mod, "sdpa_math_only", lambda target: seen.append(target) or True)
    records = _capture_generate_failures(monkeypatch)

    with pytest.raises(RuntimeError):
        backend.generate(prompt = "a clip")

    # The run's dtype reaches the record, which is what the probe target is built from.
    assert any("dtype=float32" in line for line in records)
    assert len(seen) == 1


def test_the_probe_target_resolves_the_recorded_dtype():
    """The other half: that recorded string becomes the real torch dtype the probe asks about."""
    import torch

    import core.inference.video as video_mod

    assert video_mod._probe_target({"device": "cuda", "dtype": "float32"}).dtype is torch.float32
    assert video_mod._probe_target({"device": "cuda", "dtype": "bfloat16"}).dtype is torch.bfloat16
    # No recorded dtype keeps the probe's own half-precision default.
    assert video_mod._probe_target({"device": "cuda"}).dtype is None
    # A junk value, or a torch attribute that is not a dtype, can never become a bogus dtype.
    assert video_mod._probe_target({"device": "cuda", "dtype": "nonsense"}).dtype is None
    assert video_mod._probe_target({"device": "cuda", "dtype": "nn"}).dtype is None


# ── text-encoder budgeting ───────────────────────────────────────────────────
# The plan sizes companions from the family's bf16 (transformer, text_encoder, vae) table. A pick
# that takes its encoder PRE-CAST from a hosted fp8 checkpoint loads roughly half that encoder, and
# for ltx-2 the encoder is Gemma3-12B at 24.4 GB, so budgeting it at bf16 over-states the resident
# requirement by ~11 GB. PR #8213 gates a hard unified-memory refusal on model_dense_mib, which
# turns that over-estimate from a conservative offload choice into a refused load that would fit.
_MIB_PER_GB = 1000.0**3 / (1024.0 * 1024.0)


def _capture_plan(monkeypatch):
    """Record every plan_diffusion_memory call's kwargs, keeping the real planner."""
    import core.inference.video as video_mod

    calls = []
    real = video_mod.plan_diffusion_memory

    def _spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(video_mod, "plan_diffusion_memory", _spy)
    return calls


def _allow_te_prequant(monkeypatch, *, injects = True):
    """Make the pre-cast encoder resolvable (device-gated on CUDA in real life) without any IO.

    Injection itself needs the Hub and a real transformers class, so it is stubbed either way;
    ``injects=False`` models the fallback (unreachable / corrupt / base-mismatched checkpoint),
    where the load ends up with the dense encoder and the budget must return to bf16."""
    import core.inference.diffusion_precision as precision
    import core.inference.diffusion_te_prequant as tpq

    # These tests are about the BUDGET the plan is built from, not about the cast. The CPU stub
    # pipeline carries no encoder the quantiser can cast, and the strict default refuses an
    # explicit precision it cannot honour, so keep the legacy silent fallback here.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ALLOW_PRECISION_FALLBACK", "1")
    monkeypatch.setattr(precision, "te_quant_supported", lambda target, mode: True)
    injected = {"text_encoder": object()} if injects else {}
    monkeypatch.setattr(tpq, "te_prequant_pipe_kwargs", lambda *a, **k: dict(injected))
    return tpq.TE_PREQUANT_BUDGET_SCALE


def test_gguf_plan_budgets_a_pre_cast_text_encoder_at_its_real_size(
    fake_runtime, monkeypatch, tmp_path
):
    from core.inference.video_families import detect_video_family

    scale = _allow_te_prequant(monkeypatch)
    transformer_gb, text_encoder_gb, vae_gb = detect_video_family(
        "Lightricks/LTX-2"
    ).bf16_components_gb

    calls = _capture_plan(monkeypatch)
    backend = VideoBackend()
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
        text_encoder_quant = "fp8",
    )
    quant_companion = calls[0]["companion_dense_mib"]
    quant_dense = calls[0]["model_dense_mib"]

    calls.clear()
    backend = VideoBackend()
    backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
    )
    bf16_companion = calls[0]["companion_dense_mib"]
    bf16_dense = calls[0]["model_dense_mib"]

    # The encoder is scaled...
    assert quant_companion == int((text_encoder_gb * scale + vae_gb) * _MIB_PER_GB)
    # ...the VAE is NOT: nothing on this path quantises it, and the Wan families widen it to fp32.
    assert quant_companion - int(text_encoder_gb * scale * _MIB_PER_GB) == int(vae_gb * _MIB_PER_GB)
    assert bf16_companion == int((text_encoder_gb + vae_gb) * _MIB_PER_GB)
    # The saving reaches model_dense_mib, which is the number PR #8213 refuses on.
    assert bf16_dense - quant_dense == bf16_companion - quant_companion
    # ~11 GB for Gemma3-12B, so this is worth a whole offload tier rather than rounding noise.
    assert (bf16_dense - quant_dense) / _MIB_PER_GB > 8.0
    # The transformer term is untouched by a TEXT-ENCODER quant.
    assert bf16_dense - bf16_companion == quant_dense - quant_companion


def test_pipeline_plan_budgets_a_pre_cast_text_encoder_at_its_real_size(fake_runtime, monkeypatch):
    from core.inference.video_families import detect_video_family

    scale = _allow_te_prequant(monkeypatch)
    transformer_gb, text_encoder_gb, vae_gb = detect_video_family(
        "Lightricks/LTX-2"
    ).bf16_components_gb

    calls = _capture_plan(monkeypatch)
    VideoBackend().load_pipeline(
        "Lightricks/LTX-2", model_kind = "pipeline", text_encoder_quant = "fp8"
    )
    assert calls[0]["model_dense_mib"] == int(
        (transformer_gb + text_encoder_gb * scale + vae_gb) * _MIB_PER_GB
    )
    # The pipeline kind budgets one total, so companion_dense_mib stays None as before.
    assert calls[0]["companion_dense_mib"] is None


def test_plan_returns_to_bf16_when_the_pre_cast_encoder_does_not_inject(fake_runtime, monkeypatch):
    # The source resolves (so the budget shrinks) but the checkpoint cannot be loaded -- an
    # unreachable Hub, a corrupt artifact, a base-model mismatch. Assembly falls back to the dense
    # encoder, so the plan must be rebuilt at bf16 before placement is applied; leaving the fp8
    # budget in place under-states ltx-2's resident requirement by ~8.5 GB and can pick resident.
    from core.inference.video_families import detect_video_family

    components = detect_video_family("Lightricks/LTX-2").bf16_components_gb
    _allow_te_prequant(monkeypatch, injects = False)

    calls = _capture_plan(monkeypatch)
    VideoBackend().load_pipeline(
        "Lightricks/LTX-2", model_kind = "pipeline", text_encoder_quant = "fp8"
    )
    assert calls[0]["model_dense_mib"] < int(sum(components) * _MIB_PER_GB)
    assert calls[-1]["model_dense_mib"] == int(sum(components) * _MIB_PER_GB)


def test_plan_keeps_the_bf16_budget_without_a_hosted_pre_cast_encoder(fake_runtime, monkeypatch):
    # Wan requests fp8 the same way, but hosts no pre-cast checkpoint: the encoder is downloaded
    # dense and cast in place AFTER assembly, so its PEAK is bf16 and the budget must not shrink.
    from core.inference.video_families import detect_video_family

    _allow_te_prequant(monkeypatch)
    components = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers").bf16_components_gb

    calls = _capture_plan(monkeypatch)
    VideoBackend().load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline", text_encoder_quant = "fp8"
    )
    assert calls[0]["model_dense_mib"] == int(sum(components) * _MIB_PER_GB)


def test_plan_is_unchanged_when_no_text_encoder_quant_is_requested(fake_runtime, monkeypatch):
    # The whole change is inert on a default load: every family keeps its bf16-table budget, so no
    # existing decision on any device moves.
    from core.inference.diffusion_families import family_pipeline_available
    from core.inference.video_families import _FAMILIES

    _allow_te_prequant(monkeypatch)
    calls = _capture_plan(monkeypatch)
    for fam in _FAMILIES:
        if fam.bf16_components_gb is None or fam.name == "wan2.2-t2v-a14b":
            continue
        # A modular-workflow family has no dense plan to be unchanged: load_pipeline dispatches to
        # the workflow's own loader before plan_diffusion_memory is ever reached, because there is
        # no single dense pipeline to budget -- each component is built by its own from_pretrained.
        if fam.modular_workflow:
            continue
        calls.clear()
        VideoBackend().load_pipeline(fam.base_repo, model_kind = "pipeline")
        assert calls[0]["model_dense_mib"] == int(
            sum(fam.bf16_components_gb) * _MIB_PER_GB
        ), fam.name


def test_dense_quant_replan_uses_the_scaled_text_encoder(fake_runtime, monkeypatch):
    # The transformer-quant re-plan rebuilds the total from the table; it must carry the same
    # scaled encoder, or it re-introduces the over-estimate the first plan just dropped.
    import core.inference.video as video_mod
    from core.inference.diffusion_auto_policy import _QUANT_STEADY_FACTOR
    from core.inference.video_families import detect_video_family

    scale = _allow_te_prequant(monkeypatch)
    transformer_gb, text_encoder_gb, vae_gb = detect_video_family(
        "Lightricks/LTX-2"
    ).bf16_components_gb
    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        video_mod, "select_transformer_quant_scheme", lambda target, mode, family = None: "int8"
    )
    monkeypatch.setattr(video_mod, "quantize_transformer", lambda *a, **k: None)
    # Force the first plan to offload so the re-plan branch runs.
    import dataclasses

    real = video_mod.plan_diffusion_memory
    calls = []

    def _spy(**kwargs):
        calls.append(kwargs)
        return dataclasses.replace(real(**kwargs), offload_policy = "model")

    monkeypatch.setattr(video_mod, "plan_diffusion_memory", _spy)
    placements = _stub_apply_memory_plan(monkeypatch, video_mod)

    VideoBackend().load_pipeline(
        "Lightricks/LTX-2",
        model_kind = "pipeline",
        transformer_quant = "int8",
        text_encoder_quant = "fp8",
    )
    _assert_placement_follows_the_target(placements, video_mod)
    assert len(calls) == 2, "the dense-quant re-plan did not run"
    assert calls[1]["model_dense_mib"] == int(
        (transformer_gb * _QUANT_STEADY_FACTOR["int8"] + text_encoder_gb * scale + vae_gb)
        * _MIB_PER_GB
    )


# ── MiniMax-H3: the denoiser partition a ref2va load actually needs ──────────


_H3_TWO_PARTITION_SIBLINGS = [
    _sibling("model_index.json", 1),
    _sibling("modular_model_index.json", 1),
    _sibling("scheduler/scheduler_config.json", 1),
    _sibling("audio_scheduler/scheduler_config.json", 1),
    _sibling("vae/diffusion_pytorch_model.safetensors", 10),
    _sibling("audio_vae/diffusion_pytorch_model.safetensors", 1),
    _sibling("text_encoder/model-00001-of-00002.safetensors", 66),
    _sibling("tokenizer/tokenizer.json", 1),
    _sibling("processor/preprocessor_config.json", 1),
    _sibling("transformer/config.json", 1),
    _sibling("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 66),
    _sibling("transformer_ref/config.json", 1),
    _sibling("transformer_ref/diffusion_pytorch_model-00001-of-00002.safetensors", 66),
    _sibling("FL2VA/single_file.safetensors", 144),
    _sibling("Ref2VA/single_file.safetensors", 144),
]


def test_a_ref2va_pick_stages_the_reference_denoiser_and_only_that_one():
    """MiniMaxAI/MiniMax-H3 ships two 66.28 GB denoisers, and the modular load builds exactly
    one: diffusers' ref2va denoise step is constructed with transformer_name="transformer_ref".
    Staging only transformer/ therefore downloaded the keyframe partition for a References load
    and left the one it opens to be fetched inline, outside the download manager."""
    info = types.SimpleNamespace(siblings = _H3_TWO_PARTITION_SIBLINGS)

    ref = [n for n, _ in VideoBackend._base_download_files(info, "pipeline", h3_task = "ref2va")]
    assert any(n.startswith("transformer_ref/") for n in ref)
    assert not any(n.startswith("transformer/") for n in ref)

    # The default (and an explicit keyframe task) keeps the fl2va partition and drops the other.
    for task in (None, "fl2va"):
        keyframes = [
            n for n, _ in VideoBackend._base_download_files(info, "pipeline", h3_task = task)
        ]
        assert any(n.startswith("transformer/") for n in keyframes)
        assert not any(n.startswith("transformer_ref/") for n in keyframes)

    # Neither partition doubles the stage, and the packaged single-file dirs stay excluded.
    assert not any(n.startswith(("FL2VA/", "Ref2VA/")) for n in ref)


def test_download_plan_keys_the_h3_denoiser_on_the_requested_task(monkeypatch):
    # The plan is what the Hub download manager stages, so the task has to reach it: without it
    # a References pick paid for 66.28 GB of keyframe denoiser it never opens.
    _cuda_bf16_target(monkeypatch)
    _plan_api(monkeypatch, {"MiniMaxAI/MiniMax-H3": _H3_TWO_PARTITION_SIBLINGS})

    def files_for(task):
        plan = VideoBackend().download_plan(
            "MiniMaxAI/MiniMax-H3",
            family_override = "minimax-h3",
            model_kind = "pipeline",
            h3_task = task,
        )
        by_repo = {entry["repo_id"]: entry for entry in plan["entries"]}
        return by_repo["MiniMaxAI/MiniMax-H3"]["files"]

    assert any(n.startswith("transformer_ref/") for n in files_for("ref2va"))
    assert not any(n.startswith("transformer/") for n in files_for("ref2va"))
    assert any(n.startswith("transformer/") for n in files_for(None))


def test_h3_records_the_guidance_that_actually_ran(monkeypatch):
    """MiniMax-H3 declares supports_cfg=False: the diffusers branch forwards no CFG kwarg and
    the native branch pins --cfg-scale 1.0, so a requested guidance reaches no sampler. Keeping
    the request in the result labelled the clip and its gallery sidecar with a number that did
    nothing, and two clips run at 1.0 read back as different recipes."""
    pytest.importorskip("PIL.Image")
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    result = backend.generate(prompt = "a fox runs through snow", width = 960, height = 544, guidance = 7.5)

    assert result["guidance"] == backend._state.family.default_guidance == 1.0
    # And that default is exactly what the engine was told to run.
    assert calls[-1]["params"].cfg_scale == 1.0


def test_h3_guidance_normalises_to_its_own_default_not_a_neighbours(monkeypatch):
    """The normalisation above has to use the FAMILY default, not the identifier-derived one.

    ``default_video_generation_params`` matches on the repo id or path, so a local H3 file under
    a folder named after another family picks up that family's guidance. Recording it writes back
    exactly the inaccurate recipe this normalisation exists to prevent, and with a number no H3
    sampler can have produced.
    """
    import dataclasses

    pytest.importorskip("PIL.Image")
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)
    local = "/models/wan/minimax_h3_fl2va-Q4_K_M.gguf"
    backend._state = dataclasses.replace(backend._state, repo_id = local)

    # Precondition: the identifier really does resolve to a different family's guidance, so the
    # test cannot pass by that lookup happening to agree with H3.
    from core.inference.video import default_video_generation_params

    fam = backend._state.family
    _steps, derived = default_video_generation_params(
        local, fallback = (fam.default_steps, fam.default_guidance)
    )
    assert derived != fam.default_guidance

    result = backend.generate(prompt = "a fox runs through snow", width = 960, height = 544)
    assert result["guidance"] == fam.default_guidance == 1.0
    assert calls[-1]["params"].cfg_scale == 1.0


def test_h3_records_no_negative_prompt_because_neither_engine_takes_one(monkeypatch):
    """The same rule as the guidance above, for the other half of the unconditional branch. A
    negative prompt IS the unconditional branch, so a guidance-distilled family consumes none:
    the diffusers call adds the kwarg only when the pipeline signature has it (H3's modular
    workflow does not) and ``SdCppVideoGenParams`` carries no field for one at all. Persisting the
    caller's string left the gallery sidecar and its restored recipe claiming conditioning that
    never reached a sampler."""
    pytest.importorskip("PIL.Image")
    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls)

    result = backend.generate(
        prompt = "a fox runs through snow",
        negative_prompt = "blurry, watermark",
        width = 960,
        height = 544,
    )

    assert result["negative_prompt"] is None
    assert not hasattr(calls[-1]["params"], "negative_prompt")


def test_the_video_sidecar_records_the_negative_prompt_that_ran(monkeypatch, tmp_path):
    """The recipe the gallery reads back comes from the RESULT, like guidance beside it, not from
    the request the worker was handed: normalising inside generate() alone would have left
    _run_generate persisting the caller's original string."""
    from core.inference import video as video_mod
    from core.inference import video_gallery

    saved: list = []
    monkeypatch.setattr(
        video_gallery, "save", lambda data, meta: (saved.append(meta), {"id": "v1"})[1]
    )

    backend = VideoBackend()
    result = {
        "mp4_bytes": b"MP4",
        "seed": 7,
        "repo_id": "unsloth/MiniMax-H3-GGUF",
        "width": 960,
        "height": 544,
        "num_frames": 124,
        "fps": 24,
        "duration_s": 5.0,
        "has_audio": True,
        "conditioning": "t2va",
        "steps": 30,
        "guidance": 1.0,
        "negative_prompt": None,
    }
    monkeypatch.setattr(backend, "generate", lambda **kw: result)
    monkeypatch.setattr(backend, "_finish_generate_job", lambda **kw: None)

    backend._run_generate(
        cancel_event = threading.Event(),
        prompt = "a fox runs through snow",
        negative_prompt = "blurry, watermark",
    )

    assert saved and saved[-1]["negative_prompt"] is None
    assert saved[-1]["prompt"] == "a fox runs through snow"


def test_a_managed_h3_native_run_holds_the_install_off(monkeypatch, tmp_path):
    """An H3 native run is an sd-cli run out of the managed tree, so it takes the same reader claim
    the one-shot image path takes. Without it an image-engine request sees no readers, starts an
    install, and the sweep unlinks the CLI this runtime resolved at load."""
    pytest.importorskip("PIL.Image")
    import core.inference.sd_cpp_backend as bk

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    managed = root / "sd-bin" / "sd-cli"
    managed.write_bytes(b"managed")

    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls, binary = managed)

    held: list = []
    inner = backend._state.pipe.engine.generate_video

    def _watching(files, params, **kwargs):
        held.append(bk._tree_readers)
        # An install decided right now must stand down rather than replace what is executing.
        with bk._tree_claimed_for_install() as claimed:
            held.append(claimed)
        return inner(files, params, **kwargs)

    backend._state.pipe.engine.generate_video = _watching
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)
    backend.generate(prompt = "p", width = 960, height = 544)
    assert held == [1, False]
    # And the claim is released afterwards, so the next install is not blocked forever.
    assert bk._tree_readers == 0


def test_an_unmanaged_h3_native_run_takes_no_claim(monkeypatch, tmp_path):
    """A user-supplied sd-cli is one the installer cannot replace, so it must not be blocked
    behind an unrelated managed install."""
    pytest.importorskip("PIL.Image")
    import core.inference.sd_cpp_backend as bk

    outside = tmp_path / "mine" / "sd-cli"
    outside.parent.mkdir(parents = True)
    outside.write_bytes(b"my own build")

    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls, binary = outside)

    held: list = []
    inner = backend._state.pipe.engine.generate_video

    def _watching(files, params, **kwargs):
        held.append(bk._tree_readers)
        return inner(files, params, **kwargs)

    backend._state.pipe.engine.generate_video = _watching
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)
    backend.generate(prompt = "p", width = 960, height = 544)
    assert held == [0]


def test_an_in_place_binary_swap_stops_the_h3_run(monkeypatch, tmp_path):
    """Existence is not identity, and the identity has to be the one recorded when
    ensure_h3_sd_cpp_binary vetted the file. An install that lands at the SAME path leaves a build
    that is not the one this runtime was checked against: a different accelerator, or one predating
    the H3 options, which aborts partway through a render nobody wants to repeat."""
    pytest.importorskip("PIL.Image")
    import contextlib
    import os

    import core.inference.sd_cpp_backend as bk

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    managed = root / "sd-bin" / "sd-cli"
    managed.write_bytes(b"the build the load checked")

    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls, binary = managed)
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)

    # Unchanged since the load: the run goes ahead.
    backend.generate(prompt = "p", width = 960, height = 544)
    assert len(calls) == 1

    # The install landed BEFORE this generation, not during it. Two reads taken now would agree
    # with each other, which is exactly why the comparison is against the load-time value.
    managed.write_bytes(b"some other accelerator's build")
    os.utime(managed, (1, 1))
    with pytest.raises(RuntimeError, match = "Reload the model"):
        backend.generate(prompt = "p", width = 960, height = 544)
    assert len(calls) == 1, "the run must not reach a build the load never checked"

    # A replacement that lands DURING the wait is caught the same way.
    @contextlib.contextmanager
    def _install_lands_during_the_wait(
        _binary,
        _cancel = None,
        _msg = None,
    ):
        managed.write_bytes(b"yet another build")
        os.utime(managed, (2, 2))
        yield

    backend = _h3_native_backend(monkeypatch, calls, binary = managed)
    monkeypatch.setattr(bk, "_tree_reader", _install_lands_during_the_wait)
    with pytest.raises(RuntimeError, match = "Reload the model"):
        backend.generate(prompt = "p", width = 960, height = 544)
    assert len(calls) == 1


def test_a_cancelled_h3_install_wait_reads_as_a_cancellation(monkeypatch, tmp_path):
    """The reader is shared with the image path, whose sentinel _run_generate does not recognise.
    Left untranslated, an ordinary H3 cancellation surfaces as "Video generation failed"."""
    pytest.importorskip("PIL.Image")
    import threading

    import core.inference.sd_cpp_backend as bk
    from core.inference.video_families import VIDEO_CANCELLED_MSG

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    managed = root / "sd-bin" / "sd-cli"
    managed.write_bytes(b"managed")

    calls: list = []
    backend = _h3_native_backend(monkeypatch, calls, binary = managed)
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)
    monkeypatch.setattr(bk, "_tree_installing", True)
    monkeypatch.setattr(bk, "_TREE_WAIT_TICK_S", 0.02)

    # Cancel shortly after the generation parks in the install wait.
    threading.Timer(0.1, lambda: backend.cancel_generate()).start()
    with pytest.raises(RuntimeError) as exc:
        backend.generate(prompt = "p", width = 960, height = 544)
    assert str(exc.value) == VIDEO_CANCELLED_MSG
    assert calls == []


# ── the refusal reads the plan the load will actually take ────────────────────
# Two ways the hard unified-memory refusal added in PR #8213 can read a number the load never
# occupies, both of which turn it from a guard into a false rejection.


def _fp32_promoted_cuda_target(monkeypatch):
    """A CUDA target whose fp16 promotes to fp32, i.e. dtype_scale == 2 in the video planner."""
    import torch

    import core.inference.video as video_mod
    from core.inference.diffusion_device import DiffusionDeviceTarget

    target = DiffusionDeviceTarget(
        device = "cuda",
        dtype = torch.float16,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = True,
    )
    monkeypatch.setattr(video_mod, "resolve_diffusion_device_target", lambda: target)
    return target


def test_the_fp32_promotion_does_not_double_an_already_fp32_wan_vae(fake_runtime, monkeypatch):
    # A device without bf16 promotes the whole plan by 2. Wan's VAE term is recorded at fp32
    # ALREADY (the family comment says so) and assembly pins that VAE to fp32 whatever the
    # promotion does, so doubling it counts 2.8 GB twice on TI2V-5B. Against a hard refusal that
    # is a load rejected over bytes it never allocates.
    from core.inference.video_families import detect_video_family

    _fp32_promoted_cuda_target(monkeypatch)
    transformer_gb, te_gb, vae_gb = detect_video_family(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    ).bf16_components_gb

    calls = _capture_plan(monkeypatch)
    VideoBackend().load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    # bf16 terms doubled, the fp32 VAE term left exactly where the table put it.
    assert calls[0]["model_dense_mib"] == int(
        ((transformer_gb + te_gb) * 2.0 + vae_gb) * _MIB_PER_GB
    )


def test_the_fp32_promotion_still_doubles_a_bf16_vae(fake_runtime, monkeypatch):
    # LTX-2 does not force fp32, so its VAE term IS a bf16 one and must keep scaling with the rest.
    from core.inference.video_families import detect_video_family

    _fp32_promoted_cuda_target(monkeypatch)
    components = detect_video_family("Lightricks/LTX-2").bf16_components_gb

    calls = _capture_plan(monkeypatch)
    VideoBackend().load_pipeline("Lightricks/LTX-2", model_kind = "pipeline")
    assert calls[0]["model_dense_mib"] == int(sum(components) * 2.0 * _MIB_PER_GB)


def test_unified_memory_refuses_on_the_dense_peak_even_when_a_quant_is_requested(
    fake_runtime, monkeypatch
):
    """The quant re-plan prices the DiT's STEADY size, but the video path has no pre-quantised
    artifact: the transformer is always built dense and quantize_transformer rewrites it in place,
    so the build PEAK is the bf16 figure. On unified memory the peak is what the OS kills for, so
    the refusal must keep reading the dense plan -- accepting the steady size here would wave
    through exactly the load this guard exists to stop."""
    import torch

    import core.inference.video as video_mod
    from core.inference.diffusion_device import DiffusionDeviceTarget
    from core.inference.diffusion_memory import DeviceMemory

    target = DiffusionDeviceTarget(
        device = "cuda",
        dtype = torch.bfloat16,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = True,
    )
    monkeypatch.setattr(video_mod, "resolve_diffusion_device_target", lambda: target)
    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda t: True)
    monkeypatch.setattr(
        video_mod, "select_transformer_quant_scheme", lambda t, q, family = None: "fp8"
    )
    # An integrated CUDA device: 48 GiB shared, so LTX-2's ~65 GB of dense weights cannot fit even
    # though the fp8 steady size would.
    total = 48 * 1024
    monkeypatch.setattr(
        video_mod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "unified_memory", int(total * 0.80), total),
    )

    with pytest.raises(RuntimeError) as excinfo:
        VideoBackend().load_pipeline(
            "Lightricks/LTX-2", model_kind = "pipeline", transformer_quant = "auto"
        )
    assert "unified memory" in str(excinfo.value)


def test_unified_memory_refuses_the_h3_modular_load_before_load_components(
    fake_runtime, monkeypatch
):
    """MiniMax-H3 returns into the modular workflow ABOVE load_pipeline's refusal, so the one
    family the matrix says must be declined on every Mac it models was the only one that never
    reached the check. load_components builds every component dense and the ComponentsManager's
    CPU offload frees nothing on unified memory, so 144.2 GB of components is an OS kill with no
    torch OOM to catch."""
    import core.inference.video as video_mod
    from core.inference.diffusion_device import DiffusionDeviceTarget

    torch = sys.modules["torch"]
    diffusers = sys.modules["diffusers"]
    diffusers.ComponentsManager = _FakeComponentsManager
    diffusers.ModularPipeline = _FakeModularPipeline
    _FakeModularPipeline.instance = None

    target = DiffusionDeviceTarget(
        device = "mps",
        dtype = torch.bfloat16,
        backend = "mps",
        vendor = "apple",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )
    monkeypatch.setattr(video_mod, "resolve_diffusion_device_target", lambda: target)
    monkeypatch.setattr(video_mod, "settled_snapshot_device_memory", _unified_snapshot(128))

    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    backend = VideoBackend()
    with pytest.raises(RuntimeError) as excinfo:
        backend._load_h3_modular_pipeline(
            diffusers = diffusers,
            torch = torch,
            fam = fam,
            repo_id = "MiniMaxAI/MiniMax-H3",
            base = fam.base_repo,
            kind = "pipeline",
            dtype = torch.bfloat16,
            device = "mps",
            hf_token = None,
            memory_mode = None,
            target = target,
        )
    message = str(excinfo.value)
    assert "minimax-h3" in message and "unified memory" in message
    # Refused BEFORE any component was built.
    assert _FakeModularPipeline.instance is not None
    assert _FakeModularPipeline.instance.load_kwargs is None
    assert backend.status()["loaded"] is False


def test_the_h3_modular_refusal_prices_a_seeded_prequant_denoiser(fake_runtime, monkeypatch):
    """A hosted pre-quantized checkpoint replaces the dense 66.3 GB denoiser, so the refusal must
    size that instead -- refusing it on the dense figure would reject a load that never builds
    those weights. It is still the whole component set: the encoder and the VAEs load dense."""
    import core.inference.video as video_mod
    from core.inference.video_families import detect_video_family

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    transformer_gb, te_gb, vae_gb = fam.bf16_components_gb
    seen: list[int] = []

    def _capture(*, model_dense_mib = None, **kwargs):
        seen.append(int(model_dense_mib or 0))
        return types.SimpleNamespace(offload_policy = "none", estimates = {}, device_memory = None)

    monkeypatch.setattr(video_mod, "plan_diffusion_memory", _capture)
    monkeypatch.setattr(video_mod, "settled_snapshot_device_memory", lambda t: None)
    monkeypatch.setattr(video_mod, "raise_on_unified_memory_shortfall", lambda *a, **k: None)

    # The MEASURED hosted size, not the generic 0.55 steady factor: H3's checkpoints are
    # quantized AND structurally pruned, so the factor reads 36.5 GB against a real ~20.3 GB and
    # the 16 GB gap refuses a supported prequant load that fits on a 128 GB Mac.
    assert fam.prequant_resident_gb == 20.3
    for scheme, expected_transformer in (
        (None, transformer_gb),
        ("fp8", fam.prequant_resident_gb),
        ("int8", fam.prequant_resident_gb),
        # An unknown scheme still takes the family's hosted size when it publishes one; the
        # generic factor is only the fallback for a family that does not.
        ("bogus", fam.prequant_resident_gb),
    ):
        VideoBackend._raise_on_modular_unified_shortfall(
            fam,
            target = None,
            dtype = sys.modules["torch"].bfloat16,
            device = "mps",
            memory_mode = None,
            scheme = scheme,
        )
        assert seen[-1] == int(
            (expected_transformer + te_gb + vae_gb) * (1000.0**3 / (1024.0 * 1024.0))
        ), scheme
    # The whole point: on a unified-memory host big enough for the ~20.3 GB checkpoint but not
    # for the 36.5 GB the generic factor invents, the fp8 pick must NOT be refused. 160 GiB at
    # 80% free is inside that window (the dense set is refused there either way).
    from core.inference.diffusion_memory import unified_memory_shortfall_message

    from core.inference.diffusion_memory import (
        DeviceMemory,
        estimate_video_runtime_mib,
        plan_diffusion_memory,
    )

    mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)
    total = 160 * 1024
    memory = DeviceMemory("mps", "mps", "unified_memory", int(total * 0.80), total)
    device = type("T", (), {"supports_model_cpu_offload": True, "device": "mps"})()
    headroom = estimate_video_runtime_mib(
        width = fam.resolution_presets[0][0],
        height = fam.resolution_presets[0][1],
        num_frames = fam.default_num_frames,
    )
    for gb, refused in (
        (fam.prequant_resident_gb + te_gb + vae_gb, False),
        # What the generic 0.55 factor would have budgeted: refused, for 16 GB that do not exist.
        (transformer_gb * 0.55 + te_gb + vae_gb, True),
        (sum(fam.bf16_components_gb), True),
    ):
        plan = plan_diffusion_memory(
            target = device,
            device_memory = memory,
            model_dense_mib = int(gb * mib_per_gb),
            runtime_headroom_mib = headroom,
        )
        assert (unified_memory_shortfall_message(plan, family = fam.name) is not None) is refused, gb


def test_the_h3_modular_refusal_reruns_when_the_prequant_checkpoint_does_not_land(
    fake_runtime, monkeypatch
):
    """The hosted-denoiser load is best-effort by contract: a missing, corrupt, stale or
    base-mismatched checkpoint drops to the released bfloat16 components. Sizing the load as
    quantized and then building dense with no second check is the OS kill this guard exists to
    prevent -- on a host whose budget fits the ~20.3 GB checkpoint but not the 66.3 GB dense one."""
    import core.inference.video as video_mod
    from core.inference.diffusion_device import DiffusionDeviceTarget

    torch = sys.modules["torch"]
    diffusers = sys.modules["diffusers"]
    diffusers.ComponentsManager = _FakeComponentsManager
    diffusers.ModularPipeline = _FakeModularPipeline
    diffusers.MiniMaxH3Transformer3DModel = _FakeTransformer
    _FakeModularPipeline.instance = None

    target = DiffusionDeviceTarget(
        device = "mps",
        dtype = torch.bfloat16,
        backend = "mps",
        vendor = "apple",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = False,
        supports_pinned_transfer = False,
    )
    monkeypatch.setattr(video_mod, "resolve_diffusion_device_target", lambda: target)
    # 160 GiB: the prequant pick fits, the dense component set does not.
    monkeypatch.setattr(video_mod, "settled_snapshot_device_memory", _unified_snapshot(160))

    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    sized: list = []
    real = VideoBackend._raise_on_modular_unified_shortfall

    def _spy(family, **kwargs):
        sized.append(kwargs["scheme"])
        return real(family, **kwargs)

    monkeypatch.setattr(VideoBackend, "_raise_on_modular_unified_shortfall", staticmethod(_spy))
    # The checkpoint resolves but does not load: exactly the best-effort None contract.
    import core.inference.diffusion_prequant as prequant_mod

    monkeypatch.setattr(
        prequant_mod,
        "resolve_prequant_source",
        lambda fam, scheme, base_repo = None, task = None: types.SimpleNamespace(
            location = "unsloth/H3-FP8"
        ),
    )
    monkeypatch.setattr(prequant_mod, "load_prequantized_transformer", lambda *a, **k: None)

    with pytest.raises(RuntimeError) as excinfo:
        VideoBackend()._load_h3_modular_pipeline(
            diffusers = diffusers,
            torch = torch,
            fam = fam,
            repo_id = "MiniMaxAI/MiniMax-H3",
            base = fam.base_repo,
            kind = "pipeline",
            dtype = torch.bfloat16,
            device = "mps",
            hf_token = None,
            memory_mode = None,
            transformer_quant = "fp8",
            target = target,
        )
    assert "minimax-h3" in str(excinfo.value) and "unified memory" in str(excinfo.value)
    # Sized twice: once for the pick that was requested, once for what actually landed.
    assert sized == ["fp8", None]
    # ... and nothing was built.
    assert _FakeModularPipeline.instance.load_kwargs is None


def test_generation_in_flight_tracks_a_background_job(fake_runtime, tmp_path, monkeypatch):
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)

    rendering = threading.Event()
    release = threading.Event()
    inside = {}

    def _block(
        self,
        *,
        cancel_event = None,
        **gen_kwargs,
    ):
        # Hold the worker open while liveness is checked.
        inside["in_flight"] = video_mod.generation_in_flight()
        rendering.set()
        release.wait(10)
        raise ValueError("stopped after the probe")

    monkeypatch.setattr(VideoBackend, "generate", _block)

    assert video_mod.generation_in_flight() is False
    backend.begin_generate(prompt = "a clip")
    assert rendering.wait(10), "the generate worker never started"
    assert (
        video_mod.generation_in_flight() is True
    ), "liveness cannot tell this backend from a dead one while it renders a clip"
    assert inside["in_flight"] is True

    release.set()
    deadline = time.monotonic() + 10
    while backend.generate_progress()["active"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert video_mod.generation_in_flight() is False, (
        "the marker stayed lit after the job ended, so the watchdog would hold the widened "
        "budget against a backend that really did hang"
    )


def test_generation_in_flight_never_builds_a_backend(fake_runtime, monkeypatch):
    import core.inference.video as video_mod

    monkeypatch.setattr(video_mod, "_backend", None)
    monkeypatch.setattr(
        video_mod,
        "VideoBackend",
        lambda *a, **k: pytest.fail("liveness constructed a video backend"),
    )
    assert video_mod.generation_in_flight() is False


def _video_result(tag: str) -> dict:
    """The result shape _run_generate_body persists; tag identifies which job produced it."""
    return {
        "mp4_bytes": tag.encode(),
        "negative_prompt": None,
        "width": 64,
        "height": 64,
        "num_frames": 9,
        "fps": 8,
        "duration_s": 1.0,
        "steps": 2,
        "guidance": 1.0,
        "seed": 1,
        "has_audio": False,
        "conditioning": "t2v",
        "flow_shift": None,
        "audio_flow_shift": None,
        "repo_id": "test/model",
    }


def _until(predicate, timeout = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_a_worker_that_never_started_does_not_hold_the_job_open(
    fake_runtime, tmp_path, monkeypatch
):
    """begin_generate reserves the slot before it spawns, so a spawn that raises leaves a
    reservation no worker will ever release: generate stays refused for the rest of the
    session, and liveness reports this backend as rendering to a watchdog that answers a
    busy backend by waiting longer, not by restarting it."""
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)

    real_start = threading.Thread.start
    monkeypatch.setattr(
        threading.Thread,
        "start",
        lambda self: (_ for _ in ()).throw(RuntimeError("can't start new thread")),
    )

    with pytest.raises(RuntimeError, match = "can't start new thread"):
        backend.begin_generate(prompt = "a clip")

    assert video_mod.generation_in_flight() is False
    assert backend._active_generate_cancel is None
    progress = backend.generate_progress()
    assert progress["active"] is False, "progress stayed queued for a job that never ran"

    # ... and the rejected spawn did not poison the next request.
    monkeypatch.setattr(threading.Thread, "start", real_start)
    monkeypatch.setattr(backend, "generate", lambda **kw: _video_result("second"))
    from core.inference import video_gallery

    monkeypatch.setattr(video_gallery, "save", lambda data, meta: {"id": "second"})

    backend.begin_generate(prompt = "second")
    assert _until(lambda: not video_mod.generation_in_flight())
    assert backend.generate_progress()["phase"] == "completed"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_worker_killed_outright_does_not_hold_the_job_open(fake_runtime, tmp_path, monkeypatch):
    """The worker names ValueError, RuntimeError and Exception. SystemExit and
    KeyboardInterrupt are none of those, so before the finally they unwound past every
    terminal path and left the marker lit for good; unload() does not clear it either."""
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)

    entered = threading.Event()

    def _die(
        self,
        *,
        cancel_event = None,
        **gen_kwargs,
    ):
        entered.set()
        raise SystemExit("worker killed")

    monkeypatch.setattr(VideoBackend, "generate", _die)

    backend.begin_generate(prompt = "a clip")
    assert entered.wait(10), "the generate worker never started"

    assert _until(lambda: not video_mod.generation_in_flight()), (
        "the marker outlived the worker, so liveness reports this backend as rendering "
        "for the rest of the process"
    )
    assert backend.generate_progress()["active"] is False


def test_a_finished_job_cannot_finalise_the_one_that_replaced_it(
    fake_runtime, tmp_path, monkeypatch
):
    """The backstop runs after the body has already published, and by then the next job may
    own the slot. Keyed on the busy flag it would finalise that successor instead: clear a
    marker that is still rendering, overwrite its progress with a generic failure, and let a
    third job past the busy guard. Keyed on the job's own token it is a no-op."""
    import core.inference.video as video_mod
    from core.inference import video_gallery
    from core.inference.video_families import VIDEO_GENERATION_BUSY_MSG

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)
    monkeypatch.setattr(video_gallery, "save", lambda data, meta: {"id": data.decode()})

    second_running = threading.Event()
    release_second = threading.Event()
    calls = []

    def _generate(*, cancel_event, **gen_kwargs):
        calls.append(1)
        if len(calls) == 1:
            return _video_result("one")
        second_running.set()
        assert release_second.wait(10)
        return _video_result("two")

    monkeypatch.setattr(backend, "generate", _generate)

    first_published = threading.Event()
    release_first = threading.Event()
    backstop_ran = threading.Event()
    real_finish = backend._finish_generate_job

    def _gated_finish(**kwargs):
        # Hold the first job between publishing its outcome and reaching its backstop,
        # which is exactly the window the second job reserves in.
        if (kwargs.get("video") or {}).get("id") == "one":
            real_finish(**kwargs)
            first_published.set()
            assert release_first.wait(10)
            return
        real_finish(**kwargs)
        if kwargs.get("error") == "Video generation failed." and first_published.is_set():
            backstop_ran.set()

    monkeypatch.setattr(backend, "_finish_generate_job", _gated_finish)

    backend.begin_generate(prompt = "one")
    assert first_published.wait(10)

    backend.begin_generate(prompt = "two")
    assert second_running.wait(10)
    assert video_mod.generation_in_flight() is True

    release_first.set()
    assert backstop_ran.wait(10), "the first job's backstop never ran"

    assert video_mod.generation_in_flight() is True, (
        "the finished job cleared the running job's marker; liveness now calls a rendering "
        "backend idle, which is the failure this marker exists to prevent"
    )
    assert backend.generate_progress()["active"] is True
    with pytest.raises(RuntimeError, match = VIDEO_GENERATION_BUSY_MSG):
        backend.begin_generate(prompt = "three")

    release_second.set()
    assert _until(lambda: not video_mod.generation_in_flight())
    progress = backend.generate_progress()
    assert progress["phase"] == "completed"
    assert progress["video"]["id"] == "two", "the running job lost its own outcome"


def test_the_backstop_leaves_a_cancelled_job_reported_as_cancelled(
    fake_runtime, tmp_path, monkeypatch
):
    """Cancellation is a RuntimeError carrying a sentinel the route maps to its own status.
    A backstop that overwrote it would turn every cancel into a generic failure."""
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)

    entered = threading.Event()

    def _wait_for_cancel(*, cancel_event, **gen_kwargs):
        entered.set()
        assert cancel_event.wait(10)
        raise RuntimeError(VIDEO_CANCELLED_MSG)

    monkeypatch.setattr(backend, "generate", _wait_for_cancel)

    backend.begin_generate(prompt = "cancel me")
    assert entered.wait(10)
    backend.cancel_generate()

    assert _until(lambda: not video_mod.generation_in_flight())
    progress = backend.generate_progress()
    assert progress["phase"] == "failed"
    assert progress["error"] == VIDEO_CANCELLED_MSG

    # The backstop runs after the body returns; give it room to prove it changed nothing.
    time.sleep(0.1)
    assert backend.generate_progress()["error"] == VIDEO_CANCELLED_MSG


def test_an_interrupted_spawn_leaves_a_live_worker_its_reservation(
    fake_runtime, tmp_path, monkeypatch
):
    """Thread.start() creates the OS thread and then waits on it, so a signal delivered in
    that wait unwinds with the worker already running. Rolling the reservation back there
    would retire a live render's token: liveness would call it idle, cancel and unload could
    not reach it, and the next request could reserve the slot underneath it."""
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)

    rendering = threading.Event()
    release = threading.Event()

    def _hold(*, cancel_event = None, **gen_kwargs):
        rendering.set()
        assert release.wait(10)
        return _video_result("held")

    monkeypatch.setattr(backend, "generate", _hold)
    from core.inference import video_gallery

    monkeypatch.setattr(video_gallery, "save", lambda data, meta: {"id": "held"})

    real_start = threading.Thread.start

    def _start_then_interrupt(self):
        real_start(self)
        raise KeyboardInterrupt("signal delivered while waiting on the child")

    monkeypatch.setattr(threading.Thread, "start", _start_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        backend.begin_generate(prompt = "a clip")
    assert rendering.wait(10), "the worker never started"

    assert video_mod.generation_in_flight() is True, (
        "the interrupted spawn retired a running render's reservation, so liveness reports "
        "this backend as idle while it renders"
    )
    assert backend._active_generate_cancel is not None, "the running job lost its cancel handle"
    monkeypatch.setattr(threading.Thread, "start", real_start)
    with pytest.raises(RuntimeError, match = "already in progress"):
        backend.begin_generate(prompt = "second")

    release.set()
    assert _until(lambda: not video_mod.generation_in_flight())
    assert backend.generate_progress()["phase"] == "completed"


def test_a_direct_worker_call_keeps_the_outcome_it_recorded(fake_runtime, tmp_path, monkeypatch):
    """_run_generate is callable without a reservation. Such a caller finalises with the same
    "unreserved" token the backstop would carry, so a backstop that fired here would match a
    second time and replace the recorded outcome with the generic failure."""
    import core.inference.video as video_mod
    from core.inference import video_gallery

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)
    monkeypatch.setattr(video_gallery, "save", lambda data, meta: {"id": "direct"})
    monkeypatch.setattr(backend, "generate", lambda **kw: _video_result("direct"))

    backend._run_generate(cancel_event = threading.Event(), prompt = "a clip")

    progress = backend.generate_progress()
    assert (
        progress["phase"] == "completed"
    ), f"a direct call recorded {progress['phase']!r}; the backstop overwrote its result"
    assert progress["video"]["id"] == "direct"


def test_a_direct_worker_call_keeps_its_cancellation(fake_runtime, tmp_path, monkeypatch):
    """Same path, the outcome that matters most: a cancellation carries a sentinel the route
    maps to its own status, and turning it into a generic failure loses that."""
    import core.inference.video as video_mod

    backend = VideoBackend()
    _load_gguf(backend, tmp_path)
    monkeypatch.setattr(video_mod, "_backend", backend)

    def _cancelled(**kwargs):
        raise RuntimeError(VIDEO_CANCELLED_MSG)

    monkeypatch.setattr(backend, "generate", _cancelled)
    backend._run_generate(cancel_event = threading.Event(), prompt = "a clip")

    progress = backend.generate_progress()
    assert progress["phase"] == "failed"
    assert (
        progress["error"] == VIDEO_CANCELLED_MSG
    ), f"a direct call reported {progress['error']!r} instead of the cancellation sentinel"
