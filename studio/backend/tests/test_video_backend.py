# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VideoBackend lifecycle on a faked torch/diffusers runtime (CPU-only, offline).
Mirrors test_diffusion_backend's fake_runtime pattern: explicit fake signatures so
the signature-gated kwargs actually exercise, sys.modules stubs so no real ML
stack loads."""

import contextlib
import sys
import types

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

    # Explicit signature so generate()'s signature-gated kwargs (negative_prompt,
    # frame_rate, callback) actually engage; **kwargs would defeat the gates.
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


# ── Wan2.2 fakes: a per-DiT trackable transformer so the dual-DiT optimisation tests can assert
# speed / cache / attention engaged on BOTH experts, plus two pipeline fakes -- single-DiT
# (TI2V-5B) and dual-DiT MoE (A14B). The MoE __call__ carries guidance_scale_2 so the cfg2
# signature-gate exercises; the single-DiT __call__ omits it so the gate proves it is NOT
# threaded there.


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
        # Real Wan / HV15 / LTX pipelines open a cache_context around the denoise loop; the
        # First-Block-Cache hook needs it, so the fake transformer provides it too.
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


# ── HunyuanVideo-1.5 fakes: __call__ has NO guidance kwarg and NO callback_on_step_end
# (matching pipeline_hunyuan_video1_5.py in diffusers 0.39), a guider object carries the CFG
# scale, and the denoise loop drives scheduler.step -- so the guider write and the scheduler-wrap
# progress/cancel paths exercise.


class _FakeHV15Scheduler:
    def __init__(self) -> None:
        self.calls = 0
        # Test hook fired from the ORIGINAL step (i.e. inside the wrapped call),
        # letting a test cancel mid-denoise exactly as a user request would land.
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
    # MP4 encode needs real frames + PyAV; the backend contract under test is the
    # byte handoff, so stub the encoder.
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
    # model_kind single_file with a .gguf file, or gguf with a non-.gguf file, must be rejected
    # BEFORE the GPU handoff (mirrors the image loader), not fail in the wrong single-file loader
    # after the route evicted the resident model.
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
    # A local FILE is handed straight to the gguf/single_file loader: _resolve_checkpoint_path
    # returns the file itself, IGNORING gguf_filename, so the file's OWN suffix must match the
    # kind. A .gguf picked as single_file (or a .safetensors picked as gguf) slips past the
    # gguf_filename suffix checks, so reject it HERE, before the route evicts the resident GPU
    # owner and fails deep in from_single_file / the GGUF reader.
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
    # A missing Windows-shaped local pick (backslash path, or a C:/ drive path) must fail HERE,
    # not be treated as a Hub repo and fail after the route evicts the resident GPU owner. Mirrors
    # the image loader's is_absolute()/backslash path-shaped check.
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
    # A local dir resolved to a video family but missing model_index.json is not a loadable
    # diffusers pipeline; it must fail preflight BEFORE the route evicts the resident model,
    # mirroring the image loader's local-pipeline shape check.
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(str(d), family_override = "ltx-2")
    # With a model_index.json it is a valid local pipeline pick and passes preflight.
    (d / "model_index.json").write_text("{}")
    fam = backend.validate_load_request(str(d), family_override = "ltx-2")
    assert fam.name == "ltx-2"


def test_validate_rejects_local_file_picked_as_pipeline(tmp_path):
    backend = VideoBackend()
    # A local FILE (a bare .safetensors) sent as a pipeline is not a diffusers directory, so
    # from_pretrained fails deep in the background load AFTER the route evicts the resident model.
    # The preflight rejects it HERE -- gating on .exists() (not .is_dir()), mirroring the image
    # loader, so it catches files as well as directories.
    f = tmp_path / "ltx-2.safetensors"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match = "model_index.json"):
        backend.validate_load_request(str(f), model_kind = "pipeline", family_override = "ltx-2")


def test_validate_rejects_local_base_repo_without_model_index(tmp_path):
    backend = VideoBackend()
    # A local base_repo dir that is NOT a diffusers pipeline (no model_index.json) passes the
    # any-existing-path trust check, but the base loads via from_pretrained (needs model_index), so
    # reject it HERE before the route hands the GPU to VIDEO -- the pipeline-kind shape check covers
    # only repo_id, and an explicit base_repo is only meaningful for a gguf/single_file load.
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
    # A -GGUF repo with no quant filename resolves to the pipeline kind and would
    # only fail minutes later in from_pretrained, AFTER evicting the GPU owner.
    with pytest.raises(ValueError, match = "pick one of its .gguf files"):
        backend.validate_load_request("unsloth/LTX-2.3-GGUF")
    with pytest.raises(ValueError, match = "pick one of its .gguf files"):
        backend.validate_load_request("unsloth/Wan2.2-TI2V-5B-GGUF/")


def test_detect_load_family_filename_fallback():
    # Repo id alone carries the family.
    fam = _detect_load_family("Lightricks/LTX-2", None, None)
    assert fam is not None and fam.name == "ltx-2"
    # Repo id is opaque but the picked filename carries it: fall back to the
    # combined path so validate and _run_load agree on the family.
    fam = _detect_load_family("someorg/quants", "ltx-2-19b-Q4_K_M.gguf", None)
    assert fam is not None and fam.name == "ltx-2"
    # No filename and no recognisable repo id: no family.
    assert _detect_load_family("someorg/quants", None, None) is None
    # An explicit override resolves by name/alias and skips the filename fallback:
    # a bogus override stays None even when the filename would have matched.
    fam = _detect_load_family("someorg/quants", "ltx-2-19b-Q4_K_M.gguf", "ltxv")
    assert fam is not None and fam.name == "ltx-2"
    assert _detect_load_family("someorg/quants", "ltx-2-19b-Q4_K_M.gguf", "bogus") is None


def test_detect_load_family_cached_hub_arch_fallback(monkeypatch):
    # A CACHED HUB GGUF is admitted to the picker by its general.architecture (the cached-gguf
    # listing tags it text-to-video), but an opaque repo id + renamed file carry no family token,
    # so name detection misses. The local-file arch read misses too (a hub repo id is not a local
    # dir), so without a cache fallback the loader 400s a SUPPORTED checkpoint the picker offered.
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

    # A recognised-but-unsupported video arch (wan has no backend family in this build) stays None,
    # so an unsupported cached pick 400s just like the local-dir case.
    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: "/fake/cache/blobs/model.gguf"
    )
    monkeypatch.setattr(
        gguf_meta, "read_gguf_general_metadata", lambda path: {"general.architecture": "wan"}
    )
    assert _detect_load_family("someorg/opaque-quants", "model.gguf", None) is None

    # The blob lives in a NON-active cache root (legacy / default): the active probe (no cache_dir)
    # misses, but the per-root probe finds it, so a GGUF the picker offered from any root resolves.
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
    # During a background load status()["loaded"] is still False, but the target repo (+ companion
    # base) is downloading, so the delete-cached guard needs loading_repo_ids to refuse deletion
    # and avoid yanking blobs from under the in-flight download/assembly.
    from core.inference.video import _VideoLoadingState

    backend = VideoBackend()
    assert backend.loading_repo_ids() == ()  # idle: nothing to guard
    backend._loading = _VideoLoadingState(repo_id = "org/ckpt", base_repo = "Lightricks/LTX-2")
    assert set(backend.loading_repo_ids()) == {"org/ckpt", "Lightricks/LTX-2"}
    # An errored load is no longer in flight -> the files are safe to delete.
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
    # The video load must hold _generate_lock across GPU placement (apply_memory_plan) so an
    # unload / arbiter eviction -- which barriers on _generate_lock before freeing -- can't hand
    # the GPU to another backend while a multi-GB pipeline is still being moved onto it (mirrors
    # the image backend). Verify unload() blocks until placement releases the lock, and the
    # superseded load then aborts without committing.
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

    # Placement is in flight, holding _generate_lock. unload() must block on its barrier.
    unload_done = []

    def do_unload():
        backend.unload()
        unload_done.append(True)

    unload_thread = threading.Thread(target = do_unload)
    unload_thread.start()
    unload_thread.join(timeout = 0.5)
    assert not unload_done, "unload() returned while placement still held _generate_lock (the race)"

    # Release placement; unload()'s barrier then passes and its teardown runs strictly AFTER
    # the load's placement+commit -- never concurrently -- so no two pipelines are ever resident.
    release_placement.set()
    unload_thread.join(timeout = 5)
    load_thread.join(timeout = 5)
    assert unload_done, "unload() did not complete after placement released _generate_lock"
    assert not load_thread.is_alive() and not load_exc
    assert backend._state is None  # unload's teardown ran after the load, leaving nothing resident


def test_load_records_engaged_speed_optims(fake_runtime, tmp_path, monkeypatch):
    # Regression: the load tail once re-ran the already-filtered speed_optims tuple through
    # ``.items()`` as if it were still the raw applied dict, so every real-GPU load (where at least
    # channels_last engages) crashed with 'tuple' object has no attribute 'items'. Fake runtime
    # forces every optim False, so this only reproduces when one is made to engage.
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
    # At the distilled default step count the calibrated ltx_core curve is passed verbatim
    # (the DiT was trained against it; the scheduler's own 8-step spacing lands far off).
    from core.inference.video_ltx2 import LTX23_DISTILLED_SIGMAS

    assert call["sigmas"] == list(LTX23_DISTILLED_SIGMAS)


def test_generate_distilled_custom_steps_keep_scheduler_spacing(fake_runtime, tmp_path):
    # A non-default step count on the distilled DiT has no calibrated list; the scheduler's
    # spacing applies and no sigmas kwarg is injected.
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
    # The dev/base DiT uses the resolution-shifted scheduler spacing even at 8 steps: the
    # calibrated list is distilled-only.
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
    # The context manager must neutralise exactly the transforms that distort explicit
    # sigmas and put the original values back afterwards, even on error.
    from core.inference.video_ltx2 import ltx23_verbatim_sigmas

    class _Cfg(dict):
        pass

    class _Sched:
        def __init__(self):
            self.config = _Cfg(
                use_dynamic_shifting = True, shift = 1.0, shift_terminal = 0.1
            )

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
    # FBCache residuals live on the long-lived DiT(s) and survive a generation, so the next clip
    # at a new resolution would crash on stale state. generate must reset them when a cache is
    # engaged (diffusers 0.39 exposes _reset_stateful_cache on the transformer; reset_stateful_hooks
    # only on the HookRegistry) and must not touch an uncached load. transformer_2 (the Wan dual
    # expert) resets too when present.
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
    # Cache engaged -> both resident DiTs reset before the pipe call.
    backend._state = dataclasses.replace(backend._state, transformer_cache = "fbcache")
    backend.generate(prompt = "a sloth")
    assert resets == ["transformer", "transformer_2"]


def test_is_ltx23_checkpoint_gguf(monkeypatch, tmp_path):
    # diffusers maps every LTX-2 single file to the 2.0 config; a 2.3 checkpoint (9-row modulation
    # tables in the header) must be detected so the loader routes to the full 2.3 assembly. A 2.0
    # header must not, and an unreadable header falls back to the stock path (False), never raise.
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
    # The Lightricks fp8 files carry .weight_scale/.input_scale companions; a plain dtype cast
    # would corrupt them, so the loader must refuse with a pointer to the supported GGUF path.
    from core.inference import video_ltx2

    # Stub the module tree so this also runs under the CI sim, which blocks the
    # real diffusers import.
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


def test_generate_without_load_raises(fake_runtime):
    backend = VideoBackend()
    with pytest.raises(RuntimeError, match = VIDEO_NOT_LOADED_MSG):
        backend.generate(prompt = "x")


def test_generate_progress_and_cancel_idle(fake_runtime):
    backend = VideoBackend()
    # Idle shape carries the image-endpoint-compatible aliases (total_steps / fraction)
    # so one poller works against both generate-progress APIs.
    assert backend.generate_progress() == {
        "active": False,
        "total_steps": 0,
        "fraction": 0.0,
    }
    assert backend.cancel_generate() is False


def test_generate_progress_derives_total_steps_and_fraction(fake_runtime):
    # A mid-denoise poll must report fraction = step / total under BOTH field names:
    # a client polling the image API's shape against video used to read
    # total_steps=null / fraction=0 while step advanced.
    backend = VideoBackend()
    backend._gen = {"active": True, "phase": "denoise", "step": 5, "total": 20}
    gen = backend.generate_progress()
    assert gen["total"] == 20 and gen["total_steps"] == 20
    assert gen["step"] == 5 and gen["fraction"] == 0.25


def test_cache_bytes_counts_incomplete_blobs(fake_runtime, tmp_path, monkeypatch):
    # scan_cache_dir skips in-flight *.incomplete blobs, so the old counter froze at the
    # last completed blob for the whole multi-GB shard pull. The walk must count both,
    # without double-counting snapshot symlinks.
    import huggingface_hub.constants as hub_constants

    repo_dir = tmp_path / "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers"
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "aa11").write_bytes(b"x" * 1000)  # completed blob
    (blobs / "bb22.incomplete").write_bytes(b"y" * 500)  # in-flight shard
    snap = repo_dir / "snapshots" / "deadbeef"
    snap.mkdir(parents = True)
    (snap / "model_index.json").symlink_to(blobs / "aa11")  # must not double-count
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))

    backend = VideoBackend()
    assert backend._cache_bytes("Wan-AI/Wan2.2-TI2V-5B-Diffusers") == 1500
    assert backend._cache_bytes("Wan-AI/absent-repo") == 0
    assert backend._cache_bytes(None) == 0


def test_hv15_guider_and_scheduler_progress(fake_runtime):
    # HunyuanVideo-1.5: no guidance kwarg (CFG set on the guider), no step
    # callback (progress via the scheduler.step wrapper, restored afterwards).
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
    # Cancel lands during the FIRST real step; the next wrapped call must raise out
    # of the denoise loop and generate() must surface the cancelled sentinel.
    pipe.scheduler.on_step = lambda n: backend.cancel_generate() if n == 1 else None
    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        backend.generate(prompt = "a fox", steps = 4)
    assert pipe.scheduler.calls == 1
    # The wrapper must restore scheduler.step even on the exception path.
    assert pipe.scheduler.step.__func__ is _FakeHV15Scheduler.step
    # The exception unwound pipe.__call__ before its own end-of-call cleanup, so generate() must
    # have freed the offload hooks itself (VRAM would otherwise stay onloaded until the next
    # request).
    assert pipe.hooks_freed == 1


def test_cancel_during_export_discards_clip(fake_runtime, monkeypatch):
    # A cancel landing during the (blocking, uncancellable) export/mux must still discard the
    # clip: cancel_generate() already reported success, so generate() must raise the cancelled
    # sentinel rather than return the clip to be persisted to the gallery.
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
    # A full-pipeline load of the single-DiT TI2V-5B repo: WanPipeline.from_pretrained,
    # no audio, tiling forced on, and the 4k+1 frame lattice surfaced.
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
    # A clip denoise amortises the one-time compile within a single run, so an UNSET speed on a
    # dense (pipeline) load resolves to `default` -- never `max`, never `off`. Explicit "off" is
    # still honored verbatim.
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
    # An explicit Speed="off" (bit-exact) pipeline load with Precision left at auto must NOT
    # promote the unset precision to auto-quant: that would engage torchao quantization (and force
    # speed back to default), breaking the bit-exact request. Mirrors the image backend. On a
    # dense-capable GPU (stubbed) quantize_transformer must not run.
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

    # Control: with speed NOT off, the auto precision promotion still engages the dense quant, so
    # the suppression above is specific to speed=off (not a blanket disable).
    backend.load_pipeline("Wan-AI/Wan2.2-TI2V-5B-Diffusers", model_kind = "pipeline")
    assert calls == [True]


def test_video_step_cache_auto_from_default_schedule(fake_runtime, tmp_path):
    # Unset step cache is AUTO, decided from the model's default schedule: Wan's 50-step default
    # engages FBCache at load; the LTX distilled 8-step default keeps it off. Both are re-checked
    # per generation (toggle test below).
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
    # The AUTO decision follows the ACTUAL step count of each generation: a few-step request drops
    # the load-time cache, a many-step request restores it. An explicit "off" never toggles.
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
    # /32 spatial snap for TI2V-5B: its VAE is 16x spatial * patch 2 = 32 (WanPipeline floors
    # H/W to 32), so 1000x700 -> 992x672 (not the /16 992x688).
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
    # The single-DiT TI2V pipeline has no guidance_scale_2 in its signature, so a request value
    # must NOT be threaded (WanPipeline raises on it when boundary_ratio is None), even if the
    # caller passes guidance_2.
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
    # The MoE pipeline's __call__ carries guidance_scale_2, so an explicit guidance_2
    # is threaded through as that kwarg (the cfg2_kwarg the family declares).
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
    # A dual-DiT MoE load must engage the step cache on BOTH experts, not just the
    # first: transformer_2 handles the low-noise steps and would otherwise run uncached.
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
    # An explicit attention backend must be set on both experts. The fake runtime is a CPU target,
    # where the NVIDIA gate drops explicit kernels; pin the gate open so the explicit-set path is
    # what this test exercises.
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
    # transformer_quant on a pipeline load quantises the dense DiT(s). On CPU the real dense path
    # is unsupported, so stub the two quant seams to record which pipe view each helper saw: BOTH
    # experts must be quantised (via the _SecondDiTView proxy), and status must report the scheme.
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
        # The helper reads view.transformer; record the object it would quantise so the
        # test proves the second expert was reached through the proxy.
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
    # Offload hooks move modules with Module.to(), which torchao quantized tensors reject
    # (observed as a hard crash on the A14B gate run). When the memory plan resolves to any offload
    # policy, quant must be SKIPPED, not attempted: the load succeeds dense and the record explains
    # why.
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
    # The CPU fake target never plans an offload, so force one at the plan seam (frozen dataclass
    # -> dataclasses.replace) and stub the apply step, which would else call offload hooks the fake
    # pipe lacks.
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
    # If the first expert quantises but the second doesn't, the pipe is left at mismatched
    # precision with no way back (in-place mutation), so the load must fail cleanly rather than run
    # mixed with quant reported off.
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
    # The two Wan base repos are trusted for non-GGUF (pipeline) loads; an unrelated
    # repo carrying the family name is not.
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
    # A single gguf/safetensors checkpoint carries only one of the A14B's two experts; the
    # pipeline would pull the other dense bf16 from the base repo outside the memory plan, so
    # validate refuses it up front (before any download).
    backend = VideoBackend()
    with pytest.raises(ValueError, match = "dual-expert"):
        backend.validate_load_request(
            "QuantStack/Wan2.2-T2V-A14B-GGUF",
            gguf_filename = "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf",
        )
    # The single-DiT 5B family still accepts GGUF.
    fam = backend.validate_load_request(
        "QuantStack/Wan2.2-TI2V-5B-GGUF",
        gguf_filename = "Wan2.2-TI2V-5B-Q4_K_M.gguf",
    )
    assert fam.name == "wan2.2-ti2v-5b"


def test_second_dit_view_write_through():
    # Attribute writes on the proxy must land on the real pipe (a helper's side effect would else
    # vanish with the temporary view); a ``transformer`` write mirrors the read property onto the
    # second expert.
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
    # A pipeline load skips the packaged root checkpoint, the duplicate
    # text-encoder shard naming, and non-weight assets -- and keeps everything else.
    info = types.SimpleNamespace(siblings = _LTX2_SIBLINGS)
    files = dict(VideoBackend._base_download_files(info, "pipeline"))
    assert "ltx-2-19b-packaged-fp8.safetensors" not in files
    assert "text_encoder/diffusion_pytorch_model-00001-of-00002.safetensors" not in files
    assert "assets/example.mp4" not in files
    assert files["text_encoder/model-00001-of-00002.safetensors"] == 25
    assert files["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"] == 20
    # The standalone chat template must survive the whitelist: apply_chat_template
    # reads it at generation time and it is not embedded in tokenizer_config.json.
    assert "tokenizer/chat_template.jinja" in files
    assert sum(files.values()) == 10 + 1 + 20 + 18 + 25 + 25 + 3 + 1 + 1


def test_base_download_files_gguf_drops_transformer():
    # A GGUF/single-file checkpoint replaces the DiT: the base transformer never pulls.
    info = types.SimpleNamespace(siblings = _LTX2_SIBLINGS)
    names = [n for n, _ in VideoBackend._base_download_files(info, "gguf")]
    assert not any(n.startswith("transformer/") for n in names)
    assert "text_encoder/model-00001-of-00002.safetensors" in names


def test_load_progress_clamps_overshoot(fake_runtime, monkeypatch):
    # The cache scan counts blobs a broader previous pull left behind; the reported
    # counter must never exceed the scoped estimate (no "282 GB of 263 GB").
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
    # When the scoped pre-download produced a local snapshot, from_pretrained must
    # receive that dir (keeping diffusers' own broader snapshot sweep off the hub).
    backend = VideoBackend()
    backend.load_pipeline(
        "Lightricks/LTX-2",
        model_kind = "pipeline",
        _base_local_dir = str(tmp_path),
    )
    assert _FakePipeline.last["base"] == str(tmp_path)
    backend.unload()


def test_base_download_files_ltx23_keeps_only_shared_components():
    # A 2.3 checkpoint supplies the DiT, connectors, both VAEs and the vocoder, so
    # the base pull shrinks to scheduler + text encoder + tokenizer (+ root manifest).
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
    # The 720p repack is trusted, but it must resolve its OWN family entry: the
    # generic hunyuanvideo-1.5 entry would default generation to 832x480.
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
    # A warm-cache sweep returns each file instantly without consulting the event,
    # so the loop must check it explicitly or an unload mid-predownload is ignored.
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
    # A local GGUF is admitted to the Video picker by its general.architecture, but its path name
    # may carry no whole-segment family token (a renamed "model.gguf"). The loader must resolve the
    # same family the picker offered by reading the arch, not only the name.
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

    # A video arch with no backend family (wan) stays None -> the loader 400s as before.
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
