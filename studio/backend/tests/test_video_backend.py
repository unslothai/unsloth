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
from core.inference.video_families import VIDEO_NOT_LOADED_MSG


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


# ── Wan2.2 fakes: a per-DiT trackable transformer so the dual-DiT optimisation
# tests can assert speed / cache / attention engaged on BOTH experts, plus two
# pipeline fakes -- single-DiT (TI2V-5B) and dual-DiT MoE (A14B). The MoE __call__
# carries guidance_scale_2 so the cfg2 signature-gate actually exercises; the
# single-DiT __call__ omits it so the gate proves it is NOT threaded there.


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

    def set_attention_backend(self, backend) -> None:
        self.attention = backend


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


def test_load_records_engaged_speed_optims(fake_runtime, tmp_path, monkeypatch):
    # Regression: the load tail once re-ran the already-filtered speed_optims
    # tuple through ``.items()`` as if it were still the raw applied dict, so
    # every real-GPU load (where at least channels_last engages) crashed with
    # 'tuple' object has no attribute 'items'. Fake runtime forces every optim
    # False, so this only reproduces when one is made to engage.
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


def test_generate_resets_step_cache_only_when_engaged(fake_runtime, tmp_path):
    # FBCache residuals live on the long-lived DiT(s) and survive a generation, so
    # the next clip at a new resolution would crash on stale state. generate must
    # reset them when a cache is engaged (diffusers 0.39 exposes
    # _reset_stateful_cache on the transformer; reset_stateful_hooks only exists on
    # the HookRegistry) and must not touch an uncached load. transformer_2 (the Wan
    # dual expert) resets too when present.
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
    # diffusers maps every LTX-2 single file to the 2.0 config; a 2.3 checkpoint
    # (9-row modulation tables in the header) must be detected so the loader
    # routes to the full 2.3 assembly. A 2.0 header must not, and an unreadable
    # header must fall back to the stock path (False), never raise.
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
    # The Lightricks fp8 files carry .weight_scale/.input_scale companions; a
    # plain dtype cast would silently corrupt them, so the loader must refuse
    # with a pointer to the supported GGUF path.
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
    assert backend.generate_progress() == {"active": False}
    assert backend.cancel_generate() is False


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
    # The single-DiT TI2V pipeline has no guidance_scale_2 in its signature, so a
    # request value must NOT be threaded (WanPipeline raises on it when boundary_ratio
    # is None), even if the caller passes guidance_2.
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
    # An explicit attention backend must be set on both experts. The fake runtime is a
    # CPU target, where the NVIDIA gate correctly drops explicit kernels; pin the gate
    # open so the explicit-set path itself is what this test exercises.
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


def test_video_quant_load_raises_fbcache_threshold(fake_runtime, monkeypatch):
    # Regression: a quantized video transformer's block residuals are larger, so it needs
    # the higher FBCache trigger threshold to cache at all. The load must thread quant_active
    # into apply_step_cache like the image path does. With transformer_quant engaged and no
    # explicit threshold the engaged config must carry QUANT_FBCACHE_THRESHOLD (0.12); a plain
    # bf16 load uses the dense default (0.08), proving the discrimination.
    import core.inference.video as video_mod
    from core.inference.diffusion_cache import (
        DEFAULT_FBCACHE_THRESHOLD,
        QUANT_FBCACHE_THRESHOLD,
    )

    monkeypatch.setattr(video_mod, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        video_mod,
        "quantize_transformer",
        lambda view, target, *, mode, family, logger = None: "int8",
    )

    quant = VideoBackend()
    quant.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_kind = "pipeline",
        transformer_quant = "int8",
        transformer_cache = "fbcache",
    )
    quant_cfg = quant._state.pipe.transformer.cache_config
    assert quant_cfg is not None
    assert quant_cfg[1] == QUANT_FBCACHE_THRESHOLD

    dense = VideoBackend()
    dense.load_pipeline(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        model_kind = "pipeline",
        transformer_cache = "fbcache",
    )
    dense_cfg = dense._state.pipe.transformer.cache_config
    assert dense_cfg is not None
    assert dense_cfg[1] == DEFAULT_FBCACHE_THRESHOLD


def test_wan_a14b_dense_quant_applies_to_both_dits(fake_runtime, monkeypatch):
    # transformer_quant on a pipeline load quantises the dense DiT(s). On CPU the real
    # dense path is unsupported, so stub the two quant seams to record which pipe view
    # each helper saw: BOTH experts must be quantised (via the _SecondDiTView proxy),
    # and status must report the engaged scheme.
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


def test_wan_a14b_partial_quant_fails_the_load(fake_runtime, monkeypatch):
    # If the first expert quantises but the second does not, the pipe is left at
    # mismatched precision with no way back (in-place mutation), so the load must
    # fail cleanly rather than run mixed with quant reported off.
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
    # A single gguf/safetensors checkpoint carries only one of the A14B's two experts;
    # the pipeline would pull the other dense bf16 from the base repo outside the
    # memory plan, so validate refuses it up front (before any download).
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
    # Attribute writes on the proxy must land on the real pipe (a helper's side
    # effect would otherwise vanish with the temporary view); a ``transformer``
    # write mirrors the read property onto the second expert.
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
