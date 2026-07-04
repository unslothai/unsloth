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

from core.inference.video import VideoBackend, get_video_backend, resolve_video_model_kind
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
    gguf.GGUFReader = _reader_for({
        "model.diffusion_model.transformer_blocks.0.scale_shift_table": (4096, 9),
    })
    monkeypatch.setitem(sys.modules, "gguf", gguf)
    path = tmp_path / "ltx23.gguf"
    path.write_bytes(b"x")
    assert is_ltx23_checkpoint(path) is True

    gguf.GGUFReader = _reader_for({
        "model.diffusion_model.transformer_blocks.0.scale_shift_table": (4096, 6),
    })
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
