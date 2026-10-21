# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Edge and Cosmos3-Nano on real weights through Studio's VideoBackend: opt-in, one CUDA GPU.

Not hermetic. The weights are nvidia/Cosmos3-Edge and nvidia/Cosmos3-Nano at the commits
fixtures/cosmos3_checkpoints.json pins, read from UNSLOTH_TEST_COSMOS3_CACHE: a hub cache dir holding both snapshots,
with refs/main at those commits, because an offline load of a hub id opens refs/main. Still offline and read-only:
HF_HUB_OFFLINE=1, local_files_only=True, and the suite's network guard stays up. Nano loads with
UNSLOTH_COSMOS3_SAFETY_CHECKER=0: its model_index.json leaves enable_safety_checker at the pipeline default (on), which
needs NVIDIA's cosmos_guardrail package, and without the package or the opt-out Studio refuses the load up front
(video_cosmos3.guardrail_refusal). Edge loads without the opt-out: its model_index.json ships the checker off.

Each checkpoint loads once, at speed "off" (Studio's bit-identical tier), and serves every test that names it. Shapes
are small and steps few: the tests check that the real pipeline assembles and renders through Studio's code path
(frame count and size, finite, not flat, same seed same bits, frame 0 is the centre-cropped conditioning image), not
quality or speed. Load time, render time and torch peak memory go to record_property. From studio/backend:

    UNSLOTH_TEST_COSMOS3_CACHE=~/.cache/huggingface/hub python -m pytest tests/test_video_cosmos3_gpu.py -m gpu -rs
"""

from __future__ import annotations

import base64
import dataclasses
import gc
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tests" / "_shared"))
from real_accelerator import has_real_cuda  # noqa: E402

CACHE_ENV = "UNSLOTH_TEST_COSMOS3_CACHE"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU"),
    pytest.mark.skipif(
        not os.environ.get(CACHE_ENV),
        reason = f"needs {CACHE_ENV}, a hub cache dir holding nvidia/Cosmos3-Nano and nvidia/Cosmos3-Edge",
    ),
    # One xdist worker for the module: under --dist loadgroup each worker would load its own copy of a checkpoint
    # onto the same GPU.
    pytest.mark.xdist_group(name = "cosmos3_gpu"),
    # The first test that names a checkpoint pays for its load (Nano reads about 35 GB of weights), which on slow
    # storage can outlast Backend CI's per-test --timeout=330.
    pytest.mark.timeout(3600),
]

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cosmos3_checkpoints.json"
_NANO = "nvidia/Cosmos3-Nano"
_EDGE = "nvidia/Cosmos3-Edge"
# The loads that set UNSLOTH_COSMOS3_SAFETY_CHECKER=0 (module docstring). Edge's load goes without it, so the guardrail
# test shows Edge needs neither the opt-out nor cosmos_guardrail.
_GUARDRAIL_OPT_OUT = frozenset({_NANO})
# The cards send their JSON-upsampled prompts as json.dumps of assets/*.json, with assets/negative_prompt.json as the
# negative (Nano README.md:878-891, Edge README.md:750-765).
_NANO_T2V_PROMPT = "example_t2v_prompt.json"
_I2V_PROMPT = "example_i2v_prompt.json"
_NEGATIVE_PROMPT = "negative_prompt.json"
_I2V_IMAGE = "example_i2v_input.jpg"
# Frame 0 of an image-to-video clip is not sampled. The pipeline encodes the conditioning image as latent frame 0,
# marks it clean (pipeline_cosmos3_omni.py:941-955 at diffusers d42ebf9) and zeroes the predicted velocity there at
# every step (:1239-1242), which UniPC's flow update carries through unchanged; the causal VAE then decodes it on its
# own (:1914). So frame 0 is a VAE round trip of the image as the pipeline fitted it: resized to cover the canvas,
# centre-cropped (:85-98). Measured on CPU with the checkpoints' own assets/ (H.264 included for the videos): frame 0
# of Edge's diffusers example output (edge_i2v_diffusers.mp4, 832x480) scores 30.8 dB against that centre crop; at
# 1280x720 the input stretched onto the canvas scores 19.4 dB against it and a crop anchored at the top or bottom
# 17.5 dB; unrelated frames score 9-10 dB and a flat frame 12.5 dB. 25 dB sits between the round trip and all of
# those. At 832x480 this 3034x1754 input already has the canvas's aspect, so there the floor only shows that the image
# conditioned the clip; at 1280x720 it also shows the fit.
_FIRST_FRAME_MIN_PSNR_DB = 25.0
# A flat frame (one value everywhere: a NaN decode cast to uint8, a dead sampler) has a pixel std of 0 on the 0-255
# scale; any picture, blurred by few steps or not, has far more.
_MIN_FRAME_STD = 1.0


# ── the checkpoints in the opted-in cache ──────────────────────────────────────


def _pinned_revision(repo: str) -> str:
    """The commit the hermetic Cosmos3 tests read their Hub metadata at; the weights here are that same commit."""
    keys = json.loads(_FIXTURE.read_text(encoding = "utf-8"))
    (revision,) = [key.split("@", 1)[1] for key in keys if key.split("@", 1)[0] == repo]
    return revision


def _snapshot(repo: str) -> Path:
    """The pinned snapshot in the opted-in cache; skips when the cache cannot serve it offline."""
    revision = _pinned_revision(repo)
    root = Path(os.environ[CACHE_ENV]).expanduser() / ("models--" + repo.replace("/", "--"))
    snapshot = root / "snapshots" / revision
    if not (snapshot / "model_index.json").is_file():
        pytest.skip(f"{CACHE_ENV} holds no {repo} snapshot at {revision}")
    ref = root / "refs" / "main"
    head = ref.read_text(encoding = "utf-8").strip() if ref.is_file() else None
    if head != revision:
        # Studio's offline load hands from_pretrained the hub id, which resolves refs/main in the cache.
        pytest.skip(f"{CACHE_ENV} has {repo} refs/main at {head}, not the pinned {revision}")
    return snapshot


def _card_prompt(repo: str, name: str) -> str:
    return json.dumps(json.loads((_snapshot(repo) / "assets" / name).read_text(encoding = "utf-8")))


def _card_image_b64(repo: str) -> str:
    """The conditioning image both cards animate, as the base64 a Studio request carries."""
    return base64.b64encode((_snapshot(repo) / "assets" / _I2V_IMAGE).read_bytes()).decode("ascii")


# ── one resident checkpoint for the module ─────────────────────────────────────


def _reset_peaks(torch: Any) -> None:
    """Peaks per phase: release what the caching allocator kept from the last phase, then restart the counters."""
    gc.collect()
    # The allocator's counters exist once CUDA is initialised; before the first allocation, resetting them raises
    # "Invalid device argument" (torch 2.13 on a GB300).
    torch.cuda.init()
    torch.cuda.empty_cache()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def _peaks(torch: Any) -> tuple[int, int]:
    devices = range(torch.cuda.device_count())
    return (
        max(torch.cuda.max_memory_allocated(index) for index in devices),
        max(torch.cuda.max_memory_reserved(index) for index in devices),
    )


def _memory(size: int) -> dict[str, Any]:
    return {"bytes": int(size), "mib": round(size / 2**20, 1), "gib": round(size / 2**30, 3)}


class _Resident:
    """At most one checkpoint on the GPU, shared by every test that names it: Nano's transformer alone is 30 GB, so a
    load per test would spend the run reading weights."""

    def __init__(self) -> None:
        self.repo: Optional[str] = None
        self.backend: Any = None
        self.status: dict[str, dict] = {}
        self.loads: dict[str, dict] = {}
        self.failed: dict[str, str] = {}

    def backend_for(self, repo: str) -> Any:
        if repo in self.failed:
            pytest.fail(f"{repo} failed to load earlier in this module: {self.failed[repo]}")
        if self.repo == repo:
            return self.backend
        _snapshot(repo)
        self.release()
        import torch

        from core.inference import video_cosmos3
        from core.inference.video import VideoBackend

        backend = VideoBackend()
        opt_out = repo in _GUARDRAIL_OPT_OUT
        _reset_peaks(torch)
        started = time.monotonic()
        try:
            # Studio reads the opt-out at load only (validate_load_request, from_pretrained's kwargs), so it is scoped
            # to the load; deleted for Edge in case the shell exports it.
            with pytest.MonkeyPatch.context() as env:
                if opt_out:
                    env.setenv(video_cosmos3.SAFETY_CHECKER_ENV, "0")
                else:
                    env.delenv(video_cosmos3.SAFETY_CHECKER_ENV, raising = False)
                # The hub id with local_files_only, as an offline (API) load reaches load_pipeline. Speed "off" keeps
                # the bit-identical reference tier (diffusion_speed.py:10); the video default resolves to "default"
                # (video.py:8872-8874), whose cudnn autotuning and free-memory-dependent untiled decode are not
                # guaranteed to repeat bit for bit.
                status = backend.load_pipeline(
                    repo,
                    local_files_only = True,
                    model_kind = "pipeline",
                    speed_mode = "off",
                )
        except Exception as exc:
            self.failed[repo] = repr(exc)
            raise
        seconds = time.monotonic() - started
        allocated, reserved = _peaks(torch)
        self.loads[repo] = {
            "guardrail_opt_out": opt_out,
            "load_seconds": round(seconds, 2),
            "peak_allocated": _memory(allocated),
            "peak_reserved": _memory(reserved),
            "resident_allocated": _memory(torch.cuda.memory_allocated()),
            "memory_mode": status.get("memory_mode"),
            "offload_policy": status.get("offload_policy"),
            "speed_optims": status.get("speed_optims"),
            "attention_backend": status.get("attention_backend"),
        }
        self.repo, self.backend, self.status[repo] = repo, backend, status
        return backend

    def release(self) -> None:
        backend, self.backend, self.repo = self.backend, None, None
        if backend is None:
            return
        import torch

        backend.unload()
        del backend
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope = "module")
def _resident():
    # Studio's video path is diffusers; a CUDA host without it has nothing to load with.
    pytest.importorskip("diffusers")
    resident = _Resident()
    yield resident
    resident.release()


@pytest.fixture
def cosmos3(_resident, monkeypatch):
    """The module's resident checkpoint, with the opted-in cache and offline; backend_for scopes the guardrail opt-out
    to Nano's load."""
    from huggingface_hub import constants as hf_constants

    from utils import hf_cache_settings

    cache = str(Path(os.environ[CACHE_ENV]).expanduser())
    # The three roots conftest's _hf_cache_is_empty points at an empty dir, pointed at the opted-in cache instead.
    monkeypatch.setitem(hf_cache_settings._EXPLICIT_CACHE_ENV, "HF_HUB_CACHE", cache)
    monkeypatch.setenv("HF_HUB_CACHE", cache)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", cache)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")
    return _resident


# ── one render ─────────────────────────────────────────────────────────────────


class _DecodeProbe:
    """Wraps ``vae.decode`` for one render: counts the calls and checks every decoded sample is finite. Restores the
    attribute as found, since Studio may keep its own gated decode in the instance dict (video_vae_untiled.py)."""

    def __init__(self, vae: Any) -> None:
        self.vae = vae
        self.calls = 0
        self.finite = True

    def __enter__(self) -> "_DecodeProbe":
        import torch

        self.had_own = "decode" in vars(self.vae)
        self.original = original = self.vae.decode

        def decode(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            sample = out.sample if hasattr(out, "sample") else out[0]
            self.calls += 1
            self.finite = self.finite and bool(torch.isfinite(sample).all())
            return out

        self.vae.decode = decode
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self.had_own:
            self.vae.decode = self.original
        else:
            vars(self.vae).pop("decode", None)


@dataclasses.dataclass
class _Clip:
    result: dict
    # (frames, height, width, 3) uint8 as Studio's MP4 encoders take them; None when they would refuse the clip.
    frames: Any
    seconds: float
    encode_seconds: float
    decode_calls: int
    decoded_finite: bool
    peak_allocated: int
    peak_reserved: int


def _render(backend: Any, **request: Any) -> _Clip:
    """``VideoBackend.generate`` with the clip's frames captured at the MP4 encoder, which still runs."""
    import torch

    from core.inference.video import VideoBackend
    from core.inference.video_nvenc import _uint8_frames

    handed: list = []
    encode_seconds: list = []

    def _encode(video_frames: Any, fps: int, audio: Any, pipe: Any, **kwargs: Any) -> bytes:
        handed.append(_uint8_frames(video_frames))
        started = time.monotonic()
        try:
            return VideoBackend._encode_mp4(video_frames, fps, audio, pipe, **kwargs)
        finally:
            encode_seconds.append(time.monotonic() - started)

    _reset_peaks(torch)
    backend._encode_mp4 = _encode
    try:
        with _DecodeProbe(backend._state.pipe.vae) as probe:
            started = time.monotonic()
            result = backend.generate(**request)
            seconds = time.monotonic() - started
    finally:
        vars(backend).pop("_encode_mp4", None)
    allocated, reserved = _peaks(torch)
    assert len(handed) == 1, f"one generate call encoded {len(handed)} clips"
    return _Clip(
        result = result,
        frames = handed[0],
        seconds = seconds,
        encode_seconds = encode_seconds[0],
        decode_calls = probe.calls,
        decoded_finite = probe.finite,
        peak_allocated = allocated,
        peak_reserved = reserved,
    )


def _frames_sha256(frames: Any) -> Optional[str]:
    """sha256 of the uint8 RGB frames (frames, height, width, 3) in C order: equal digests, equal pixels."""
    if frames is None:
        return None
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(frames).reshape(-1).data).hexdigest()


def _mp4_shape(data: bytes) -> tuple[int, int, int]:
    """(frames, width, height) of the returned MP4, decoded with PyAV, which Studio's own encoders write through."""
    import av
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
        return frames, stream.codec_context.width, stream.codec_context.height


def _summary(result: dict) -> dict:
    """A generate result without the MP4 bytes and the effective negative prompt (15-17 kB for the cards' JSON)."""
    return {
        key: value for key, value in result.items() if key not in ("mp4_bytes", "negative_prompt")
    }


def _assert_clip(clip: _Clip, *, num_frames: int, width: int, height: int) -> None:
    import numpy as np

    result = clip.result
    shape = (result["num_frames"], result["width"], result["height"])
    assert shape == (num_frames, width, height), _summary(result)
    assert clip.frames is not None, "frames Studio's uint8 MP4 encoders refuse"
    assert clip.frames.shape == (num_frames, height, width, 3), clip.frames.shape
    # Cosmos3OmniPipeline decodes through vae.decode (pipeline_cosmos3_omni.py:1914).
    assert clip.decode_calls >= 1, "the render never went through pipe.vae.decode"
    assert clip.decoded_finite, "the VAE decoded NaN or inf"
    spread = np.array([float(frame.std()) for frame in clip.frames])
    assert float(spread.min()) > _MIN_FRAME_STD, f"flat frame: pixel std {spread.round(2).tolist()}"
    assert _mp4_shape(result["mp4_bytes"]) == (num_frames, width, height)


def _centre_crop(image: Any, width: int, height: int) -> Any:
    """The pipeline's own fit of a conditioning image (pipeline_cosmos3_omni.py:85-98): resize to cover the canvas
    keeping the aspect, then crop the centre. Studio hands Cosmos3 the decoded image unfitted. PIL's bilinear resize
    matches the pipeline's antialiased torch resize to 57 dB at 1280x720 and 58 dB at 832x480 on this input (measured
    on CPU), so the reference spends none of the floor's margin."""
    import numpy as np
    from PIL import Image

    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (math.ceil(scale * image.width), math.ceil(scale * image.height)), Image.BILINEAR
    )
    top = round((resized.height - height) / 2)
    left = round((resized.width - width) / 2)
    return np.asarray(resized.crop((left, top, left + width, top + height)))


def _first_frame_psnr_db(clip: _Clip, image_b64: str) -> float:
    if clip.frames is None:
        # No frames to compare: _assert_clip says why, and NaN fails the floor.
        return math.nan
    import numpy as np

    from core.inference.diffusion import decode_b64_image

    # The same decode Studio applies to the request's first_frame (RGB, EXIF-upright).
    reference = _centre_crop(
        decode_b64_image(image_b64, mode = "RGB"), clip.result["width"], clip.result["height"]
    )
    error = float(np.mean((clip.frames[0].astype(np.float64) - reference.astype(np.float64)) ** 2))
    return math.inf if error == 0 else 10.0 * math.log10(255.0**2 / error)


def _record(record_property: Any, name: str, clip: _Clip, load: dict, **extra: Any) -> None:
    """One clip's numbers, with the load that served it (load seconds and peaks), as a JSON user property."""
    record_property(
        name,
        json.dumps(
            {
                "load": load,
                "generate_seconds": round(clip.seconds, 2),
                "encode_seconds": round(clip.encode_seconds, 2),
                "peak_allocated": _memory(clip.peak_allocated),
                "peak_reserved": _memory(clip.peak_reserved),
                "num_frames": clip.result["num_frames"],
                "width": clip.result["width"],
                "height": clip.result["height"],
                "steps": clip.result["steps"],
                "guidance": clip.result["guidance"],
                "fps": clip.result["fps"],
                "seed": clip.result["seed"],
                "frames_sha256": _frames_sha256(clip.frames),
                **extra,
            },
            sort_keys = True,
        ),
    )


# ── Cosmos3-Edge ───────────────────────────────────────────────────────────────


def test_cosmos3_edge_loads_offline_with_its_card_sampler_and_no_guardrail(
    cosmos3, record_property
):
    import torch

    backend = cosmos3.backend_for(_EDGE)
    record_property("load", json.dumps(cosmos3.loads[_EDGE], sort_keys = True))
    status = cosmos3.status[_EDGE]
    pipe = backend._state.pipe
    assert (status["family"], status["model_kind"], status["has_audio"]) == (
        "cosmos3-edge",
        "pipeline",
        False,
    )
    assert status["supports_first_frame"] is True
    assert str(status["device"]).startswith("cuda")
    assert type(pipe).__name__ == "Cosmos3OmniPipeline"
    # "Only BF16 precision is tested" (Edge README.md:156).
    assert pipe.transformer.dtype == torch.bfloat16
    # The card rebuilds UniPC at flow_shift 12 with Karras sigmas off (README.md:759-761); the shipped scheduler
    # config keeps flow_shift 1.0 and Karras on.
    config = pipe.scheduler.config
    assert (type(pipe.scheduler).__name__, config.flow_shift, config.use_karras_sigmas) == (
        "UniPCMultistepScheduler",
        12.0,
        False,
    )
    # Edge's model_index.json ships the guardrail off, the native flow schedule on and the system prompt off. Loaded
    # without UNSLOTH_COSMOS3_SAFETY_CHECKER, Studio neither refuses Edge nor builds CosmosSafetyChecker: the load
    # keeps all three.
    assert cosmos3.loads[_EDGE]["guardrail_opt_out"] is False
    assert pipe.config.enable_safety_checker is False and pipe.safety_checker is None
    assert pipe.config.use_native_flow_schedule is True
    assert pipe.config.default_use_system_prompt is False


def test_cosmos3_edge_animates_its_card_image_into_a_finite_clip_of_the_requested_shape(
    cosmos3, record_property
):
    backend = cosmos3.backend_for(_EDGE)
    image = _card_image_b64(_EDGE)
    # 53 frames is Edge's floor (the card's 50-150 on the 4k+1 lattice, README.md:102); 832x480 its card size.
    clip = _render(
        backend,
        prompt = _card_prompt(_EDGE, _I2V_PROMPT),
        negative_prompt = _card_prompt(_EDGE, _NEGATIVE_PROMPT),
        first_frame = image,
        width = 832,
        height = 480,
        num_frames = 53,
        steps = 8,
        seed = 0,
    )
    psnr = _first_frame_psnr_db(clip, image)
    _record(
        record_property, "edge_i2v", clip, cosmos3.loads[_EDGE], first_frame_psnr_db = round(psnr, 2)
    )
    _assert_clip(clip, num_frames = 53, width = 832, height = 480)
    assert psnr >= _FIRST_FRAME_MIN_PSNR_DB, f"frame 0 vs the centre crop: {psnr:.2f} dB"


# ── Cosmos3-Nano ───────────────────────────────────────────────────────────────


def test_cosmos3_nano_loads_offline_with_its_card_sampler_and_no_sound_decoder(
    cosmos3, record_property
):
    import torch

    backend = cosmos3.backend_for(_NANO)
    record_property("load", json.dumps(cosmos3.loads[_NANO], sort_keys = True))
    status = cosmos3.status[_NANO]
    pipe = backend._state.pipe
    assert (status["family"], status["model_kind"], status["has_audio"]) == (
        "cosmos3-nano",
        "pipeline",
        False,
    )
    assert status["supports_first_frame"] is True
    assert str(status["device"]).startswith("cuda")
    # Nano's index names Cosmos3OmniDiffusersPipeline, a class diffusers does not have; Studio asks for the real one.
    assert type(pipe).__name__ == "Cosmos3OmniPipeline"
    # "Only BF16 precision is tested" (Nano README.md:168).
    assert pipe.transformer.dtype == torch.bfloat16
    # The card's UniPC.from_config(cfg, flow_shift=10.0) (README.md:887) leaves the shipped Karras sigmas on, under
    # which the shift is inert; pinned as is, so a change to either is noticed.
    config = pipe.scheduler.config
    assert (type(pipe.scheduler).__name__, config.flow_shift, config.use_karras_sigmas) == (
        "UniPCMultistepScheduler",
        10.0,
        True,
    )
    # Video only: Studio passes sound_tokenizer=None, so the 2 GB audio decoder never loads.
    assert pipe.sound_tokenizer is None
    # Nano's index leaves enable_safety_checker unset, so the pipeline default (on) would build CosmosSafetyChecker;
    # the opt-out reached the pipeline instead.
    assert pipe.config.enable_safety_checker is False and pipe.safety_checker is None


def test_cosmos3_nano_renders_text_to_video_as_a_finite_clip_of_the_requested_shape(
    cosmos3, record_property
):
    backend = cosmos3.backend_for(_NANO)
    clip = _render(
        backend,
        prompt = _card_prompt(_NANO, _NANO_T2V_PROMPT),
        negative_prompt = _card_prompt(_NANO, _NEGATIVE_PROMPT),
        width = 832,
        height = 480,
        num_frames = 17,
        steps = 8,
        seed = 123,
    )
    _record(record_property, "nano_t2v", clip, cosmos3.loads[_NANO])
    _assert_clip(clip, num_frames = 17, width = 832, height = 480)


def test_cosmos3_nano_same_seed_repeats_bit_for_bit_and_a_new_seed_does_not(
    cosmos3, record_property
):
    import numpy as np

    backend = cosmos3.backend_for(_NANO)
    request = {
        "prompt": _card_prompt(_NANO, _NANO_T2V_PROMPT),
        "negative_prompt": _card_prompt(_NANO, _NEGATIVE_PROMPT),
        "width": 832,
        "height": 480,
        "num_frames": 9,
        "steps": 4,
    }
    first = _render(backend, seed = 123, **request)
    again = _render(backend, seed = 123, **request)
    other = _render(backend, seed = 124, **request)
    clips = {"first": first, "again": again, "other": other}
    for name, clip in clips.items():
        _record(record_property, f"nano_repeat_{name}", clip, cosmos3.loads[_NANO])
    for clip in clips.values():
        _assert_clip(clip, num_frames = 9, width = 832, height = 480)
    differing = int(np.count_nonzero(first.frames != again.frames))
    assert differing == 0, f"same seed, {differing} differing values"
    # Not a seed the sampler ignores: another one moves the clip.
    assert _frames_sha256(other.frames) != _frames_sha256(first.frames)


def test_cosmos3_nano_first_frame_is_the_centre_cropped_conditioning_image(
    cosmos3, record_property
):
    backend = cosmos3.backend_for(_NANO)
    image = _card_image_b64(_NANO)
    # 1280x720 is the card's canvas (README.md:359) and where the fit shows: the 3034x1754 input loses 20 rows to the
    # crop there, so a stretched or top/bottom-anchored fit lands below the PSNR floor (_FIRST_FRAME_MIN_PSNR_DB).
    clip = _render(
        backend,
        prompt = _card_prompt(_NANO, _I2V_PROMPT),
        negative_prompt = _card_prompt(_NANO, _NEGATIVE_PROMPT),
        first_frame = image,
        width = 1280,
        height = 720,
        num_frames = 17,
        steps = 8,
        seed = 123,
    )
    psnr = _first_frame_psnr_db(clip, image)
    _record(
        record_property, "nano_i2v", clip, cosmos3.loads[_NANO], first_frame_psnr_db = round(psnr, 2)
    )
    _assert_clip(clip, num_frames = 17, width = 1280, height = 720)
    assert psnr >= _FIRST_FRAME_MIN_PSNR_DB, f"frame 0 vs the centre crop: {psnr:.2f} dB"
