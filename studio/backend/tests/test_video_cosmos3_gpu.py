# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cosmos3-Edge and Cosmos3-Nano on real weights: opt-in, one CUDA GPU, offline.

Each checkpoint loads once through VideoBackend at speed "off" (Studio's bit-identical tier) and renders a small
text-to-video and image-to-video clip; then plain diffusers' Cosmos3OmniPipeline loads the same snapshot with the card's
scheduler and renders the same requests, which must match bit for bit. Weights come from UNSLOTH_TEST_COSMOS3_CACHE, a
hub cache holding both repos with refs/main at the revisions fixtures/cosmos3_checkpoints.json pins. From
studio/backend:

    UNSLOTH_TEST_COSMOS3_CACHE=~/.cache/huggingface/hub python -m pytest tests/test_video_cosmos3_gpu.py -m gpu -rs
"""

from __future__ import annotations

import base64
import gc
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests" / "_shared"))
from real_accelerator import has_real_cuda  # noqa: E402

CACHE_ENV = "UNSLOTH_TEST_COSMOS3_CACHE"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU"),
    pytest.mark.skipif(
        not os.environ.get(CACHE_ENV),
        reason = f"needs {CACHE_ENV}, a hub cache holding nvidia/Cosmos3-Nano and nvidia/Cosmos3-Edge",
    ),
    # One worker loads the checkpoints; Nano reads ~35 GB, past Backend CI's per-test timeout.
    pytest.mark.xdist_group(name = "cosmos3_gpu"),
    pytest.mark.timeout(3600),
]

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "cosmos3_checkpoints.json"
# Card scheduler rebuilds (Nano README.md:887, Edge README.md:759-761) and Edge's prompt-template flags (:774-775).
_CARD = {
    "nvidia/Cosmos3-Nano": ({"flow_shift": 10.0}, {}),
    "nvidia/Cosmos3-Edge": (
        {"flow_shift": 12.0, "use_karras_sigmas": False},
        {"add_resolution_template": False, "add_duration_template": False},
    ),
}


def _snapshot(repo):
    keys = json.loads(_FIXTURE.read_text(encoding = "utf-8"))
    (revision,) = [k.split("@", 1)[1] for k in keys if k.split("@", 1)[0] == repo]
    root = Path(os.environ[CACHE_ENV]).expanduser() / ("models--" + repo.replace("/", "--"))
    ref = root / "refs" / "main"
    if not (root / "snapshots" / revision / "model_index.json").is_file() or (
        not ref.is_file() or ref.read_text(encoding = "utf-8").strip() != revision
    ):
        pytest.skip(f"{CACHE_ENV} has no {repo} snapshot with refs/main at {revision}")
    return root / "snapshots" / revision


def _requests(snapshot):
    assets = snapshot / "assets"

    def prompt(name):
        return json.dumps(json.loads((assets / name).read_text(encoding = "utf-8")))

    image = base64.b64encode((assets / "example_i2v_input.jpg").read_bytes()).decode("ascii")
    common = {"negative_prompt": prompt("negative_prompt.json"), "steps": 4, "seed": 123}
    t2v = {
        "prompt": prompt("example_i2v_prompt.json"),
        "width": 832,
        "height": 480,
        "num_frames": 53,  # Edge's minimum; a 4k+1 count Nano takes too
    }
    return {"t2v": t2v, "i2v": {**t2v, "first_frame": image}}, common


@pytest.fixture
def offline_cache(monkeypatch):
    from huggingface_hub import constants as hf_constants

    from utils import hf_cache_settings

    pytest.importorskip("diffusers", reason = "Studio's video path is diffusers")
    cache = str(Path(os.environ[CACHE_ENV]).expanduser())
    monkeypatch.setitem(hf_cache_settings._EXPLICIT_CACHE_ENV, "HF_HUB_CACHE", cache)
    monkeypatch.setenv("HF_HUB_CACHE", cache)
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", cache)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_NO_MIRROR", "1")


def _free():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _studio_frames(repo, requests, common, monkeypatch):
    from core.inference.video import VideoBackend
    from core.inference.video_nvenc import _uint8_frames

    handed = []
    real_encode = VideoBackend._encode_mp4

    def _encode(frames, fps, audio, pipe, **kwargs):
        handed.append(_uint8_frames(frames))
        return real_encode(frames, fps, audio, pipe, **kwargs)

    monkeypatch.setattr(VideoBackend, "_encode_mp4", staticmethod(_encode))
    if repo.endswith("Nano"):
        monkeypatch.setenv("UNSLOTH_COSMOS3_SAFETY_CHECKER", "0")
    else:
        monkeypatch.delenv("UNSLOTH_COSMOS3_SAFETY_CHECKER", raising = False)
    backend = VideoBackend()
    status = backend.load_pipeline(
        repo, local_files_only = True, model_kind = "pipeline", speed_mode = "off"
    )
    try:
        assert type(backend._state.pipe).__name__ == "Cosmos3OmniPipeline"
        assert status["has_audio"] is False and backend._state.pipe.sound_tokenizer is None
        results = {}
        for name, request in requests.items():
            result = backend.generate(**request, **common)
            results[name] = (result, handed[-1])
        return results
    finally:
        backend.unload()
        del backend
        _free()


def _diffusers_frames(snapshot, repo, requests, common):
    import numpy as np
    import torch
    from diffusers import Cosmos3OmniPipeline, UniPCMultistepScheduler

    from core.inference.diffusion import decode_b64_image

    scheduler, call_kwargs = _CARD[repo]
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(snapshot),
        torch_dtype = torch.bfloat16,
        sound_tokenizer = None,
        enable_safety_checker = False,
        local_files_only = True,
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, **scheduler)
    try:
        frames = {}
        for name, request in requests.items():
            image = request.get("first_frame")
            out = pipe(
                prompt = request["prompt"],
                negative_prompt = common["negative_prompt"],
                image = decode_b64_image(image, mode = "RGB") if image else None,
                num_frames = request["num_frames"],
                height = request["height"],
                width = request["width"],
                fps = 24.0,
                num_inference_steps = common["steps"],
                guidance_scale = 6.0,
                generator = torch.Generator(device = "cuda").manual_seed(common["seed"]),
                **call_kwargs,
            )
            frames[name] = np.stack([np.asarray(frame) for frame in out.video])
        return frames
    finally:
        del pipe
        _free()


@pytest.mark.parametrize("repo", ["nvidia/Cosmos3-Edge", "nvidia/Cosmos3-Nano"])
def test_studio_renders_what_plain_diffusers_renders(
    offline_cache, monkeypatch, record_property, repo
):
    import numpy as np

    snapshot = _snapshot(repo)
    requests, common = _requests(snapshot)
    studio = _studio_frames(repo, requests, common, monkeypatch)
    reference = _diffusers_frames(snapshot, repo, requests, common)
    for name, (result, frames) in studio.items():
        request = requests[name]
        assert (result["num_frames"], result["width"], result["height"]) == (
            request["num_frames"],
            request["width"],
            request["height"],
        )
        assert frames.shape == (request["num_frames"], request["height"], request["width"], 3)
        assert min(float(frame.std()) for frame in frames) > 1.0, name
        differing = int(np.count_nonzero(frames != reference[name]))
        record_property(f"{repo}:{name}:differing_values", differing)
        assert differing == 0, f"{name}: {differing} values differ from plain diffusers"
