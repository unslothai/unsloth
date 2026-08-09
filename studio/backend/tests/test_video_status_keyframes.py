# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How a keyframe-capable family is LOADED, and what /video/status says about it.

The load is the part of keyframe support that is easy to get subtly wrong, because
`ModularPipeline.from_pretrained(workflow=...)` prunes the auto block graph statically:
an fl2va-pruned pipeline cannot serve a text-only request at all. These tests pin the
three decisions that follow from that, with a fake `diffusers` module so nothing is
downloaded and no GPU is touched:

  * the block graph is left whole for a keyframe-capable family;
  * `load_components` is still bounded to the keyframe workflow's component set, so the
    61.7 GB Ref2VA partition is no more loaded than it was before;
  * the video VAE keeps its encoder, which is what encodes a keyframe (the text-only
    load frees it).
"""

from __future__ import annotations

import types

import pytest

import core.inference.video as video_module
from core.inference.video_families import detect_video_family


def _h3():
    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam is not None
    return fam


class _FakeModule:
    """A weightless stand-in: trim_h3_video_vae sizes a module through parameters/buffers."""

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())


class _FakeVae:
    """Only the attributes trim_h3_video_vae reaches for."""

    def __init__(self):
        self.encoder = _FakeModule()
        self.quant_conv = _FakeModule()
        self.decoder = None
        self.post_quant_conv = None


class _FakePipe:
    def __init__(self):
        self.vae = _FakeVae()
        self.load_components_calls: list[dict] = []

    def load_components(self, **kwargs):
        self.load_components_calls.append(kwargs)

    def update_components(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


class _FakeManager:
    def enable_auto_cpu_offload(self, **kwargs):  # pragma: no cover -- device is cpu here
        raise AssertionError("a cpu load must not enable offload")


def _fake_diffusers(pipe):
    calls: dict = {}

    class _ModularPipeline:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["from_pretrained"] = {"path": path, **kwargs}
            return pipe

    return (
        types.SimpleNamespace(ComponentsManager = _FakeManager, ModularPipeline = _ModularPipeline),
        calls,
    )


def _load(fam, monkeypatch, pipe = None):
    pipe = pipe or _FakePipe()
    diffusers, calls = _fake_diffusers(pipe)
    backend = video_module.VideoBackend()
    monkeypatch.setattr(video_module, "hub_cache_dir", lambda: "/tmp/hub", raising = False)
    backend._load_h3_modular_pipeline(
        diffusers = diffusers,
        torch = types.SimpleNamespace(),
        fam = fam,
        repo_id = "MiniMaxAI/MiniMax-H3",
        base = "MiniMaxAI/MiniMax-H3",
        kind = "pipeline",
        dtype = "torch.bfloat16",
        device = "cpu",
        hf_token = None,
        memory_mode = None,
        _load_token = None,
        _base_local_dir = None,
    )
    return backend, pipe, calls


def test_keyframe_load_keeps_the_whole_block_graph(monkeypatch):
    """No `workflow=` reaches from_pretrained: that argument prunes the auto blocks once and
    for all, and a pruned fl2va pipeline cannot run a text-only request."""
    _, _, calls = _load(_h3(), monkeypatch)
    assert "workflow" not in calls["from_pretrained"]


def test_keyframe_load_bounds_components_to_the_keyframe_workflow(monkeypatch):
    """Leaving the graph whole must not widen the DOWNLOAD: the component set is still bounded,
    just at load_components instead, so transformer_ref is never pulled."""
    _, pipe, _ = _load(_h3(), monkeypatch)
    assert pipe.load_components_calls[0]["workflow"] == "fl2va"


def test_keyframe_load_keeps_the_vae_encoder(monkeypatch):
    """The text-only load frees vae.encoder as dead weight. It is exactly what encodes a
    keyframe, so a keyframe-capable load has to keep it."""
    _, pipe, _ = _load(_h3(), monkeypatch)
    assert pipe.vae.encoder is not None
    assert pipe.vae.quant_conv is not None


def test_text_only_family_still_prunes_and_frees_the_encoder(monkeypatch):
    """The old behaviour is unchanged for a family with no keyframe workflow: prune at
    from_pretrained, no workflow at load_components, encoder freed."""
    import dataclasses

    fam = dataclasses.replace(_h3(), keyframe_workflow = None)
    _, pipe, calls = _load(fam, monkeypatch)
    assert calls["from_pretrained"]["workflow"] == "t2va"
    assert "workflow" not in pipe.load_components_calls[0]
    assert pipe.vae.encoder is None


def test_status_reports_keyframe_support_for_h3(monkeypatch):
    backend, _, _ = _load(_h3(), monkeypatch)
    status = backend.status()
    assert status["loaded"] is True
    assert status["supports_keyframes"] is True


def test_status_reports_no_keyframe_support_for_a_text_only_family(monkeypatch):
    import dataclasses

    fam = dataclasses.replace(_h3(), keyframe_workflow = None)
    backend, _, _ = _load(fam, monkeypatch)
    assert backend.status()["supports_keyframes"] is False


def test_unloaded_status_reports_no_keyframe_support():
    assert video_module.VideoBackend().status()["supports_keyframes"] is False


def test_status_response_model_accepts_the_flag():
    """The route serialises through the response model, so a field it does not declare is
    silently dropped before the frontend ever sees it."""
    from models.inference import VideoStatusResponse

    assert VideoStatusResponse(loaded = False).supports_keyframes is False
    assert VideoStatusResponse(loaded = True, supports_keyframes = True).supports_keyframes is True


@pytest.mark.parametrize("field", ["image", "last_image"])
def test_generate_request_model_carries_the_keyframe_fields(field):
    from models.inference import VideoGenerateRequest

    request = VideoGenerateRequest(prompt = "p", **{field: "AAAA"})
    assert getattr(request, field) == "AAAA"
    assert VideoGenerateRequest(prompt = "p").image is None


def test_generate_request_bounds_the_keyframe_payload():
    """Same 32 MiB cap the image routes put on init_image, so one request cannot buffer a
    multi-GB body."""
    from pydantic import ValidationError

    from models.inference import VideoGenerateRequest

    with pytest.raises(ValidationError):
        VideoGenerateRequest(prompt = "p", image = "A" * (32 * 1024 * 1024 + 1))


def test_gallery_record_carries_the_keyframe_anchors():
    from models.inference import GalleryVideo

    base = dict(
        id = "v1",
        url = "/u",
        prompt = "p",
        width = 1344,
        height = 768,
        num_frames = 124,
        fps = 24,
        duration_s = 5.0,
        steps = 30,
        guidance = 1.0,
        seed = 1,
        created_at = "2026-01-01T00:00:00Z",
    )
    assert GalleryVideo(**base).keyframe_anchors is None
    assert GalleryVideo(**base, keyframe_anchors = ["first", "last"]).keyframe_anchors == [
        "first",
        "last",
    ]
