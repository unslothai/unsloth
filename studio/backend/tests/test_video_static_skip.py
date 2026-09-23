# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic CPU tests for the static step skip on the VIDEO backend (``transformer_cache="static"``).

The layer itself is covered by ``test_diffusion_static_skip``; these pin the video wiring. They run
on the faked torch/diffusers runtime of ``test_video_backend`` (no GPU, offline), with Wan and
HunyuanVideo-1.5 fakes whose ``__call__`` runs a real CFG denoise loop over their transformer, so
the schedule, the per-branch counting and the step callback are exercised end to end.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from core.inference import diffusion_cache as dcache
from core.inference import diffusion_step_skip as ss
from core.inference.video import VideoBackend

from .test_video_backend import (  # noqa: F401 - fake_runtime is a fixture
    _FakeComponentsManager,
    _FakeHV15Pipe,
    _FakeModularPipeline,
    _FakeWanDiT,
    _FakeWanPipeSingle,
    _FakeWanPipelineSingle,
    fake_runtime,
)

WAN_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
HV15 = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"

# 50 steps: head round(10) = 10 and tail round(5) = 5 compute, the middle 35 compute every other
# step, so 17 steps per CFG branch are skipped.
STEPS = 50
SKIPPED_PER_BRANCH = 17


class _Tensorish:
    """Enough of a tensor for the layer under the fake runtime's stub torch."""

    def __init__(self, value):
        self.value = value
        self.shape = (1, 4)

    def detach(self):
        return self

    def clone(self):
        return _Tensorish(self.value)


class _LoopDiT(_FakeWanDiT):
    """A Wan-style denoiser with a class ``forward`` (what the layer wraps) and a class
    ``cache_context`` that records the branch names the pipeline opens."""

    def __init__(self) -> None:
        super().__init__()
        self.computed = 0
        self.contexts: list = []

    @contextlib.contextmanager
    def cache_context(self, name, **kwargs):
        self.contexts.append(name)
        yield

    def forward(
        self,
        hidden_states = None,
        timestep = None,
        encoder_hidden_states = None,
        return_dict = True,
    ):
        self.computed += 1
        return (_Tensorish(self.computed),)


class _LoopWanPipe(_FakeWanPipeSingle):
    """WanPipeline's order: both CFG branches, then ``callback_on_step_end`` for that step."""

    raise_at = None

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _LoopDiT()
        self.components["transformer"] = self.transformer

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
        self.last_kwargs = {"num_inference_steps": num_inference_steps, **kwargs}
        # Spelled out, not factored: the FBCache gate reads this source for a cache_context.
        for i in range(int(num_inference_steps)):
            if self.raise_at is not None and i == self.raise_at:
                raise RuntimeError("denoise failed")
            for name in ("cond", "uncond"):
                with self.transformer.cache_context(name, step_index = i):
                    self.transformer.forward(hidden_states = None, timestep = None, return_dict = False)
            if callback_on_step_end is not None:
                callback_on_step_end(self, i, 0, {})
        self.vae.decode(object())
        frames = [[object() for _ in range(int(num_frames or 1))]]
        return types.SimpleNamespace(frames = frames, audio = None)


class _LoopHV15Pipe(_FakeHV15Pipe):
    """HunyuanVideo-1.5: no step callback, CFG branches named by its guider's cache contexts."""

    def __init__(self) -> None:
        super().__init__()
        self.transformer = _LoopDiT()
        self.components["transformer"] = self.transformer

    def __call__(
        self,
        *,
        num_inference_steps = None,
        num_frames = None,
        **kwargs,
    ):
        self.last_kwargs = {"num_inference_steps": num_inference_steps, **kwargs}
        for _ in range(int(num_inference_steps)):
            for name in ("pred_cond", "pred_uncond"):
                with self.transformer.cache_context(name):
                    self.transformer.forward(hidden_states = None, timestep = None, return_dict = False)
            self.scheduler.step()
        self.vae.decode(object())
        frames = [[object() for _ in range(int(num_frames or 1))]]
        return types.SimpleNamespace(frames = frames, audio = None)


@pytest.fixture
def loop_runtime(fake_runtime, monkeypatch):
    """The fake runtime with loop-running Wan / HV15 pipelines and a stub torch.is_tensor."""
    diffusers = sys.modules["diffusers"]
    made: dict = {}

    class _WanLoader:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            if "a14b" in str(repo).lower():
                pipe = _FakeWanPipelineSingle.from_pretrained(repo, **kwargs)
            else:
                pipe = _LoopWanPipe()
            made["pipe"] = pipe
            return pipe

    class _HV15Loader:
        @classmethod
        def from_pretrained(cls, repo, **kwargs):
            made["pipe"] = _LoopHV15Pipe()
            return made["pipe"]

    monkeypatch.setattr(diffusers, "WanPipeline", _WanLoader, raising = False)
    monkeypatch.setattr(diffusers, "HunyuanVideo15Pipeline", _HV15Loader, raising = False)
    monkeypatch.setattr(
        sys.modules["torch"], "is_tensor", lambda obj: isinstance(obj, _Tensorish), raising = False
    )
    return made


def _record_speed(monkeypatch):
    import core.inference.video as video_mod

    seen: list = []

    def fake_speed(pipe, target, **kwargs):
        seen.append(kwargs)
        return {"compiled": True, "cuda_graph": False}

    monkeypatch.setattr(video_mod, "apply_speed_optims", fake_speed)
    return seen


# ── load ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("request_cache", ["off", "static"])
def test_static_load_keeps_the_uncached_speed_decisions(loop_runtime, monkeypatch, request_cache):
    seen = _record_speed(monkeypatch)
    backend = VideoBackend()
    status = backend.load_pipeline(
        WAN_5B, model_kind = "pipeline", speed_mode = "default", transformer_cache = request_cache
    )
    # Static decides outside the forward: the compile keeps fullgraph and the graph arm sees no cache.
    assert [kw["cache_active"] for kw in seen] == [False]
    assert status["speed_optims"] == ["compiled"]
    pipe = loop_runtime["pipe"]
    if request_cache == "static":
        assert status["transformer_cache"] == "static"
        entry = status["resolved"]["transformer_cache"]
        assert entry["value"] == "static" and entry["requested"] == "static"
        assert entry["status"] == "applied" and entry["reason"] == "requested"
        assert isinstance(pipe.transformer.__dict__["forward"], ss.StaticStepSkip)
        # Never the FBCache marker, which is what would run a CUDA graph eager.
        assert getattr(pipe.transformer, "_unsloth_step_cache", None) is None
        assert pipe.transformer.cache_config is None
        assert status["transformer_cache_stats"]["mode"] == "taylor1"
    else:
        assert status["transformer_cache"] is None
        assert status["transformer_cache_stats"] is None
        assert "forward" not in pipe.transformer.__dict__
    backend.unload()


def test_fbcache_load_still_reports_an_active_cache(loop_runtime, monkeypatch):
    seen = _record_speed(monkeypatch)
    backend = VideoBackend()
    status = backend.load_pipeline(
        WAN_5B, model_kind = "pipeline", speed_mode = "default", transformer_cache = "fbcache"
    )
    assert status["transformer_cache"] == "fbcache"
    assert [kw["cache_active"] for kw in seen] == [True]
    assert status["transformer_cache_stats"] is None
    backend.unload()


@pytest.mark.parametrize("request_cache", [None, "auto"])
def test_auto_never_installs_static(loop_runtime, monkeypatch, request_cache):
    import core.inference.video as video_mod

    monkeypatch.setattr(
        video_mod, "install_static_step_skip", lambda *a, **k: pytest.fail("auto picked static")
    )
    backend = VideoBackend()
    status = backend.load_pipeline(WAN_5B, model_kind = "pipeline", transformer_cache = request_cache)
    assert status["transformer_cache"] == "fbcache"
    backend.unload()


def test_two_expert_moe_declines_static_with_a_reason(loop_runtime, monkeypatch):
    import core.inference.video as video_mod

    seen = _record_speed(monkeypatch)
    monkeypatch.setattr(
        video_mod, "install_static_step_skip", lambda *a, **k: pytest.fail("static on an MoE")
    )
    backend = VideoBackend()
    status = backend.load_pipeline(
        WAN_A14B, model_kind = "pipeline", speed_mode = "default", transformer_cache = "static"
    )
    assert status["transformer_cache"] is None
    entry = status["resolved"]["transformer_cache"]
    assert entry["requested"] == "static" and entry["value"] == "off"
    assert entry["status"] == "unsupported"
    assert "transformer_2" in entry["reason"]
    # Uncached on both experts, and the speed layers see an uncached load.
    pipe = loop_runtime["pipe"]
    assert pipe.transformer.cache_config is None and pipe.transformer_2.cache_config is None
    assert all(kw["cache_active"] is False for kw in seen)
    backend.unload()


def test_joint_audio_video_family_declines_static_with_a_reason(
    fake_runtime, tmp_path, monkeypatch
):
    import core.inference.video as video_mod

    monkeypatch.setattr(
        video_mod, "install_static_step_skip", lambda *a, **k: pytest.fail("static on LTX-2")
    )
    (tmp_path / "model.gguf").write_bytes(b"weights")
    backend = VideoBackend()
    status = backend.load_pipeline(
        str(tmp_path),
        gguf_filename = "model.gguf",
        base_repo = "Lightricks/LTX-2",
        family_override = "ltx-2",
        transformer_cache = "static",
    )
    assert status["transformer_cache"] is None
    entry = status["resolved"]["transformer_cache"]
    assert entry["requested"] == "static" and entry["status"] == "unsupported"
    assert "audio" in entry["reason"]
    backend.unload()


def test_a_transformer_without_forward_runs_uncached_with_a_reason(fake_runtime):
    # The stock Wan fake has no forward, which is what install_static_step_skip refuses.
    backend = VideoBackend()
    status = backend.load_pipeline(WAN_5B, model_kind = "pipeline", transformer_cache = "static")
    assert status["transformer_cache"] is None
    entry = status["resolved"]["transformer_cache"]
    assert entry["status"] == "unsupported" and "unavailable" in entry["reason"]
    backend.unload()


# ── generate ────────────────────────────────────────────────────────────────────────
def _static_backend(loop_runtime):
    backend = VideoBackend()
    backend.load_pipeline(WAN_5B, model_kind = "pipeline", transformer_cache = "static")
    return backend, loop_runtime["pipe"]


def test_generate_skips_the_scheduled_steps_per_cfg_branch(loop_runtime):
    backend, pipe = _static_backend(loop_runtime)
    backend.generate(prompt = "a sloth", steps = STEPS)
    computed = 2 * (STEPS - SKIPPED_PER_BRANCH)
    assert pipe.transformer.computed == computed
    stats = backend.status()["transformer_cache_stats"]
    assert stats["planned_skips"] == 0  # disarmed after the clip...
    # ...but the clip's own counts survive the post-render reset.
    assert stats["stats"] == {
        "calls": 2 * STEPS,
        "computed": computed,
        "skipped": 2 * SKIPPED_PER_BRANCH,
    }
    # A second clip re-arms the schedule.
    backend.generate(prompt = "a sloth", steps = STEPS)
    assert pipe.transformer.computed == 2 * computed
    # Under the bar every step computes.
    backend.generate(prompt = "a sloth", steps = 8)
    assert pipe.transformer.computed == 2 * computed + 16
    backend.unload()
    assert "forward" not in pipe.transformer.__dict__
    assert "cache_context" not in pipe.transformer.__dict__


def test_generate_arms_with_the_step_callback_and_disarms_after(loop_runtime, monkeypatch):
    import core.inference.video as video_mod

    backend, pipe = _static_backend(loop_runtime)
    armed, marks = [], []
    real_reset, real_mark = video_mod.reset_static_step_skip, video_mod.mark_step_end
    monkeypatch.setattr(
        video_mod,
        "reset_static_step_skip",
        lambda p, steps, **k: armed.append((steps, k.get("step_signal")))
        or real_reset(p, steps, **k),
    )
    monkeypatch.setattr(video_mod, "mark_step_end", lambda p: marks.append(p) or real_mark(p))
    resets = []
    monkeypatch.setattr(
        type(backend), "_reset_step_cache", staticmethod(lambda p: resets.append(p))
    )
    backend.generate(prompt = "a sloth", steps = STEPS)
    assert armed == [(STEPS, True), (None, None)]
    assert len(marks) == STEPS and all(p is pipe for p in marks)
    # The FBCache reset is not what a static clip runs.
    assert resets == []
    backend.unload()


def test_a_failed_clip_drops_the_kept_outputs(loop_runtime):
    backend, pipe = _static_backend(loop_runtime)
    layer = pipe.transformer.__dict__["forward"]
    pipe.raise_at = 20
    with pytest.raises(RuntimeError, match = "denoise failed"):
        backend.generate(prompt = "a sloth", steps = STEPS)
    assert layer.history == {} and layer.plan == ()
    # The next clip runs a full schedule from a clean slate.
    pipe.raise_at = None
    before = pipe.transformer.computed
    backend.generate(prompt = "a sloth", steps = STEPS)
    assert pipe.transformer.computed - before == 2 * (STEPS - SKIPPED_PER_BRANCH)
    backend.unload()


def test_hv15_counts_branches_by_context_without_a_step_callback(loop_runtime, monkeypatch):
    import core.inference.video as video_mod

    armed = []
    real_reset = video_mod.reset_static_step_skip
    monkeypatch.setattr(
        video_mod,
        "reset_static_step_skip",
        lambda p, steps, **k: armed.append((steps, k.get("step_signal")))
        or real_reset(p, steps, **k),
    )
    backend = VideoBackend()
    status = backend.load_pipeline(HV15, model_kind = "pipeline", transformer_cache = "static")
    assert status["transformer_cache"] == "static"
    pipe = loop_runtime["pipe"]
    backend.generate(prompt = "a fox", steps = STEPS, num_frames = 9)
    assert "callback_on_step_end" not in pipe.last_kwargs
    assert armed[0] == (STEPS, False)
    assert pipe.transformer.computed == 2 * (STEPS - SKIPPED_PER_BRANCH)
    assert set(pipe.transformer.contexts) == {"pred_cond", "pred_uncond"}
    backend.unload()


# ── modular workflow ────────────────────────────────────────────────────────────────
def _load_h3(backend, transformer_cache):
    from core.inference.video import _detect_load_family

    diffusers = sys.modules["diffusers"]
    diffusers.ComponentsManager = _FakeComponentsManager
    diffusers.ModularPipeline = _FakeModularPipeline
    fam = _detect_load_family("MiniMaxAI/MiniMax-H3", None, "minimax-h3")
    return backend._load_h3_modular_pipeline(
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
        _load_token = None,
        _base_local_dir = None,
        transformer_cache = transformer_cache,
    )


def test_modular_workflow_reports_a_static_ask_as_unsupported(fake_runtime):
    backend = VideoBackend()
    status = _load_h3(backend, "static")
    entry = status["resolved"]["transformer_cache"]
    assert status["transformer_cache"] is None
    assert entry["requested"] == "static" and entry["value"] == "off"
    assert entry["status"] == "unsupported" and "modular" in entry["reason"]
    backend.unload()


@pytest.mark.parametrize("request_cache", [None, "off", "fbcache", "bogus"])
def test_modular_workflow_keeps_its_record_for_every_other_ask(fake_runtime, request_cache):
    backend = VideoBackend()
    status = _load_h3(backend, request_cache)
    entry = status["resolved"]["transformer_cache"]
    assert entry["requested"] is None and entry["value"] == "off"
    assert entry["reason"] == "not supported by this modular workflow"
    backend.unload()


# ── API ─────────────────────────────────────────────────────────────────────────────
def test_video_api_accepts_static_and_reports_its_stats():
    from pydantic import ValidationError

    from models.inference import VideoLoadRequest, VideoStatusResponse

    assert (
        VideoLoadRequest(model_path = "org/model", transformer_cache = "static").transformer_cache
        == "static"
    )
    with pytest.raises(ValidationError):
        VideoLoadRequest(model_path = "org/model", transformer_cache = "magic")
    stats = {"mode": "taylor1", "every": 2, "stats": {"calls": 100, "computed": 66, "skipped": 34}}
    status = VideoStatusResponse(
        loaded = True, transformer_cache = "static", transformer_cache_stats = stats
    )
    assert status.transformer_cache_stats == stats
    assert VideoStatusResponse(loaded = False).transformer_cache_stats is None


def test_static_is_a_graph_keeping_mode():
    assert dcache.normalize_transformer_cache("static") == dcache.TC_STATIC
    assert dcache.cache_breaks_graph(dcache.TC_STATIC) is False
    assert dcache.cache_breaks_graph(dcache.TC_FBCACHE) is True
