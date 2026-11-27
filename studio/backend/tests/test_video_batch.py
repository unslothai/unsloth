# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Queued Wan clips that share one denoise (video_batch): every clip is its own solo render, a neighbour
never changes it, each clip decodes alone, a Stop is its own clip's, an out-of-memory batch renders its
clips one at a time, and the queue hands over only the compatible clips right behind its head.

The numeric tests run a tiny real Diffusers Wan pipeline on the CPU (random weights, a word tokenizer),
through the real VideoBackend; they skip where this Diffusers has no Wan pipeline."""

from __future__ import annotations

import asyncio
import pickle
import threading
import time
import types

import pytest

import core.inference.video as video_module
from core.inference import video_batch, video_gallery
from core.inference.video_families import (
    VIDEO_CANCELLED_MSG,
    VIDEO_GENERATION_BUSY_MSG,
    detect_video_family,
)

_SHAPE = dict(width = 32, height = 32, num_frames = 5, steps = 3, guidance = 4.0, fps = 16)


def _wan_classes():
    try:
        import torch  # noqa: F401
        from diffusers import (
            AutoencoderKLWan,
            UniPCMultistepScheduler,
            WanPipeline,
            WanTransformer3DModel,
        )
        from transformers import T5Config, T5EncoderModel
    except Exception as exc:  # noqa: BLE001 -- an old or broken Diffusers: no Wan pipeline to run
        pytest.skip(f"no Diffusers Wan pipeline here ({type(exc).__name__})")
    return (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
        T5Config,
        T5EncoderModel,
    )


class _WordTokenizer:
    """Enough of a tokenizer for WanPipeline: one id per word, right-padded to ``max_length``."""

    model_max_length = 512

    def __call__(
        self,
        texts,
        *,
        padding = None,
        max_length = 16,
        **_kwargs,
    ):
        import torch

        texts = [texts] if isinstance(texts, str) else list(texts)
        ids = torch.zeros(len(texts), max_length, dtype = torch.long)
        mask = torch.zeros(len(texts), max_length, dtype = torch.long)
        for row, text in enumerate(texts):
            words = [sum(map(ord, word)) % 90 + 5 for word in text.split()][: max_length - 1] + [1]
            ids[row, : len(words)] = torch.tensor(words)
            mask[row, : len(words)] = 1
        return types.SimpleNamespace(input_ids = ids, attention_mask = mask)


def _tiny_wan(two_experts: bool = False):
    import torch

    (
        AutoencoderKLWan,
        UniPCMultistepScheduler,
        WanPipeline,
        WanTransformer3DModel,
        T5Config,
        T5EncoderModel,
    ) = _wan_classes()

    def transformer(seed):
        torch.manual_seed(seed)
        return WanTransformer3DModel(
            patch_size = (1, 2, 2),
            num_attention_heads = 2,
            attention_head_dim = 12,
            in_channels = 16,
            out_channels = 16,
            text_dim = 32,
            freq_dim = 256,
            ffn_dim = 32,
            num_layers = 2,
            cross_attn_norm = True,
            qk_norm = "rms_norm_across_heads",
            rope_max_seq_len = 32,
        ).eval()

    torch.manual_seed(0)
    vae = AutoencoderKLWan(
        base_dim = 3,
        z_dim = 16,
        dim_mult = [1, 1, 1, 1],
        num_res_blocks = 1,
        temperal_downsample = [False, True, True],
    ).eval()
    torch.manual_seed(0)
    text_encoder = T5EncoderModel(
        T5Config(
            vocab_size = 100,
            d_model = 32,
            d_kv = 8,
            d_ff = 37,
            num_layers = 2,
            num_heads = 4,
            relative_attention_num_buckets = 8,
        )
    ).eval()
    scheduler = UniPCMultistepScheduler(
        prediction_type = "flow_prediction", use_flow_sigmas = True, flow_shift = 3.0
    )
    extra = {}
    if two_experts:
        # The first step (t = 1000) is high noise, the rest low noise.
        extra = {"transformer_2": transformer(1), "boundary_ratio": 0.95}
    return WanPipeline(
        transformer = transformer(0),
        vae = vae,
        scheduler = scheduler,
        text_encoder = text_encoder,
        tokenizer = _WordTokenizer(),
        **extra,
    )


@pytest.fixture
def saved(monkeypatch):
    """Clips the backend saved, by id: (frames, recipe)."""
    clips: dict = {}

    def _save(
        mp4,
        meta,
        video_id = None,
    ):
        clips[video_id] = (pickle.loads(mp4), dict(meta))
        return {"id": video_id}

    monkeypatch.setattr(video_gallery, "save", _save)
    monkeypatch.setattr(video_gallery, "record_job_outcome", lambda *a, **k: None)
    return clips


def _backend(
    monkeypatch,
    *,
    two_experts = False,
    family = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
):
    monkeypatch.setenv(video_batch.ENV, "force")
    from core.inference import diffusion_attention

    # The real CPU SDPA probe answers are this test's own: a later fake-torch test would read them back.
    monkeypatch.setattr(diffusion_attention, "_SDPA_PROBE_CACHE", {})
    # The tiny canvas is far below the trained sizes the shape check enforces.
    monkeypatch.setattr(video_module, "validate_video_request_shape", lambda *a, **k: None)
    backend = video_module.VideoBackend()
    pipe = _tiny_wan(two_experts)
    backend._state = video_module._VideoLoadState(
        pipe = pipe,
        family = detect_video_family(family),
        repo_id = "tiny/wan",
        base_repo = "tiny/wan",
        device = "cpu",
        dtype = "float32",
        kind = "pipeline",
    )
    # The frames themselves stand in for the mp4, so a comparison is exact.
    backend._encode_mp4 = lambda frames, fps, audio, pipe, **kw: pickle.dumps(frames)
    return backend


def _wait_idle(backend, timeout = 120.0) -> None:
    deadline = time.monotonic() + timeout
    # A single clip hands its slot on before its encode (_hand_off_encode); it is saved once its tail ends.
    while backend._generate_job_active or backend._encode_tail is not None:
        assert time.monotonic() < deadline, backend._gen
        time.sleep(0.02)


def _solo(backend, prompt, seed, video_id, **extra):
    backend.begin_generate(prompt = prompt, seed = seed, video_id = video_id, **_SHAPE, **extra)
    _wait_idle(backend)


def _batch(backend, rows, **extra):
    """rows: (prompt, seed, video_id); the first starts the job, the rest join it."""
    (prompt, seed, video_id), *rest = rows
    backend.begin_generate(
        prompt = prompt, seed = seed, video_id = video_id, batch_open = True, **_SHAPE, **extra
    )
    for prompt, seed, video_id in rest:
        backend.begin_generate(
            prompt = prompt, seed = seed, video_id = video_id, join = True, **_SHAPE, **extra
        )
    assert backend.seal_batch() == len(rows)
    _wait_idle(backend)


def _frames(saved, video_id):
    import numpy as np
    return np.asarray(saved[video_id][0])


# ---- numerics on a tiny real pipeline ----------------------------------------------------------


def test_each_batched_clip_is_byte_identical_to_its_solo_render(monkeypatch, saved):
    backend = _backend(monkeypatch)
    _solo(backend, "a cat on a sofa", 11, "solo-cat")
    _solo(backend, "a dog in the snow", 22, "solo-dog")
    decodes: list = []
    vae = backend._state.pipe.vae
    real_decode = type(vae).decode

    def _decode(self, z, *args, **kwargs):
        decodes.append(int(z.shape[0]))
        return real_decode(self, z, *args, **kwargs)

    monkeypatch.setattr(type(vae), "decode", _decode)
    rows_seen: list = []
    real_forward = type(backend._state.pipe.transformer).forward

    def _forward(self, hidden_states, *args, **kwargs):
        rows_seen.append(int(hidden_states.shape[0]))
        return real_forward(self, hidden_states, *args, **kwargs)

    monkeypatch.setattr(type(backend._state.pipe.transformer), "forward", _forward)
    encoder_rows: list = []
    backend._state.pipe.text_encoder.register_forward_pre_hook(
        lambda module, args: encoder_rows.append(int(args[0].shape[0]))
    )
    _batch(backend, [("a cat on a sofa", 11, "b-cat"), ("a dog in the snow", 22, "b-dog")])
    # The text encoder saw one prompt per call (each clip's prompt and negative prompt), as in a solo render.
    assert encoder_rows == [1, 1, 1, 1]

    assert (_frames(saved, "b-cat") == _frames(saved, "solo-cat")).all()
    assert (_frames(saved, "b-dog") == _frames(saved, "solo-dog")).all()
    assert (_frames(saved, "solo-cat") != _frames(saved, "solo-dog")).any()
    # One denoiser call per step and guidance branch holds both clips; each clip decodes alone.
    assert rows_seen and set(rows_seen) == {2}
    assert decodes == [1, 1]
    assert saved["b-dog"][1]["prompt"] == "a dog in the snow"
    assert saved["b-dog"][1]["seed"] == 22
    assert backend.generate_outcome("b-dog", video_module.current_account_id())["phase"] == (
        "completed"
    )


@pytest.mark.parametrize("target_row", [0, 1, 2])
def test_a_neighbours_prompt_and_seed_never_change_a_clip(monkeypatch, saved, target_row):
    """Fixed execution shape (three rows); only the neighbours change."""
    backend = _backend(monkeypatch)
    target = ("a red kite over a beach", 7)

    def rows(tag, neighbours):
        out = list(neighbours)
        out.insert(target_row, target)
        return [(prompt, seed, f"{tag}-{i}") for i, (prompt, seed) in enumerate(out)]

    _batch(backend, rows("x", [("a cat", 1), ("a dog", 2)]))
    _batch(backend, rows("y", [("a long winding mountain road at dusk", 99), ("rain", 3)]))
    assert (_frames(saved, f"x-{target_row}") == _frames(saved, f"y-{target_row}")).all()


def test_two_experts_switch_together_and_each_clip_matches_its_solo_render(monkeypatch, saved):
    backend = _backend(monkeypatch, two_experts = True, family = "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    pipe = backend._state.pipe
    seen: list = []
    for name in ("transformer", "transformer_2"):
        module = getattr(pipe, name)
        real = module.forward

        def _forward(
            hidden_states,
            *args,
            _real = real,
            _name = name,
            **kwargs,
        ):
            seen.append((_name, int(hidden_states.shape[0])))
            return _real(hidden_states, *args, **kwargs)

        module.forward = _forward
    _solo(backend, "a cat", 5, "solo-a", guidance_2 = 3.0)
    _solo(backend, "a dog", 6, "solo-b", guidance_2 = 3.0)
    seen.clear()
    _batch(backend, [("a cat", 5, "b-a"), ("a dog", 6, "b-b")], guidance_2 = 3.0)
    experts = [name for name, _ in seen]
    # Both experts ran, the high-noise one first, and every call held both clips.
    assert "transformer" in experts and "transformer_2" in experts
    assert experts.index("transformer_2") > experts.index("transformer")
    assert experts[experts.index("transformer_2") :].count("transformer") == 0
    assert {rows for _, rows in seen} == {2}
    assert (_frames(saved, "b-a") == _frames(saved, "solo-a")).all()
    assert (_frames(saved, "b-b") == _frames(saved, "solo-b")).all()


def test_an_out_of_memory_batch_renders_its_clips_one_at_a_time(monkeypatch, saved):
    backend = _backend(monkeypatch)
    _solo(backend, "a cat", 1, "solo-cat")
    _solo(backend, "a dog", 2, "solo-dog")
    pipe_cls = type(backend._state.pipe)
    real_call = pipe_cls.__call__
    calls: list = []

    def _call(self, *args, **kwargs):
        prompt = kwargs.get("prompt")
        calls.append(prompt)
        if isinstance(prompt, list) and len(prompt) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(pipe_cls, "__call__", _call)
    _batch(backend, [("a cat", 1, "b-cat"), ("a dog", None, "b-dog")])
    assert calls[0] == ["a cat", "a dog"] and calls[1:] == ["a cat", "a dog"]
    assert (_frames(saved, "b-cat") == _frames(saved, "solo-cat")).all()
    # Unseeded: the rerun replays the seed drawn for the batch, and the recipe returns it.
    assert set(saved) == {"solo-cat", "solo-dog", "b-cat", "b-dog"}
    assert isinstance(saved["b-dog"][1]["seed"], int)


def test_an_oom_retry_after_streaming_redraws_every_clips_own_noise(monkeypatch, saved):
    """A batch whose first attempt drew its noise and then ran out of memory is retried streamed: every row's
    generator is reseeded to its own clip's seed, so the retry still renders each clip's solo noise."""
    import torch
    from core.inference import video_stream_residency

    backend = _backend(monkeypatch)
    _solo(backend, "a cat", 1, "solo-cat")
    _solo(backend, "a dog", 2, "solo-dog")
    monkeypatch.setattr(video_stream_residency, "may_stream_after_oom", lambda pipe: True)
    streamed: list = []
    monkeypatch.setattr(
        video_stream_residency,
        "stream_after_oom",
        lambda pipe, logger = None: streamed.append(1) or True,
    )
    pipe_cls = type(backend._state.pipe)
    real_call = pipe_cls.__call__
    calls: list = []

    def _call(self, *args, **kwargs):
        calls.append(kwargs.get("prompt"))
        generator = kwargs.get("generator")
        if len(calls) == 1 and isinstance(generator, list):
            # The failed attempt consumed noise from every row's generator before running out of memory.
            for row_generator in generator:
                torch.randn(64, generator = row_generator)
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(pipe_cls, "__call__", _call)
    _batch(backend, [("a cat", 1, "b-cat"), ("a dog", 2, "b-dog")])
    # Retried once as the same batch, not split into solo renders.
    assert calls == [["a cat", "a dog"], ["a cat", "a dog"]] and streamed == [1]
    assert (_frames(saved, "b-cat") == _frames(saved, "solo-cat")).all()
    assert (_frames(saved, "b-dog") == _frames(saved, "solo-dog")).all()
    assert saved["b-dog"][1]["seed"] == 2


@pytest.mark.parametrize("path", ["batch", "serial", "failed"])
def test_a_rider_from_another_account_is_persisted_as_that_account(monkeypatch, saved, path):
    """The job's worker thread acts for the job's account; a rider queued by another account has its clip,
    recipe and job outcome written to its own account's storage on every path: the shared batch, the
    one at a time rerun after an out of memory batch, and a batch that failed."""
    from utils.account_context import AccountContext, bind_account, reset_account

    saved_as: dict = {}
    outcomes_as: dict = {}
    real_save = video_gallery.save

    def _save(
        mp4,
        meta,
        video_id = None,
    ):
        saved_as[video_id] = video_module.current_account_id()
        return real_save(mp4, meta, video_id = video_id)

    def _record(
        video_id,
        *,
        completed_at,
        error = None,
    ):
        outcomes_as[video_id] = (video_module.current_account_id(), error)

    monkeypatch.setattr(video_gallery, "save", _save)
    monkeypatch.setattr(video_gallery, "record_job_outcome", _record)
    backend = _backend(monkeypatch)
    if path != "batch":
        pipe_cls = type(backend._state.pipe)
        real_call = pipe_cls.__call__

        def _call(self, *args, **kwargs):
            prompt = kwargs.get("prompt")
            if isinstance(prompt, list) and len(prompt) > 1:
                if path == "serial":
                    raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
                raise RuntimeError("the denoiser broke")
            return real_call(self, *args, **kwargs)

        monkeypatch.setattr(pipe_cls, "__call__", _call)
    leader = video_module.current_account_id()
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "lead", batch_open = True, **_SHAPE)
    token = bind_account(AccountContext("acct-bob", "bob"))
    try:
        backend.begin_generate(prompt = "a dog", seed = 2, video_id = "ride", join = True, **_SHAPE)
    finally:
        reset_account(token)
    assert backend.seal_batch() == 2
    _wait_idle(backend)
    assert leader != "acct-bob"
    if path == "failed":
        assert set(saved_as) == set()
        assert outcomes_as["ride"][0] == "acct-bob" and outcomes_as["ride"][1]
        assert outcomes_as["lead"][0] == leader
        return
    assert saved_as == {"lead": leader, "ride": "acct-bob"}
    assert outcomes_as == {"lead": (leader, None), "ride": ("acct-bob", None)}
    assert backend.generate_outcome("ride", "acct-bob")["phase"] == "completed"


def test_a_compiled_and_graphed_denoiser_batches_eager_without_compiling_or_capturing(
    monkeypatch, saved
):
    pytest.importorskip("torch")
    backend = _backend(monkeypatch)
    pipe = backend._state.pipe
    compiles: list = []

    def _backend_fn(gm, example_inputs):
        compiles.append(len(example_inputs))
        return gm.forward

    for block in pipe.transformer.blocks:
        block.compile(backend = _backend_fn)

    class _Graphs:
        bypassed = False
        seen: list = []

        def set_bypass(self, on):
            self.bypassed = bool(on)
            self.seen.append(bool(on))
            return self

    graphs = _Graphs()
    pipe._unsloth_cuda_graphs = (graphs,)
    _solo(backend, "a cat", 1, "solo-cat")
    _solo(backend, "a dog", 2, "solo-dog")
    compiled_before = len(compiles)
    assert compiled_before > 0
    bypassed_during: list = []
    pipe.transformer.register_forward_pre_hook(
        lambda module, args: bypassed_during.append(
            (int(args[0].shape[0]) if args else None, graphs.bypassed)
        )
    )
    _batch(backend, [("a cat", 1, "b-cat"), ("a dog", 2, "b-dog")])
    # No compile for the batch shape; the graphs are bypassed while the batch denoises, restored after.
    assert len(compiles) == compiled_before
    assert bypassed_during and all(on for _, on in bypassed_during)
    assert graphs.bypassed is False
    assert (_frames(saved, "b-cat") == _frames(saved, "solo-cat")).all()
    assert (_frames(saved, "b-dog") == _frames(saved, "solo-dog")).all()


def test_auto_batches_a_guided_clip_on_a_low_vram_load(monkeypatch, saved):
    """Auto at the Wan default guidance (CFG above 1): the measured placement (Low VRAM, eager) batches."""
    backend = _backend(monkeypatch)
    monkeypatch.delenv(video_batch.ENV)
    object.__setattr__(backend._state, "memory_mode", "low_vram")
    object.__setattr__(backend._state, "offload_policy", "group")
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "l2", batch_open = True, **_SHAPE)
    backend.begin_generate(prompt = "a dog", seed = 2, video_id = "r2", join = True, **_SHAPE)
    assert backend.seal_batch() == 2
    _wait_idle(backend)
    assert {"l2", "r2"} <= set(saved)


def test_auto_step_cache_keeps_clips_apart(monkeypatch, saved):
    """FBCache's residual test spans the whole batch: Auto's cache refuses a join at its step count, and a cache
    that engages anyway renders the clips one at a time, each its own solo render."""
    backend = _backend(monkeypatch)
    engaged = {"mode": None}
    monkeypatch.setattr(video_module, "maybe_toggle_step_cache", lambda view, **kw: engaged["mode"])
    monkeypatch.setattr(
        "core.inference.diffusion_autoopt.runtime.auto_cache_steps",
        lambda base, runner, steps: (steps, None),
    )
    monkeypatch.setattr(backend, "_reset_step_cache", lambda pipe: None)
    object.__setattr__(backend._state, "cache_auto", True)
    reached, release = _hold_steps(monkeypatch, backend)
    long = {**_SHAPE, "steps": 20}
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "lead", batch_open = True, **long)
    with pytest.raises(RuntimeError, match = VIDEO_GENERATION_BUSY_MSG):
        backend.begin_generate(prompt = "a dog", seed = 2, video_id = "ride", join = True, **long)
    backend.seal_batch()
    release.set()
    _wait_idle(backend)
    _solo(backend, "a cat", 1, "solo-cat")
    _solo(backend, "a dog", 2, "solo-dog")
    # Below Auto's step count the clips join, then the request-time toggle engages a cache anyway.
    engaged["mode"] = "fbcache"
    calls: list = []
    pipe_cls = type(backend._state.pipe)
    real_call = pipe_cls.__call__

    def _call(self, *args, **kwargs):
        calls.append(kwargs.get("prompt"))
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(pipe_cls, "__call__", _call)
    _batch(backend, [("a cat", 1, "b-cat"), ("a dog", 2, "b-dog")])
    # No batched denoise ran: the clips rendered one at a time.
    assert calls == ["a cat", "a dog"]
    assert (_frames(saved, "b-cat") == _frames(saved, "solo-cat")).all()
    assert (_frames(saved, "b-dog") == _frames(saved, "solo-dog")).all()


# ---- one clip's Stop, progress and outcome -------------------------------------------------------


def _hold_steps(monkeypatch, backend):
    """Pause the shared denoise after its first step until released."""
    release = threading.Event()
    reached = threading.Event()
    pipe = backend._state.pipe
    real = pipe.scheduler.step

    def _step(*args, **kwargs):
        out = real(*args, **kwargs)
        reached.set()
        release.wait(30)
        return out

    monkeypatch.setattr(pipe.scheduler, "step", _step)
    return reached, release


def _start_pair(backend):
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "lead", batch_open = True, **_SHAPE)
    backend.begin_generate(prompt = "a dog", seed = 2, video_id = "ride", join = True, **_SHAPE)
    backend.seal_batch()


def test_a_riders_stop_settles_only_its_clip_at_once(monkeypatch, saved):
    backend = _backend(monkeypatch)
    reached, release = _hold_steps(monkeypatch, backend)
    _start_pair(backend)
    assert reached.wait(60)
    account = video_module.current_account_id()
    live = backend.member_progress("ride", account)
    assert live["active"] is True and live["video_id"] == "ride" and "preview" not in live
    assert backend.cancel_generate(expected_video_id = "ride", expected_account = account)
    # Settled before the shared denoise ends.
    assert backend.generate_outcome("ride", account)["error"] == VIDEO_CANCELLED_MSG
    assert backend._generate_job_active
    release.set()
    _wait_idle(backend)
    assert set(saved) == {"lead"}
    assert backend.generate_outcome("lead", account)["phase"] == "completed"


def test_the_jobs_own_stop_keeps_its_riders_rendering(monkeypatch, saved):
    backend = _backend(monkeypatch)
    reached, release = _hold_steps(monkeypatch, backend)
    _start_pair(backend)
    assert reached.wait(60)
    account = video_module.current_account_id()
    assert backend.cancel_generate(expected_video_id = "lead", expected_account = account)
    assert backend.progress_for_video("lead")["error"] == VIDEO_CANCELLED_MSG
    release.set()
    _wait_idle(backend)
    assert set(saved) == {"ride"}
    assert backend.generate_outcome("lead", account)["error"] == VIDEO_CANCELLED_MSG


def test_the_jobs_own_stop_before_the_seal_keeps_its_rider_and_refuses_new_ones(monkeypatch, saved):
    backend = _backend(monkeypatch)
    account = video_module.current_account_id()
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "lead", batch_open = True, **_SHAPE)
    backend.begin_generate(prompt = "a dog", seed = 2, video_id = "ride", join = True, **_SHAPE)
    assert backend.cancel_generate(expected_video_id = "lead", expected_account = account)
    with pytest.raises(RuntimeError, match = VIDEO_GENERATION_BUSY_MSG):
        backend.begin_generate(prompt = "a bird", seed = 3, video_id = "late", join = True, **_SHAPE)
    backend.seal_batch()
    _wait_idle(backend)
    assert set(saved) == {"ride"}
    assert backend.generate_outcome("ride", account)["phase"] == "completed"
    assert backend.generate_outcome("lead", account)["error"] == VIDEO_CANCELLED_MSG


def test_everyone_stopping_stops_the_denoise(monkeypatch, saved):
    backend = _backend(monkeypatch)
    reached, release = _hold_steps(monkeypatch, backend)
    _start_pair(backend)
    assert reached.wait(60)
    account = video_module.current_account_id()
    assert backend.cancel_generate(expected_video_id = "ride", expected_account = account)
    assert backend.cancel_generate(expected_video_id = "lead", expected_account = account)
    release.set()
    _wait_idle(backend)
    assert saved == {}
    assert backend._gen.get("error") == VIDEO_CANCELLED_MSG


# ---- who may join ---------------------------------------------------------------------------------


def test_join_is_refused_without_a_gathering_batch_or_with_another_shape(monkeypatch, saved):
    backend = _backend(monkeypatch)
    with pytest.raises(RuntimeError, match = VIDEO_GENERATION_BUSY_MSG):
        backend.begin_generate(prompt = "x", video_id = "j0", join = True, **_SHAPE)
    reached, release = _hold_steps(monkeypatch, backend)
    backend.begin_generate(prompt = "a", seed = 1, video_id = "lead", batch_open = True, **_SHAPE)
    for change in ({"num_frames": 9}, {"steps": 4}, {"guidance": 2.0}, {"width": 64}):
        with pytest.raises(RuntimeError, match = VIDEO_GENERATION_BUSY_MSG):
            backend.begin_generate(prompt = "b", video_id = "j1", join = True, **{**_SHAPE, **change})
    backend.seal_batch()
    with pytest.raises(RuntimeError, match = VIDEO_GENERATION_BUSY_MSG):
        backend.begin_generate(prompt = "c", video_id = "j2", join = True, **_SHAPE)
    release.set()
    _wait_idle(backend)
    assert set(saved) == {"lead"}


def test_a_single_clip_never_waits_or_forms_a_batch(monkeypatch, saved):
    backend = _backend(monkeypatch)
    sealed: list = []
    monkeypatch.setattr(backend, "_seal_batch_rows", lambda *a, **k: sealed.append(a) or [])
    started = time.monotonic()
    _solo(backend, "a cat", 1, "solo")
    assert sealed == [] and backend._forming is None
    assert time.monotonic() - started < video_batch.SEAL_TIMEOUT_S


def test_a_gathering_batch_nobody_joined_renders_the_single_clip(monkeypatch, saved):
    backend = _backend(monkeypatch)
    _solo(backend, "a cat", 1, "solo")
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "alone", batch_open = True, **_SHAPE)
    assert backend.seal_batch() == 1
    _wait_idle(backend)
    assert (_frames(saved, "alone") == _frames(saved, "solo")).all()


# ---- the job slot around the encode (#19817's handoff) ---------------------------------------------


def _watch_encodes(backend, monkeypatch):
    """Record, per encode, whether the generate lock and the job slot were held, and every slot handoff."""
    encodes: list = []
    handoffs: list = []
    inner = backend._encode_mp4

    def _encode(frames, fps, audio, pipe, **kw):
        encodes.append(
            {
                "locked": backend._generate_lock.locked(),
                "slot_busy": backend._generate_job_active,
                "tail": backend._encode_tail is not None,
            }
        )
        return inner(frames, fps, audio, pipe, **kw)

    backend._encode_mp4 = _encode
    real_hand_off = backend._hand_off_encode

    def _hand_off(*args, **kwargs):
        handed = real_hand_off(*args, **kwargs)
        handoffs.append(handed)
        return handed

    monkeypatch.setattr(backend, "_hand_off_encode", _hand_off)
    monkeypatch.delenv("UNSLOTH_VIDEO_ENCODE_OUTSIDE_LOCK", raising = False)
    return encodes, handoffs


def test_a_batched_job_keeps_its_slot_until_every_clip_is_saved(monkeypatch, saved):
    backend = _backend(monkeypatch)
    encodes, handoffs = _watch_encodes(backend, monkeypatch)
    _batch(backend, [("a cat", 1, "lead"), ("a dog", 2, "ride")])
    assert set(saved) == {"lead", "ride"}
    assert saved["lead"][1]["seed"] == 1 and saved["ride"][1]["seed"] == 2
    # Never handed on: the next queued clip starts only after both clips are saved and settled.
    assert handoffs == [] and backend._encode_tail is None
    # A shared denoise encodes every clip in the job, under the lock, as before #19817.
    assert encodes == [{"locked": True, "slot_busy": True, "tail": False}] * 2
    account = video_module.current_account_id()
    assert backend.generate_outcome("lead", account)["phase"] == "completed"
    assert backend.generate_outcome("ride", account)["phase"] == "completed"
    assert backend._forming is None


def test_a_gathering_batch_nobody_joined_hands_its_slot_on_like_a_single_clip(monkeypatch, saved):
    backend = _backend(monkeypatch)
    encodes, handoffs = _watch_encodes(backend, monkeypatch)
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "alone", batch_open = True, **_SHAPE)
    assert backend.seal_batch() == 1
    _wait_idle(backend)
    assert set(saved) == {"alone"}
    assert handoffs == [True]
    # The encode ran after the lock, with the slot already free for the next clip.
    assert encodes == [{"locked": False, "slot_busy": False, "tail": True}]
    assert backend._forming is None and backend._job_cancel is None
    assert (
        backend.generate_outcome("alone", video_module.current_account_id())["phase"] == "completed"
    )


def test_a_job_whose_rider_stopped_before_the_seal_keeps_its_slot_and_records_the_rider(
    monkeypatch, saved
):
    backend = _backend(monkeypatch)
    encodes, handoffs = _watch_encodes(backend, monkeypatch)
    recorded: list = []
    monkeypatch.setattr(
        video_gallery, "record_job_outcome", lambda video_id, **kw: recorded.append((video_id, kw))
    )
    account = video_module.current_account_id()
    backend.begin_generate(prompt = "a cat", seed = 1, video_id = "lead", batch_open = True, **_SHAPE)
    backend.begin_generate(prompt = "a dog", seed = 2, video_id = "ride", join = True, **_SHAPE)
    assert backend.cancel_generate(expected_video_id = "ride", expected_account = account)
    backend.seal_batch()
    _wait_idle(backend)
    assert set(saved) == {"lead"}
    # The leader renders alone and encodes after the lock, but the job keeps its slot, so its end
    # settles and persists the stopped rider.
    assert handoffs == [] and encodes == [{"locked": False, "slot_busy": True, "tail": False}]
    assert [video_id for video_id, kw in recorded if kw.get("error") == VIDEO_CANCELLED_MSG] == [
        "ride"
    ]
    assert backend.generate_outcome("ride", account)["error"] == VIDEO_CANCELLED_MSG


def test_an_oom_batch_encodes_each_clip_after_the_lock_without_handing_off(monkeypatch, saved):
    backend = _backend(monkeypatch)
    encodes, handoffs = _watch_encodes(backend, monkeypatch)
    pipe_cls = type(backend._state.pipe)
    real_call = pipe_cls.__call__

    def _call(self, *args, **kwargs):
        prompt = kwargs.get("prompt")
        if isinstance(prompt, list) and len(prompt) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.00 GiB")
        return real_call(self, *args, **kwargs)

    monkeypatch.setattr(pipe_cls, "__call__", _call)
    _batch(backend, [("a cat", 1, "lead"), ("a dog", 2, "ride")])
    assert set(saved) == {"lead", "ride"}
    assert saved["ride"][1]["seed"] == 2
    assert handoffs == []
    assert encodes == [{"locked": False, "slot_busy": True, "tail": False}] * 2


# ---- which loads batch ------------------------------------------------------------------------------


def _state(**fields):
    base = dict(
        family = detect_video_family("Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        engine = "diffusers",
        transformer_cache = None,
        vram_split_cards = None,
        offload_policy = "group",
        memory_mode = "low_vram",
        denoiser_streamed = False,
        pipe = types.SimpleNamespace(),
    )
    base.update(fields)
    return types.SimpleNamespace(**base)


def test_auto_batches_a_streamed_wan_load_and_keeps_the_rest_serial(monkeypatch):
    monkeypatch.delenv(video_batch.ENV, raising = False)
    assert video_batch.load_decline_reason(_state()) is None
    assert "Low VRAM" in video_batch.load_decline_reason(
        _state(offload_policy = "none", memory_mode = "auto")
    )
    # The default group offload keeps most groups on the card: measured no faster batched.
    assert "Low VRAM" in video_batch.load_decline_reason(_state(memory_mode = "auto"))
    # Unmeasured tiers stay serial.
    assert "Low VRAM" in video_batch.load_decline_reason(
        _state(memory_mode = "auto", offload_policy = "streaming")
    )
    assert "step cache" in video_batch.load_decline_reason(_state(transformer_cache = "fbcache"))
    assert "family" in video_batch.load_decline_reason(
        _state(family = detect_video_family("Lightricks/LTX-2"))
    )
    monkeypatch.setattr(video_batch, "eager_batch_available", lambda: False)
    assert "CUDA graphs" in video_batch.load_decline_reason(
        _state(pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (object(),)))
    )
    monkeypatch.setattr(video_batch, "eager_batch_available", lambda: True)
    # Auto keeps a graph-captured load serial (measured faster); force batches it eager.
    assert "CUDA graphs" in video_batch.load_decline_reason(
        _state(pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (object(),)))
    )
    monkeypatch.setenv(video_batch.ENV, "force")
    assert (
        video_batch.load_decline_reason(
            _state(pipe = types.SimpleNamespace(_unsloth_cuda_graphs = (object(),)))
        )
        is None
    )
    monkeypatch.delenv(video_batch.ENV)
    monkeypatch.setenv(video_batch.ENV, "off")
    assert video_batch.load_decline_reason(_state()) == f"{video_batch.ENV}=off"
    monkeypatch.setenv(video_batch.ENV, "force")
    assert video_batch.load_decline_reason(_state(offload_policy = "none")) is None


def test_a_compiled_denoiser_or_whole_call_activation_scales_render_alone(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(video_batch.ENV, "force")
    compiled = torch.nn.Linear(2, 2)
    compiled.compile()
    monkeypatch.setattr(video_batch, "eager_batch_available", lambda: False)
    assert "compiled" in video_batch.load_decline_reason(
        _state(pipe = types.SimpleNamespace(transformer = compiled))
    )
    monkeypatch.setattr(video_batch, "eager_batch_available", lambda: True)
    assert (
        video_batch.load_decline_reason(_state(pipe = types.SimpleNamespace(transformer = compiled)))
        is None
    )
    monkeypatch.delenv(video_batch.ENV)
    assert "compiled" in video_batch.load_decline_reason(
        _state(pipe = types.SimpleNamespace(transformer = compiled))
    )

    class DynamicNVFP4Linear(torch.nn.Linear):
        pass

    holder = torch.nn.Sequential(DynamicNVFP4Linear(2, 2))
    assert "activation scales" in video_batch.load_decline_reason(
        _state(pipe = types.SimpleNamespace(transformer = holder))
    )


def test_a_sampler_that_renoises_every_step_renders_alone(monkeypatch):
    monkeypatch.setenv(video_batch.ENV, "force")

    class EulerAncestralDiscreteScheduler:
        config = {}

    class FlowMatchEulerDiscreteScheduler:
        def __init__(self, stochastic):
            self.config = {"stochastic_sampling": stochastic}

    class DPMSolverMultistepScheduler:
        config = {"algorithm_type": "sde-dpmsolver++"}

    for scheduler in (
        EulerAncestralDiscreteScheduler(),
        FlowMatchEulerDiscreteScheduler(True),
        DPMSolverMultistepScheduler(),
    ):
        assert "fresh noise" in video_batch.load_decline_reason(
            _state(pipe = types.SimpleNamespace(scheduler = scheduler))
        )
    assert (
        video_batch.load_decline_reason(
            _state(pipe = types.SimpleNamespace(scheduler = FlowMatchEulerDiscreteScheduler(False)))
        )
        is None
    )


def test_conditioned_requests_render_alone():
    assert video_batch.request_decline_reason(first_frame = object()) is not None
    assert video_batch.request_decline_reason(references = [object()]) is not None
    assert video_batch.request_decline_reason(continuation = (object(), 1)) is not None
    assert video_batch.request_decline_reason() is None


def test_the_batch_denoise_price_is_per_row_and_never_below_one_clip():
    from core.inference import video_stream_residency as vsr

    key = vsr.calibration_key("wan2.2-ti2v-5b", "r", "group", "cuda:0", probe = "batch-price")
    assert vsr.batch_denoise_mib(key, 100, 3, 4096) == 3 * 4096
    vsr._PRE_DECODE[key] = {100: 6000}
    try:
        assert vsr.batch_denoise_mib(key, 100, 2, 4096) == 12000
        assert vsr.batch_denoise_mib(key, 50, 1, 4096) == 6000
    finally:
        vsr._PRE_DECODE.pop(key, None)


# ---- the queue hands over the compatible clips right behind its head -------------------------------


def _entry(
    name,
    account = "a",
    *,
    frames = 121,
    steps = 30,
    kind = "page",
    **page,
):
    from routes import video as video_routes

    request = types.SimpleNamespace(
        first_frame = None,
        last_frame = None,
        reference_images = None,
        reference_videos = None,
        reference_audios = None,
        continuation_video = None,
        draft_decode = None,
        guidance = 5.0,
        guidance_2 = None,
        flow_shift = None,
        fps = 24,
        model = None,
    )
    for key, value in page.items():
        setattr(request, key, value)
    return types.SimpleNamespace(
        id = name,
        account = types.SimpleNamespace(account_id = account),
        kind = kind,
        page = request,
        body = None,
        inputs = None,
        reference = None,
        clip_key = ("wan2.2-ti2v-5b", 1280, 704, frames),
        steps = steps,
        pinned_model = ("wan",),
        starting = False,
        arrival = 0,
    ), video_routes


def test_batch_mates_are_the_compatible_clips_right_behind_the_head(monkeypatch):
    monkeypatch.delenv(video_batch.ENV, raising = False)
    head, routes = _entry("h")
    same, _ = _entry("s1", "b")
    other_shape, _ = _entry("x", "c", frames = 81)
    later_same, _ = _entry("s2", "d")
    ordered = [head, same, other_shape, later_same]
    # Contiguous only: the clip behind an incompatible one is never started ahead of it.
    assert routes._batch_mates(head, ordered) == [same]
    image, _ = _entry("i", "b", first_frame = "data:image/png;base64,AA")
    assert routes._batch_mates(head, [head, image, same]) == []
    monkeypatch.setenv(video_batch.MAX_ENV, "2")
    assert routes._batch_mates(head, [head, same, later_same]) == [same]
    monkeypatch.setenv(video_batch.ENV, "off")
    assert routes._batch_mates(head, ordered) == []
    ltx, _ = _entry("l")
    ltx.clip_key = ("ltx-2.5", 1280, 704, 121)
    assert routes._batch_mates(ltx, [ltx, same]) == []


def test_the_drainer_opens_a_batch_hands_over_its_mates_and_seals(monkeypatch):
    head, routes = _entry("h")
    mate, _ = _entry("m", "b")
    blocked, _ = _entry("z", "c", frames = 81)
    calls: list = []

    async def _start(entry, batch = None):
        calls.append((entry.id, batch))
        routes._VIDEO_QUEUE.remove(entry)
        return routes._QUEUE_MOVED

    sealed: list = []
    fake = types.SimpleNamespace(
        forming_batch = lambda: len(sealed) == 0,
        seal_batch = lambda: sealed.append(True) or 2,
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: fake)
    monkeypatch.setattr(routes, "_VIDEO_QUEUE", [head, mate, blocked])
    monkeypatch.setattr(routes, "_ordered_queue", lambda: list(routes._VIDEO_QUEUE))
    monkeypatch.setattr(routes, "_start_queued_video", _start)
    assert asyncio.run(routes._drain_video_queue_once()) is True
    assert calls == [("h", "open"), ("m", "join")]
    assert sealed and routes._VIDEO_QUEUE == [blocked]


def test_a_refused_seat_keeps_the_clip_queued_and_still_seals(monkeypatch):
    head, routes = _entry("h")
    mate, _ = _entry("m", "b")
    calls: list = []

    async def _start(entry, batch = None):
        calls.append((entry.id, batch))
        if batch == "join":
            return routes._QUEUE_WAIT
        routes._VIDEO_QUEUE.remove(entry)
        return routes._QUEUE_MOVED

    sealed: list = []
    fake = types.SimpleNamespace(
        forming_batch = lambda: True, seal_batch = lambda: sealed.append(True) or 1
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: fake)
    monkeypatch.setattr(routes, "_VIDEO_QUEUE", [head, mate])
    monkeypatch.setattr(routes, "_ordered_queue", lambda: list(routes._VIDEO_QUEUE))
    monkeypatch.setattr(routes, "_start_queued_video", _start)
    asyncio.run(routes._drain_video_queue_once())
    assert calls == [("h", "open"), ("m", "join")]
    assert sealed and routes._VIDEO_QUEUE == [mate]
