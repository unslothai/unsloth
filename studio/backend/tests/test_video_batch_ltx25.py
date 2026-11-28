# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Queued LTX-2.5 clips that share one denoise (video_batch): the euler-ancestral sampler draws each row's
renoise from that row's own generator, so a clip's frames and soundtrack never depend on its neighbours;
one denoiser call per step holds every clip; each clip's prompt is encoded alone, and each clip is
upsampled and decoded (video and audio) alone, through one stage or two.

The numeric tests run a tiny real Diffusers LTX-2 pipeline on the CPU (random weights; a stub stands in
for the Gemma 4 encoder worker, which runs in another process in Studio) through the real VideoBackend,
with LTX-2.5's own sampler and render. They skip where this Diffusers has no LTX-2 pipeline."""

from __future__ import annotations

import pickle
import time
import types

import pytest

import core.inference.video as video_module
from core.inference import video_batch, video_gallery
from core.inference.video_families import detect_video_family

# The distilled recipe: 8 stage-one sigmas (plus 3 in stage two), CFG off.
_SHAPE = dict(width = 64, height = 64, num_frames = 9, steps = 8, guidance = 1.0, fps = 24)
_CAPTION, _LAYERS, _TOKENS = 32, 3, 16


def _ltx2_modules():
    try:
        import torch  # noqa: F401

        from core.inference.ltx2_import_compat import ensure_ltx2_pipelines_importable

        ensure_ltx2_pipelines_importable()
        from diffusers import (
            AutoencoderKLLTX2Audio,
            AutoencoderKLLTX2Video,
            FlowMatchEulerDiscreteScheduler,
            LTX2Pipeline,
            LTX2VideoTransformer3DModel,
        )
        from diffusers.pipelines.ltx2 import LTX2LatentUpsamplerModel, LTX2TextConnectors
        from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
    except Exception as exc:  # noqa: BLE001 -- an old or broken Diffusers: no LTX-2 pipeline to run
        pytest.skip(f"no Diffusers LTX-2 pipeline here ({type(exc).__name__})")
    return types.SimpleNamespace(
        audio_vae = AutoencoderKLLTX2Audio,
        vae = AutoencoderKLLTX2Video,
        scheduler = FlowMatchEulerDiscreteScheduler,
        pipeline = LTX2Pipeline,
        transformer = LTX2VideoTransformer3DModel,
        upsampler = LTX2LatentUpsamplerModel,
        connectors = LTX2TextConnectors,
        vocoder = LTX2Vocoder,
    )


def _stub_encoder(pipe, calls: list):
    """The encoder worker's contract (``install_remote_encode_prompt``): left-padded packed hidden states and their
    mask per prompt, deterministic in the prompt text. ``calls`` records the prompts of each call."""
    import torch

    def _one(texts):
        embeds, masks = [], []
        for text in texts:
            words = len((text or "").split()) + 1
            valid = min(_TOKENS, words)
            g = torch.Generator().manual_seed(sum(map(ord, text or "")) + 7)
            row = torch.zeros(_TOKENS, _CAPTION * _LAYERS)
            row[-valid:] = torch.randn(valid, _CAPTION * _LAYERS, generator = g)
            mask = torch.zeros(_TOKENS, dtype = torch.long)
            mask[-valid:] = 1
            embeds.append(row)
            masks.append(mask)
        return torch.stack(embeds), torch.stack(masks)

    def encode_prompt(
        prompt = None,
        negative_prompt = None,
        do_classifier_free_guidance = True,
        num_videos_per_prompt = 1,
        prompt_embeds = None,
        negative_prompt_embeds = None,
        prompt_attention_mask = None,
        negative_prompt_attention_mask = None,
        max_sequence_length = _TOKENS,
        scale_factor = 8,
        device = None,
        dtype = None,
    ):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        calls.append(list(prompts))
        positive, positive_mask = _one(prompts)
        negative = negative_mask = None
        if do_classifier_free_guidance:
            texts = negative_prompt or ""
            texts = [texts] * len(prompts) if isinstance(texts, str) else list(texts)
            negative, negative_mask = _one(texts)
        return positive, positive_mask, negative, negative_mask

    pipe.encode_prompt = encode_prompt


def _tiny_ltx25(calls: list):
    import torch

    from core.inference import video_ltx25

    m = _ltx2_modules()
    torch.manual_seed(0)
    transformer = m.transformer(
        in_channels = 4,
        out_channels = 4,
        patch_size = 1,
        patch_size_t = 1,
        num_attention_heads = 2,
        attention_head_dim = 8,
        cross_attention_dim = 16,
        audio_in_channels = 4,
        audio_out_channels = 4,
        audio_num_attention_heads = 2,
        audio_attention_head_dim = 4,
        audio_cross_attention_dim = 8,
        num_layers = 2,
        qk_norm = "rms_norm_across_heads",
        caption_channels = _CAPTION,
        rope_double_precision = False,
        rope_type = "split",
    ).eval()
    torch.manual_seed(0)
    connectors = m.connectors(
        caption_channels = _CAPTION,
        text_proj_in_factor = _LAYERS,
        video_connector_num_attention_heads = 4,
        video_connector_attention_head_dim = 8,
        video_connector_num_layers = 1,
        video_connector_num_learnable_registers = None,
        audio_connector_num_attention_heads = 4,
        audio_connector_attention_head_dim = 8,
        audio_connector_num_layers = 1,
        audio_connector_num_learnable_registers = None,
        connector_rope_base_seq_len = 32,
        rope_theta = 10000.0,
        rope_double_precision = False,
        causal_temporal_positioning = False,
        rope_type = "split",
    ).eval()
    torch.manual_seed(0)
    vae = m.vae(
        in_channels = 3,
        out_channels = 3,
        latent_channels = 4,
        block_out_channels = (8,),
        decoder_block_out_channels = (8,),
        layers_per_block = (1,),
        decoder_layers_per_block = (1, 1),
        spatio_temporal_scaling = (True,),
        decoder_spatio_temporal_scaling = (True,),
        decoder_inject_noise = (False, False),
        downsample_type = ("spatial",),
        upsample_residual = (False,),
        upsample_factor = (1,),
        timestep_conditioning = False,
        patch_size = 1,
        patch_size_t = 1,
        encoder_causal = True,
        decoder_causal = False,
    ).eval()
    vae.use_framewise_encoding = False
    vae.use_framewise_decoding = False
    torch.manual_seed(0)
    audio_vae = m.audio_vae(
        base_channels = 4,
        output_channels = 2,
        ch_mult = (1,),
        num_res_blocks = 1,
        attn_resolutions = None,
        in_channels = 2,
        resolution = 32,
        latent_channels = 2,
        norm_type = "pixel",
        causality_axis = "height",
        dropout = 0.0,
        mid_block_add_attention = False,
        sample_rate = 16000,
        mel_hop_length = 160,
        is_causal = True,
        mel_bins = 8,
    ).eval()
    torch.manual_seed(0)
    vocoder = m.vocoder(
        in_channels = 16,
        hidden_channels = 32,
        out_channels = 2,
        upsample_kernel_sizes = [4, 4],
        upsample_factors = [2, 2],
        resnet_kernel_sizes = [3],
        resnet_dilations = [[1, 3, 5]],
        leaky_relu_negative_slope = 0.1,
        output_sampling_rate = 16000,
    ).eval()
    pipe = m.pipeline(
        scheduler = m.scheduler(),
        vae = vae,
        audio_vae = audio_vae,
        text_encoder = None,
        tokenizer = None,
        connectors = connectors,
        transformer = transformer,
        vocoder = vocoder,
    )
    try:
        pipe.set_progress_bar_config(disable = True)
    except Exception:  # noqa: BLE001 -- cosmetic
        pass
    # What the load installs: the distilled DiT's euler-ancestral sampler and the stage-two upsampler.
    assert video_ltx25.prepare_scheduler(pipe, dev = False) == "euler_ancestral"
    torch.manual_seed(0)
    pipe._unsloth_ltx25_upsampler = m.upsampler(
        in_channels = 4, mid_channels = 32, num_blocks_per_stage = 1
    ).eval()
    _stub_encoder(pipe, calls)
    return pipe


@pytest.fixture
def saved(monkeypatch):
    """Clips the backend saved, by id: ((frames, audio), recipe)."""
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


@pytest.fixture(params = ["one_stage", "two_stage"])
def stages(request, monkeypatch):
    from core.inference import video_ltx25
    monkeypatch.setenv(video_ltx25.TWO_STAGE_ENV, "1" if request.param == "two_stage" else "0")
    return request.param


def _backend(monkeypatch, calls: list):
    monkeypatch.setenv(video_batch.ENV, "force")
    from core.inference import diffusion_attention

    # The real CPU SDPA probe answers are this test's own: a later fake-torch test would read them back.
    monkeypatch.setattr(diffusion_attention, "_SDPA_PROBE_CACHE", {})
    # The tiny canvas is far below the trained sizes the shape check enforces.
    monkeypatch.setattr(video_module, "validate_video_request_shape", lambda *a, **k: None)
    backend = video_module.VideoBackend()
    backend._state = video_module._VideoLoadState(
        pipe = _tiny_ltx25(calls),
        family = detect_video_family("Lightricks/LTX-2.5-Diffusers"),
        repo_id = "tiny/ltx25",
        base_repo = "tiny/ltx25",
        device = "cpu",
        dtype = "float32",
        kind = "pipeline",
    )

    def _encode(frames, fps, audio, pipe, **kw):
        import numpy as np
        track = None if audio is None else np.asarray(getattr(audio, "cpu", lambda: audio)())
        return pickle.dumps((np.asarray(frames), track))

    # The frames and the soundtrack themselves stand in for the mp4, so a comparison is exact.
    backend._encode_mp4 = _encode
    return backend


def _wait_idle(backend, timeout = 120.0) -> None:
    deadline = time.monotonic() + timeout
    while backend._generate_job_active or backend._encode_tail is not None:
        assert time.monotonic() < deadline, backend._gen
        time.sleep(0.02)


def _solo(backend, prompt, seed, video_id):
    backend.begin_generate(prompt = prompt, seed = seed, video_id = video_id, **_SHAPE)
    _wait_idle(backend)


def _batch(backend, rows):
    """rows: (prompt, seed, video_id); the first starts the job, the rest join it."""
    (prompt, seed, video_id), *rest = rows
    backend.begin_generate(prompt = prompt, seed = seed, video_id = video_id, batch_open = True, **_SHAPE)
    for prompt, seed, video_id in rest:
        backend.begin_generate(prompt = prompt, seed = seed, video_id = video_id, join = True, **_SHAPE)
    assert backend.seal_batch() == len(rows)
    _wait_idle(backend)


def _clip(saved, video_id):
    assert video_id in saved, (video_id, sorted(saved))
    return saved[video_id][0]


# ---- numerics on a tiny real pipeline ----------------------------------------------------------


def test_each_batched_clip_is_its_own_solo_render(monkeypatch, saved, stages):
    import numpy as np
    import torch

    calls: list = []
    backend = _backend(monkeypatch, calls)
    pipe = backend._state.pipe
    _solo(backend, "a cat on a sofa", 11, "solo-cat")
    _solo(backend, "a dog in the snow", 22, "solo-dog")
    calls.clear()
    rows_seen: list = []
    pipe.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: rows_seen.append(int(kwargs["hidden_states"].shape[0])),
        with_kwargs = True,
    )
    decodes: list = []
    for module in (pipe.vae, pipe.audio_vae):
        real = module.decode
        module.decode = lambda z, *a, _real = real, _name = type(module).__name__, **k: (
            decodes.append((_name, int(z.shape[0]))) or _real(z, *a, **k)
        )
    vocoder_rows: list = []
    real_vocoder = type(pipe.vocoder).forward

    def _vocoder(self, mel, *args, **kwargs):
        vocoder_rows.append(int(mel.shape[0]))
        return real_vocoder(self, mel, *args, **kwargs)

    monkeypatch.setattr(type(pipe.vocoder), "forward", _vocoder)
    upsampler_rows: list = []
    pipe._unsloth_ltx25_upsampler.register_forward_pre_hook(
        lambda module, args: upsampler_rows.append(int(args[0].shape[0]))
    )
    before = video_batch.counts()
    _batch(backend, [("a cat on a sofa", 11, "b-cat"), ("a dog in the snow", 22, "b-dog")])
    after = video_batch.counts()
    assert after["batches"] == before["batches"] + 1
    assert after["rows"] == before["rows"] + 2

    # One denoiser call per step holds both clips (8 distilled steps, plus 3 in stage two).
    steps = 8 + (3 if stages == "two_stage" else 0)
    assert rows_seen == [2] * steps
    # Each clip's prompt is encoded alone, as in its own render.
    assert all(len(prompts) == 1 for prompts in calls), calls
    # Each clip is upsampled, decoded and vocoded alone.
    assert upsampler_rows == ([1, 1] if stages == "two_stage" else [])
    assert sorted(set(decodes)) == [("AutoencoderKLLTX2Audio", 1), ("AutoencoderKLLTX2Video", 1)]
    assert vocoder_rows == [1, 1]

    for batched, solo in (("b-cat", "solo-cat"), ("b-dog", "solo-dog")):
        frames, audio = _clip(saved, batched)
        solo_frames, solo_audio = _clip(saved, solo)
        assert frames.shape == solo_frames.shape and audio.shape == solo_audio.shape
        # A batch-shape GEMM floor (about 1e-6 in float32 here), never a neighbour's noise or prompt.
        assert np.abs(frames.astype(np.int16) - solo_frames.astype(np.int16)).max() <= 1
        assert np.abs(audio - solo_audio).max() < 1e-4
        assert saved[batched][1]["has_audio"] is True
    cat, dog = _clip(saved, "solo-cat"), _clip(saved, "solo-dog")
    assert np.abs(cat[0].astype(np.int16) - dog[0].astype(np.int16)).max() > 8
    assert np.abs(cat[1] - dog[1]).max() > 1e-3
    assert saved["b-dog"][1]["prompt"] == "a dog in the snow"
    assert saved["b-dog"][1]["seed"] == 22
    assert (
        backend.generate_outcome("b-dog", video_module.current_account_id())["phase"] == "completed"
    )
    assert torch.is_tensor(pipe.scheduler.sigmas)


@pytest.mark.parametrize("target_row", [0, 1, 2])
def test_a_neighbours_prompt_and_seed_never_change_a_clip(monkeypatch, saved, stages, target_row):
    import numpy as np

    backend = _backend(monkeypatch, [])
    target = ("a red fox in fresh snow", 7)

    def rows(tag, neighbours):
        out = []
        for i in range(3):
            prompt, seed = target if i == target_row else neighbours[i]
            out.append((prompt, seed, f"{tag}-{i}"))
        return out

    _batch(backend, rows("a", {0: ("a boat at sea", 1), 1: ("a city at night", 2), 2: ("rain", 3)}))
    _batch(
        backend,
        rows(
            "b",
            {
                0: ("a very long and loud parade with drums", 91),
                1: ("an empty desert road", 92),
                2: ("fireworks over a lake", 93),
            },
        ),
    )
    frames_a, audio_a = _clip(saved, f"a-{target_row}")
    frames_b, audio_b = _clip(saved, f"b-{target_row}")
    # Same execution shape, only the neighbours changed: the target clip is bit-identical, frames and soundtrack.
    assert np.array_equal(frames_a, frames_b)
    assert np.array_equal(audio_a, audio_b)
    neighbour = (target_row + 1) % 3
    assert not np.array_equal(_clip(saved, f"a-{neighbour}")[0], _clip(saved, f"b-{neighbour}")[0])


# ---- the sampler's per-row state ----------------------------------------------------------------


def _ancestral():
    pytest.importorskip("diffusers")
    from diffusers import FlowMatchEulerDiscreteScheduler

    from core.inference import video_ltx25

    return video_ltx25.make_ancestral_scheduler(FlowMatchEulerDiscreteScheduler())


def _step_all(
    scheduler,
    sample,
    generator,
    keep = None,
):
    import torch

    scheduler.set_timesteps(sigmas = [1.0, 0.9, 0.5], device = "cpu")
    scheduler.noise_generator = generator
    scheduler.keep_tokens = keep
    out = sample
    for t in scheduler.timesteps[:2]:
        out = scheduler.step(torch.ones_like(out) * 0.1, t, out, return_dict = False)[0]
    return out


def test_the_ancestral_sampler_draws_each_rows_noise_from_its_own_generator():
    import torch

    scheduler = _ancestral()
    assert scheduler.per_row_noise is True
    assert video_batch.stochastic_sampler(scheduler) is None
    sample = torch.randn(2, 6, 4, generator = torch.Generator().manual_seed(0))
    solo = [
        _step_all(_ancestral(), sample[i : i + 1], torch.Generator().manual_seed(seed))
        for i, seed in enumerate((5, 6))
    ]
    batched = _step_all(
        scheduler, sample, [torch.Generator().manual_seed(5), torch.Generator().manual_seed(6)]
    )
    assert torch.equal(batched[0:1], solo[0])
    assert torch.equal(batched[1:2], solo[1])
    # One generator for the whole batch (the old shared stream) is not row 1's solo draw.
    shared = _step_all(_ancestral(), sample, torch.Generator().manual_seed(5))
    assert not torch.equal(shared[1:2], solo[1])


def test_keep_tokens_apply_per_row():
    import torch

    sample = torch.randn(2, 6, 4, generator = torch.Generator().manual_seed(1))
    generators = lambda: [torch.Generator().manual_seed(5), torch.Generator().manual_seed(6)]  # noqa: E731
    free = _step_all(_ancestral(), sample, generators())
    kept = _step_all(_ancestral(), sample, generators(), keep = [(2, 6), None])
    whole = _step_all(_ancestral(), sample, generators(), keep = (2, 6))
    # Row 0 keeps its first two tokens on the deterministic step; row 1 is untouched.
    assert torch.equal(kept[0, :2], whole[0, :2])
    assert not torch.equal(kept[0, :2], free[0, :2])
    assert torch.equal(kept[0, 2:], free[0, 2:])
    assert torch.equal(kept[1], free[1])
    # A tuple (every row) is the single-clip behaviour, unchanged: both rows keep.
    assert not torch.equal(whole[1, :2], free[1, :2])


# ---- which loads batch ------------------------------------------------------------------------------


def _state(**fields):
    base = dict(
        family = detect_video_family("Lightricks/LTX-2.5-Diffusers"),
        engine = "diffusers",
        transformer_cache = None,
        vram_split_cards = None,
        offload_policy = "streaming",
        memory_mode = "low_vram",
        denoiser_streamed = True,
        pipe = types.SimpleNamespace(),
    )
    base.update(fields)
    return types.SimpleNamespace(**base)


def test_auto_batches_a_two_stage_low_vram_ltx25_load_and_keeps_the_rest_serial(monkeypatch):
    from core.inference import video_ltx25

    monkeypatch.delenv(video_batch.ENV, raising = False)
    monkeypatch.delenv(video_ltx25.TWO_STAGE_ENV, raising = False)
    # Measured faster batched under Low VRAM, and both gated sizes pass the quality gate.
    assert video_batch.load_decline_reason(_state()) is None
    # A DiT kept on the card is compute bound: an eager batch loses to one compiled clip.
    assert "Low VRAM" in video_batch.load_decline_reason(
        _state(memory_mode = "auto", offload_policy = "none")
    )
    # The one-stage recipe passes the same gate at 2 and 4 clips (test_auto_batches_the_one_stage_recipe_too).
    # Auto runs only the gated sizes; force runs any.
    state = _state()
    assert video_batch.auto_size_decline(state, 1) is None
    assert video_batch.auto_size_decline(state, 2) is None
    assert video_batch.auto_size_decline(state, 4) is None
    assert "not 3" in video_batch.auto_size_decline(state, 3)
    monkeypatch.setenv(video_batch.ENV, "force")
    assert video_batch.auto_size_decline(state, 3) is None
    assert video_batch.load_decline_reason(_state()) is None
    assert (
        video_batch.load_decline_reason(_state(memory_mode = "auto", offload_policy = "none")) is None
    )
    # The dev DiT (CFG + STG, 30+ steps) is unmeasured: it renders alone even when forced.
    assert "dev DiT" in video_batch.load_decline_reason(
        _state(pipe = types.SimpleNamespace(_unsloth_ltx25_dev = True))
    )
    # LTX-2.5's own sampler renoises from per-row generators: not the shared-stream refusal.
    assert (
        video_batch.load_decline_reason(_state(pipe = types.SimpleNamespace(scheduler = _ancestral())))
        is None
    )
    monkeypatch.setenv(video_batch.ENV, "off")
    assert video_batch.load_decline_reason(_state()) == f"{video_batch.ENV}=off"


def test_auto_batches_the_one_stage_recipe_too(monkeypatch):
    from core.inference import video_ltx25

    monkeypatch.delenv(video_batch.ENV, raising = False)
    # One stage under Low VRAM, eager: the paired gate against a bf16 / fp32-attention reference passed at 2 and 4.
    monkeypatch.setenv(video_ltx25.TWO_STAGE_ENV, "0")
    state = _state()
    assert video_batch.load_decline_reason(state) is None
    assert video_batch.auto_size_decline(state, 2) is None
    assert video_batch.auto_size_decline(state, 4) is None
    assert "not 3" in video_batch.auto_size_decline(state, 3)
    # Still Low VRAM only: a DiT kept on the card stays serial whatever the recipe.
    assert "Low VRAM" in video_batch.load_decline_reason(
        _state(memory_mode = "auto", offload_policy = "none")
    )


def test_the_batch_key_carries_the_audio_shift():
    common = dict(width = 64, height = 64, num_frames = 9, fps = 24, steps = 8, guidance = 1.0, guidance_2 = None)
    assert video_batch.request_key(flow_shift = None, audio_flow_shift = 2.0, **common) != (
        video_batch.request_key(flow_shift = None, audio_flow_shift = 3.0, **common)
    )
    # Families without an audio stream keep the key they had (the extra field is None).
    assert video_batch.request_key(flow_shift = 5.0, **common)[-1] is None


def test_per_row_encode_stacks_every_output_of_a_four_part_encoder():
    import torch

    seen: list = []

    def encode_prompt(
        prompt = None,
        negative_prompt = None,
        **kwargs,
    ):
        seen.append(prompt)
        value = float(len(prompt))
        return (
            torch.full((1, 2, 3), value),
            torch.full((1, 2), value),
            None,
            None,
        )

    pipe = types.SimpleNamespace(encode_prompt = encode_prompt)
    with video_batch.per_row_encode(pipe):
        out = pipe.encode_prompt(prompt = ["ab", "abcd"], negative_prompt = None)
    assert seen == ["ab", "abcd"]
    assert len(out) == 4 and out[2] is None and out[3] is None
    assert out[0][:, 0, 0].tolist() == [2.0, 4.0]
    assert out[1][:, 0].tolist() == [2.0, 4.0]
    assert pipe.encode_prompt is encode_prompt


def test_auto_renders_an_ungated_batch_size_one_clip_at_a_time(monkeypatch, saved):
    import numpy as np

    from core.inference import video_ltx25

    # The two-stage recipe (Auto's gated one).
    monkeypatch.setenv(video_ltx25.TWO_STAGE_ENV, "1")
    calls: list = []
    backend = _backend(monkeypatch, calls)
    import dataclasses

    backend._state = dataclasses.replace(backend._state, memory_mode = "low_vram")
    monkeypatch.delenv(video_batch.ENV)
    rows = [
        ("a cat on a sofa", 11, "t-0"),
        ("a dog in the snow", 22, "t-1"),
        ("rain on a window", 33, "t-2"),
    ]
    for prompt, seed, video_id in rows:
        _solo(backend, prompt, seed, f"solo-{video_id}")
    rows_seen: list = []
    backend._state.pipe.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: rows_seen.append(int(kwargs["hidden_states"].shape[0])),
        with_kwargs = True,
    )
    _batch(backend, rows)
    # Three clips is not a gated size: every denoiser call holds one clip, and each clip is its solo render.
    assert rows_seen and set(rows_seen) == {1}
    for _prompt, _seed, video_id in rows:
        frames, audio = _clip(saved, video_id)
        solo_frames, solo_audio = _clip(saved, f"solo-{video_id}")
        assert np.array_equal(frames, solo_frames) and np.array_equal(audio, solo_audio)
