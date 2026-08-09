# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 LoRA training: the clip dataset layer, the packed-sequence geometry, the
two-schedule sigma coupling, the LoRA target surface, the routing, and the forward contract.

CPU-only and free of a diffusers import wherever the contract allows it: the two places that
genuinely need the pipeline's own layout builders are exercised against a fake transformer, so
the forward contract is checkable without the 66 GB checkpoint.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.training.diffusion_h3_clips import (  # noqa: E402
    H3_AUDIO_CHANNELS,
    H3_AUDIO_LATENTS_PER_SECOND,
    H3_AUDIO_SAMPLING_RATE,
    H3_CANVAS_MULTIPLE,
    H3_FPS,
    H3_FRAMES_PER_CHUNK,
    H3_LATENTS_PER_CHUNK,
    H3_SPATIAL_COMPRESSION,
    H3_TRAIN_NUM_FRAMES,
    discover_clip_caption_pairs,
    h3_align_num_frames,
    h3_audio_latent_count,
    h3_audio_sample_count,
    h3_packed_sequence_length,
    h3_rows_per_latent_frame,
    h3_train_canvas,
    h3_video_latent_frames,
)
from core.training.diffusion_train_common import (  # noqa: E402
    TRAINABLE_VIDEO_FAMILIES,
    DiffusionLoraConfig,
    _FLOW_TRAIN_FAMILIES,
    bf16_unsupported_reason,
    dit_accelerator_missing_reason,
    get_trainer,
    resolve_trainable_family,
    train_defaults,
)


# ── frame / latent arithmetic ────────────────────────────────────────────────
def test_align_num_frames_snaps_up_to_the_vae_grid():
    # The video VAE encodes 17 * n + 5 frames; anything else is not encodable at all.
    assert h3_align_num_frames(22) == 22
    assert h3_align_num_frames(1) == 5
    assert h3_align_num_frames(6) == 22
    assert h3_align_num_frames(124) == 124
    assert h3_align_num_frames(125) == 141
    for n in range(1, 400):
        aligned = h3_align_num_frames(n)
        assert aligned >= n
        assert aligned % H3_FRAMES_PER_CHUNK == H3_LATENTS_PER_CHUNK


def test_align_num_frames_refuses_a_non_positive_count():
    with pytest.raises(ValueError):
        h3_align_num_frames(0)


def test_video_latent_frames_is_five_per_chunk_plus_two():
    assert h3_video_latent_frames(5) == 2
    assert h3_video_latent_frames(22) == 7
    assert h3_video_latent_frames(124) == 37
    assert h3_video_latent_frames(345) == 102


def test_video_latent_frames_refuses_an_unaligned_count():
    # Silently accepting one would reserve the wrong number of video rows in the layout.
    with pytest.raises(ValueError):
        h3_video_latent_frames(24)


def test_the_training_clip_is_exactly_one_vae_chunk():
    assert H3_TRAIN_NUM_FRAMES == H3_FRAMES_PER_CHUNK + H3_LATENTS_PER_CHUNK
    assert h3_align_num_frames(H3_TRAIN_NUM_FRAMES) == H3_TRAIN_NUM_FRAMES
    # ... and is the SHORTEST encodable clip above the VAE's 5-frame head.
    assert h3_align_num_frames(H3_LATENTS_PER_CHUNK + 1) == H3_TRAIN_NUM_FRAMES


# ── audio arithmetic ─────────────────────────────────────────────────────────
def test_audio_latent_count_follows_the_forty_per_second_grid():
    assert h3_audio_latent_count(H3_FPS) == H3_AUDIO_LATENTS_PER_SECOND
    assert h3_audio_latent_count(124) == 207
    assert h3_audio_latent_count(22) == 37


def test_audio_sample_count_is_a_whole_number_of_hops():
    # The audio VAE hops 800 samples and right-pads a short tail, so handing it exactly
    # latents * hop is what makes the encode produce the row count the layout reserves.
    hop = H3_AUDIO_SAMPLING_RATE // H3_AUDIO_LATENTS_PER_SECOND
    assert hop == 800
    for frames in (22, 124, 345):
        samples = h3_audio_sample_count(frames)
        assert samples % hop == 0
        assert samples // hop == h3_audio_latent_count(frames)


# ── the packed sequence ──────────────────────────────────────────────────────
def test_rows_per_latent_frame_applies_the_two_by_two_patch():
    assert h3_rows_per_latent_frame(48, 84) == 24 * 42


def test_packed_sequence_length_counts_text_audio_and_video_rows():
    text, frames, height, width = 48, 22, 768, 1344
    latent_h, latent_w = height // H3_SPATIAL_COMPRESSION, width // H3_SPATIAL_COMPRESSION
    expected = (
        text
        + h3_audio_latent_count(frames) * H3_AUDIO_CHANNELS
        + h3_video_latent_frames(frames) * h3_rows_per_latent_frame(latent_h, latent_w)
    )
    assert h3_packed_sequence_length(text, frames, height, width) == expected
    # The figure the attention cost is quadratic in, at the released canvas.
    assert expected == 7178


def test_a_five_second_clip_is_five_times_the_sequence_of_a_training_clip():
    # The reason the trainer uses 22-frame clips rather than the 5 s floor H3 generates at.
    short = h3_packed_sequence_length(48, 22, 768, 1344)
    full = h3_packed_sequence_length(48, 124, 768, 1344)
    assert full > 5 * short


# ── the canvas rule ──────────────────────────────────────────────────────────
def test_train_canvas_reproduces_the_released_sixteen_by_nine_canvas():
    assert h3_train_canvas(16, 9) == (1344, 768)


def test_train_canvas_snaps_both_axes_to_the_multiple():
    for aspect in ((16, 9), (4, 3), (1, 1), (3, 4), (9, 16), (2048, 872)):
        width, height = h3_train_canvas(*aspect)
        assert width % H3_CANVAS_MULTIPLE == 0
        assert height % H3_CANVAS_MULTIPLE == 0


def test_train_canvas_scales_its_area_cap_with_the_short_edge():
    # A smaller training canvas must keep the released ASPECT budget, not the released pixel
    # count: a fixed cap would silently letterbox every wide clip at a small short edge.
    wide_768 = h3_train_canvas(21, 9, short_edge = 768)
    wide_384 = h3_train_canvas(21, 9, short_edge = 384)
    assert wide_768[0] / wide_768[1] == pytest.approx(wide_384[0] / wide_384[1], rel = 0.05)


def test_train_canvas_refuses_an_untrained_aspect_ratio():
    with pytest.raises(ValueError, match = "1:4 to 4:1"):
        h3_train_canvas(10, 1)
    with pytest.raises(ValueError):
        h3_train_canvas(0, 9)


# ── clip discovery ───────────────────────────────────────────────────────────
def _clip(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"not a real container")
    return path


def test_discover_clip_pairs_reads_a_sidecar_caption(tmp_path):
    _clip(tmp_path, "a.mp4")
    (tmp_path / "a.txt").write_text("a rabbit in a meadow")
    assert discover_clip_caption_pairs(tmp_path) == [
        (str(tmp_path / "a.mp4"), "a rabbit in a meadow")
    ]


def test_discover_clip_pairs_reads_metadata_jsonl(tmp_path):
    _clip(tmp_path, "a.mov")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.mov", "text": "from metadata"}) + "\n"
    )
    assert discover_clip_caption_pairs(tmp_path)[0][1] == "from metadata"


def test_discover_clip_pairs_accepts_a_video_key_in_metadata(tmp_path):
    _clip(tmp_path, "a.webm")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"video": "a.webm", "text": "keyed by video"}) + "\n"
    )
    assert discover_clip_caption_pairs(tmp_path)[0][1] == "keyed by video"


def test_a_sidecar_beats_a_metadata_row(tmp_path):
    _clip(tmp_path, "a.mp4")
    (tmp_path / "a.txt").write_text("the explicit edit")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.mp4", "text": "the bulk caption"}) + "\n"
    )
    assert discover_clip_caption_pairs(tmp_path)[0][1] == "the explicit edit"


def test_an_empty_sidecar_is_a_tombstone_that_falls_back_to_the_instance_prompt(tmp_path):
    _clip(tmp_path, "a.mp4")
    (tmp_path / "a.txt").write_text("   ")
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"file_name": "a.mp4", "text": "suppressed"}) + "\n"
    )
    pairs = discover_clip_caption_pairs(tmp_path, instance_prompt = "fallback")
    assert pairs[0][1] == "fallback"


def test_discover_clip_pairs_ignores_still_images(tmp_path):
    # H3 has no still-image milestone: a 1-frame clip is not a valid video VAE input.
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "a.txt").write_text("a still")
    with pytest.raises(ValueError, match = "clips with sound"):
        discover_clip_caption_pairs(tmp_path)


def test_discover_clip_pairs_raises_for_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_clip_caption_pairs(tmp_path / "nope")


def test_discover_clip_pairs_is_sorted_and_stable(tmp_path):
    for name in ("c.mp4", "a.mp4", "b.mp4"):
        _clip(tmp_path, name)
    pairs = discover_clip_caption_pairs(tmp_path, instance_prompt = "p")
    assert [Path(p).name for p, _ in pairs] == ["a.mp4", "b.mp4", "c.mp4"]


# ── the two coupled schedules ────────────────────────────────────────────────
def test_the_two_shifts_come_from_the_released_scheduler_configs():
    from core.training.diffusion_h3_trainer import _H3_AUDIO_SHIFT, _H3_VIDEO_SHIFT

    assert _H3_VIDEO_SHIFT == 12.0
    assert _H3_AUDIO_SHIFT == 3.0


def test_shifted_sigma_matches_the_schedulers_exponential_shift():
    from core.training.diffusion_h3_trainer import _shifted_sigma

    for shift in (3.0, 12.0):
        # The endpoints are fixed points of the shift, which is what keeps t = 0 clean and
        # t = 1 pure noise.
        assert _shifted_sigma(0.0, shift) == 0.0
        assert _shifted_sigma(1.0, shift) == 1.0
        for u in (0.1, 0.25, 0.5, 0.9):
            assert _shifted_sigma(u, shift) == pytest.approx(shift * u / (1 + (shift - 1) * u))


def test_a_larger_shift_pushes_sigma_higher_so_video_is_always_noisier_than_audio():
    # Both streams are indexed by the SAME step at inference, so the pair (video, audio) walks
    # one curve. Drawing one u and pushing it through both shifts is what reproduces it.
    from core.training.diffusion_h3_trainer import (
        _H3_AUDIO_SHIFT,
        _H3_VIDEO_SHIFT,
        _shifted_sigma,
    )

    for u in (0.05, 0.2, 0.5, 0.8, 0.95):
        assert _shifted_sigma(u, _H3_VIDEO_SHIFT) > _shifted_sigma(u, _H3_AUDIO_SHIFT)


def test_shifted_sigma_is_monotonic_in_u():
    from core.training.diffusion_h3_trainer import _shifted_sigma

    previous = -1.0
    for i in range(101):
        value = _shifted_sigma(i / 100, 12.0)
        assert value > previous
        previous = value


# ── LoRA targets ─────────────────────────────────────────────────────────────
def test_h3_targets_are_qualified_so_the_text_refiner_is_not_adapted():
    # MiniMaxH3TokenRefinerBlock carries `attn` and `ff` under the same leaf names as a
    # transformer block, so a bare "to_q" would also adapt the two refiner blocks, i.e. the
    # text stream rather than the denoiser.
    from core.training.diffusion_h3_trainer import _H3_TARGETS

    for target in _H3_TARGETS:
        assert target.startswith("transformer_blocks."), target
    assert "to_q" not in _H3_TARGETS
    assert "to_out.0" not in _H3_TARGETS


def test_h3_targets_cover_attention_and_the_feed_forward_and_nothing_else():
    from core.training.diffusion_h3_trainer import _H3_TARGETS

    leaves = {t.split("transformer_blocks.*.", 1)[1] for t in _H3_TARGETS}
    assert leaves == {
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }
    # adaln_proj is 40% of the checkpoint but its input carries two or three rows per step.
    assert not any("adaln" in t for t in _H3_TARGETS)
    # The patch and text projections are fp32 in the checkpoint's own _keep_in_fp32_modules.
    assert not any("proj_in" in t or "proj_out" in t or "context_embedder" in t for t in _H3_TARGETS)


def test_the_nf4_skip_list_covers_every_dtype_reading_module():
    # The transformer aligns activations with `x.to(self.<m>.weight.dtype)` in five places. A
    # bitsandbytes Params4bit reports uint8, so quantizing one of those casts the activation
    # to Byte and the next RMSNorm raises. proj_in / audio_proj_in / proj_out /
    # audio_proj_out / time_embedder are already excluded by _keep_in_fp32_modules; the three
    # here are not.
    from core.training.diffusion_h3_trainer import _H3_NF4_SKIP_MODULES

    assert set(_H3_NF4_SKIP_MODULES) == {"context_embedder", "adaln_proj", "norm_out"}


# ── routing ──────────────────────────────────────────────────────────────────
def test_minimax_h3_is_a_trainable_video_family():
    assert "minimax-h3" in TRAINABLE_VIDEO_FAMILIES


def test_minimax_h3_routes_to_its_own_trainer():
    from core.training import diffusion_h3_trainer

    assert get_trainer("minimax-h3") is diffusion_h3_trainer.run_h3_lora_training


def test_minimax_h3_does_not_route_to_the_dit_trainer():
    # Its objective is a two-schedule joint loss over one packed sequence, which the DiT
    # loop's single sigma and single target cannot express.
    from core.training import diffusion_dit_trainer

    assert get_trainer("minimax-h3") is not diffusion_dit_trainer.run_dit_lora_training


def test_the_official_h3_base_resolves_to_the_h3_family():
    assert resolve_trainable_family("MiniMaxAI/MiniMax-H3") == "minimax-h3"


def test_minimax_h3_is_a_flow_family_for_the_preflight_gates():
    # Without this the start route skips the bf16 / accelerator checks for H3, evicts the
    # resident GPU models, and only then fails inside the child.
    assert "minimax-h3" in _FLOW_TRAIN_FAMILIES
    assert bf16_unsupported_reason("minimax-h3") is None or isinstance(
        bf16_unsupported_reason("minimax-h3"), str
    )
    assert dit_accelerator_missing_reason("minimax-h3") is None or isinstance(
        dit_accelerator_missing_reason("minimax-h3"), str
    )


def test_h3_defaults_train_at_the_released_short_edge():
    defaults = train_defaults("minimax-h3")
    assert defaults["resolution"] == 768
    assert defaults["train_batch_size"] == 1
    assert defaults["lora_rank"] == 16


# ── config validation ────────────────────────────────────────────────────────
def _cfg(**kw) -> DiffusionLoraConfig:
    base = {
        "base_model": "MiniMaxAI/MiniMax-H3",
        "data_dir": "/tmp/clips",
        "output_dir": "/tmp/out",
        "mixed_precision": "bf16",
        "resolution": 768,
    }
    base.update(kw)
    return DiffusionLoraConfig(**base)


def test_h3_refuses_fp16():
    with pytest.raises(ValueError, match = "bf16"):
        _cfg(mixed_precision = "fp16").normalized()


def test_h3_refuses_an_off_grid_resolution():
    with pytest.raises(ValueError, match = "multiple of 32"):
        _cfg(resolution = 760).normalized()


def test_h3_refuses_the_float8_precisions():
    for mode in ("fp8", "mxfp8"):
        with pytest.raises(ValueError, match = "minimax-h3"):
            _cfg(base_precision = mode).normalized()


def test_h3_accepts_the_precisions_it_supports():
    for mode in ("nf4", "auto"):
        assert _cfg(base_precision = mode).normalized().base_precision == mode


def test_h3_resolves_to_its_family_through_normalized():
    assert _cfg().normalized().resolved_family == "minimax-h3"


# ── the forward contract, against a fake transformer ─────────────────────────
torch = pytest.importorskip("torch")

try:
    # diffusers' lazy module machinery makes a bare importorskip on the PACKAGE succeed even
    # when the real module cannot import, so pull the symbol these tests actually use. Only
    # the handful of tests below need it; the rest of this file stays runnable on a host
    # without the MiniMax-H3 diffusers revision.
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: F401
        MiniMaxH3PrepareLayoutStep,
    )

    _H3_BLOCKS = True
    _H3_BLOCKS_WHY = ""
except Exception as _exc:  # noqa: BLE001 -- a host without the H3 diffusers revision
    _H3_BLOCKS = False
    _H3_BLOCKS_WHY = f"diffusers MiniMax-H3 blocks unavailable: {_exc}"

needs_h3_blocks = pytest.mark.skipif(not _H3_BLOCKS, reason = _H3_BLOCKS_WHY)


def _layout(text_tokens = 6, latent_frames = 2, latent_h = 4, latent_w = 6, audio_latents = 3):
    from core.training.diffusion_h3_trainer import _build_layout

    return _build_layout(
        text_tokens, latent_frames, latent_h, latent_w, audio_latents, (1, 2, 2), "cpu"
    )


@needs_h3_blocks
def test_the_layout_reserves_one_row_per_token_of_every_modality():
    layout = _layout()
    rows = h3_rows_per_latent_frame(4, 6)
    assert layout["text_indices"].numel() == 6
    assert layout["audio_indices"].numel() == 3 * H3_AUDIO_CHANNELS
    assert layout["video_indices"].numel() == 2 * rows
    assert layout["position_ids"].shape == (
        6 + 3 * H3_AUDIO_CHANNELS + 2 * rows,
        3,
    )


@needs_h3_blocks
def test_the_layout_has_no_conditioning_rows():
    # The trainer trains the t2va layout: every media row is a generated row, so nothing is
    # pinned at the keyframe noise-augmentation level.
    layout = _layout()
    assert layout["num_condition_video_rows"] == 0
    assert layout["num_condition_audio_rows"] == 0


@needs_h3_blocks
def test_every_row_of_the_layout_is_claimed_by_exactly_one_modality():
    layout = _layout()
    claimed = torch.cat(
        [layout["text_indices"], layout["audio_indices"], layout["video_indices"]]
    )
    assert claimed.numel() == layout["position_ids"].shape[0]
    assert torch.equal(claimed.sort().values, torch.arange(claimed.numel()))


@needs_h3_blocks
def test_the_row_timestep_plan_carries_exactly_the_two_generated_timesteps():
    from core.training.diffusion_h3_trainer import _row_timesteps

    layout = _layout()
    timestep, indices = _row_timesteps(layout, 6, 0.25, 0.75, "cpu")
    # Two distinct noise levels in one forward: the video rows and the audio rows.
    assert sorted(round(float(t), 6) for t in timestep) == [0.25, 0.75]
    assert indices.shape == (layout["position_ids"].shape[0],)
    video_t = timestep[indices[layout["video_indices"]]]
    audio_t = timestep[indices[layout["audio_indices"]]]
    assert torch.allclose(video_t, torch.full_like(video_t, 0.25))
    assert torch.allclose(audio_t, torch.full_like(audio_t, 0.75))


@needs_h3_blocks
def test_the_text_rows_inherit_the_video_timestep():
    # Text rows never reach an output head; the reference gives them the video level.
    from core.training.diffusion_h3_trainer import _row_timesteps

    layout = _layout()
    timestep, indices = _row_timesteps(layout, 6, 0.25, 0.75, "cpu")
    text_t = timestep[indices[layout["text_indices"]]]
    assert torch.allclose(text_t, torch.full_like(text_t, 0.25))


@needs_h3_blocks
def test_patchify_round_trips_through_the_decoder_unpack():
    # The trainer packs the target the same way the sampler packs its noise, so the loss is
    # taken in the transformer's own row order.
    from core.training.diffusion_h3_trainer import _patchify

    latents = torch.randn(1, 24, 2, 4, 6)
    rows = _patchify(latents, (1, 2, 2))
    assert rows.shape == (2 * h3_rows_per_latent_frame(4, 6), 24 * 4)
    back = rows.reshape(-1, 2, 2, 3, 24, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7)
    assert torch.allclose(back.reshape(1, 24, 2, 4, 6), latents)


def test_audio_rows_are_channel_major_like_the_decoder_expects():
    # (channels, latent_channels, n) -> rows, one block per stereo channel.
    audio = torch.randn(H3_AUDIO_CHANNELS, 32, 3)
    rows = audio.permute(0, 2, 1).reshape(-1, 32)
    assert rows.shape == (H3_AUDIO_CHANNELS * 3, 32)
    assert torch.allclose(rows[0], audio[0, :, 0])
    assert torch.allclose(rows[3], audio[1, :, 0])


def test_the_velocity_target_is_data_ward():
    # MiniMax-H3 predicts x0 = x_t + sigma * v, so v = latents - noise: the NEGATION of the
    # convention in diffusion_dit_trainer. Getting this backwards trains a model that
    # reliably converges and then generates noise.
    import inspect

    from core.training import diffusion_h3_trainer

    source = inspect.getsource(diffusion_h3_trainer._train_h3)
    assert "clean_video - noise_video" in source
    assert "noise_video - clean_video" not in source
    assert "clean_audio - noise_audio" in source


def test_the_loss_includes_both_modalities():
    # A LoRA on the shared block stack changes the audio prediction whether or not audio is in
    # the loss, so leaving audio out would degrade it silently.
    import inspect

    from core.training import diffusion_h3_trainer

    source = inspect.getsource(diffusion_h3_trainer._train_h3)
    assert "loss_video + loss_audio" in source


def test_one_base_u_drives_both_sigmas():
    import inspect

    from core.training import diffusion_h3_trainer

    source = inspect.getsource(diffusion_h3_trainer._train_h3)
    assert "u = rng.random()" in source
    assert "_shifted_sigma(u, video_shift)" in source
    assert "_shifted_sigma(u, _H3_AUDIO_SHIFT)" in source


def test_the_conditioning_phase_never_names_the_transformer():
    # Naming it in load_components would put the 66 GB denoiser on the device alongside the
    # 63 GiB conditioner, which is the whole point of the phased load.
    from core.training.diffusion_h3_trainer import _H3_CONDITIONING_COMPONENTS

    assert "transformer" not in _H3_CONDITIONING_COMPONENTS
    assert "transformer_ref" not in _H3_CONDITIONING_COMPONENTS
    assert set(_H3_CONDITIONING_COMPONENTS) == {
        "text_encoder",
        "tokenizer",
        "processor",
        "vae",
        "audio_vae",
    }


def test_the_component_grid_assertion_catches_a_moved_constant():
    from core.training.diffusion_h3_trainer import _assert_component_grid

    class _Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    good_vae = types.SimpleNamespace(
        spatial_compression_ratio = H3_SPATIAL_COMPRESSION,
        tokens_chunk_size = H3_LATENTS_PER_CHUNK,
        config = _Cfg(clip_length = H3_FRAMES_PER_CHUNK, latent_channels = 24),
    )
    good_audio = types.SimpleNamespace(config = _Cfg(latent_channels = 32, sampling_rate = 32000))
    _assert_component_grid(types.SimpleNamespace(vae = good_vae, audio_vae = good_audio))

    bad_vae = types.SimpleNamespace(
        spatial_compression_ratio = 8,
        tokens_chunk_size = H3_LATENTS_PER_CHUNK,
        config = _Cfg(clip_length = H3_FRAMES_PER_CHUNK, latent_channels = 24),
    )
    with pytest.raises(ValueError, match = "spatial compression"):
        _assert_component_grid(types.SimpleNamespace(vae = bad_vae, audio_vae = good_audio))


def test_the_saved_adapter_carries_the_diffusers_transformer_prefix(tmp_path):
    # Diffusers ships no MiniMaxH3LoraLoaderMixin, so the file is written directly. It must
    # still be the ordinary single-file layout, or nothing can load it.
    from safetensors.torch import load_file

    from core.training.diffusion_h3_trainer import _save_lora

    _save_lora(str(tmp_path), {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 2)})
    saved = load_file(str(tmp_path / "pytorch_lora_weights.safetensors"))
    assert list(saved) == ["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"]
