# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 LoRA training: the clip dataset layer, the packed-sequence geometry, the
two-schedule sigma coupling, the LoRA target surface, the routing, and the forward contract.

CPU-only and free of a diffusers import wherever the contract allows it: the two places that
genuinely need the pipeline's own layout builders are exercised against a fake transformer, so
the forward contract is checkable without the 66 GB checkpoint.
"""

from __future__ import annotations

import importlib
import json
import random
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
    # A smaller training canvas must keep the released AREA budget in units of the short edge,
    # not the released pixel count: with a fixed cap a 384-edge canvas never reaches the cap at
    # all, so a wide clip trains at a completely different area-to-edge ratio than it would at
    # 768 and the geometry stops being a scaled-down version of the released one.
    def area_ratio(short_edge: int) -> float:
        width, height = h3_train_canvas(21, 9, short_edge = short_edge)
        return width * height / short_edge**2

    assert area_ratio(384) == pytest.approx(area_ratio(768), rel = 0.10)
    assert area_ratio(192) == pytest.approx(area_ratio(768), rel = 0.10)


def test_train_canvas_refuses_an_untrained_aspect_ratio():
    with pytest.raises(ValueError, match = "1:4 to 4:1"):
        h3_train_canvas(10, 1)


def test_train_canvas_names_a_degenerate_size_as_such():
    # A zero-sized source has no aspect ratio at all. Folding it into the trained-range message
    # would tell the user to "crop it first", which cannot help.
    with pytest.raises(ValueError, match = "must be positive"):
        h3_train_canvas(0, 9)
    with pytest.raises(ValueError, match = "must be positive"):
        h3_train_canvas(16, -1)


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


# ── the preflight has to run the trainer's own discovery ─────────────────────


def test_discover_training_pairs_routes_h3_to_the_clip_discovery(tmp_path):
    """The gap this closes: /diffusion/start preflights the dataset BEFORE freeing the resident
    GPU models, and it ran the IMAGE discovery unconditionally. An H3 dataset is captioned clips
    -- the only thing its trainer accepts -- so a perfectly valid one was rejected at the route
    with "No captioned images found" and the advertised H3 trainer could not be reached at all.
    """
    from core.training.diffusion_train_common import discover_training_pairs

    _clip(tmp_path, "a.mp4")
    (tmp_path / "a.txt").write_text("a rabbit in a meadow")
    pairs = discover_training_pairs("minimax-h3", tmp_path, verify_images = True)
    assert [Path(p).name for p, _ in pairs] == ["a.mp4"]
    assert pairs[0][1] == "a rabbit in a meadow"


def test_discover_training_pairs_leaves_an_image_family_on_the_image_discovery(tmp_path):
    """The other half: nothing about an existing image family moves, verify_images included."""
    from PIL import Image

    from core.training.diffusion_train_common import discover_training_pairs

    Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    _clip(tmp_path, "b.mp4")  # present, and must be ignored for an image family
    pairs = discover_training_pairs("sdxl", tmp_path, instance_prompt = "p", verify_images = True)
    assert [Path(p).name for p, _ in pairs] == ["a.png"]


def test_the_clip_families_are_exactly_the_ones_whose_trainer_takes_clips():
    """LTX-2 trains a style LoRA FROM STILLS, so it must keep the image discovery even though it
    is a video family -- the split is by what the TRAINER reads, not by output modality."""
    from core.training.diffusion_train_common import (
        CLIP_TRAINED_FAMILIES,
        TRAINABLE_VIDEO_FAMILIES,
    )

    assert CLIP_TRAINED_FAMILIES == {"minimax-h3"}
    assert CLIP_TRAINED_FAMILIES <= TRAINABLE_VIDEO_FAMILIES
    assert "ltx-2" not in CLIP_TRAINED_FAMILIES


# ── the two coupled schedules ────────────────────────────────────────────────
def test_the_two_shifts_come_from_the_released_scheduler_configs():
    from core.training.diffusion_h3_trainer import _H3_AUDIO_SHIFT, _H3_VIDEO_SHIFT
    assert _H3_VIDEO_SHIFT == 12.0
    assert _H3_AUDIO_SHIFT == 3.0


def test_an_omitted_flow_shift_reaches_the_trainer_as_the_released_video_shift():
    """The gap this closes: ``normalized()`` resolved an omitted flow_shift to the identity 1.0
    for every family outside AUTO_FLOW_SHIFT_FAMILIES, and the H3 loop only falls back to
    _H3_VIDEO_SHIFT when the value is NOT a number. 1.0 is a number, so it won over the 12.0 the
    released scheduler uses and every default run trained against an unshifted video-noise
    distribution the sampler never visits -- silently, at full cost."""
    from core.training.diffusion_train_common import AUTO_FLOW_SHIFT_FAMILIES, DiffusionLoraConfig

    assert "minimax-h3" in AUTO_FLOW_SHIFT_FAMILIES
    cfg = DiffusionLoraConfig(
        base_model = "MiniMaxAI/MiniMax-H3",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
    ).normalized()
    assert cfg.resolved_family == "minimax-h3"
    # Not a number, which is exactly what routes the trainer to its own pair of shifts.
    assert cfg.flow_shift == "auto"
    assert not isinstance(cfg.flow_shift, (int, float))


def test_an_explicit_flow_shift_still_overrides_the_video_default():
    """The escape hatch stays: a caller who names a shift gets it, not 12.0."""
    from core.training.diffusion_train_common import DiffusionLoraConfig

    cfg = DiffusionLoraConfig(
        base_model = "MiniMaxAI/MiniMax-H3",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
        flow_shift = 7.0,
    ).normalized()
    assert cfg.flow_shift == 7.0


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
# Every Linear of the released checkpoint, by name (num_layers and num_refiner_layers cut
# down; the names are otherwise exactly what MiniMaxH3Transformer3DModel builds).
_H3_LINEARS = (
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_embedder.linear_1",
    "time_embedder.linear_2",
    "token_refiner.refiner_blocks.0.attn.to_q",
    "token_refiner.refiner_blocks.0.attn.to_k",
    "token_refiner.refiner_blocks.0.attn.to_v",
    "token_refiner.refiner_blocks.0.attn.to_out.0",
    "token_refiner.refiner_blocks.0.ff.net.0.proj",
    "token_refiner.refiner_blocks.0.ff.net.2",
    "token_refiner.refiner_blocks.1.attn.to_q",
    "transformer_blocks.0.attn.to_q",
    "transformer_blocks.0.attn.to_k",
    "transformer_blocks.0.attn.to_v",
    "transformer_blocks.0.attn.to_out.0",
    "transformer_blocks.0.ff.net.0.proj",
    "transformer_blocks.0.ff.net.2",
    "transformer_blocks.0.adaln_proj.linear",
    "transformer_blocks.49.attn.to_q",
    "transformer_blocks.49.ff.net.2",
    "norm_out.linear",
    "proj_out",
    "audio_proj_out",
)


def _selected(target_regex: str) -> set:
    """PEFT's own rule for a STRING target_modules: re.fullmatch on the module name."""
    import re
    return {name for name in _H3_LINEARS if re.fullmatch(target_regex, name)}


def test_h3_targets_are_a_regex_because_peft_does_not_glob():
    # PEFT either suffix-matches a LIST of names or re.fullmatch-es a STRING. A list entry
    # written "transformer_blocks.*.attn.to_q" matches nothing at all, so the adapter would
    # train zero parameters while every step still reported a loss.
    from core.training.diffusion_h3_trainer import _H3_TARGETS
    assert isinstance(_H3_TARGETS, str)
    assert "*" not in _H3_TARGETS


def test_h3_targets_never_adapt_the_text_refiner():
    # MiniMaxH3TokenRefinerBlock carries `attn` and `ff` under the SAME leaf names as a
    # transformer block, so a bare "to_q" would also adapt the two refiner blocks, i.e. the
    # text stream rather than the denoiser.
    from core.training.diffusion_h3_trainer import _H3_TARGETS
    assert not any(name.startswith("token_refiner") for name in _selected(_H3_TARGETS))


def test_h3_targets_cover_attention_and_the_feed_forward_and_nothing_else():
    from core.training.diffusion_h3_trainer import _H3_TARGET_LEAVES, _H3_TARGETS

    assert set(_H3_TARGET_LEAVES) == {
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    }
    selected = _selected(_H3_TARGETS)
    assert selected == {
        f"transformer_blocks.{index}.{leaf}"
        for index in (0, 49)
        for leaf in _H3_TARGET_LEAVES
        if f"transformer_blocks.{index}.{leaf}" in _H3_LINEARS
    }
    # adaln_proj is 40% of the checkpoint but its input carries two or three rows per step.
    assert not any("adaln" in name for name in selected)
    # The patch and text projections are fp32 in the checkpoint's own _keep_in_fp32_modules.
    assert not any(
        "proj_in" in name or "proj_out" in name or "context_embedder" in name for name in selected
    )
    # The final norm's modulation projection is a dtype-reading module, not an adapter site.
    assert "norm_out.linear" not in selected


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


def test_the_official_h3_base_is_trusted_for_training():
    # The image-side inference allowlist never covered a video family, so without the training
    # allowlist entry the official repo is refused as untrusted and only a local path trains.
    from core.training.diffusion_train_common import _assert_trusted_base_model
    _assert_trusted_base_model("MiniMaxAI/MiniMax-H3")
    with pytest.raises(ValueError, match = "untrusted"):
        _assert_trusted_base_model("some-random-user/minimax-h3-repack")


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


# ── entrypoint refusals, all of which must fire BEFORE anything loads ─────────
def _run(**kw):
    from core.training.diffusion_h3_trainer import run_h3_lora_training
    return run_h3_lora_training(_cfg(**kw))


def test_h3_refuses_a_batch_larger_than_one():
    # The batch axis of an H3 forward is a pure replication axis: the row layout is set by the
    # clip's geometry AND its caption's length, so two clips cannot share one packed sequence.
    with pytest.raises(ValueError, match = "batch size 1"):
        _run(train_batch_size = 2)


def test_h3_refuses_a_cfg_dropout():
    # The checkpoint is guidance-distilled: there is no unconditional branch to train.
    with pytest.raises(ValueError, match = "guidance-distilled"):
        _run(cfg_dropout = 0.1)


def test_h3_refuses_a_weighted_loss():
    # Two schedules put video and audio at different sigmas in the same step, so a single
    # weight over "the" timestep is ambiguous.
    with pytest.raises(ValueError, match = "weighting_scheme"):
        _run(weighting_scheme = "bell")


def test_the_entrypoint_refusals_fire_before_the_data_directory_is_read():
    # _cfg points data_dir at a path that does not exist, so any of these reaching discovery
    # would raise FileNotFoundError instead -- and in a real run would have already evicted the
    # resident GPU models.
    with pytest.raises(ValueError):
        _run(train_batch_size = 4)


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


def _layout(
    text_tokens = 6,
    latent_frames = 2,
    latent_h = 4,
    latent_w = 6,
    audio_latents = 3,
):
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
    assert layout["position_ids"].shape == (6 + 3 * H3_AUDIO_CHANNELS + 2 * rows, 3)


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
    claimed = torch.cat([layout["text_indices"], layout["audio_indices"], layout["video_indices"]])
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


def test_the_row_timestep_plan_is_built_video_first():
    # build_row_timesteps takes (video_timestep, audio_timestep) in that order, and both are
    # plain floats, so swapping them is silent: the video rows would be conditioned at the
    # audio schedule's noise level and vice versa.
    import inspect

    from core.training import diffusion_h3_trainer

    source = inspect.getsource(diffusion_h3_trainer._train_h3)
    assert "1.0 - sigma_video, 1.0 - sigma_audio" in source
    assert "1.0 - sigma_audio, 1.0 - sigma_video" not in source


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


# ── the hosted denoiser is a component, not a base ───────────────────────────


def test_the_hosted_prequant_denoiser_is_refused_as_a_training_base():
    """The two registries mean opposite things by ``prequant_repos``: an image family's entry is a
    full quantized PIPELINE mirror, a video family's is the pre-quantized DENOISER alone. Reading
    both as bases let the H3 denoiser repo through the preflight -- its name resolves to the
    family, the unsloth/* trust gate passes it -- so the run evicted the resident GPU workloads
    and only then failed inside ModularPipeline.from_pretrained."""
    from core.inference.video_families import detect_video_family
    from core.training.diffusion_train_common import (
        _component_only_repos,
        resolve_trainable_family,
    )

    fam = detect_video_family("", override = "minimax-h3")
    denoiser_repo = fam.prequant_repos[0][1]
    assert _component_only_repos()[denoiser_repo.lower()][:2] == ("minimax-h3", "transformer")
    with pytest.raises(ValueError) as exc:
        resolve_trainable_family(denoiser_repo)
    detail = str(exc.value)
    assert "not a full model" in detail
    assert fam.base_repo in detail  # and names what to train instead


def test_the_real_h3_base_is_still_trainable():
    """The other side of the same coin: the base itself must not be caught by the refusal."""
    from core.inference.video_families import detect_video_family
    from core.training.diffusion_train_common import resolve_trainable_family

    fam = detect_video_family("", override = "minimax-h3")
    assert resolve_trainable_family(fam.base_repo) == "minimax-h3"


# ── the checkpoint contract the H3 loop does not implement ───────────────────


def _h3_cfg(**kw):
    from core.training.diffusion_train_common import DiffusionLoraConfig
    return DiffusionLoraConfig(
        base_model = "MiniMaxAI/MiniMax-H3",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
        **kw,
    )


def test_resume_is_refused_rather_than_silently_starting_over():
    """The dangerous half: run_h3_lora_training neither writes a resume bundle nor restores one,
    so a caller handing one over got a FRESH optimization that then overwrote the outputs it was
    meant to continue -- discovered only after an expensive run."""
    with pytest.raises(ValueError, match = "resume_from_checkpoint is not supported"):
        _h3_cfg(resume_from_checkpoint = "/tmp/o/checkpoint-100").normalized()


def test_save_steps_is_refused_rather_than_silently_ignored():
    """The quieter half: periodic saves were accepted and never happened, so a stopped run had
    nothing to go back to."""
    with pytest.raises(ValueError, match = "save_steps is not supported"):
        _h3_cfg(save_steps = 50).normalized()


def test_the_defaults_are_untouched():
    """Neither knob set is the normal case and must stay valid, or the family becomes untrainable."""
    cfg = _h3_cfg().normalized()
    assert cfg.save_steps == 0
    assert cfg.resume_from_checkpoint is None


def test_an_image_family_keeps_its_checkpointing():
    """The refusal is scoped to the loop that lacks the support, not to training at large."""
    from core.training.diffusion_train_common import DiffusionLoraConfig

    cfg = DiffusionLoraConfig(
        base_model = "stabilityai/stable-diffusion-xl-base-1.0",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
        save_steps = 50,
    ).normalized()
    assert cfg.save_steps == 50


# ── the four review fixes ────────────────────────────────────────────────────


def test_int8_training_applies_h3s_small_m_guards():
    """Without the family, ``adaln_proj`` (Linear(2688 -> 96768) on the dense checkpoint) clears
    the 512-feature floor, gets quantized, then runs at M = 1 and raises
    ``self.size(0) needs to be greater than 16`` on the first step -- after the whole 66.3 GB
    base has loaded. The family also carries the PAD list, which is the other half of the same
    contract: context_embedder and the two token_refiner blocks are in neither exclusion list
    precisely because they are meant to be padded instead."""
    from core.inference.diffusion_transformer_quant import (
        exclude_tokens_for_scheme,
        pad_tokens_for_scheme,
    )
    from core.training import diffusion_dit_trainer as dit
    from core.training import diffusion_h3_trainer as h3

    assert "adaln_proj" in exclude_tokens_for_scheme("int8", "minimax-h3")
    assert "adaln_proj" not in exclude_tokens_for_scheme("int8", None)
    assert pad_tokens_for_scheme("int8", "minimax-h3")

    # The trainer hands the family down, and the shared helper reads BOTH lists off it.
    src = Path(h3.__file__).read_text()
    assert "_int8_quantize_base(transformer, cfg.resolved_family)" in src
    dit_src = Path(dit.__file__).read_text()
    body = dit_src.split("def _int8_quantize_base", 1)[1].split("\ndef ", 1)[0]
    assert 'exclude_tokens_for_scheme("int8", family)' in body
    assert 'apply_small_m_padding(transformer, "int8", family)' in body


def test_int8_quantize_base_pads_the_families_small_m_linears(monkeypatch):
    # The behavioural half: the helper must call the padding pass, with the family, AFTER
    # quantize_ (which is what reparents the Linears the pass wraps).
    import torch.nn as nn

    from core.training import diffusion_dit_trainer as dit

    order: list = []
    fake_ao = types.ModuleType("torchao.quantization")
    fake_ao.Int8WeightOnlyConfig = lambda: "int8cfg"
    fake_ao.quantize_ = lambda model, cfg, filter_fn = None: order.append("quantize")
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setitem(sys.modules, "torchao.quantization", fake_ao)
    monkeypatch.setattr(
        "core.inference.diffusion_transformer_quant.apply_small_m_padding",
        lambda transformer, scheme, family, **kw: order.append(("pad", scheme, family)),
    )

    dit._int8_quantize_base(nn.Linear(4, 4), "minimax-h3")
    assert order == ["quantize", ("pad", "int8", "minimax-h3")]


def test_a_non_default_lora_alpha_survives_the_export(tmp_path):
    """``load_lora_adapter`` reads the rank off the B matrices and then sets
    ``lora_alpha = r`` unless the file carries adapter metadata, so an adapter trained at
    rank 16 / alpha 32 used to come back at half its trained strength. The metadata layout is
    diffusers' own: JSON under ``lora_adapter_metadata``, every key packed with the same
    ``transformer.`` prefix the tensors carry, because the loader strips it off both."""
    from safetensors import safe_open

    from core.training.diffusion_h3_trainer import _save_lora

    _save_lora(
        str(tmp_path),
        {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 2)},
        {"r": 16, "lora_alpha": 32, "target_modules": {"to_q"}},
    )
    with safe_open(str(tmp_path / "pytorch_lora_weights.safetensors"), framework = "pt") as f:
        meta = f.metadata()
    assert meta["format"] == "pt"
    recorded = json.loads(meta["lora_adapter_metadata"])
    assert recorded["transformer.lora_alpha"] == 32
    assert recorded["transformer.r"] == 16
    # A set is not JSON-serialisable; diffusers lists them, and so must this.
    assert recorded["transformer.target_modules"] == ["to_q"]


def test_the_export_without_a_config_still_writes_a_plain_file(tmp_path):
    # The EMA path and the existing callers pass no config; that must stay a valid file.
    from safetensors.torch import load_file

    from core.training.diffusion_h3_trainer import _save_lora

    _save_lora(str(tmp_path), {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 2)})
    assert list(load_file(str(tmp_path / "pytorch_lora_weights.safetensors"))) == [
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
    ]


def test_a_mostly_silent_soundtrack_is_refused_rather_than_padded(tmp_path):
    """The mandatory-audio check only asks whether the container declares an audio stream, and
    the pad had no ceiling, so a clip carrying a fraction of a second of sound was accepted and
    zero-padded out to the whole window -- training the shared adapter on the silence the check
    exists to keep out."""
    import numpy as np

    from core.training import diffusion_h3_clips as clips

    target = clips.h3_audio_sample_count(clips.H3_FRAMES_PER_CHUNK)

    class _Resampler:
        def resample(self, frame):
            return [] if frame is None else [frame]

    class _Container:
        def __init__(self, n):
            self._n = n

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def decode(self, audio = 0):
            block = types.SimpleNamespace(
                # A real signal, not zeros: this test is about DURATION, and an all-zero window
                # is separately refused for being silent, which would mask what it checks.
                to_ndarray = lambda: np.full(
                    (self._n * clips.H3_AUDIO_CHANNELS,), 0.25, dtype = "float32"
                )
            )
            return [block]

    def fake_av(n):
        return types.SimpleNamespace(
            AudioResampler = lambda **kw: _Resampler(),
            open = lambda path: _Container(n),
        )

    # A tail a few milliseconds short is still padded, as the comment promises.
    short_tail = target - int(0.002 * clips.H3_AUDIO_SAMPLING_RATE)
    out = clips._decode_clip_audio(tmp_path / "a.mp4", target, fake_av(short_tail), np)
    assert out.shape == (clips.H3_AUDIO_CHANNELS, target)

    # A soundtrack that is materially shorter is refused instead.
    with pytest.raises(ValueError, match = "of audio for a"):
        clips._decode_clip_audio(tmp_path / "a.mp4", target, fake_av(target // 10), np)


def test_a_soundtrack_that_is_silent_throughout_is_refused(tmp_path):
    """A muted track is full length, so the duration checks above all pass and the window comes
    back all zeros -- the very target the short-audio refusal exists to keep out, arriving by a
    route that refusal cannot see. Training on it teaches the shared adapter to stop making
    sound, so it is refused with the rest of the audio validation rather than accepted.

    The control matters as much as the refusal: a track that is quiet, or silent for almost all
    of its length with one real sound in it, is still a usable soundtrack and must be kept."""
    import types

    import numpy as np

    from core.training import diffusion_h3_clips as clips

    target = clips.h3_audio_sample_count(41)

    class _Resampler:
        def resample(self, frame):
            return [] if frame is None else [frame]

    def container_of(fill):
        class _Container:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def decode(self, audio = 0):
                buf = fill(target)
                return [types.SimpleNamespace(to_ndarray = lambda b = buf: b)]

        return _Container()

    def fake_av(fill):
        return types.SimpleNamespace(
            AudioResampler = lambda **kw: _Resampler(),
            open = lambda path, _f = fill: container_of(_f),
        )

    def silent(n):
        return np.zeros((n * clips.H3_AUDIO_CHANNELS,), dtype = "float32")

    def dithered(n):
        # Not exact zeros: an encode/decode round trip leaves rounding dust on digital silence,
        # and a threshold that only caught 0.0 would wave that through.
        return (silent(n) + 1e-7).astype("float32")

    def one_real_sound(n):
        buf = silent(n)
        buf[: 64 * clips.H3_AUDIO_CHANNELS] = 0.4
        return buf

    def quiet(n):
        return (silent(n) + 0.01).astype("float32")

    for fill in (silent, dithered):
        with pytest.raises(ValueError, match = "silent all the way through"):
            clips._decode_clip_audio(tmp_path / "a.mp4", target, fake_av(fill), np)

    for fill in (one_real_sound, quiet):
        out = clips._decode_clip_audio(tmp_path / "a.mp4", target, fake_av(fill), np)
        assert out.shape == (clips.H3_AUDIO_CHANNELS, target)


def test_the_knobs_h3_cannot_honour_are_refused_or_normalised():
    """Same rule as the other pins this trainer applies: a setting the loop does not implement
    must not be accepted and then silently dropped. Refused where the value can only be an
    explicit non-default, normalised where the SCHEMA DEFAULT is the one the loop disagrees with
    -- refusing there would 422 every untouched request.

    Read off the shared preflight, which is also what the START ROUTE calls before it evicts the
    user's resident models."""
    from dataclasses import replace as _replace

    from core.training.diffusion_train_common import h3_train_unsupported_reason

    base = _h3_cfg().normalized()
    assert h3_train_unsupported_reason(base) is None, "a default H3 request must start"

    refused = {
        "mixed_precision": ("no", "requires bf16"),
        "train_batch_size": (2, "batch size 1"),
        "compile_transformer": ("on", "does not compile"),
        "cond_cache_dir": ("/tmp/cond", "conditioning cache"),
        "cfg_dropout": (0.1, "guidance-distilled"),
        "weighting_scheme": ("bell", "timestep-weighted"),
        "resolution": (700, "multiples of"),
    }
    for field, (value, fragment) in refused.items():
        reason = h3_train_unsupported_reason(_replace(base, **{field: value}))
        assert reason and fragment in reason, f"{field} was accepted: {reason!r}"

    # Config-only, so it never answers for another family and never touches the host.
    from core.training.diffusion_train_common import DiffusionLoraConfig

    other = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
        mixed_precision = "no",
    )
    assert h3_train_unsupported_reason(other.normalized()) is None


def test_the_h3_preflight_runs_before_the_start_route_evicts_anything():
    # The whole point of the shared helper: these used to reach the worker and 400 there, with
    # the resident models already freed for a run that never began.
    import inspect

    import routes.training as tr

    src = inspect.getsource(tr.start_diffusion_training)
    # The CALL, not the docstring's mention of it.
    assert src.index("h3_train_unsupported_reason") < src.index(
        "asyncio.to_thread(_free_gpu_for_diffusion_training)"
    )


def test_the_augmentation_knobs_record_what_h3_actually_does():
    """Every frame goes through the same centre cover-crop and nothing is flipped, but the
    schema defaults say the opposite (center_crop=False, random_flip=True), so an untouched
    request described augmentation that never happened. Normalised rather than refused: a
    refusal would 422 every default request."""
    from dataclasses import replace as _replace

    from core.training import diffusion_h3_trainer as h3
    from core.training.diffusion_train_common import train_recipe_overrides

    src = Path(h3.__file__).read_text()
    # Applied from the shared table, which the service also reads for the persisted run record.
    assert "replace(cfg, **train_recipe_overrides(cfg))" in src
    overrides = train_recipe_overrides(_h3_cfg().normalized())
    assert overrides["center_crop"] is True and overrides["random_flip"] is False
    # And the two fields really are settable on the config the trainer normalises.
    cfg = _replace(_h3_cfg(), center_crop = True, random_flip = False)
    assert cfg.center_crop is True and cfg.random_flip is False


def test_h3_is_advertised_as_trainable_with_the_precisions_it_has():
    """The Train panel reads an EMPTY precision_modes on a non-SDXL family as "this GPU cannot
    train this family" and disables Start with "Not supported on this GPU". MiniMax-H3 was only
    in _FLOW_TRAIN_FAMILIES, while the info builder keyed the precision branch on
    _DIT_TRAIN_FAMILIES, so it reported [] even on a host that can train, and the trainer this
    PR adds was unreachable from Unsloth.

    Judged against a reference DiT family rather than against a hardcoded list, so the test
    describes the host it runs on: on a GPU-less runner BOTH are legitimately empty, and the
    invariant under test is that H3 is never the only one that is. fp8/mxfp8 stay out either
    way -- the trainer refuses both outright."""
    from core.training.diffusion_train_common import family_train_infos

    infos = {i["name"]: i for i in family_train_infos()}
    h3 = infos.get("minimax-h3")
    reference = next((infos[n] for n in ("flux.1", "ltx-2", "qwen-image") if n in infos), None)
    if h3 is None or reference is None:
        pytest.skip("this diffusers carries neither H3 nor a reference DiT family")

    expected = [m for m in reference["precision_modes"] if m not in ("fp8", "mxfp8")]
    assert h3["precision_modes"] == expected
    # The case the bug was: a host that CAN train the DiT families must be able to train H3.
    if reference["precision_modes"]:
        assert h3["precision_modes"], "an empty list disables Start in the Train panel"
    # compile is the one DiT lever that must NOT follow: "on" is refused by the trainer.
    assert h3["supports_compile"] is False
    assert infos["sdxl"]["supports_compile"] is True
    if reference["precision_modes"]:
        assert reference["supports_compile"] is True


def test_an_over_long_clip_says_that_only_its_opening_trains(tmp_path, monkeypatch):
    """The window is the clip's FIRST num_frames and the latents are cached once for the run, so
    a longer source trains only its opening while its caption describes the whole scene. That is
    the dataset contract, but it was silent, which is how the mismatch went unnoticed."""
    import numpy as np

    from core.training import diffusion_h3_clips as clips

    notes: list = []

    class _Frame:
        def to_image(self):
            from PIL import Image
            return Image.new("RGB", (64, 64))

    class _Stream:
        average_rate = clips.H3_FPS
        guessed_rate = clips.H3_FPS
        duration = 10 * clips.H3_FPS
        time_base = 1 / clips.H3_FPS

    class _Container:
        streams = types.SimpleNamespace(video = [_Stream()], audio = [object()])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def decode(self, video = 0):
            return [_Frame() for _ in range(10 * clips.H3_FPS)]

    monkeypatch.setattr(
        clips, "av", types.SimpleNamespace(open = lambda p: _Container()), raising = False
    )
    monkeypatch.setitem(sys.modules, "av", types.SimpleNamespace(open = lambda p: _Container()))
    monkeypatch.setattr(
        clips,
        "_decode_clip_audio",
        lambda path, target, av, np_: np.zeros((clips.H3_AUDIO_CHANNELS, target), "float32"),
    )

    clips.decode_clip(
        tmp_path / "a.mp4",
        num_frames = clips.H3_FRAMES_PER_CHUNK,
        width = 64,
        height = 64,
        on_note = notes.append,
    )
    assert notes and "trains its first" in notes[0]

    # A clip already at the training duration says nothing.
    notes.clear()
    _Stream.duration = clips.H3_FRAMES_PER_CHUNK
    clips.decode_clip(
        tmp_path / "a.mp4",
        num_frames = clips.H3_FRAMES_PER_CHUNK,
        width = 64,
        height = 64,
        on_note = notes.append,
    )
    assert notes == []


# ── what the run RECORDS vs what the loop RUNS ───────────────────────────────
def test_the_precision_recorded_for_h3_is_the_one_its_loop_runs_in():
    """``identity_for_config`` records the EFFECTIVE mixed precision, and the helper it reads it
    from keyed the "this loop ignores the knob" branch on _DIT_TRAIN_FAMILIES, which H3 is not in.

    H3's loop is byte-identical to the DiT loop here -- bf16 on CUDA, fp32 otherwise, and the
    string "mixed_precision" appears nowhere in it -- so the two families must resolve to the same
    answer. Judged against a reference DiT family rather than a hardcoded value, so the test
    describes the host it runs on (a CPU runner legitimately answers "no" for both).

    Not reachable as a live mismatch today: an H3 request may only be bf16 (two independent gates
    below), and H3 writes no checkpoint at all, so nothing consults the identity. Pinned anyway
    because both of those are documented as temporary."""
    import inspect

    from core.training import diffusion_dit_trainer, diffusion_h3_trainer
    from core.training.diffusion_train_common import effective_mixed_precision

    h3 = _cfg().normalized()
    reference = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
        mixed_precision = "bf16",
    ).normalized()
    assert effective_mixed_precision(h3) == effective_mixed_precision(reference)

    # The claim the equality rests on: the same weight-dtype rule in both loops, and no reader of
    # mixed_precision in the H3 one.
    dtype_rule = 'weight_dtype = torch.bfloat16 if device == "cuda" else torch.float32'
    h3_src = inspect.getsource(diffusion_h3_trainer)
    assert dtype_rule in h3_src
    assert dtype_rule in inspect.getsource(diffusion_dit_trainer)
    assert "mixed_precision" not in h3_src


def test_an_h3_run_can_never_record_a_precision_it_did_not_use():
    """The three gates that keep the identity honest today, so a change to any of them fails here
    rather than in a resume refusal: fp16 is refused in validation, any other non-bf16 value is
    refused by the shared start preflight, and the family writes no checkpoint to record into."""
    from dataclasses import replace as _replace

    from core.training.diffusion_train_common import h3_train_unsupported_reason

    with pytest.raises(ValueError, match = "bf16"):
        _cfg(mixed_precision = "fp16").normalized()
    reason = h3_train_unsupported_reason(_replace(_cfg().normalized(), mixed_precision = "no"))
    assert reason and "requires bf16" in reason
    with pytest.raises(ValueError, match = "resume_from_checkpoint"):
        _cfg(resume_from_checkpoint = "/tmp/out").normalized()
    with pytest.raises(ValueError, match = "save_steps"):
        _cfg(save_steps = 100).normalized()


def test_the_persisted_h3_recipe_is_the_one_the_loop_runs():
    """The loop replaces center_crop / random_flip / snr_gamma, but the run record is written by
    the PARENT from the config handed to ``service.start`` -- the child's ``replace`` never
    reached it, so Previous runs described cropping, flipping and min-SNR weighting that no step
    used. Both sides read one shared table now."""
    import inspect

    from core.training.diffusion_training_service import DiffusionTrainingService
    from core.training.diffusion_train_common import train_recipe_overrides

    assert train_recipe_overrides(_cfg().normalized()) == {
        "center_crop": True,
        "random_flip": False,
        "snr_gamma": None,
        # The loop encodes each clip once into one cached tuple and frees the VAEs; it reads
        # neither field, so a request for no cache, or for the schema's four variants, ran
        # cached-with-one anyway and the record said otherwise.
        "cache_latents": True,
        "cache_variants": 1,
    }
    # Config-only, and no other family's loop disagrees with its request.
    other = DiffusionLoraConfig(
        base_model = "black-forest-labs/FLUX.1-dev",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
    ).normalized()
    assert train_recipe_overrides(other) == {}

    # The two appliers: the trainer for what runs, the service for what is recorded.
    from core.training import diffusion_h3_trainer

    assert "train_recipe_overrides" in inspect.getsource(diffusion_h3_trainer)
    start_src = inspect.getsource(DiffusionTrainingService.start)
    assert "train_recipe_overrides" in start_src
    assert "self._config.update(" in start_src


# ── start-route gates ────────────────────────────────────────────────────────
def test_the_strict_start_gate_probes_the_h3_transformer_not_modular_pipeline(monkeypatch):
    """A diffusers old enough to predate H3's blocks still exports the generic ``ModularPipeline``,
    so the listing probe (which reads the family's own transformer class) hid H3 while a direct
    POST /diffusion/start sailed through both training gates, evicted the resident GPU models and
    failed in the child. Both gates read the same probe class now."""
    from core.inference.diffusion_families import family_pipeline_available, family_probe_class
    from core.training.diffusion_train_common import (
        _trainable_family_spec,
        training_pipeline_import_error,
    )

    fam = _trainable_family_spec("minimax-h3")
    assert fam is not None and fam.pipeline_class == "ModularPipeline"
    assert family_probe_class(fam) == "MiniMaxH3Transformer3DModel"

    old = types.ModuleType("diffusers")
    old.__version__ = "0.36.0"
    old.ModularPipeline = object  # the generic entry point, present for several releases
    old.FluxPipeline = object
    monkeypatch.setitem(sys.modules, "diffusers", old)

    assert family_pipeline_available(fam) is False
    reason = training_pipeline_import_error("minimax-h3")
    assert reason and "MiniMaxH3Transformer3DModel" in reason
    # A conventional family on the same install is untouched.
    assert training_pipeline_import_error("flux.1") is None

    # And the trainer-side half refuses before it can reach a download.
    with pytest.raises(ValueError, match = "MiniMaxH3Transformer3DModel"):
        _cfg().normalized()


def test_the_probe_still_fails_open_for_a_record_that_names_no_pipeline_class():
    """``family_pipeline_available`` hides nothing it cannot judge. The probe helper reads
    ``pipeline_class`` with a default, so a record without one arrives as the empty string, and
    ``hasattr(diffusers, "")`` is False -- which would drop a usable repo out of a picker over a
    stand-in family object rather than over a real missing class."""
    from core.inference.diffusion_families import family_pipeline_available, family_probe_class

    assert family_probe_class(object()) == ""
    assert family_pipeline_available(object()) is True


def test_a_local_modular_h3_pipeline_is_an_acceptable_training_base(tmp_path):
    """A local MiniMax-H3 pipeline carries ``modular_model_index.json`` and NO
    ``model_index.json`` -- that is the layout ``ModularPipeline.from_pretrained`` reads and the
    one the local-model scanners already count. The shared shape check knew only the conventional
    index, so it rejected the only local layout the family has."""
    from core.training.diffusion_train_common import _assert_trusted_base_model

    modular = tmp_path / "MiniMax-H3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text("{}")

    _assert_trusted_base_model(str(modular), allow_modular = True)
    # Off by default: a conventional DiffusionPipeline load still needs the conventional index.
    with pytest.raises(ValueError, match = "model_index.json"):
        _assert_trusted_base_model(str(modular))
    # And a directory that is neither is still refused on both paths.
    empty = tmp_path / "not-a-pipeline"
    empty.mkdir()
    for allow in (True, False):
        with pytest.raises(ValueError, match = "not a diffusers pipeline directory"):
            _assert_trusted_base_model(str(empty), allow_modular = allow)


def test_the_start_route_reaches_the_same_modular_verdict_as_the_trainer():
    """The route runs the trainers' trust gate first, so an untrusted base 400s before the
    resident GPU models are freed. It called it with the default conventional-only shape check,
    while the H3 loop called it with allow_modular -- so a local modular pipeline, the only local
    layout the family HAS, was rejected at the route and the loop that can load it never ran.
    Both sides now select off one set, which is the only way they cannot disagree again."""
    import inspect

    from core.training import diffusion_h3_trainer
    from core.training.diffusion_train_common import MODULAR_BASE_FAMILIES
    from routes import training as training_routes

    assert "minimax-h3" in MODULAR_BASE_FAMILIES
    # A family whose trainer loads a conventional pipeline must NOT be in it, or the shape check
    # would start accepting a modular directory its loader cannot read.
    assert MODULAR_BASE_FAMILIES.isdisjoint({"flux.1", "qwen-image", "sdxl", "ltx-2"})

    # Neither call site may hard-code its answer: that is what let them drift apart.
    route_src = inspect.getsource(training_routes.start_diffusion_training)
    assert "MODULAR_BASE_FAMILIES" in route_src
    assert "_assert_trusted_base_model(\n            config.get" in route_src
    trainer_src = inspect.getsource(diffusion_h3_trainer)
    assert "MODULAR_BASE_FAMILIES" in trainer_src
    assert "allow_modular = True" not in trainer_src


def test_h3_advertises_that_it_cannot_checkpoint():
    """save_steps is REFUSED for a checkpointless family, not ignored, so a panel that keeps
    offering "Checkpoint every" offers a value that rejects Start with nothing on the control
    saying why. The family metadata already carries supports_compile for the same reason."""
    from core.training.diffusion_train_common import CHECKPOINTLESS_FAMILIES, family_train_infos

    infos = {info["name"]: info for info in family_train_infos()}
    assert "minimax-h3" in CHECKPOINTLESS_FAMILIES
    for name, info in infos.items():
        assert info["supports_checkpoints"] == (name not in CHECKPOINTLESS_FAMILIES), name
    # The flag has to agree with the validation, or the panel hides a control that works or
    # offers one that does not.
    with pytest.raises(ValueError, match = "save_steps is not supported"):
        _cfg(save_steps = 50).normalized()
    _cfg(save_steps = 0).normalized()


def test_the_batch_cap_survives_the_response_model():
    """family_train_infos emitting it is not enough: the route builds a DiffusionTrainableFamily
    from that dict, and Pydantic drops any field the model does not declare. Undeclared, the
    panel reads the cap as undefined, keeps rendering Batch, and sends a carried-over value the
    validation refuses -- the clamp exists but never receives its input."""
    from core.training.diffusion_train_common import (
        SINGLE_SEQUENCE_FAMILIES,
        family_train_infos,
    )
    from models.training import DiffusionTrainableFamily

    infos = {info["name"]: info for info in family_train_infos()}
    assert "minimax-h3" in SINGLE_SEQUENCE_FAMILIES
    for name, info in infos.items():
        expected = 1 if name in SINGLE_SEQUENCE_FAMILIES else None
        assert info["max_train_batch_size"] == expected, name
        # Through the wire model, which is where it was being dropped.
        assert DiffusionTrainableFamily(**info).max_train_batch_size == expected, name
    # And the cap has to agree with the validation, or the panel hides a control that works.
    from core.training.diffusion_train_common import h3_train_unsupported_reason

    assert "batch size 1" in (
        h3_train_unsupported_reason(_cfg(train_batch_size = 2).normalized()) or ""
    )
    assert h3_train_unsupported_reason(_cfg(train_batch_size = 1).normalized()) is None


def test_the_h3_conditioner_load_carries_the_hub_token(monkeypatch):
    """``ModularPipeline.from_pretrained``'s token opens the modular INDEX only: every component
    is fetched by its own ``from_pretrained`` inside ``load_components``, which swallows a failure
    as a logger.warning and leaves the attribute unset. A gated or private base therefore loaded
    its components anonymously and died on a None, after the route's authenticated preflight had
    passed. The inference H3 loader forwards it again for exactly this reason."""
    from core.training import diffusion_h3_trainer as h3

    seen: list[dict] = []

    class _Placed:
        def to(self, device):
            return self

    class _Pipe:
        text_encoder = _Placed()
        vae = _Placed()
        audio_vae = _Placed()

        def load_components(self, **kwargs):
            seen.append(kwargs)

    class _Modular:
        @staticmethod
        def from_pretrained(
            path,
            token = None,
            **kw,
        ):
            seen.append({"index_token": token, **kw})
            return _Pipe()

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(ModularPipeline = _Modular))
    monkeypatch.setattr(h3, "_assert_component_grid", lambda pipe: None)
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: "/live/hub")

    cfg = types.SimpleNamespace(base_model = "unsloth/private-h3", hf_token = "hf_secret")
    h3._load_conditioners(cfg, "cpu")

    assert seen[0] == {"index_token": "hf_secret", "cache_dir": "/live/hub"}
    component_loads = seen[1:]
    assert len(component_loads) == 2
    assert all(call.get("token") == "hf_secret" for call in component_loads), component_loads
    # And every one of them pinned to the LIVE cache root. An unset cache_dir resolves through
    # huggingface_hub's import-time constant, which a mid-session cache-folder change does not
    # update, and the training subprocess is spawned without the cache-environment wrapper: the
    # components already in the selected root would be missed and re-downloaded into the old one.
    assert all(call.get("cache_dir") == "/live/hub" for call in component_loads), component_loads

    # No token configured -> the kwarg is omitted entirely rather than sent as None.
    seen.clear()
    h3._load_conditioners(
        types.SimpleNamespace(base_model = "MiniMaxAI/MiniMax-H3", hf_token = None), "cpu"
    )
    assert all("token" not in call for call in seen[1:]), seen


@pytest.mark.parametrize("base_precision", ["bf16", "nf4"])
def test_the_h3_denoiser_load_is_pinned_to_the_live_cache(monkeypatch, base_precision):
    """The denoiser is the 145 GB half, so this is the expensive one to leave unpinned."""
    from core.training import diffusion_h3_trainer as h3

    seen: list[dict] = []

    class _Model:
        def to(self, device):
            return self

    class _Transformer:
        @staticmethod
        def from_pretrained(path, **kw):
            seen.append(kw)
            return _Model()

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            MiniMaxH3Transformer3DModel = _Transformer,
            BitsAndBytesConfig = lambda **kw: kw,
        ),
    )
    monkeypatch.setattr("core.inference.diffusion.hub_cache_dir", lambda: "/live/hub")

    cfg = types.SimpleNamespace(base_model = "MiniMaxAI/MiniMax-H3", hf_token = None)
    h3._load_transformer(cfg, "cpu", base_precision)

    assert len(seen) == 1
    assert seen[0]["cache_dir"] == "/live/hub"
    assert seen[0]["subfolder"] == "transformer"


def test_the_audio_decode_stops_at_the_training_window(tmp_path):
    """An over-long source is accepted input -- only its first num_frames train, and the caller is
    warned -- so the soundtrack past that window is never used. The video loop already breaks at
    the window; the audio one decoded and resampled the whole recording and kept every chunk
    before truncating, so a long clip cost a recording's worth of time and memory to build a
    sub-second sample (and failed on damage in a region that is never read)."""
    import numpy as np

    from core.training import diffusion_h3_clips as clips

    target = clips.h3_audio_sample_count(clips.H3_FRAMES_PER_CHUNK)
    decoded = 0

    class _Resampler:
        def resample(self, frame):
            return [] if frame is None else [frame]

    class _Container:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def decode(self, audio = 0):
            nonlocal decoded
            for _ in range(4000):  # an hour of audio behind a one-second window
                decoded += 1
                yield types.SimpleNamespace(
                    # Nonzero for the same reason as above: the subject here is where decoding
                    # STOPS, and a silent window is refused before that can be asserted.
                    to_ndarray = lambda: np.full(
                        (target * clips.H3_AUDIO_CHANNELS,), 0.25, dtype = "float32"
                    )
                )

    av = types.SimpleNamespace(
        AudioResampler = lambda **_kw: _Resampler(), open = lambda _p: _Container()
    )
    out = clips._decode_clip_audio(tmp_path / "a.mp4", target, av, np)

    assert out.shape == (clips.H3_AUDIO_CHANNELS, target)
    assert decoded == 1, f"decoded {decoded} blocks for a window one block wide"


def _stub_training_stack(monkeypatch):
    """Satisfy the four training-stack names ``_train_h3`` binds on entry.

    The backend CI image installs neither diffusers nor peft, so the function-level
    ``from diffusers.optimization import get_scheduler`` aborted the cache-gate tests before they
    reached the gate. None of the four is called on the path under test -- the gate raises, or
    the faked ``_load_transformer`` does -- so binding them to placeholders keeps the tests
    running everywhere rather than skipping them on the only host that runs them in CI."""
    scheduler = types.ModuleType("diffusers.optimization")
    scheduler.get_scheduler = lambda *_a, **_k: None
    training_utils = types.ModuleType("diffusers.training_utils")
    training_utils.cast_training_params = lambda *_a, **_k: None
    peft_utils = types.ModuleType("peft.utils")
    peft_utils.get_peft_model_state_dict = lambda *_a, **_k: {}
    peft = types.ModuleType("peft")
    peft.LoraConfig = object
    peft.utils = peft_utils
    for name, module in (
        # The SUBMODULES only, never a bare ``diffusers`` parent. ``from a.b import c`` is
        # satisfied straight from ``sys.modules["a.b"]`` without importing ``a``, and a stub
        # parent would be worse than none here: ``resolve_trainable_family`` absorbs an
        # unimportable diffusers but REFUSES one that imports and lacks
        # MiniMaxH3Transformer3DModel, so faking the package turns a skipped probe into a raise.
        ("diffusers.optimization", scheduler),
        ("diffusers.training_utils", training_utils),
        ("peft", peft),
        ("peft.utils", peft_utils),
    ):
        # Only where the real one is absent, so a developer host keeps running the real imports.
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 -- absent or broken both mean "use the placeholder"
            monkeypatch.setitem(sys.modules, name, module)


def _h3_cache_run(monkeypatch, *, num_clips: int):
    """Drive ``_train_h3`` through phase 2 only, with every model call faked.

    ``_load_transformer`` raises a sentinel, so "got past the cache gate" is observable without
    the 66 GB base."""
    import torch

    from core.training import diffusion_h3_trainer as h3

    _stub_training_stack(monkeypatch)

    class _Placed:
        def to(self, _device):
            return self

    pipe = types.SimpleNamespace(
        text_encoder = _Placed(), processor = None, vae = _Placed(), audio_vae = _Placed()
    )
    monkeypatch.setattr(h3, "_load_conditioners", lambda cfg, device: pipe)
    monkeypatch.setattr(h3, "_encode_prompt", lambda *_a, **_k: torch.zeros(1, 8, 16))
    monkeypatch.setattr(h3, "_dataset_canvas", lambda _path, _short_edge: (64, 64))
    monkeypatch.setattr(h3, "decode_clip", lambda *_a, **_k: (None, None))
    monkeypatch.setattr(
        h3,
        "_encode_video_stats",
        lambda *_a: (torch.zeros(1, 4, 2, 8, 8), torch.zeros(1, 4, 2, 8, 8)),
    )
    num_audio = h3.h3_audio_latent_count(h3.H3_TRAIN_NUM_FRAMES)
    monkeypatch.setattr(
        h3,
        "_encode_audio_latents",
        lambda *_a: torch.zeros(1, h3.H3_AUDIO_LATENT_CHANNELS, num_audio),
    )

    def _no_transformer(*_a, **_k):
        raise RuntimeError("reached phase 3")

    monkeypatch.setattr(h3, "_load_transformer", _no_transformer)

    cfg = _cfg(output_dir = "/tmp/h3-out").normalized()
    pairs = [(f"/clips/{i}.mp4", "a caption") for i in range(num_clips)]
    return lambda: h3._train_h3(
        cfg,
        pairs,
        random.Random(0),
        "cpu",
        torch.float32,
        lambda *_a, **_k: None,
        lambda: False,
        lambda: True,
    )


def test_the_h3_latent_cache_is_size_gated_like_the_image_trainers(monkeypatch):
    """Both shared image trainers measure the FIRST real entry and refuse to build a latent cache
    over the host-memory budget. H3 built one entry per discovered clip unconditionally, so a
    large clip dataset was OOM-killed at the end of the whole preparation with nothing saved.

    It cannot answer the way they do -- they drop the cache and encode per step, and H3 frees both
    VAEs to make room for the 66 GB transformer -- so it says so up front instead, with the
    numbers and the same explicit override the size gate already has."""
    from core.training import diffusion_train_common as common

    monkeypatch.delenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", raising = False)
    monkeypatch.setattr(common, "_LATENT_CACHE_BUDGET_BYTES", 1)

    with pytest.raises(ValueError, match = "latent cache") as exc:
        _h3_cache_run(monkeypatch, num_clips = 4)()
    assert "UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE" in str(exc.value)


def test_the_h3_latent_cache_gate_honours_the_explicit_override(monkeypatch):
    """The gate is the AUTOMATIC default only, exactly as it is for the image trainers: a user who
    has the RAM and says so gets the cache, and the run carries on into phase 3."""
    from core.training import diffusion_train_common as common

    monkeypatch.setenv("UNSLOTH_DIFFUSION_FORCE_LATENT_CACHE", "1")
    monkeypatch.setattr(common, "_LATENT_CACHE_BUDGET_BYTES", 1)

    with pytest.raises(RuntimeError, match = "reached phase 3"):
        _h3_cache_run(monkeypatch, num_clips = 4)()


def _write_rotated_clip(
    path,
    *,
    theta: int,
    width: int = 640,
    height: int = 360,
    seconds: int = 2,
):
    """A real H.264+AAC clip whose display matrix says ``theta``, coded ``width`` x ``height``."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")

    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate = H3_FPS)
        video.width, video.height = width, height
        video.pix_fmt = "yuv420p"
        audio = out.add_stream("aac", rate = 48000)
        if theta:
            video.set_display_rotation(theta)
        for i in range(H3_FPS * seconds):
            # An asymmetric picture, so a wrong rotation cannot look like a right one.
            img = np.zeros((height, width, 3), dtype = "uint8")
            img[: height // 3, :, 0] = 255
            img[:, : max(1, i * 4), 1] = 255
            for packet in video.encode(av.VideoFrame.from_ndarray(img, format = "rgb24")):
                out.mux(packet)
        # An audible tone, not silence: these clips go through the real decode, which refuses a
        # soundtrack that is silent end to end. A rotation test must not depend on that refusal
        # being absent, and a clip with a working soundtrack is the realistic input anyway.
        t = np.arange(48000 * seconds, dtype = "float32") / 48000.0
        tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype("float32").reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(tone, format = "fltp", layout = "mono")
        frame.sample_rate = 48000
        for packet in audio.encode(frame):
            out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        for packet in audio.encode():
            out.mux(packet)


def test_a_rotated_clip_reports_its_display_rotation(tmp_path):
    """PyAV hands back the CODED frame: unlike the ffmpeg CLI it does not apply the display
    matrix. A portrait phone clip is stored landscape with a 90 degree matrix, so the rotation
    has to be read off the frame or every target trains sideways."""
    av = pytest.importorskip("av")
    from core.training.diffusion_h3_clips import display_rotation_degrees

    clip = tmp_path / "portrait.mp4"
    _write_rotated_clip(clip, theta = 90)
    with av.open(str(clip)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(video = 0))
        assert (frame.width, frame.height) == (640, 360), "the coded frame is still landscape"
        assert display_rotation_degrees(frame, stream) == 270


def test_an_unrotated_clip_reports_no_rotation(tmp_path):
    """The common case must stay exactly as it was, matrix or no matrix."""
    av = pytest.importorskip("av")
    from core.training.diffusion_h3_clips import display_rotation_degrees

    clip = tmp_path / "landscape.mp4"
    _write_rotated_clip(clip, theta = 0)
    with av.open(str(clip)) as container:
        frame = next(container.decode(video = 0))
        assert display_rotation_degrees(frame, container.streams.video[0]) == 0


def test_the_canvas_follows_the_displayed_orientation_not_the_coded_one(tmp_path):
    """``_dataset_canvas`` used to read ``codec_context`` alone, so a portrait clip picked a
    LANDSCAPE canvas and every frame was then cover-cropped down to it -- the sides of the real
    picture thrown away, on top of training it sideways."""
    pytest.importorskip("av")
    from core.training.diffusion_h3_trainer import _dataset_canvas

    landscape = tmp_path / "landscape.mp4"
    portrait = tmp_path / "portrait.mp4"
    _write_rotated_clip(landscape, theta = 0)
    _write_rotated_clip(portrait, theta = 90)

    wide_w, wide_h = _dataset_canvas(str(landscape), 512)
    tall_w, tall_h = _dataset_canvas(str(portrait), 512)
    assert wide_w > wide_h, (wide_w, wide_h)
    assert tall_h > tall_w, "a rotated clip must train on a portrait canvas"
    assert (tall_w, tall_h) == (wide_h, wide_w)


def test_decode_clip_returns_the_rotated_picture(tmp_path):
    """End to end: the cached frames are the picture a player shows, not the coded one."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    from core.training.diffusion_h3_clips import decode_clip

    clip = tmp_path / "portrait.mp4"
    _write_rotated_clip(clip, theta = 90, seconds = 3)
    frames, _waveform = decode_clip(clip, num_frames = 8, width = 64, height = 128)

    assert frames.shape == (8, 128, 64, 3)
    # The red band occupies the top third of the CODED frame. Rotated 90 CCW for display it
    # moves to the left third, so the rotation is observable in the pixels rather than assumed.
    first = frames[0].astype("int32")
    left = first[:, : 64 // 3, 0].mean()
    right = first[:, -(64 // 3) :, 0].mean()
    assert left > right + 40, f"red band did not move to the left edge (left {left}, right {right})"
    top = first[: 128 // 3, :, 0].mean()
    assert left > top + 40, "the band is still along the top, so no rotation was applied"


def test_a_local_modular_pipeline_under_a_gguf_path_is_not_refused_as_gguf(tmp_path):
    """``resolve_trainable_family`` exempted a local checkout by ``model_index.json`` only. A
    MODULAR_BASE_FAMILIES base has ``modular_model_index.json`` and no ``model_index.json``, so
    the one local layout MiniMax-H3 HAS was refused as a GGUF repo whenever its path contained
    'gguf', before model_family was even consulted."""
    from core.training.diffusion_train_common import resolve_trainable_family

    base = tmp_path / "gguf" / "MiniMax-H3"
    base.mkdir(parents = True)
    (base / "modular_model_index.json").write_text("{}", encoding = "utf-8")

    assert resolve_trainable_family(str(base), model_family = "minimax-h3") == "minimax-h3"


def test_a_real_gguf_path_without_a_pipeline_is_still_refused(tmp_path):
    """The exemption is the index file, not the word: an actual GGUF pick must still be refused."""
    from core.training.diffusion_train_common import resolve_trainable_family

    base = tmp_path / "gguf" / "MiniMax-H3-GGUF"
    base.mkdir(parents = True)
    with pytest.raises(ValueError, match = "GGUF"):
        resolve_trainable_family(str(base), model_family = "minimax-h3")


def test_the_advertised_h3_vram_floor_covers_the_measured_run_peak():
    """The field's own contract is the WHOLE-RUN peak, because that is what a card has to hold.
    A real run on this branch peaked at 77.76 GB, so the advertised floor may not sit under it:
    the chip is what a user sizes a host from."""
    from core.training.diffusion_train_common import _FAMILY_TRAIN_SPECS
    assert _FAMILY_TRAIN_SPECS["minimax-h3"]["qlora_vram_gb"] >= 78
