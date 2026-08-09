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
                to_ndarray = lambda: np.zeros((self._n * clips.H3_AUDIO_CHANNELS,), dtype = "float32")
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

    src = Path(h3.__file__).read_text()
    assert "center_crop = True, random_flip = False" in src
    # And the two fields really are settable on the config the trainer normalises.
    cfg = _replace(_h3_cfg(), center_crop = True, random_flip = False)
    assert cfg.center_crop is True and cfg.random_flip is False


def test_h3_is_advertised_as_trainable_with_the_precisions_it_has():
    """The Train panel reads an EMPTY precision_modes on a non-SDXL family as "this GPU cannot
    train this family" and disables Start with "Not supported on this GPU". MiniMax-H3 was only
    in _FLOW_TRAIN_FAMILIES, while the info builder keyed the precision branch on
    _DIT_TRAIN_FAMILIES, so it reported [] even on a host that can train, and the trainer this
    PR adds was unreachable from Studio.

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
