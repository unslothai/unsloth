# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 helpers shared by the Diffusers and stable-diffusion.cpp paths."""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Must stay equal to the minimax-h3 family's `gguf_repo`. They are the same one-click pick, and
# main's test_curated_gguf_repos_are_unsloth_mirrors only checks the family field, so a divergence
# here would let that test pass while the actual download still came from a community repack.
# tests/test_video_backend.py::test_the_h3_native_repo_matches_the_family_gguf_repo pins the pair.
# The mirror carries the Qwen3-VL encoder quants as well as the denoisers, so this repo alone
# satisfies h3_native_hub_files' first two entries.
H3_GGUF_REPO = "unsloth/MiniMax-H3-GGUF"
# The VAEs live beside the denoisers, so the native pick is one repo we control end to end. It was
# Comfy-Org/MiniMax-H3, which put a community repack in the download path of BOTH H3 paths; an
# install that already holds those bytes keeps using them through h3_component_source below,
# because the HF cache is keyed by repo id and repointing alone re-downloads ~6 GB.
H3_COMPONENT_REPO = "unsloth/MiniMax-H3-GGUF"
# Where the component files came from originally. Only for reusing an existing cache entry: a
# fresh install never reads it. The pairing itself lives in diffusion_families'
# _SD_CPP_LEGACY_SOURCES, which owns this decision for every mirrored asset; this name is what the
# delete-cached claims read, and a test pins the two together.
H3_LEGACY_COMPONENT_REPO = "Comfy-Org/MiniMax-H3"
H3_VIDEO_VAE = "vae/minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "vae/minimax_h3_audio_vae_fp32.safetensors"
H3_QWEN_Q2 = "qwen3vl_32b_minimax_h3-Q2_K_M.gguf"
H3_QWEN_Q4 = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"

# Measured with the merged Diffusers T2VA workflow and component-level CPU
# offload. The base is the largest component plus runtime overhead; activation
# memory scales with spatiotemporal volume across the tested 960x544 and
# 1344x768, 124-345 frame matrix. The guard covers allocator variation around
# the measured success and OOM boundaries.
H3_DIFFUSERS_VRAM_BASE_GB = 68.5
H3_DIFFUSERS_VRAM_GB_PER_MPIXEL_FRAME = 0.08

# The terms H3_DIFFUSERS_VRAM_BASE_GB is built from, so a load that shrinks one of the big
# components can rebuild the floor from what it actually holds instead of the released sizes.
#
# With everything under enable_auto_cpu_offload the base is the LARGEST SINGLE RESIDENT COMPONENT
# plus runtime overhead, not the sum: at any instant one component is on the device and the rest
# are parked on the host. That is exactly why seeding a 20 GB pre-quantized denoiser moved this
# number by nothing -- the 66.7 GB conditioner was already the larger of the two and simply took
# over as the maximum. Both have to shrink before the floor does.
#
# ONE case breaks the max, and it is the case a pre-quantized denoiser creates. A torchao module
# does not survive being moved mid-block, so _load_h3_modular_pipeline PINS it to the device and
# takes it out of the offload rotation (see pin_prequantized_module). It is then resident for the
# whole generation and the floor becomes additive: denoiser + whichever offloaded component is
# largest. Measured at 960x544x124 with the int8 denoiser pinned, torch.cuda.max_memory_allocated:
#
#   bfloat16 conditioner   94.62 GB   =  20.3 + 66.7 + 5.18 activations + 2.44
#   int8 conditioner       55.20 GB   =  20.3 + 27.1 + 5.18 activations + 2.58
#
# so 2.6 covers the pinned overhead on the conservative side of both. Note what the first row says
# about the shipped constant: a pinned denoiser and a dense conditioner really need ~95 GB, and the
# flat 68.5 under-states that by 26 GB. Rebuilding the floor from the resident components fixes
# that under-estimate in the same stroke as crediting the saving.
H3_DIFFUSERS_VRAM_OVERHEAD_GB = 1.8
H3_DIFFUSERS_VRAM_PINNED_OVERHEAD_GB = 2.6
H3_TEXT_ENCODER_BF16_GB = 66.7
H3_TRANSFORMER_BF16_GB = 66.3
# Video + audio VAE, from the family's bf16_components_gb. Only a floor for the offloaded term: it
# stops a very small conditioner from claiming a base no component rotation could actually fit in.
H3_VAE_RESIDENT_GB = 11.1


# Resident decimal GB of each hosted pre-quantized denoiser, from the artifact sizes in
# unsloth/MiniMax-H3-FP8 (MiniMax-H3-INT8.pt 18.86 GiB, MiniMax-H3-FP8.pt 18.87 GiB). Both are the
# PRUNED (curve-form adaLN) partition, which is why they are so far under half the 66.3 GB dense
# denoiser rather than at it.
H3_TRANSFORMER_PREQUANT_GB: dict[str, float] = {"int8": 20.3, "fp8": 20.3}


def h3_transformer_resident_gb(scheme: Optional[str]) -> float:
    """Resident decimal GB of the denoiser this load holds: the hosted pre-quantized size for an
    ENGAGED scheme, else the released bfloat16 one."""
    return H3_TRANSFORMER_PREQUANT_GB.get(scheme or "", H3_TRANSFORMER_BF16_GB)


def h3_diffusers_vram_base_gb(
    *,
    text_encoder_gb: Optional[float] = None,
    transformer_gb: Optional[float] = None,
    transformer_pinned: bool = False,
) -> float:
    """The resident-weights floor for an H3 Diffusers load.

    Every argument defaults to today's behaviour -- the RELEASED bfloat16 sizes and no pinning --
    so the no-argument call reproduces ``H3_DIFFUSERS_VRAM_BASE_GB`` exactly and nothing that does
    not pass a size changes.

    ``transformer_pinned`` is the ENGAGED fact, not the request: only a denoiser that
    ``pin_prequantized_module`` actually pinned is resident alongside the offload rotation."""
    text_encoder = H3_TEXT_ENCODER_BF16_GB if text_encoder_gb is None else float(text_encoder_gb)
    transformer = H3_TRANSFORMER_BF16_GB if transformer_gb is None else float(transformer_gb)
    if transformer_pinned:
        offloaded = max(text_encoder, H3_VAE_RESIDENT_GB)
        return transformer + offloaded + H3_DIFFUSERS_VRAM_PINNED_OVERHEAD_GB
    return max(text_encoder, transformer) + H3_DIFFUSERS_VRAM_OVERHEAD_GB


def estimate_h3_diffusers_vram_gb(
    width: int,
    height: int,
    num_frames: int,
    *,
    text_encoder_gb: Optional[float] = None,
    transformer_gb: Optional[float] = None,
    transformer_pinned: bool = False,
) -> float:
    """Measured available-VRAM floor for an H3 Diffusers generation.

    ``text_encoder_gb`` / ``transformer_gb`` are the RESIDENT sizes this load actually holds and
    ``transformer_pinned`` whether the denoiser was taken out of the offload rotation; all unset
    keeps the released-bfloat16 floor this shipped with."""
    volume_mpixel_frames = width * height * num_frames / 1_000_000
    base = h3_diffusers_vram_base_gb(
        text_encoder_gb = text_encoder_gb,
        transformer_gb = transformer_gb,
        transformer_pinned = transformer_pinned,
    )
    return base + (H3_DIFFUSERS_VRAM_GB_PER_MPIXEL_FRAME * volume_mpixel_frames)


# The VRAM at which the offload tier changes, and the host floor of the tier above it. Both are
# the shipped values, unchanged: that tier is only reachable on a >= 132 GB device, where the
# component sizes below are not what stands between a load and a generation, and there is no
# measurement here to justify moving it.
H3_DIFFUSERS_HOST_RAM_TIER_VRAM_GB = 132.0
H3_DIFFUSERS_HOST_RAM_HIGH_VRAM_GB = 85.0
# The offload tier parks every component on the host, so its floor is their SUM (unlike the VRAM
# floor, which is the largest resident one). Derived, not newly measured: it is the shipped 150.0
# minus the released component sum, so the released configuration still asks for exactly 150.0 and
# only a load holding smaller components asks for less.
H3_DIFFUSERS_HOST_RAM_HEADROOM_GB = 5.9


def estimate_h3_diffusers_host_ram_gb(
    available_vram_gb: float,
    *,
    text_encoder_gb: Optional[float] = None,
    transformer_gb: Optional[float] = None,
) -> float:
    """Host-RAM floor for the offload tier selected at the available VRAM.

    ``text_encoder_gb`` / ``transformer_gb`` are the RESIDENT sizes this load actually holds, as
    for the VRAM floor; unset keeps the released bfloat16 sizes, so the no-argument call is the
    number this shipped with.

    Sizing this from the released pair while the VRAM floor is sized from the engaged one refuses
    the exact configuration the hosted quantized components exist for: an 80 GB device holding the
    int8 conditioner and the int8 denoiser clears the VRAM check at 55 GB and is then told it needs
    150 GB of system RAM for 47.5 GB of weights.

    A pinned denoiser is still counted here. It lives on the device during the generation, but it
    was built on the host to get there, and keeping it in the sum errs toward refusing a load that
    would have fitted rather than admitting one that will not."""
    if available_vram_gb >= H3_DIFFUSERS_HOST_RAM_TIER_VRAM_GB:
        return H3_DIFFUSERS_HOST_RAM_HIGH_VRAM_GB
    text_encoder = H3_TEXT_ENCODER_BF16_GB if text_encoder_gb is None else float(text_encoder_gb)
    transformer = H3_TRANSFORMER_BF16_GB if transformer_gb is None else float(transformer_gb)
    return text_encoder + transformer + H3_VAE_RESIDENT_GB + H3_DIFFUSERS_HOST_RAM_HEADROOM_GB


# torch.autocast casts the weight and bias of these module types to the autocast dtype on
# entry. Norms sit on autocast's float32 promote list and bare parameters are read directly,
# so both must keep their source precision.
_AUTOCAST_WEIGHT_MODULE_NAMES = ("Linear", "Conv1d", "Conv2d", "Conv3d")


def _module_bytes(module: Any) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def trim_h3_video_vae(vae: Any, *, workflow: str) -> dict[str, int]:
    """Drop what the H3 video VAE cannot use and pre-cast what autocast casts anyway.

    Two thirds of an H3 render's peak is not activations. Measured at 640x384 across 124
    frames, a 20.25 GB int8 denoiser peaks at 36.96 GB, and the gap is almost all weights:
    the video VAE alone is 10.42 GB because diffusers pins it to float32, and a further
    4.91 GB is autocast's own float16 copy of those weights.

    ``MiniMaxH3VideoDecodeStep`` wraps ``vae.decode`` in ``torch.autocast(float16)``. Autocast
    casts every Linear and Conv weight it meets and caches the copy for the lifetime of the
    region, so the float32 original and its float16 twin are both resident through the whole
    decode. Storing those weights as float16 up front makes the cast a no-op and removes both:
    ``x.to(float16).to(float16)`` is ``x.to(float16)``, so the arithmetic is unchanged rather
    than merely close. The audio VAE decode is NOT under autocast, so it is left alone.

    The encoder half goes only for a workflow that never encodes. ``t2va`` starts from noise,
    so ``vae.encoder`` and ``vae.quant_conv`` are dead weight; a future image-conditioned
    workflow needs them, hence the explicit check rather than an unconditional drop.

    Returns a byte report for the caller to log. Never raises: a diffusers release that
    renames these attributes should cost the saving, not the render.
    """
    import torch

    # Every lookup below is a getattr with a default, so a None vae and a vae whose
    # attributes moved both fall through to a zero report without a separate guard.
    report = {"encoder_freed": 0, "decoder_freed": 0}

    if workflow == "t2va":
        for name in ("encoder", "quant_conv"):
            module = getattr(vae, name, None)
            if module is None:
                continue
            report["encoder_freed"] += _module_bytes(module)
            setattr(vae, name, None)

    for holder in ("decoder", "post_quant_conv"):
        module = getattr(vae, holder, None)
        if module is None:
            continue
        before = _module_bytes(module)
        for sub in module.modules():
            if type(sub).__name__ not in _AUTOCAST_WEIGHT_MODULE_NAMES:
                continue
            for name, param in list(sub.named_parameters(recurse = False)):
                if param.dtype is not torch.float32:
                    continue
                setattr(
                    sub,
                    name,
                    torch.nn.Parameter(param.data.to(torch.float16), requires_grad = False),
                )
        report["decoder_freed"] += before - _module_bytes(module)

    return report


# ── canvas geometry ──────────────────────────────────────────────────────────
#
# MiniMax-H3's upstream canvas rule, shared by both engines.
H3_CANVAS_SHORT_EDGE = 768
H3_CANVAS_MAX_PIXELS = 768 * 1344
H3_CANVAS_MULTIPLE = 32
# Trained aspect-ratio range.
H3_MIN_ASPECT_RATIO = 1 / 4
H3_MAX_ASPECT_RATIO = 4

# Upstream stretches the first frame and center cover-crops the last.
H3_ANCHOR_FIRST = "first"
H3_ANCHOR_LAST = "last"


def h3_canvas_for_aspect(aspect_width: float, aspect_height: float) -> tuple[int, int]:
    """Resolve MiniMax-H3's canvas for an aspect ratio.

    Raises ValueError outside the trained 1:4 to 4:1 range.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(
            f"The source image has no usable aspect ratio ({aspect_width}x{aspect_height})."
        )
    ratio = aspect_width / aspect_height
    if not H3_MIN_ASPECT_RATIO <= ratio <= H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1; this image is "
            f"{aspect_width:g}x{aspect_height:g} ({ratio:.2f}:1). Crop it first."
        )
    if ratio >= 1.0:
        width, height = H3_CANVAS_SHORT_EDGE * ratio, float(H3_CANVAS_SHORT_EDGE)
    else:
        width, height = float(H3_CANVAS_SHORT_EDGE), H3_CANVAS_SHORT_EDGE / ratio
    area = width * height
    if area > H3_CANVAS_MAX_PIXELS:
        scale = math.sqrt(H3_CANVAS_MAX_PIXELS / area)
        width, height = width * scale, height * scale
    snap = lambda v: max(  # noqa: E731
        H3_CANVAS_MULTIPLE, round(v / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE
    )
    return snap(width), snap(height)


def fit_h3_keyframe(image: Any, width: int, height: int, *, anchor: str) -> Any:
    """Fit a keyframe using MiniMax-H3's asymmetric first/last-frame rules."""
    from PIL import Image

    target = (max(1, int(width)), max(1, int(height)))
    if anchor == H3_ANCHOR_FIRST:
        return image.resize(target, Image.LANCZOS)
    if anchor != H3_ANCHOR_LAST:
        raise ValueError(f"Unknown keyframe anchor {anchor!r}.")
    source_w, source_h = image.size
    scale = max(target[0] / source_w, target[1] / source_h)
    crop_w = min(source_w, math.ceil(target[0] / scale))
    crop_h = min(source_h, math.ceil(target[1] / scale))
    left = (source_w - crop_w) // 2
    top = (source_h - crop_h) // 2
    return image.resize(target, Image.LANCZOS, box = (left, top, left + crop_w, top + crop_h))


# ── omni references (Ref2VA) ─────────────────────────────────────────────────
#
# Ref2VA uses a separate transformer partition selected at load time.
H3_TASK_KEYFRAMES = "fl2va"
H3_TASK_REFERENCES = "ref2va"

# Upstream request limits.
H3_MAX_REF_IMAGES = 9
H3_MAX_REF_VIDEOS = 3
H3_MAX_REF_AUDIOS = 3
H3_MAX_REFERENCES = 12
# A reference video's trained window, in seconds.
H3_REF_VIDEO_MIN_SECONDS = 2.0
H3_REF_VIDEO_MAX_SECONDS = 15.0
# How far a trim may reach past the video track before it is refused. A container reports its
# longest track, so a file whose audio outruns its video reads as longer than it can show, and
# a client picking an interval from that duration (HTMLMediaElement.duration, in Unsloth's case)
# asks for slightly more video than exists. Within this margin the last frame is held instead.
H3_REF_TRIM_COVERAGE_SLACK_SECONDS = 0.5
H3_FPS = 24

# "match" uses the generation area. Diffusers-only "max" uses a 2048px short edge.
H3_REF_SIZE_MATCH = "match"
H3_REF_SIZE_MAX = "max"
H3_REF_IMAGE_SHORT_EDGE = 2048
# H3 downscales references immediately, so this path can accept larger bounded sources.
H3_REF_IMAGE_SOURCE_MAX_SIDE = 8192
H3_REF_IMAGE_SOURCE_MAX_PIXELS = 32_000_000


def validate_h3_reference_trim(
    start_seconds: Optional[float], end_seconds: Optional[float]
) -> Optional[tuple[float, float]]:
    """Validate an explicit reference-media interval, or return None for the whole source."""
    if start_seconds is None and end_seconds is None:
        return None
    if start_seconds is None or end_seconds is None:
        raise ValueError("Reference video trim start and end must be provided together.")
    start, end = float(start_seconds), float(end_seconds)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError("Reference video trim must satisfy 0 <= start < end.")
    duration = end - start
    if duration + 1e-6 < H3_REF_VIDEO_MIN_SECONDS or duration - 1e-6 > H3_REF_VIDEO_MAX_SECONDS:
        raise ValueError(
            f"MiniMax-H3 reference video trims must be {H3_REF_VIDEO_MIN_SECONDS:g} to "
            f"{H3_REF_VIDEO_MAX_SECONDS:g} seconds; this one is {duration:.1f}s."
        )
    return start, end


def h3_transformer_task(filename: str) -> str:
    """Which H3 task a picked GGUF denoiser serves, from its published name."""
    name = Path(filename).name.lower()
    return H3_TASK_REFERENCES if name.startswith("minimax_h3_ref2va") else H3_TASK_KEYFRAMES


def h3_denoiser_component(task: Optional[str]) -> str:
    """The pipeline component / base-repo subfolder holding ``task``'s denoiser partition.

    One repo, two partitions: the keyframe workflows (fl2va, and text-only through it) denoise
    against ``transformer``, the reference workflow against ``transformer_ref``. Diffusers names
    the components that way too, so this single answer serves the seed target, the offload pin and
    the config subfolder alike -- and seeding ``transformer`` for a reference load would leave the
    denoise step with no denoiser and pull the dense 66.28 GB partition anyway."""
    return (
        "transformer_ref" if (task or "").strip().lower() == H3_TASK_REFERENCES else "transformer"
    )


def fit_h3_reference_image(image: Any, *, width: int, height: int, policy: str) -> Any:
    """Scale a reference to its policy limit without changing its aspect ratio or upscaling."""
    from PIL import Image

    source_w, source_h = image.size
    if policy == H3_REF_SIZE_MAX:
        scale = min(1.0, H3_REF_IMAGE_SHORT_EDGE / max(1, min(source_w, source_h)))
    elif policy == H3_REF_SIZE_MATCH:
        scale = min(1.0, math.sqrt((width * height) / max(1, source_w * source_h)))
    else:
        raise ValueError(f"Unknown reference image size policy {policy!r}.")
    snap = lambda v: max(  # noqa: E731
        H3_CANVAS_MULTIPLE, round(v * scale / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE
    )
    return image.resize((snap(source_w), snap(source_h)), Image.LANCZOS)


def h3_reference_frame_size(source_w: int, source_h: int) -> tuple[int, int]:
    """Resolve a reference-video frame size without upscaling."""
    width, height = h3_canvas_for_aspect(source_w, source_h)
    if source_w * source_h < width * height:
        snap = lambda v: max(  # noqa: E731
            H3_CANVAS_MULTIPLE, round(v / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE
        )
        width, height = snap(source_w), snap(source_h)
    return width, height


def decode_h3_reference_video(
    blob: bytes,
    *,
    trim_start_seconds: Optional[float] = None,
    trim_end_seconds: Optional[float] = None,
    decode_audio: bool = True,
) -> tuple[list, Optional[Any], Optional[int]]:
    """Decode one uploaded video to 24 fps frames plus its soundtrack, if it carries one.

    Returns ``(frames, waveform, sample_rate)``. The frames land on MiniMax-H3's own 24 fps by
    whole-frame drop and duplicate -- the selection ffmpeg's fps filter made in the reference
    implementation, and the one the Diffusers blocks make from a declared rate -- so both
    engines receive a stream that is already on the model's clock. The waveform is float32
    ``(samples, channels)`` at the container's own rate; both engines resample it themselves.
    ``decode_audio=False`` avoids touching an embedded track that a replacement will supersede.
    """
    import io

    import av
    import numpy as np

    trim = validate_h3_reference_trim(trim_start_seconds, trim_end_seconds)
    video_timeline_start = None
    if trim is None:
        frames, duration, video_timeline_start = _decode_h3_video_without_trim(blob, av)
    else:
        try:
            frames, video_timeline_start = _decode_h3_video_trim_by_timestamp(blob, av, trim)
        except _H3MediaTimestampsUnavailable:
            # Ordinal selection must restart because a seek makes frame zero keyframe-relative.
            frames, video_timeline_start = _decode_h3_video_trim_by_ordinal(blob, av, trim)
        duration = trim[1] - trim[0]
    if not frames or duration + 1e-6 < H3_REF_VIDEO_MIN_SECONDS:
        raise ValueError(
            f"MiniMax-H3 reference videos run {H3_REF_VIDEO_MIN_SECONDS:g} to "
            f"{H3_REF_VIDEO_MAX_SECONDS:g} seconds; this one is {duration:.1f}s."
        )
    expected_frames = int(round(duration * H3_FPS))
    if trim is not None and len(frames) < expected_frames:
        # Hold the last frame across a shortfall the slack allows, as the trim decoders do at
        # their own endpoint; a larger gap means the range really was not there.
        if len(frames) + math.ceil(H3_REF_TRIM_COVERAGE_SLACK_SECONDS * H3_FPS) < expected_frames:
            raise ValueError("That reference video did not cover the selected range.")
        frames.extend([frames[-1]] * (expected_frames - len(frames)))
    frames = frames[:expected_frames]

    waveform, sample_rate = (None, None)
    if decode_audio:
        with av.open(io.BytesIO(blob)) as container:
            if container.streams.audio:
                waveform, sample_rate = _decode_audio_stream(
                    container,
                    np,
                    trim_start_seconds = trim[0] if trim else None,
                    trim_end_seconds = trim[1] if trim else None,
                    timeline_start_seconds = video_timeline_start,
                    untrimmed_duration_seconds = len(frames) / H3_FPS if trim is None else None,
                )
    return frames, waveform, sample_rate


class _H3MediaTimestampsUnavailable(Exception):
    """Internal signal to retry a trim from the beginning using the declared rate."""


def _stream_start_seconds(stream: Any) -> Optional[float]:
    start_time = getattr(stream, "start_time", None)
    time_base = getattr(stream, "time_base", None)
    if start_time is None or time_base is None:
        return None
    return float(start_time * time_base)


def _frame_timestamp_seconds(frame: Any) -> Optional[float]:
    pts = getattr(frame, "pts", None)
    time_base = getattr(frame, "time_base", None)
    if pts is None or time_base is None:
        return None
    return float(pts * time_base)


def _seek_media(container: Any, stream: Any, absolute_seconds: float, *, label: str) -> bool:
    """Seek backward near an absolute stream timestamp when its time base permits it."""
    time_base = getattr(stream, "time_base", None)
    if time_base is None:
        return False
    offset = math.floor(absolute_seconds / float(time_base) + 1e-6)
    try:
        container.seek(offset, backward = True, any_frame = False, stream = stream)
    except Exception as exc:  # noqa: BLE001 -- decoder-specific seek errors become input feedback
        raise ValueError(f"Could not seek to the requested reference {label} trim start.") from exc
    return True


def _fit_h3_video_frame(
    frame: Any, fitted_size: Optional[tuple[int, int]]
) -> tuple[Any, tuple[int, int]]:
    from PIL import Image

    image = frame.to_image().convert("RGB")
    if fitted_size is None:
        fitted_size = h3_reference_frame_size(*image.size)
    if image.size != fitted_size:
        image = image.resize(fitted_size, Image.LANCZOS)
    return image, fitted_size


def _decode_h3_video_without_trim(blob: bytes, av: Any) -> tuple[list, float, Optional[float]]:
    """Preserve the existing whole-file rate-based behavior and its 15-second refusal."""
    import io

    with av.open(io.BytesIO(blob)) as container:
        if not container.streams.video:
            raise ValueError("That reference file carries no video track.")
        stream = container.streams.video[0]
        timeline_start = _stream_start_seconds(stream)
        source_fps = float(stream.average_rate or stream.guessed_rate or H3_FPS)
        if source_fps <= 0:
            source_fps = float(H3_FPS)
        max_source_frames = math.floor(H3_REF_VIDEO_MAX_SECONDS * source_fps + 1e-6)
        frames = []
        decoded_count = 0
        next_target = 0
        fitted_size = None
        for source_index, frame in enumerate(container.decode(video = 0)):
            decoded_count = source_index + 1
            if decoded_count > max_source_frames:
                raise ValueError(
                    f"MiniMax-H3 reference videos run {H3_REF_VIDEO_MIN_SECONDS:g} to "
                    f"{H3_REF_VIDEO_MAX_SECONDS:g} seconds; this one is longer than "
                    f"{H3_REF_VIDEO_MAX_SECONDS:g}s. Trim it first."
                )
            target_source = int(next_target * source_fps / H3_FPS)
            if target_source > source_index:
                continue
            image, fitted_size = _fit_h3_video_frame(frame, fitted_size)
            while int(next_target * source_fps / H3_FPS) <= source_index:
                frames.append(image)
                next_target += 1
    if decoded_count == 0:
        raise ValueError("That reference video decoded to no frames.")
    return frames, decoded_count / source_fps, timeline_start


def _decode_h3_video_trim_by_timestamp(
    blob: bytes, av: Any, trim: tuple[float, float]
) -> tuple[list, Optional[float]]:
    """Select an explicit interval from presentation timestamps and stop at its endpoint."""
    import io

    start, end = trim
    target_count = int(round((end - start) * H3_FPS))
    with av.open(io.BytesIO(blob)) as container:
        if not container.streams.video:
            raise ValueError("That reference file carries no video track.")
        stream = container.streams.video[0]
        timeline_start = _stream_start_seconds(stream)
        if timeline_start is not None and start > 0:
            _seek_media(container, stream, timeline_start + start, label = "video")

        frames = []
        fitted_size = None
        candidate = None
        last_relative = None
        last_duration = None

        def _fill_until(relative_limit: float) -> None:
            nonlocal fitted_size
            if candidate is None:
                return
            wanted = min(
                target_count,
                max(0, math.ceil((relative_limit - start) * H3_FPS - 1e-6)),
            )
            if wanted <= len(frames):
                return
            image, fitted_size = _fit_h3_video_frame(candidate, fitted_size)
            frames.extend([image] * (wanted - len(frames)))

        for frame in container.decode(video = 0):
            timestamp = _frame_timestamp_seconds(frame)
            if timestamp is None:
                raise _H3MediaTimestampsUnavailable
            if timeline_start is None:
                # Without a declared start or seek, the first frame is the timeline origin.
                timeline_start = timestamp
            relative = timestamp - timeline_start
            if candidate is not None:
                _fill_until(relative)
                if len(frames) >= target_count:
                    return frames, timeline_start
            candidate = frame
            last_relative = relative
            duration = getattr(frame, "duration", None)
            time_base = getattr(frame, "time_base", None)
            last_duration = (
                float(duration * time_base)
                if duration is not None and time_base is not None and duration > 0
                else None
            )

        if candidate is None or last_relative is None:
            raise ValueError("That reference video decoded to no frames in the selected range.")
        declared_duration = getattr(stream, "duration", None)
        stream_time_base = getattr(stream, "time_base", None)
        source_end = (
            float(declared_duration * stream_time_base)
            if declared_duration is not None and stream_time_base is not None
            else last_relative + (last_duration or 0.0)
        )
        if source_end + H3_REF_TRIM_COVERAGE_SLACK_SECONDS < end:
            raise ValueError("Reference video trim ends after the source video.")
        _fill_until(end)
        if len(frames) < target_count:
            raise ValueError("That reference video decoded to no frames in the selected range.")
        return frames, timeline_start


def _decode_h3_video_trim_by_ordinal(
    blob: bytes, av: Any, trim: tuple[float, float]
) -> tuple[list, Optional[float]]:
    """Fallback for streams whose decoded frames carry no presentation timestamps."""
    import io
    with av.open(io.BytesIO(blob)) as container:
        if not container.streams.video:
            raise ValueError("That reference file carries no video track.")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.guessed_rate or H3_FPS)
        if source_fps <= 0:
            source_fps = float(H3_FPS)
        # The frame on screen at t is the last one starting at or before it, so the start
        # floors where the exclusive end ceils. Ceiling both skips the frame straddling a
        # fractional start, drifting this fallback ahead of the timestamp path.
        start_source_frame = math.floor(trim[0] * source_fps + 1e-6)
        end_source_frame = math.ceil(trim[1] * source_fps - 1e-6)
        target_count = int(round((trim[1] - trim[0]) * H3_FPS))
        frames = []
        decoded_count = 0
        next_target = 0
        fitted_size = None
        for source_index, frame in enumerate(container.decode(video = 0)):
            decoded_count = source_index + 1
            if source_index < start_source_frame:
                continue
            if source_index >= end_source_frame:
                break
            selected_index = source_index - start_source_frame
            target_source = int(next_target * source_fps / H3_FPS)
            if target_source > selected_index:
                continue
            image, fitted_size = _fit_h3_video_frame(frame, fitted_size)
            while (
                len(frames) < target_count
                and int(next_target * source_fps / H3_FPS) <= selected_index
            ):
                frames.append(image)
                next_target += 1
        slack_frames = math.ceil(H3_REF_TRIM_COVERAGE_SLACK_SECONDS * source_fps)
        if decoded_count + slack_frames < end_source_frame:
            raise ValueError("Reference video trim ends after the source video.")
        if not frames:
            raise ValueError("That reference video decoded to no frames in the selected range.")
        return frames, _stream_start_seconds(stream)


def decode_h3_reference_audio(
    blob: bytes,
    *,
    trim_start_seconds: Optional[float] = None,
    trim_end_seconds: Optional[float] = None,
) -> tuple[Any, int]:
    """Decode one uploaded audio file to a float32 ``(samples, channels)`` waveform + its rate."""
    import io

    import av
    import numpy as np

    with av.open(io.BytesIO(blob)) as container:
        if not container.streams.audio:
            raise ValueError("That reference file carries no audio track.")
        waveform, sample_rate = _decode_audio_stream(
            container,
            np,
            trim_start_seconds = trim_start_seconds,
            trim_end_seconds = trim_end_seconds,
        )
    if waveform is None:
        raise ValueError("That reference audio decoded to no samples.")
    return waveform, sample_rate


def _decode_audio_stream(
    container: Any,
    np: Any,
    *,
    trim_start_seconds: Optional[float] = None,
    trim_end_seconds: Optional[float] = None,
    timeline_start_seconds: Optional[float] = None,
    untrimmed_duration_seconds: Optional[float] = None,
) -> tuple[Optional[Any], Optional[int]]:
    """Decode the first audio stream as float32 samples at its native rate.

    Decode incrementally to enforce H3's 15-second limit. For untrimmed video, cap embedded audio
    to the video timeline.
    """
    import av

    trim = validate_h3_reference_trim(trim_start_seconds, trim_end_seconds)
    stream = container.streams.audio[0]
    sample_rate = int(stream.codec_context.sample_rate or 48_000)
    if trim is not None:
        try:
            return _decode_audio_trim_by_timestamp(
                container,
                stream,
                np,
                av,
                sample_rate,
                trim,
                timeline_start_seconds = timeline_start_seconds,
            )
        except _H3MediaTimestampsUnavailable:
            start_time = getattr(stream, "start_time", None)
            time_base = getattr(stream, "time_base", None)
            if time_base is None:
                raise ValueError(
                    "Reference audio timestamps are unavailable and its stream cannot be reset."
                ) from None
            try:
                container.seek(int(start_time or 0), backward = True, any_frame = False, stream = stream)
            except Exception as exc:  # noqa: BLE001 -- decoder-specific reset errors
                raise ValueError(
                    "Reference audio timestamps are unavailable and its stream cannot be reset."
                ) from exc

    # One resampler pass gives interleaved float32 whatever the source layout/format was.
    resampler = av.AudioResampler(format = "flt", layout = stream.layout.name, rate = sample_rate)
    channels = len(stream.layout.channels)
    max_samples = math.floor(H3_REF_VIDEO_MAX_SECONDS * sample_rate + 1e-6)
    start_sample = math.floor(trim[0] * sample_rate + 1e-6) if trim else 0
    untrimmed_end_sample = (
        int(round(untrimmed_duration_seconds * sample_rate))
        if untrimmed_duration_seconds is not None
        else None
    )
    end_sample = math.floor(trim[1] * sample_rate + 1e-6) if trim else untrimmed_end_sample
    chunks = []
    source_total = 0

    def _take(resampled: Any) -> bool:
        nonlocal source_total
        block = resampled.to_ndarray().reshape(-1, channels)
        block_start = source_total
        block_end = block_start + block.shape[0]
        source_total = block_end
        if trim is None and untrimmed_duration_seconds is None and source_total > max_samples:
            raise ValueError(
                f"MiniMax-H3 reference audio runs up to {H3_REF_VIDEO_MAX_SECONDS:g} seconds; "
                f"this one is longer. Trim it first."
            )
        take_start = max(block_start, start_sample)
        take_end = min(block_end, end_sample) if end_sample is not None else block_end
        if take_start < take_end:
            chunks.append(block[take_start - block_start : take_end - block_start])
        # A track running past its video is clamped, not refused: encoder padding overshoots
        # routinely, and longer tracks decoded fine before trimming existed. Stopping here
        # also skips a tail no engine receives.
        return end_sample is not None and block_end >= end_sample

    stopped = False
    for frame in container.decode(audio = 0):
        for resampled in resampler.resample(frame):
            if _take(resampled):
                stopped = True
                break
        if stopped:
            break
    if not stopped:
        for resampled in resampler.resample(None):
            _take(resampled)
    # Short soundtracks are kept, not refused, as in the timestamp path above.
    if not chunks:
        return None, None
    return np.concatenate(chunks, axis = 0).astype("float32"), sample_rate


def _decode_audio_trim_by_timestamp(
    container: Any,
    stream: Any,
    np: Any,
    av: Any,
    sample_rate: int,
    trim: tuple[float, float],
    *,
    timeline_start_seconds: Optional[float],
) -> tuple[Any, int]:
    """Place decoded samples on the requested media timeline and stop at the endpoint."""
    start, end = trim
    origin = timeline_start_seconds
    if origin is None:
        origin = _stream_start_seconds(stream)
    if origin is not None and start > 0:
        _seek_media(container, stream, origin + start, label = "audio")

    channels = len(stream.layout.channels)
    target_count = int(round((end - start) * sample_rate))
    output = np.zeros((target_count, channels), dtype = "float32")
    resampler = av.AudioResampler(format = "flt", layout = stream.layout.name, rate = sample_rate)
    copied_any = False
    stopped = False

    def _take(resampled: Any) -> bool:
        nonlocal origin, copied_any
        timestamp = _frame_timestamp_seconds(resampled)
        if timestamp is None:
            raise _H3MediaTimestampsUnavailable
        if origin is None:
            origin = timestamp
        block = resampled.to_ndarray().reshape(-1, channels)
        block_start = timestamp - origin
        block_end = block_start + block.shape[0] / sample_rate
        if block_start >= end:
            return True
        destination_start = int(round((block_start - start) * sample_rate))
        source_start = max(0, -destination_start)
        destination_start = max(0, destination_start)
        take = min(block.shape[0] - source_start, target_count - destination_start)
        if take > 0:
            output[destination_start : destination_start + take] = block[
                source_start : source_start + take
            ]
            copied_any = True
        return block_end >= end

    for frame in container.decode(audio = 0):
        for resampled in resampler.resample(frame):
            if _take(resampled):
                stopped = True
                break
        if stopped:
            break
    if not stopped:
        for resampled in resampler.resample(None):
            if _take(resampled):
                break
    # Nothing copied means no soundtrack here, whether the track ended before the interval
    # or starts after it. Silence instead would be a fabricated track, and would hide the
    # gap from stage_h3_references' positional pairing. A track that merely runs out partway
    # did copy something, so it keeps its silent tail rather than failing.
    if not copied_any:
        return None, None
    return output, sample_rate


def write_h3_reference_wav(path: Path, waveform: Any, sample_rate: int) -> None:
    """Write a reference waveform as the 16-bit PCM WAV sd-cli's --ref-audio loader reads."""
    import wave

    import numpy as np

    samples = np.clip(np.asarray(waveform, dtype = "float32"), -1.0, 1.0)
    if samples.ndim == 1:
        samples = samples[:, None]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(samples.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes((samples * 32767.0).astype("<i2").tobytes())


@dataclass(frozen = True)
class MiniMaxH3References:
    """Decoded references in model order: images, videos, then standalone audio."""

    # Canvas-sized reference images, in <Picture i> order.
    images: tuple = ()
    # (frames, waveform, sample_rate) per <Video k>; waveform is None when silent.
    videos: tuple = ()
    # (waveform, sample_rate) per standalone reference, in <Audio j> order.
    audios: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.images or self.videos or self.audios)

    def count(self) -> int:
        return len(self.images) + len(self.videos) + len(self.audios)


@dataclass(frozen = True)
class MiniMaxH3StagedReferences:
    """``MiniMaxH3References`` written to disk the way sd-cli reads them back."""

    images: tuple[str, ...] = ()
    # Frame DIRECTORIES: sd-cli reads a reference video as images sorted lexicographically.
    videos: tuple[str, ...] = ()
    video_audios: tuple[str, ...] = ()
    audios: tuple[str, ...] = ()


def stage_h3_references(
    references: MiniMaxH3References, scratch: Path
) -> MiniMaxH3StagedReferences:
    """Stage references in sd-cli's positional file layout.

    Video soundtracks must form a prefix because sd-cli pairs them by index.
    """
    images: list[str] = []
    for index, image in enumerate(references.images):
        path = scratch / f"ref-image-{index:02d}.png"
        image.save(path, format = "PNG")
        images.append(str(path))

    videos: list[str] = []
    video_audios: list[str] = []
    pairing_holds = True
    for index, (frames, waveform, sample_rate) in enumerate(references.videos):
        directory = scratch / f"ref-video-{index:02d}"
        directory.mkdir()
        for frame_index, frame in enumerate(frames):
            frame.save(directory / f"{frame_index:05d}.png", format = "PNG")
        videos.append(str(directory))
        if waveform is None:
            # Positional pairing cannot express "skip this one", so stop pairing here.
            pairing_holds = False
            continue
        if not pairing_holds:
            raise ValueError(
                "stable-diffusion.cpp reference-video soundtracks must form a leading "
                "sequence without silent gaps."
            )
        path = scratch / f"ref-video-audio-{index:02d}.wav"
        write_h3_reference_wav(path, waveform, sample_rate)
        video_audios.append(str(path))

    audios: list[str] = []
    for index, (waveform, sample_rate) in enumerate(references.audios):
        path = scratch / f"ref-audio-{index:02d}.wav"
        write_h3_reference_wav(path, waveform, sample_rate)
        audios.append(str(path))

    return MiniMaxH3StagedReferences(
        images = tuple(images),
        videos = tuple(videos),
        video_audios = tuple(video_audios),
        audios = tuple(audios),
    )


def h3_diffusers_references(references: MiniMaxH3References) -> list:
    """``MiniMaxH3References`` as the Diffusers blocks' reference dataclasses, same order.

    Every rate travels with its media: the frames are already on the model's 24 fps and each
    waveform carries the rate it was decoded at, so nothing is re-guessed downstream.
    """
    import torch
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    def waveform_tensor(waveform: Any) -> Any:
        # The blocks take a (channels, samples) tensor; the decoder produces (samples, channels).
        return torch.from_numpy(waveform).transpose(0, 1).contiguous()

    built: list = []
    for image in references.images:
        built.append(MiniMaxH3ImageReference(image = image))
    for frames, waveform, sample_rate in references.videos:
        built.append(
            MiniMaxH3VideoReference(
                frames = list(frames),
                fps = float(H3_FPS),
                audio = None if waveform is None else waveform_tensor(waveform),
                sample_rate = sample_rate,
            )
        )
    for waveform, sample_rate in references.audios:
        built.append(
            MiniMaxH3AudioReference(audio = waveform_tensor(waveform), sample_rate = sample_rate)
        )
    return built


def h3_conditioning_mode(
    *,
    has_first: bool = False,
    has_last: bool = False,
    has_references: bool = False,
) -> str:
    """The task name for one request's conditioning, as MiniMax-H3 and sd.cpp name them.

    Recorded on the gallery clip so a restored recipe says which workflow produced it, and
    used in messages, so the five spellings live in one place.
    """
    if has_references:
        return H3_TASK_REFERENCES
    if has_first and has_last:
        return H3_TASK_KEYFRAMES
    if has_first:
        return "i2va"
    if has_last:
        return "l2va"
    return "t2va"


def is_h3_native(family: Any, kind: str) -> bool:
    return getattr(family, "name", None) == "minimax-h3" and kind == "gguf"


def h3_text_encoder_filename(transformer_filename: str) -> str:
    return H3_QWEN_Q2 if "-q2_" in transformer_filename.lower() else H3_QWEN_Q4


def validate_h3_transformer_filename(filename: str) -> None:
    """Accept either released denoiser partition, and nothing else from the same repo.

    FL2VA serves text-to-video and first/last-frame video; Ref2VA serves omni-reference video.
    Which one is picked IS the task, so both are valid picks -- but the Qwen3-VL encoder and the
    VAEs share the repo and are companions, never denoisers."""
    name = Path(filename).name.lower()
    partitions = ("minimax_h3_fl2va", "minimax_h3_ref2va")
    if not name.startswith(partitions) or not name.endswith(".gguf"):
        raise ValueError(
            "MiniMax-H3 needs a minimax_h3_fl2va*.gguf transformer (text-to-video and "
            "first/last-frame video) or a minimax_h3_ref2va*.gguf one (reference video). The "
            "Qwen encoder and the VAEs are companion models, not picks for either."
        )


def h3_download_error(repo_id: str, filename: str, exc: Exception) -> Exception:
    """Turn a Hub download failure on an H3 component into something a user can act on.

    The Hub says "Repository Not Found ... make sure you are authenticated" for a repo that is
    private or gated as well as for one that genuinely does not exist. For H3 that message is
    actively misleading in both directions: the mirror is real, and the user's own token is
    usually fine. Name the repo, say which of the four components it was, and say what to do.

    Returns the exception to raise (never raises), so the caller keeps ``raise ... from exc`` and
    the original traceback survives. Anything that is not a recognised access error is passed back
    unchanged rather than reworded, so a timeout or a disk-full still reads as itself.
    """
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    if not isinstance(exc, (RepositoryNotFoundError, GatedRepoError)):
        return exc

    role = {
        H3_VIDEO_VAE: "video VAE",
        H3_AUDIO_VAE: "audio VAE",
        H3_QWEN_Q2: "text encoder",
        H3_QWEN_Q4: "text encoder",
    }.get(filename, "denoiser")
    gated = isinstance(exc, GatedRepoError)
    detail = (
        "accept its licence on the Hub, then set a token in Settings"
        if gated
        else "it may be private or not published yet, or your token may not cover it"
    )
    return RuntimeError(
        f"MiniMax-H3 could not download its {role} ({filename}) from {repo_id}: {detail}. "
        f"The other H3 components are unaffected."
    )


def h3_component_source(*files: str) -> str:
    """The repo to fetch the shared VAEs from: our mirror, or the repack it was mirrored from
    when an existing install already holds those exact bytes under the old id.

    The HF cache is keyed by repo id, so moving the id alone would re-download ~5.8 GB on upgrade
    and fail outright offline. The mirror is byte identical (same sha256), so reusing the old
    entry loads the same weights. Fresh installs never take this branch.

    ``prefer_cached_legacy_source`` rather than a probe of our own: it already owns this exact
    mirror-to-repack decision for every other sd.cpp asset, and it counts BOTH cache roots.
    That second part is the reason it has to be this one. The native fetch below passes
    ``reuse_other_cache_root``, so a repack left behind by a cache-folder change is still
    perfectly usable -- but only the OLD repo id can reach it, and a live-root-only probe would
    call it absent and re-pull ~5.8 GB (offline, fail).

    Answered for the files GIVEN, and the native loop asks one at a time because that is how it
    downloads them. A pre-move pull interrupted between the two VAEs leaves only one under the old
    id; asking for the pair calls the old id useless and re-pulls the 5.2 GB already on disk,
    while asking per file reuses it and takes only the other from the mirror. With no argument it
    answers for the pair, for callers describing the pair rather than fetching it.

    PURE: table lookup plus a local stat, no network, so a download plan and the fetch that
    follows it agree on the source.
    """
    wanted = files or (H3_VIDEO_VAE, H3_AUDIO_VAE)
    try:
        from .diffusion_families import prefer_cached_legacy_source
        return prefer_cached_legacy_source(H3_COMPONENT_REPO, wanted)
    except Exception:  # noqa: BLE001 -- an unreadable cache just means "not cached"
        return H3_COMPONENT_REPO


def h3_component_metadata_repo(repo_id: str) -> str:
    """Which repo to read a shared VAE's SIZE from, given the repo its bytes will come from.

    The repack is a source of bytes ALREADY ON DISK, never of network metadata. It may since have
    been renamed or taken down -- the exact failure this move exists to survive -- and a
    ``model_info`` against it would fail the download plan for a load its own cache can still
    satisfy. The mirror's copy is byte identical, so the number is the same.
    """
    return H3_COMPONENT_REPO if repo_id == H3_LEGACY_COMPONENT_REPO else repo_id


def h3_native_hub_files(transformer_filename: str) -> tuple[tuple[str, str], ...]:
    validate_h3_transformer_filename(transformer_filename)
    return (
        (H3_GGUF_REPO, transformer_filename),
        (H3_GGUF_REPO, h3_text_encoder_filename(transformer_filename)),
        (h3_component_source(H3_VIDEO_VAE), H3_VIDEO_VAE),
        (h3_component_source(H3_AUDIO_VAE), H3_AUDIO_VAE),
    )


@dataclass(frozen = True)
class MiniMaxH3NativeRuntime:
    engine: Any
    files: Any
    offload_flags: tuple[str, ...]
    # (size, mtime_ns) of the sd-cli this runtime was built on, taken at load, right after
    # ensure_h3_sd_cpp_binary vetted it for H3 support and accelerator. Every generation compares
    # against THIS, not against whatever the path holds when it starts: an install that lands
    # between the load and a generation replaces the binary in place, and two reads taken after it
    # agree with each other while agreeing with nothing that was ever checked. None means the
    # identity could not be taken, which reads as "cannot vouch" rather than "unchanged".
    binary_identity: Optional[tuple[int, int]] = None


def transcode_video_to_mp4(source: Path, *, fps: int) -> bytes:
    """Convert an sd.cpp WebM into a gallery-compatible H.264/AAC MP4.

    The native backend is available in Unsloth's no-torch runtime, so keep this
    export entirely in PyAV rather than routing decoded frames through Diffusers.
    """
    import av

    tmp = tempfile.NamedTemporaryFile(suffix = ".mp4", delete = False)
    tmp.close()
    target = Path(tmp.name)
    try:
        with av.open(str(source)) as src, av.open(str(target), mode = "w", format = "mp4") as dst:
            source_video = src.streams.video[0]
            output_video = dst.add_stream("libx264", rate = fps)
            output_video.width = int(source_video.codec_context.width)
            output_video.height = int(source_video.codec_context.height)
            output_video.pix_fmt = "yuv420p"

            source_audio = src.streams.audio[0] if src.streams.audio else None
            output_audio = None
            audio_resampler = None
            if source_audio is not None:
                sample_rate = int(source_audio.codec_context.sample_rate or 32000)
                output_audio = dst.add_stream("aac", rate = sample_rate)
                output_audio.layout = "stereo"
                audio_resampler = av.AudioResampler(
                    format = "fltp", layout = "stereo", rate = sample_rate
                )

            selected = (source_video,) if source_audio is None else (source_video, source_audio)
            for packet in src.demux(*selected):
                for frame in packet.decode():
                    if packet.stream.type == "video":
                        for encoded in output_video.encode(frame):
                            dst.mux(encoded)
                    elif output_audio is not None and audio_resampler is not None:
                        for resampled in audio_resampler.resample(frame):
                            for encoded in output_audio.encode(resampled):
                                dst.mux(encoded)

            if output_audio is not None and audio_resampler is not None:
                for resampled in audio_resampler.resample(None):
                    for encoded in output_audio.encode(resampled):
                        dst.mux(encoded)
            for encoded in output_video.encode():
                dst.mux(encoded)
            if output_audio is not None:
                for encoded in output_audio.encode():
                    dst.mux(encoded)
        return target.read_bytes()
    finally:
        target.unlink(missing_ok = True)


def inspect_video(path: Path) -> tuple[int, int, int, bool]:
    """Return width, height, decoded frame count, and audio presence."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width = int(stream.codec_context.width)
        height = int(stream.codec_context.height)
        frames = int(stream.frames or 0)
        has_audio = bool(container.streams.audio)
        if frames <= 0:
            frames = sum(1 for _ in container.decode(video = 0))
    return width, height, frames, has_audio
