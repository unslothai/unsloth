# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The clip dataset layer for MiniMax-H3 LoRA training.

MiniMax-H3 is the first family Unsloth trains from **clips with sound** rather than stills,
and it has no still-image shortcut: its video VAE encodes ``17 * n + 5`` pixel frames at a
time (a 1-frame clip is not a valid input, unlike LTX-2's), and every forward carries audio
rows in the same packed sequence as the video rows. So the dataset unit is a video file with
a soundtrack, and this module is the part of that which needs no torch and no diffusers:
discovery, captions, the frame/latent/row arithmetic, and the canvas rule.

The decode itself (``decode_clip``) needs PyAV, which the video backend already depends on.
Everything above it is pure filesystem + arithmetic so the geometry contract is unit-testable
on a CPU-only box.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional
from utils.paths.path_utils import drop_appledouble_metadata

# Checkpoint contracts mirrored from diffusers.modular_pipelines.minimax_h3, which the trainer
# cannot import; asserted against the live components at load, so a move is caught.
H3_FPS = 24
# The video VAE's clip_length and tokens_chunk_size: 17 pixel frames per chunk, 5 latent frames
# kept per chunk, plus a 2-frame head.
H3_FRAMES_PER_CHUNK = 17
H3_LATENTS_PER_CHUNK = 5
H3_SPATIAL_COMPRESSION = 16
# The transformer's (t, h, w) patch. Only the spatial half is > 1.
H3_PATCH_T, H3_PATCH_H, H3_PATCH_W = 1, 2, 2
# Both axes have to survive the VAE's 16x spatial compression AND still be a whole number of 2x2
# patch rows, so the canvas multiple is their product.
H3_CANVAS_MULTIPLE = H3_SPATIAL_COMPRESSION * H3_PATCH_W
H3_AUDIO_SAMPLING_RATE = 32000
H3_AUDIO_LATENTS_PER_SECOND = 40
H3_AUDIO_CHANNELS = 2
# 1% of a 5.17s window is ~52ms, covering the tail a container routinely ends short by while still
# refusing a mostly-silent stream.
_MAX_AUDIO_PAD_FRACTION = 0.01
# About -80 dBFS on PyAV's "flt" scale: below the noise floor of any real recording, above the
# rounding dust an encode/decode round trip leaves on authored digital silence.
_SILENT_AUDIO_PEAK = 1e-4
H3_AUDIO_LATENT_CHANNELS = 32
H3_VIDEO_LATENT_CHANNELS = 24
# Per-row modality tags, which index the transformer's AdaLN table.
H3_VIDEO_TAG, H3_TEXT_TAG, H3_AUDIO_TAG = 0, 1, 2
# The canvas the released checkpoint generates on.
H3_CANVAS_SHORT_EDGE = 768
H3_CANVAS_MAX_PIXELS = 768 * 1344
H3_MIN_ASPECT_RATIO = 1 / 4
H3_MAX_ASPECT_RATIO = 4
# ImageNet, the video VAE's pixel convention.
H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
H3_PIXEL_STD = (0.229, 0.224, 0.225)

# ONE VAE chunk, the shortest clip the video VAE can encode at all, which is 0.917 s at 24 fps.
# Deliberately far below the 5 s floor MiniMax-H3 generates at: the packed sequence is quadratic
# in its own length, and training short clips at the native canvas keeps the SPATIAL statistics on
# distribution. A 22-frame clip's temporal rotary grid is a strict PREFIX of a generated one's.
H3_TRAIN_NUM_FRAMES = H3_FRAMES_PER_CHUNK + H3_LATENTS_PER_CHUNK

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
_CAPTION_EXTS = (".txt", ".caption")


def h3_align_num_frames(num_frames: int) -> int:
    """Snap a frame count UP to the next ``17 * n + 5`` the video VAE can encode."""
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    while num_frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


def h3_video_latent_frames(num_frames: int) -> int:
    """Latent frames the video VAE produces for an aligned frame count: ``5 * n + 2``."""
    if num_frames % H3_FRAMES_PER_CHUNK != H3_LATENTS_PER_CHUNK:
        raise ValueError(
            f"num_frames must be of the form {H3_FRAMES_PER_CHUNK} * n + {H3_LATENTS_PER_CHUNK}, "
            f"got {num_frames}."
        )
    return (num_frames - H3_LATENTS_PER_CHUNK) // H3_FRAMES_PER_CHUNK * H3_LATENTS_PER_CHUNK + 2


def h3_audio_latent_count(num_frames: int) -> int:
    """Audio latents (per channel) covering ``num_frames`` frames at 24 fps / 40 latents per s."""
    return int(round(num_frames / H3_FPS * H3_AUDIO_LATENTS_PER_SECOND))


def h3_audio_sample_count(num_frames: int) -> int:
    """Waveform samples per channel the audio VAE must be handed for ``num_frames`` frames.

    The audio VAE hops 800 samples (32 kHz / 40 latents per second) and right-pads a short
    tail, so handing it exactly ``latents * hop`` samples produces exactly the latent count the
    packed layout reserves rows for -- no pad, no truncation."""
    hop = H3_AUDIO_SAMPLING_RATE // H3_AUDIO_LATENTS_PER_SECOND
    return h3_audio_latent_count(num_frames) * hop


def h3_rows_per_latent_frame(latent_height: int, latent_width: int) -> int:
    return (latent_height // H3_PATCH_H) * (latent_width // H3_PATCH_W)


def h3_packed_sequence_length(
    num_text_tokens: int, num_frames: int, height: int, width: int
) -> int:
    """Rows of the packed ``[text | audio | video]`` sequence for one training sample.

    The trainer builds no conditioning rows (it trains the ``t2va`` layout), so this is the
    whole sequence and the figure the attention cost is quadratic in."""
    latent_frames = h3_video_latent_frames(num_frames)
    rows_per_frame = h3_rows_per_latent_frame(
        height // H3_SPATIAL_COMPRESSION, width // H3_SPATIAL_COMPRESSION
    )
    audio_rows = h3_audio_latent_count(num_frames) * H3_AUDIO_CHANNELS
    return num_text_tokens + audio_rows + latent_frames * rows_per_frame


def h3_train_canvas(
    aspect_width: float,
    aspect_height: float,
    short_edge: int = H3_CANVAS_SHORT_EDGE,
    max_pixels: Optional[int] = None,
) -> tuple[int, int]:
    """MiniMax-H3's canvas rule, as ``(width, height)``.

    Identical arithmetic to the pipeline's ``resolve_canvas_size`` (which returns
    ``(height, width)``), re-expressed here so the trainer can size a dataset before any
    diffusers import. ``short_edge`` is the run's ``resolution``; the area cap scales with it
    so a smaller training canvas keeps the released aspect budget rather than the released
    pixel count.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"The aspect ratio must be positive, got {aspect_width}:{aspect_height}.")
    ratio = aspect_width / aspect_height
    if not H3_MIN_ASPECT_RATIO <= ratio <= H3_MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 was trained on aspect ratios from 1:4 to 4:1; this clip is "
            f"{aspect_width:g}x{aspect_height:g} ({ratio:.2f}:1). Crop it first."
        )
    if max_pixels is None:
        # The released cap, rescaled to the requested short edge: (1344/768) * short_edge^2.
        max_pixels = int(H3_CANVAS_MAX_PIXELS * (short_edge / H3_CANVAS_SHORT_EDGE) ** 2)
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        width, height = width * scale, height * scale

    def snap(value: float) -> int:
        return max(H3_CANVAS_MULTIPLE, round(value / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE)

    return snap(width), snap(height)


def discover_clip_caption_pairs(
    data_dir: str | os.PathLike[str],
    *,
    instance_prompt: Optional[str] = None,
    caption_column: str = "text",
) -> list[tuple[str, str]]:
    """Resolve ``(clip_path, caption)`` pairs from a dataset directory.

    The caption rules are exactly ``discover_image_caption_pairs``' -- a per-clip ``<stem>.txt``
    / ``<stem>.caption`` sidecar wins, then a ``metadata.jsonl`` / ``captions.jsonl`` row keyed
    by ``file_name`` (or ``video`` / ``image`` / ``file``) carrying ``caption_column``, then the
    dreambooth ``instance_prompt`` -- so a user who has captioned an image dataset already knows
    this layout. Only the file extensions differ.

    An empty sidecar is the same deliberate tombstone it is for images: it suppresses the
    metadata caption and leaves the clip uncaptioned, so the ``instance_prompt`` fallback applies.
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {data_dir}")

    clips = drop_appledouble_metadata(
        sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTS)
    )

    meta_caption: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = root / meta_name
        if not meta_path.is_file():
            continue
        try:
            meta_lines = meta_path.read_text(encoding = "utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line in meta_lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(row, dict):
                continue
            key = row.get("file_name") or row.get("video") or row.get("image") or row.get("file")
            value = row.get(caption_column)
            if key and value is not None:
                meta_caption[str(key)] = str(value)

    pairs: list[tuple[str, str]] = []
    for clip in clips:
        caption: Optional[str] = None
        sidecar_present = False
        for ext in _CAPTION_EXTS:
            sidecar = clip.with_suffix(ext)
            if sidecar.is_file():
                sidecar_present = True
                try:
                    caption = sidecar.read_text(encoding = "utf-8").strip()
                except (OSError, UnicodeError):
                    caption = ""
                break
        if not sidecar_present:
            caption = meta_caption.get(clip.name) or meta_caption.get(
                clip.relative_to(root).as_posix()
            )
        if not caption and instance_prompt:
            caption = instance_prompt
        if caption:
            pairs.append((str(clip), caption))

    if not pairs:
        raise ValueError(
            "No captioned video clips found. MiniMax-H3 trains from clips with sound, not "
            "stills: provide .mp4 / .mov / .mkv / .webm files plus a metadata.jsonl / "
            "captions.jsonl, per-clip .txt captions, or an instance prompt."
        )
    return pairs


def decode_clip(
    path: str | os.PathLike[str],
    *,
    num_frames: int,
    width: int,
    height: int,
    on_note: Optional[Callable[[str], None]] = None,
) -> tuple[Any, Any]:
    """Decode one training clip to ``(frames, waveform)``.

    ``frames`` is a uint8 numpy array of shape ``(num_frames, height, width, 3)`` resampled onto
    MiniMax-H3's own 24 fps by whole-frame drop/duplicate -- the same selection the inference
    reference-video decoder makes, so a clip reaches the model on the model's clock however it
    was authored. Each frame is cover-cropped to the canvas aspect ratio and then resized, so
    nothing is letterboxed and nothing is stretched.

    The window is the FIRST ``num_frames`` of the source, and the latents are cached once for
    the run, so a longer clip trains only its opening and its caption is paired with that. That
    is the dataset contract -- pre-trim to the training duration -- but it used to be silent,
    which is how a caption describing a whole scene ended up on its first second. ``on_note``
    is called once per over-long clip with the numbers, so the run reports it.

    ``waveform`` is a float32 array of shape ``(2, h3_audio_sample_count(num_frames))`` at
    32 kHz. A mono source is duplicated to both channels; a clip with **no** audio track is
    refused rather than silently trained as silence, because the audio rows are in the objective
    and a silent target teaches the model to stop generating sound.
    """
    import av
    import numpy as np
    from PIL import Image

    target_samples = h3_audio_sample_count(num_frames)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{Path(path).name} carries no video track.")
        if not container.streams.audio:
            raise ValueError(
                f"{Path(path).name} carries no audio track. MiniMax-H3 denoises video and audio "
                f"in one packed sequence, so its training clips must have sound."
            )
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.guessed_rate or H3_FPS) or float(H3_FPS)
        # Best effort: an unknown duration means no note, never a failed decode.
        source_duration_s = 0.0
        try:
            if stream.duration is not None and stream.time_base is not None:
                source_duration_s = float(stream.duration * stream.time_base)
            elif getattr(container, "duration", None):
                source_duration_s = float(container.duration) / 1_000_000.0
        except Exception:  # noqa: BLE001 -- a note is not worth failing a decode over
            source_duration_s = 0.0

        frames: list[Any] = []
        next_target = 0
        for source_index, frame in enumerate(container.decode(video = 0)):
            if len(frames) >= num_frames:
                break
            if int(next_target * source_fps / H3_FPS) > source_index:
                continue
            image = frame.to_image().convert("RGB")
            # Before the crop: the canvas is in display orientation, so cropping the coded frame would trim the
            # wrong edges as well as train it sideways.
            image = apply_display_rotation(image, display_rotation_degrees(frame, stream), Image)
            image = _cover_resize(image, width, height, Image)
            while (
                int(next_target * source_fps / H3_FPS) <= source_index and len(frames) < num_frames
            ):
                frames.append(np.asarray(image, dtype = "uint8"))
                next_target += 1

    if len(frames) < num_frames:
        raise ValueError(
            f"{Path(path).name} decoded to {len(frames)} frames at {H3_FPS} fps, but a training "
            f"clip needs {num_frames} ({num_frames / H3_FPS:.2f}s). Use longer clips."
        )
    if on_note is not None and source_duration_s > (num_frames / H3_FPS) * 1.05:
        on_note(
            f"{Path(path).name} is {source_duration_s:.1f}s; MiniMax-H3 trains its first "
            f"{num_frames / H3_FPS:.2f}s and its caption is paired with that. Trim the clip to "
            f"the part the caption describes."
        )

    waveform = _decode_clip_audio(path, target_samples, av, np)
    return np.stack(frames), waveform


def display_rotation_degrees(frame: Any, stream: Any) -> int:
    """The clip's display rotation, one of 0/90/180/270, as a PLAYER would apply it.

    PyAV hands back the CODED frame: unlike the ffmpeg CLI, ``to_image()`` and ``to_ndarray()``
    do not honour the display matrix, and PyAV declines to do so by design. A phone clip shot
    in portrait is stored landscape with a 90 degree matrix, so without this the trainer caches
    sideways frames, on a canvas taken from the equally sideways coded size, and cover-crops
    away the sides of the real picture.

    The angle is FFmpeg's own: ``av_display_rotation_get`` on the 16.16 fixed-point 3x3, then
    ``theta = -round(...) mod 360``, which is what ffmpeg's autorotate applies. Returns 0 for a
    clip with no matrix, and for any PyAV too old to expose one -- previous behaviour, never an
    exception, since a decode must not fail over orientation metadata.
    """
    import math
    import struct

    matrix = None
    try:
        from av.sidedata.sidedata import Type
        entry = frame.side_data.get(Type.DISPLAYMATRIX)
        if entry is not None:
            raw = bytes(entry)
            if len(raw) >= 36:
                # Native byte order: the matrix is an in-memory int32[9], not a serialised field.
                matrix = struct.unpack("=9i", raw[:36])
    except Exception:  # noqa: BLE001 -- no side-data API, or no matrix on this frame
        matrix = None
    if matrix is None:
        # Legacy MOV/MP4 tag, still what older files carry.
        try:
            tag = (stream.metadata or {}).get("rotate")
            return int(float(tag)) % 360 if tag is not None else 0
        except Exception:  # noqa: BLE001 -- an unparsable tag is not a decode failure
            return 0
    try:
        conv = lambda v: v / (1 << 16)  # noqa: E731
        scale_x = math.hypot(conv(matrix[0]), conv(matrix[3]))
        scale_y = math.hypot(conv(matrix[1]), conv(matrix[4]))
        if not scale_x or not scale_y:
            return 0
        degrees = -math.degrees(math.atan2(conv(matrix[1]) / scale_y, conv(matrix[0]) / scale_x))
    except Exception:  # noqa: BLE001 -- a degenerate matrix means "no rotation", not a failure
        return 0
    theta = int(-round(degrees)) % 360
    # Only the four square turns; anything else cannot be applied without resampling, and no camera writes one.
    return theta if theta in (90, 180, 270) else 0


def apply_display_rotation(image: Any, theta: int, Image: Any) -> Any:
    """Rotate a decoded frame into display orientation. Verified against ffmpeg's autorotate."""
    if theta == 90:
        return image.transpose(Image.ROTATE_270)
    if theta == 180:
        return image.transpose(Image.ROTATE_180)
    if theta == 270:
        return image.transpose(Image.ROTATE_90)
    return image


def _cover_resize(image: Any, width: int, height: int, Image: Any) -> Any:
    """Center cover-crop to the canvas aspect ratio, then resize to it. Never stretches."""
    source_w, source_h = image.size
    scale = max(width / source_w, height / source_h)
    crop_w = min(source_w, math.ceil(width / scale))
    crop_h = min(source_h, math.ceil(height / scale))
    left = (source_w - crop_w) // 2
    top = (source_h - crop_h) // 2
    return image.resize(
        (width, height), Image.LANCZOS, box = (left, top, left + crop_w, top + crop_h)
    )


def _decode_clip_audio(path: Any, target_samples: int, av: Any, np: Any) -> Any:
    """The clip's soundtrack as float32 ``(2, target_samples)`` at 32 kHz.

    Resampled by PyAV to the audio VAE's own rate and stereo layout, then trimmed or
    zero-padded to exactly the sample count the packed layout reserves audio rows for. A short
    tail is padded rather than refused: the video stream is the authority on the clip's length
    and containers routinely end their audio a few milliseconds early.

    That tolerance is bounded. Padding a materially short soundtrack out to the full window
    trains the shared adapter on a target that is mostly silence, which is the exact failure the
    "must have sound" check above exists to prevent -- and a stream carrying a fraction of a
    second passed it, because the check only asks whether the container declares one.
    ``_MAX_AUDIO_PAD_FRACTION`` is the container-tail allowance, not an augmentation budget.
    """
    resampler = av.AudioResampler(format = "flt", layout = "stereo", rate = H3_AUDIO_SAMPLING_RATE)
    chunks = []
    have = 0
    with av.open(str(path)) as container:
        # Stops at the training window: only the first num_frames are trained, so decoding the rest of the
        # soundtrack would spend a whole recording's time and fail on damage in an unused region.
        for frame in container.decode(audio = 0):
            for resampled in resampler.resample(frame):
                block = resampled.to_ndarray().reshape(-1, H3_AUDIO_CHANNELS)
                chunks.append(block)
                have += block.shape[0]
            if have >= target_samples:
                break
        if have < target_samples:
            # Only when the stream ran out: the resampler holds a partial block back and that tail is what the
            # pad allowance measures; after an early break the window is already full.
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1, H3_AUDIO_CHANNELS))
    if not chunks:
        raise ValueError(f"{Path(path).name} decoded to no audio samples.")
    samples = np.concatenate(chunks, axis = 0).astype("float32")[:target_samples]
    if samples.shape[0] < target_samples:
        missing = target_samples - samples.shape[0]
        if missing > _MAX_AUDIO_PAD_FRACTION * target_samples:
            have_s = samples.shape[0] / H3_AUDIO_SAMPLING_RATE
            want_s = target_samples / H3_AUDIO_SAMPLING_RATE
            raise ValueError(
                f"{Path(path).name} carries {have_s:.2f}s of audio for a {want_s:.2f}s clip. "
                f"MiniMax-H3 denoises video and audio together, so padding the rest with "
                f"silence would train the adapter to stop generating sound. Use a clip whose "
                f"soundtrack runs its full length."
            )
        samples = np.pad(samples, ((0, missing), (0, 0)))
    # A muted track passes every check above and comes back all zeros.
    if float(np.max(np.abs(samples))) <= _SILENT_AUDIO_PEAK:
        raise ValueError(
            f"{Path(path).name} has a soundtrack that is silent all the way through. "
            f"MiniMax-H3 denoises video and audio together, so training on it would teach the "
            f"adapter to stop generating sound. Use a clip whose soundtrack has audio in it, or "
            f"take this one out of the dataset."
        )
    return np.ascontiguousarray(samples.T)
