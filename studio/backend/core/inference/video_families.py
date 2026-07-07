# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure helpers for text-to-video model identification.

The video registry mirrors ``diffusion_families`` (no torch/diffusers imports, so
everything unit-tests without the heavy runtime) but is a SEPARATE registry with a
separate backend: video pipelines take frame/fps arguments, return frame stacks
(and, for LTX-2, synchronized audio) instead of PIL images, and their artifacts are
MP4s. Keeping the registries apart means neither picker can mis-route a checkpoint
to the wrong engine.

A video checkpoint published as a single-file GGUF only carries the DiT weights;
the VAE / text encoder / connectors / vocoder come from the companion diffusers
base repo, exactly like the image GGUF path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Runtime->route contract, mirroring the diffusion sentinels: the routes match
# these EXACTLY to return 409 (client-recoverable) instead of a sanitized 500.
VIDEO_NOT_LOADED_MSG = "No video model is loaded."
VIDEO_CANCELLED_MSG = "Video generation was cancelled."


@dataclass(frozen = True)
class VideoFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Pipeline kwarg carrying the guidance value.
    cfg_kwarg: str = "guidance_scale"
    # The pipe attribute holding the denoiser (all current video families are DiTs).
    denoiser_attr: str = "transformer"
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)
    # True when the pipeline returns synchronized audio alongside frames (LTX-2):
    # export must mux the audio track into the MP4 and size estimates must count
    # the audio VAE + vocoder companions.
    has_audio: bool = False
    # Wan2.2-A14B style dual-expert MoE: a second DiT (``transformer_2``) handles
    # the low-noise steps, with its own guidance kwarg. None/False for single-DiT
    # families. Declared now so adding the A14B family later does not churn the
    # schema every module already imports.
    transformer2_class: Optional[str] = None
    is_moe: bool = False
    cfg2_kwarg: Optional[str] = None
    # Generation defaults + shape constraints. ``frame_step`` is the temporal
    # compression: a valid frame count is k * frame_step + 1 (the +1 is the
    # anchor frame), so requests are snapped BEFORE latents are allocated.
    default_steps: int = 40
    default_guidance: float = 4.0
    default_num_frames: int = 121
    default_fps: int = 24
    frame_step: int = 8
    # Width/height must be divisible by this (LTX-2's pipeline rejects non-/32).
    resolution_multiple: int = 32
    # (width, height) presets the UI offers, landscape first, including a vertical
    # option. The first preset is the default.
    resolution_presets: tuple[tuple[int, int], ...] = ((768, 512),)
    # Component bf16-RESIDENT sizes in decimal GB (denoiser(s), text encoder,
    # VAE + audio companions), the video analogue of the image auto-policy table.
    # These are what sits on device after the dtype cast, not the download size.
    bf16_components_gb: Optional[tuple[float, float, float]] = None
    # True when the family's DiT compiles cleanly with regional torch.compile
    # (Wan/LTX-2 declare _repeated_blocks; set False until verified per family).
    supports_torch_compile: bool = True
    # Families whose activations overflow float16 -> the loader promotes fp16 to
    # float32. Video DiTs are bf16-native, so this defaults True (fp16 is never
    # the right resolution for them; bf16 or float32 only).
    fp16_incompatible: bool = True
    # Curated GGUF repo for the picker (the DiT as single-file GGUF quants).
    gguf_repo: Optional[str] = None


_FAMILIES: tuple[VideoFamily, ...] = (
    # LTX-2 (diffusers >= 0.39): a ~19B single-stream video DiT generating
    # synchronized audio + video in one pass (audio VAE + vocoder + text
    # connectors ride the base repo; the classes are vendored inside
    # diffusers.pipelines.ltx2). The Gemma3-27B text encoder is the memory
    # heavyweight: ~50 GB bf16-resident, more than the DiT itself. The diffusers
    # base repo carries the dev-style config (40 steps, CFG 4); the distilled
    # single-file/GGUF checkpoints run few-step (see default_video_generation_params).
    VideoFamily(
        name = "ltx-2",
        pipeline_class = "LTX2Pipeline",
        transformer_class = "LTX2VideoTransformer3DModel",
        base_repo = "Lightricks/LTX-2",
        aliases = ("ltx-2.3", "ltx2", "ltx-video", "ltxv", "ltx"),
        has_audio = True,
        default_steps = 40,
        default_guidance = 4.0,
        default_num_frames = 121,
        default_fps = 24,
        frame_step = 8,
        resolution_multiple = 32,
        # The pipeline's native default is 768x512; 1216x704 is the model card's
        # quality target; 704x1216 is the vertical variant.
        resolution_presets = ((768, 512), (1216, 704), (704, 1216), (512, 768)),
        # transformer 37.8 stored bf16; Gemma3-27B TE ~50.4; video VAE 2.4 +
        # connectors 2.9 + audio VAE/vocoder 0.2 (sibling metadata, duplicates
        # removed -- the repo ships the TE twice under two shard namings).
        bf16_components_gb = (37.8, 50.4, 5.5),
        gguf_repo = "unsloth/LTX-2.3-GGUF",
    ),
)


def _token_in_needle(token: str, needle: str) -> bool:
    """Whole path/name segment match, as in diffusion_families (a short alias like
    'ltx' must not match inside an unrelated word)."""
    return re.search(r"(?:^|[-_./\\])" + re.escape(token) + r"(?:$|[-_./\\])", needle) is not None


def detect_video_family(repo_id: str, override: Optional[str] = None) -> Optional[VideoFamily]:
    """Resolve a ``VideoFamily`` from a repo id, or an explicit override.

    Same contract as ``diffusion_families.detect_family``: an override matches a
    name/alias exactly; otherwise the longest name/alias appearing as a whole
    segment of the repo id wins.
    """
    if override:
        key = override.strip().lower()
        for fam in _FAMILIES:
            if key == fam.name or key in fam.aliases:
                return fam
        return None
    needle = repo_id.lower()
    best: Optional[tuple[VideoFamily, int]] = None
    for fam in _FAMILIES:
        for token in (fam.name, *fam.aliases):
            if _token_in_needle(token, needle) and (best is None or len(token) > best[1]):
                best = (fam, len(token))
    return best[0] if best else None


def supported_video_family_names() -> tuple[str, ...]:
    return tuple(fam.name for fam in _FAMILIES)


def resolve_video_base_repo(fam: VideoFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


def snap_num_frames(fam: VideoFamily, num_frames: int) -> int:
    """The nearest valid frame count at or below the request (k * frame_step + 1).

    Video latents are allocated as (num_frames - 1) / temporal_compression + 1, so
    an off-lattice count wastes a partial latent frame at best and trips shape
    checks at worst; snapping mirrors the image path's silent /16 size snap.
    """
    step = max(1, fam.frame_step)
    return max(1, ((max(1, num_frames) - 1) // step) * step + 1)


def snap_video_size(fam: VideoFamily, width: int, height: int) -> tuple[int, int]:
    """Width/height floored to the family's required multiple (minimum one unit)."""
    multiple = max(1, fam.resolution_multiple)
    snap = lambda v: max(multiple, (max(1, v) // multiple) * multiple)  # noqa: E731
    return snap(width), snap(height)


# Default (steps, guidance) per checkpoint variant, matched by substring against
# the picked id (then the base repo), most specific first: the distilled LTX-2.3
# checkpoints run few-step with CFG off, while the dev-config base repo wants the
# full 40-step CFG schedule. Mirrors default_generation_params on the image side.
_VIDEO_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    ("distilled", 8, 1.0),
    ("ltx", 40, 4.0),
)


def default_video_generation_params(*identifiers: Optional[str]) -> tuple[int, float]:
    """Default ``(steps, guidance)`` for a loaded video model; the first identifier
    naming a known variant wins, so a GGUF filename ('...distilled...Q4_K_M.gguf')
    beats the family base repo."""
    for identifier in identifiers:
        needle = (identifier or "").lower()
        for key, steps, guidance in _VIDEO_GENERATION_DEFAULTS:
            if key in needle:
                return steps, guidance
    return 40, 4.0
