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

# Runtime->route contract: routes match these EXACTLY for a 409 instead of a 500.
VIDEO_NOT_LOADED_MSG = "No video model is loaded."
VIDEO_CANCELLED_MSG = "Video generation was cancelled."
VIDEO_GENERATION_BUSY_MSG = "A video generation is already in progress."


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
    # True when the pipeline returns synchronized audio (LTX-2): export muxes the track and size estimates count the audio VAE + vocoder.
    has_audio: bool = False
    # Wan2.2-A14B dual-expert MoE: a second DiT (transformer_2) handles the low-noise steps with its own guidance kwarg.
    transformer2_class: Optional[str] = None
    is_moe: bool = False
    cfg2_kwarg: Optional[str] = None
    # HunyuanVideo-1.5 guidance: __call__ takes NO guidance kwarg; CFG lives on a ``guider`` whose scale is set per request.
    guidance_via_guider: bool = False
    # Generation defaults + shape. ``frame_step`` is the temporal compression: a valid frame count is k*frame_step + 1.
    default_steps: int = 40
    default_guidance: float = 4.0
    default_num_frames: int = 121
    default_fps: int = 24
    frame_step: int = 8
    # Width/height must be divisible by this (LTX-2's pipeline rejects non-/32).
    resolution_multiple: int = 32
    # (width, height) UI presets, landscape first; the first is the default.
    resolution_presets: tuple[tuple[int, int], ...] = ((768, 512),)
    # Component bf16-RESIDENT sizes in decimal GB (denoiser(s), text encoder, VAE + audio): what sits on device after the cast.
    bf16_components_gb: Optional[tuple[float, float, float]] = None
    # True when the DiT compiles cleanly with regional torch.compile (declares _repeated_blocks).
    supports_torch_compile: bool = True
    # Video DiTs are bf16-native, so fp16 promotes to float32; defaults True.
    fp16_incompatible: bool = True
    # Wan VAE decodes in float32 (bf16 causes banding / black frames), so the loader pins it back. Its size term is already fp32.
    vae_force_fp32: bool = False
    # Curated GGUF repo for the picker (the DiT as single-file GGUF quants).
    gguf_repo: Optional[str] = None
    # Hosted PRE-CAST text-encoder checkpoints as (scheme, component, repo_id); same semantics as DiffusionFamily.te_prequant_repos.
    te_prequant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)


_FAMILIES: tuple[VideoFamily, ...] = (
    # LTX-2 (diffusers >= 0.39): ~19B single-stream video DiT generating synchronized audio + video in one pass. The Gemma3-12B
    # encoder is stored fp32 on the hub (~49 GB download, ~24 GB resident bf16). The base repo carries the dev config (40 steps, CFG 4).
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
        # 768x512 native default; 1216x704 the card's quality target; 704x1216 vertical.
        resolution_presets = ((768, 512), (1216, 704), (704, 1216), (512, 768)),
        # transformer 37.8 bf16; Gemma3-12B TE ~24.4 RESIDENT (the hub stores it fp32 but the pipeline loads bf16); VAE 2.4 + connectors 2.9 + audio 0.2. The old 50.4 figure double-counted the fp32 store.
        bf16_components_gb = (37.8, 24.4, 5.5),
        gguf_repo = "unsloth/LTX-2.3-GGUF",
        # Pre-cast Gemma3-12B TE (fp32 ~49 GB on the hub, pre-cast ~13.2 GB): the biggest download win.
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/LTX-2-FP8"),),
    ),
    # Wan2.2-TI2V-5B (diffusers >= 0.35, verified on 0.39): ~5B single-stream DiT (UMT5 encoder), no audio. Its VAE's temporal compression 4 gives valid frame counts 4k+1. Defaults 50 steps / CFG 5.
    VideoFamily(
        name = "wan2.2-ti2v-5b",
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        # "wan2.2-5b"/"wan-ti2v" are the picker/GGUF short ids; "wan2.2-ti2v" catches the repo stem.
        aliases = ("wan2.2-5b", "wan-ti2v", "wan2.2-ti2v", "wan-ti2v-5b"),
        has_audio = False,
        default_steps = 50,
        default_guidance = 5.0,
        # 121 frames at 24 fps ~5s; on the 4k+1 lattice (121 = 4*30 + 1) it needs no snapping.
        default_num_frames = 121,
        default_fps = 24,
        # Wan VAE temporal factor 4, so valid counts are 4k+1.
        frame_step = 4,
        # TI2V-5B's VAE is 16x spatial + patch 2, so WanPipeline floors H/W to 32; snap to 32 so the recorded size matches the clip.
        resolution_multiple = 32,
        # TI2V-5B is a 720P-only checkpoint: upstream SUPPORTED_SIZES is exactly
        # ('704*1280', '1280*704') and generate.py asserts membership, so nothing else is offered.
        # First is the default the loader plans against.
        resolution_presets = ((1280, 704), (704, 1280)),
        # bf16-RESIDENT. transformer + VAE ship FP32 on disk (20.0 GB = 5B x 4), so bf16 transformer ~10.0; UMT5 TE bf16 (11.4); VAE fp32 (2.8).
        bf16_components_gb = (10.0, 11.4, 2.8),
        vae_force_fp32 = True,
        # Mirrors QuantStack/Wan2.2-TI2V-5B-GGUF (all 13 quants + the companion VAE), byte for byte.
        gguf_repo = "unsloth/Wan2.2-TI2V-5B-GGUF",
    ),
    # Wan2.2-T2V-A14B (diffusers >= 0.35, verified on 0.39): the dual-expert MoE. Both transformers are WanTransformer3DModel with boundary_ratio 0.875; high-noise steps route through transformer, low-noise through transformer_2, so cfg2_kwarg is threaded only here.
    VideoFamily(
        name = "wan2.2-t2v-a14b",
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        aliases = ("wan2.2-14b", "wan-t2v", "wan2.2-t2v", "wan-t2v-a14b", "wan-a14b"),
        has_audio = False,
        # is_moe drives the dual-DiT optimisation layers; cfg2_kwarg names the pipeline kwarg for transformer_2's guidance.
        transformer2_class = "WanTransformer3DModel",
        is_moe = True,
        cfg2_kwarg = "guidance_scale_2",
        default_steps = 50,
        default_guidance = 5.0,
        # 81 frames at 16 fps ~5s (81 = 4*20 + 1), the A14B card's default clip.
        default_num_frames = 81,
        default_fps = 16,  # A14B runs at 16 fps (vs TI2V-5B's 24)
        frame_step = 4,
        resolution_multiple = 16,
        # 480p + 720p presets. A14B's VAE is 8x so multiple 16 renders 720 exactly (TI2V-5B's 16x VAE floors it to 704).
        resolution_presets = ((1280, 720), (832, 480), (480, 832), (720, 1280)),
        # bf16-RESIDENT. Each expert ships FP32 (57.15 GB = 14.3B x 4), so ~28.6 bf16 each and ~57.2 for BOTH, not the 114.3 fp32 sum. UMT5 TE bf16 (11.4); VAE fp32 (0.5).
        bf16_components_gb = (57.2, 11.4, 0.5),
        vae_force_fp32 = True,
        # No gguf_repo: community GGUFs split the experts, and a single-file load covers only one.
    ),
    # HunyuanVideo-1.5 (diffusers >= 0.39): 8.3B DiT, Qwen2.5-VL + ByT5 encoders. Three quirks: no guidance kwarg (CFG on the ``guider``), no callback_on_step_end (generate() wraps scheduler.step), and no upstream model_index.json, so only the community repacks load.
    VideoFamily(
        name = "hunyuanvideo-1.5",
        pipeline_class = "HunyuanVideo15Pipeline",
        transformer_class = "HunyuanVideo15Transformer3DModel",
        base_repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        # No bare "hunyuanvideo" alias: it would also claim the incompatible 1.0 repos.
        aliases = ("hunyuanvideo-1-5", "hunyuanvideo1.5", "hunyuanvideo1-5", "hv15"),
        has_audio = False,
        guidance_via_guider = True,
        default_steps = 50,
        default_guidance = 6.0,
        # 121 frames at 24 fps ~5s, the pipeline's own default.
        default_num_frames = 121,
        default_fps = 24,
        # HV15 VAE compresses 16x spatial / 4x temporal, patch-1, so sizes snap /16, frames 4k+1.
        frame_step = 4,
        resolution_multiple = 16,
        # 480p-class presets (the base is the 480p variant): landscape, vertical, square. Every
        # entry is a real bucket of this tier (generate_crop_size_list(base_size=640)); 624x624
        # was not one, so the square option snapped off-tier at 1521 tokens against the trained
        # 1600.
        resolution_presets = ((832, 480), (480, 832), (640, 640)),
        # DiT fp32 on disk (32.0 to 16.6 bf16); VAE (4.7 to 2.4); Qwen2.5-VL TE bf16 14.0 + ByT5 0.8.
        bf16_components_gb = (16.6, 14.8, 2.4),
    ),
    # The 720p t2v repack: same architecture and footprint as the 480p entry, only the trained resolution differs. Its own family so a 720p load defaults to 720p sizes; the full-path alias outranks the generic token.
    VideoFamily(
        name = "hunyuanvideo-1.5-720p",
        pipeline_class = "HunyuanVideo15Pipeline",
        transformer_class = "HunyuanVideo15Transformer3DModel",
        base_repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        aliases = ("hunyuanvideo-1.5-diffusers-720p_t2v", "hv15-720p"),
        has_audio = False,
        guidance_via_guider = True,
        default_steps = 50,
        default_guidance = 6.0,
        default_num_frames = 121,
        default_fps = 24,
        frame_step = 4,
        resolution_multiple = 16,
        # 720p-class presets: landscape, vertical, square (all /16).
        resolution_presets = ((1280, 720), (720, 1280), (960, 960)),
        bf16_components_gb = (16.6, 14.8, 2.4),
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
    if best is None:
        return None
    fam = best[0]
    # HunyuanVideo-1.5's resolution tier is baked into the weights: the 480p and 720p repacks ship
    # transformer target_size 640 vs 960 and scheduler shift 5.0 vs 9.0, and their bucket lists are
    # disjoint. Only the literal 720p_t2v path was aliased, so 720p_i2v and every GGUF repack fell
    # through to the generic "hunyuanvideo-1.5" token and inherited the 480p base repo -- which is
    # also where the VAE and text encoder come from. Re-route on the tier marker instead.
    if fam.name == "hunyuanvideo-1.5" and re.search(r"(?:^|[-_./\\])720p", needle):
        for candidate in _FAMILIES:
            if candidate.name == "hunyuanvideo-1.5-720p":
                return candidate
    return fam


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


# Default (steps, guidance) per checkpoint variant, matched by substring (picked id then base repo), most specific first.
_VIDEO_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    ("distilled", 8, 1.0),
    ("ltx", 40, 4.0),
    # Wan2.2 pipelines default to 50 steps / CFG 5.0; both TI2V-5B and A14B share these.
    ("wan", 50, 5.0),
    # HunyuanVideo-1.5: 50 steps with the guider's shipped CFG 6.0.
    ("hunyuanvideo", 50, 6.0),
)


def default_video_generation_params(
    *identifiers: Optional[str], fallback: tuple[int, float] = (40, 4.0)
) -> tuple[int, float]:
    """Default ``(steps, guidance)`` for a loaded video model; the first identifier
    naming a known variant wins, so a GGUF filename ('...distilled...Q4_K_M.gguf')
    beats the family base repo. ``fallback`` is used when no identifier names a variant --
    callers pass the resolved family's own default so a Wan model loaded from an opaque local
    path under an explicit family_override still gets 50/5.0, not the hardcoded LTX 40/4.0."""
    for identifier in identifiers:
        needle = (identifier or "").lower()
        for key, steps, guidance in _VIDEO_GENERATION_DEFAULTS:
            # Match the key as a name segment: reject a preceding ASCII letter so "swan-video" does not false-match "wan".
            if re.search(r"(?<![a-z])" + re.escape(key), needle):
                return steps, guidance
    return fallback
