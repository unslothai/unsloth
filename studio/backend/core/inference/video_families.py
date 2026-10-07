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

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .diffusion_nvfp4_flag import nvfp4_blocked, without_nvfp4

# num_frames ceiling; VideoGenerateRequest imports it for its `le` so gate and bound cannot drift.
MAX_VIDEO_NUM_FRAMES = 1024

# Routes match these EXACTLY to return 409 instead of 500.
VIDEO_NOT_LOADED_MSG = "No video model is loaded."
VIDEO_CANCELLED_MSG = "Video generation was cancelled."
VIDEO_GENERATION_BUSY_MSG = "A video generation is already in progress."
VIDEO_MODEL_CHANGED_MSG = "The requested video model changed before generation was reserved."


@dataclass(frozen = True)
class VideoFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    cfg_kwarg: str = "guidance_scale"
    denoiser_attr: str = "transformer"
    aliases: tuple[str, ...] = field(default_factory = tuple)
    has_audio: bool = False
    transformer2_class: Optional[str] = None
    is_moe: bool = False
    cfg2_kwarg: Optional[str] = None
    # HunyuanVideo-1.5: __call__ takes no guidance kwarg; CFG is set on ``guider`` per request.
    guidance_via_guider: bool = False
    default_steps: int = 40
    default_guidance: float = 4.0
    default_num_frames: int = 121
    default_fps: int = 24
    frame_step: int = 8
    frame_offset: int = 1
    min_num_frames: int = 1
    max_num_frames: Optional[int] = None
    snap_frames_up: bool = False
    # Width/height must be divisible by this (LTX-2's pipeline rejects non-/32).
    resolution_multiple: int = 32
    resolution_presets: tuple[tuple[int, int], ...] = ((768, 512),)
    duration_presets: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
    # Component bf16-RESIDENT sizes in decimal GB (denoiser(s), text encoder, VAE + audio).
    bf16_components_gb: Optional[tuple[float, float, float]] = None
    supports_torch_compile: bool = True
    # CUDA-graph capture: only for one forward per step, no CFG and no step cache.
    supports_cuda_graph: bool = False
    offload_cuda_graph: bool = False
    cuda_graph_decline: Optional[str] = None
    # Video DiTs are bf16-native, so fp16 promotes to float32; defaults True.
    fp16_incompatible: bool = True
    fp16_guard: Optional[str] = None
    # Wan VAE decodes in float32 (bf16 causes banding / black frames).
    vae_force_fp32: bool = False
    # False holds cudnn.benchmark off: its per-process conv pick makes servers decode the same latents differently.
    cudnn_benchmark: bool = True
    # Pin one config per reduction (render determinism); off by default as it can be slower.
    filter_reduction_configs: bool = False
    gguf_repo: Optional[str] = None
    te_prequant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Hosted PRE-QUANTIZED DENOISER checkpoints as (scheme, repo_id); separate from the text encoder.
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Resident GB of a hosted denoiser when _QUANT_STEADY_FACTOR is wrong (H3 is also pruned).
    prequant_resident_gb: Optional[float] = None
    prequant_resident_gb_by_scheme: tuple[tuple[str, float], ...] = field(default_factory = tuple)
    # (base_repo, scheme, repo_id) keyed on lowercased base id: a prequant only loads on its own base.
    prequant_variant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # (scheme, filename) or (scheme, task, filename); derived name stays as fallback for old builds.
    prequant_filenames: tuple[tuple[str, ...], ...] = field(default_factory = tuple)
    # Tasks with a separate denoiser partition, served ONLY by their own (scheme, task, filename) row.
    prequant_partition_tasks: tuple[str, ...] = field(default_factory = tuple)
    modular_workflow: Optional[str] = None
    default_flow_shift: Optional[float] = None
    comfy_flow_shift: Optional[float] = None
    default_audio_flow_shift: Optional[float] = None
    supports_keyframes: bool = False
    supports_references: bool = False
    supports_cfg: bool = True


# ComfyUI default 864x480: multiples of 32, inside the trained 1:4..4:1 range.
def _h3_480p_presets() -> tuple[tuple[int, int], ...]:
    flag = os.environ.get("UNSLOTH_VIDEO_H3_480P", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return ()
    return ((864, 480), (480, 864))


_FAMILIES: tuple[VideoFamily, ...] = (
    VideoFamily(
        name = "minimax-h3",
        pipeline_class = "ModularPipeline",
        transformer_class = "MiniMaxH3Transformer3DModel",
        base_repo = "MiniMaxAI/MiniMax-H3",
        aliases = ("minimax_h3", "minimaxh3", "h3"),
        has_audio = True,
        default_steps = 30,
        default_guidance = 1.0,
        default_num_frames = 124,
        default_fps = 24,
        frame_step = 17,
        frame_offset = 5,
        min_num_frames = 124,
        max_num_frames = 345,
        snap_frames_up = True,
        resolution_multiple = 32,
        resolution_presets = (
            (1344, 768),
            (1536, 672),
            (1024, 768),
            (1024, 1024),
            (768, 1024),
            (768, 1344),
            (960, 544),
            (544, 960),
        )
        + _h3_480p_presets(),
        duration_presets = (5.0, 10.0, 14.4),
        bf16_components_gb = (66.3, 66.8, 11.1),
        # Regionally compilable; dynamic=True traces once across caption lengths.
        supports_torch_compile = True,
        # CUDA graph declined: no measurable win (GPU-bound) for ~8 GB extra memory and 31 s capture.
        supports_cuda_graph = False,
        # Streamed, H3's prequant groups onload through diffusers' stream path, which waits on the host per group: a
        # capture cannot hold them (arm_after_placement refuses it by name when the family is forced).
        cuda_graph_decline = (
            "measured GPU-bound: 1.0018x resident for 3.93 GB held; streamed, its group onloads wait on the host "
            "(27 per step at 16 GB), which a graph cannot record"
        ),
        gguf_repo = "unsloth/MiniMax-H3-GGUF",
        # Hosted prequant denoisers are the only way to run H3 quantized (modular from_pretrained).
        prequant_repos = (("int8", "unsloth/MiniMax-H3-FP8"), ("fp8", "unsloth/MiniMax-H3-FP8")),
        # ConvRot INT8 (v2 tag) ships under its own name so older installs keep the plain one.
        prequant_filenames = (
            ("int8", "MiniMax-H3-INT8-ConvRot.pt"),
            ("int8", "ref2va", "MiniMax-H3-Ref2VA-INT8-ConvRot.pt"),
            ("fp8", "ref2va", "MiniMax-H3-Ref2VA-FP8.pt"),
        ),
        # Equal to H3_TASK_REFERENCES; literal avoids importing the H3 module into the registry.
        prequant_partition_tasks = ("ref2va",),
        # Measured from Hub file sizes (FP8 / INT8 .pt ~20.26 GB each).
        prequant_resident_gb = 20.3,
        modular_workflow = "fl2va",
        default_flow_shift = 12.0,
        default_audio_flow_shift = 3.0,
        supports_keyframes = True,
        supports_references = True,
        supports_cfg = False,
    ),
    # LTX-2 (diffusers >= 0.39); Gemma3-12B TE is fp32 on the hub but loads bf16.
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
        resolution_presets = ((768, 512), (1216, 704), (704, 1216), (512, 768)),
        # Gemma3 TE counted at bf16 resident (~24.4), not its fp32 hub size.
        bf16_components_gb = (37.8, 24.4, 5.5),
        gguf_repo = "unsloth/LTX-2.3-GGUF",
        te_prequant_repos = (("fp8", "text_encoder", "unsloth/LTX-2-FP8"),),
        # Hosted 2.3 DISTILLED DiT, used only by the 2.3 single-file assembly. fp8 only: LTX-2.3-INT8.pt predates the int8 excludes.
        prequant_variant_repos = (("lightricks/ltx-2.3", "fp8", "unsloth/LTX-2.3-FP8"),),
        # no steady gain on LTX's VAE / vocoder convs, but a per-shape re-tune (first render 26 s vs 10 s)
        cudnn_benchmark = False,
        # Block RMSNorm (hidden 4096) R0_BLOCK 4096 vs 2048 tie on B200: 3 of 6 cold servers rendered another clip.
        filter_reduction_configs = True,
    ),
    # Wan2.2-TI2V-5B (diffusers >= 0.35): VAE temporal compression 4 gives frame counts 4k+1.
    VideoFamily(
        name = "wan2.2-ti2v-5b",
        comfy_flow_shift = 8.0,
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        prequant_repos = (("nvfp4", "unsloth/Wan2.2-TI2V-5B-NVFP4"),),
        prequant_filenames = (("nvfp4", "Wan2.2-TI2V-5B-NVFP4.pt"),),
        prequant_resident_gb_by_scheme = (("nvfp4", 2.9),),
        aliases = ("wan2.2-5b", "wan-ti2v", "wan2.2-ti2v", "wan-ti2v-5b"),
        has_audio = False,
        default_steps = 20,
        default_guidance = 5.0,
        default_num_frames = 121,
        default_fps = 24,
        frame_step = 4,
        resolution_multiple = 32,
        # TI2V-5B is 720P-only: upstream SUPPORTED_SIZES is exactly 704x1280 / 1280x704.
        resolution_presets = ((1280, 704), (704, 1280)),
        # bf16-RESIDENT; transformer + VAE ship fp32 on disk, VAE stays fp32.
        bf16_components_gb = (10.0, 11.4, 2.8),
        vae_force_fp32 = True,
        cudnn_benchmark = False,
        offload_cuda_graph = True,
        # UMT5 keeps its overflowing `wo` in fp32 itself; the VAE stays fp32 (vae_force_fp32).
        fp16_guard = "native",
        gguf_repo = "unsloth/Wan2.2-TI2V-5B-GGUF",
    ),
    # Wan2.2-T2V-A14B dual-expert MoE: low-noise steps go to transformer_2 (cfg2_kwarg).
    VideoFamily(
        name = "wan2.2-t2v-a14b",
        comfy_flow_shift = 5.0,
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        prequant_repos = (("nvfp4", "unsloth/Wan2.2-T2V-A14B-NVFP4"),),
        prequant_filenames = (
            ("nvfp4", "Wan2.2-T2V-A14B-NVFP4.pt"),
            ("nvfp4", "transformer_2", "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"),
        ),
        # BOTH experts: the plan subtracts one denoiser term and this family builds two.
        prequant_resident_gb_by_scheme = (("nvfp4", 16.2),),
        aliases = ("wan2.2-14b", "wan-t2v", "wan2.2-t2v", "wan-t2v-a14b", "wan-a14b"),
        has_audio = False,
        transformer2_class = "WanTransformer3DModel",
        is_moe = True,
        cfg2_kwarg = "guidance_scale_2",
        default_steps = 20,
        default_guidance = 3.5,
        default_num_frames = 81,
        default_fps = 16,
        frame_step = 4,
        resolution_multiple = 16,
        resolution_presets = ((1280, 720), (832, 480), (480, 832), (720, 1280)),
        # bf16-RESIDENT: each expert ~28.6 GB bf16 (ships fp32), ~57.2 for both.
        bf16_components_gb = (57.2, 11.4, 0.5),
        vae_force_fp32 = True,
        cudnn_benchmark = False,
        # No gguf_repo: community GGUFs split the experts and a single-file load covers one.
    ),
    # HunyuanVideo-1.5: CFG via guider, no callback_on_step_end, no model_index.json (repacks only).
    VideoFamily(
        name = "hunyuanvideo-1.5",
        comfy_flow_shift = 7.0,
        pipeline_class = "HunyuanVideo15Pipeline",
        transformer_class = "HunyuanVideo15Transformer3DModel",
        base_repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        prequant_repos = (("nvfp4", "unsloth/HunyuanVideo-1.5-NVFP4"),),
        prequant_filenames = (("nvfp4", "HunyuanVideo-1.5-Diffusers-480p_t2v-NVFP4.pt"),),
        prequant_resident_gb_by_scheme = (("nvfp4", 4.8),),
        # No bare "hunyuanvideo" alias: it would also claim the incompatible 1.0 repos.
        aliases = ("hunyuanvideo-1-5", "hunyuanvideo1.5", "hunyuanvideo1-5", "hv15"),
        has_audio = False,
        guidance_via_guider = True,
        default_steps = 20,
        default_guidance = 6.0,
        default_num_frames = 121,
        default_fps = 24,
        frame_step = 4,
        resolution_multiple = 16,
        # Every entry is a real bucket of the 480p tier (generate_crop_size_list(base_size=640)).
        resolution_presets = ((832, 480), (480, 832), (640, 640)),
        bf16_components_gb = (16.6, 14.8, 2.4),
        fp16_guard = "native",
        cudnn_benchmark = False,
        offload_cuda_graph = True,
    ),
    # 720p t2v repack: own family so 720p loads default to 720p sizes; full-path alias wins.
    VideoFamily(
        name = "hunyuanvideo-1.5-720p",
        comfy_flow_shift = 7.0,
        pipeline_class = "HunyuanVideo15Pipeline",
        transformer_class = "HunyuanVideo15Transformer3DModel",
        base_repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        prequant_repos = (("nvfp4", "unsloth/HunyuanVideo-1.5-NVFP4"),),
        prequant_filenames = (("nvfp4", "HunyuanVideo-1.5-Diffusers-720p_t2v-NVFP4.pt"),),
        prequant_resident_gb_by_scheme = (("nvfp4", 4.8),),
        aliases = ("hunyuanvideo-1.5-diffusers-720p_t2v", "hv15-720p"),
        has_audio = False,
        guidance_via_guider = True,
        default_steps = 20,
        default_guidance = 6.0,
        default_num_frames = 121,
        default_fps = 24,
        frame_step = 4,
        resolution_multiple = 16,
        resolution_presets = ((1280, 720), (720, 1280), (960, 960)),
        bf16_components_gb = (16.6, 14.8, 2.4),
        fp16_guard = "native",
        cudnn_benchmark = False,
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
    # HV1.5 resolution tier is baked into the weights (480p vs 720p); route on the tier marker.
    if fam.name == "hunyuanvideo-1.5" and re.search(r"(?:^|[-_./\\])720p", needle):
        for candidate in _FAMILIES:
            if candidate.name == "hunyuanvideo-1.5-720p":
                return candidate
    return fam


def supported_video_family_names() -> tuple[str, ...]:
    return tuple(fam.name for fam in _FAMILIES)


def pipeline_available_video_families(*, device: Optional[str] = None) -> tuple[VideoFamily, ...]:
    from .diffusion_families import family_selectable
    return tuple(
        fam
        for fam in _FAMILIES
        if family_selectable(fam) and not (device == "mps" and fam.modular_workflow)
    )


def resolve_video_base_repo(fam: VideoFamily, base_repo: Optional[str]) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback."""
    base = (base_repo or "").strip()
    return base or fam.base_repo


def _prequant_base_key(repo_id: Optional[str]) -> str:
    """The lookup key ``prequant_variant_repos`` is written against: trimmed and lowercased.

    Deliberately local rather than reusing ``diffusion_families.canonical_base``: this module
    header keeps the two registries apart so neither picker can reach the other's tables, and the
    image mirror map holds image repos only, so importing it would buy nothing but the coupling.
    """
    return (repo_id or "").strip().lower()


def video_family_prequant_repo(
    fam: VideoFamily,
    scheme: str,
    base_repo: Optional[str] = None,
) -> Optional[str]:
    """The hosted pre-quantized DENOISER repo for ``scheme`` in this family, or None.

    Mirrors ``diffusion_families.family_prequant_repo``: ``base_repo`` (when known) selects a
    variant-specific checkpoint first, then the family default. Pure -- no IO, no torch -- so
    validation and download planning can both ask before anything is downloaded.

    Reads the tables through ``getattr`` and skips malformed rows instead of raising: this runs on
    the refusal path of a load request, and a table typo must not turn a legitimate pick into a
    500. A family object that predates these fields simply has no hosted checkpoint.
    """
    if nvfp4_blocked(scheme):
        return None
    base = _prequant_base_key(base_repo)
    if base:
        for entry in getattr(fam, "prequant_variant_repos", ()) or ():
            if not isinstance(entry, (tuple, list)) or len(entry) != 3:
                continue
            entry_base, entry_scheme, repo_id = entry
            if _prequant_base_key(entry_base) == base and entry_scheme == scheme and repo_id:
                return repo_id
    for entry in getattr(fam, "prequant_repos", ()) or ():
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        entry_scheme, repo_id = entry
        if entry_scheme == scheme and repo_id:
            return repo_id
    return None


def video_family_prequant_resident_gb(fam: VideoFamily, scheme: str) -> Optional[float]:
    """The MEASURED resident size in decimal GB of this family's hosted ``scheme`` denoiser."""
    if nvfp4_blocked(scheme):
        return None
    for entry in getattr(fam, "prequant_resident_gb_by_scheme", ()) or ():
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        entry_scheme, resident_gb = entry
        if entry_scheme == scheme and resident_gb:
            try:
                return float(resident_gb)
            except (TypeError, ValueError):  # a malformed row is "not measured", never a 500
                continue
    measured = getattr(fam, "prequant_resident_gb", None)
    try:
        return float(measured) if measured else None
    except (TypeError, ValueError):
        return None


def video_family_prequant_task_specific(fam: VideoFamily, scheme: str, task: str) -> bool:
    """True when the family names an artifact for exactly this ``(scheme, task)`` pair.

    Reads the same ``prequant_filenames`` table ``resolve_prequant_source`` reads, through the
    shared resolver, so the answer here and the file the load asks for cannot drift."""
    wanted = (task or "").strip().lower()
    if not wanted:
        return False
    try:
        from .diffusion_families import family_prequant_filename
        specific = family_prequant_filename(fam, scheme, task = wanted)
    except Exception:  # noqa: BLE001 -- a bad table is "no artifact", never a 500
        return False
    return specific is not None and specific != family_prequant_filename(fam, scheme)


def video_family_prequant_available(
    fam: VideoFamily,
    scheme: str,
    *,
    task: Optional[str] = None,
    base_repo: Optional[str] = None,
) -> bool:
    """True when a hosted pre-quantized denoiser really covers ``(scheme, task)``.

    ``video_family_prequant_repo`` answers "is there a checkpoint for this scheme"; this answers
    the question a load actually has, which also names the PARTITION. A task listed in
    ``prequant_partition_tasks`` is served only by its own ``(scheme, task, filename)`` row, so a
    scheme that has the repo but not that row is unavailable for it -- the alternative is loading
    another partition's denoiser, which passes every check and generates the wrong thing.

    Every other task, and every family that declares no partition tasks, gets exactly the old
    answer. Pure, and never raises: this runs on the refusal and download-planning paths."""
    if video_family_prequant_repo(fam, scheme, base_repo) is None:
        return False
    wanted = (task or "").strip().lower()
    partition_tasks = {
        (t or "").strip().lower() for t in (getattr(fam, "prequant_partition_tasks", ()) or ())
    }
    if wanted and wanted in partition_tasks:
        return video_family_prequant_task_specific(fam, scheme, wanted)
    return True


def video_family_prequant_schemes(fam: VideoFamily, task: Optional[str] = None) -> tuple[str, ...]:
    """Every scheme this family has a hosted denoiser checkpoint for, in table order.

    Used to name the workable schemes in a refusal message, so a rejected request tells the caller
    what to pick instead of only what failed. With ``task``, the list is narrowed to the schemes
    that cover THAT task, so a reference-video refusal cannot advertise a keyframe-only scheme.
    Malformed rows are skipped, as above."""
    schemes: list[str] = []
    for entry in getattr(fam, "prequant_repos", ()) or ():
        if isinstance(entry, (tuple, list)) and len(entry) == 2 and entry[0] not in schemes:
            schemes.append(entry[0])
    for entry in getattr(fam, "prequant_variant_repos", ()) or ():
        if isinstance(entry, (tuple, list)) and len(entry) == 3 and entry[1] not in schemes:
            schemes.append(entry[1])
    if task:
        schemes = [s for s in schemes if video_family_prequant_available(fam, s, task = task)]
    return without_nvfp4(schemes)


def snap_num_frames(fam: VideoFamily, num_frames: int) -> int:
    """The nearest valid frame count at or below the request (k * step + offset).

    Video latents are allocated as (num_frames - 1) / temporal_compression + 1, so
    an off-lattice count wastes a partial latent frame at best and trips shape
    checks at worst; snapping mirrors the image path's silent /16 size snap.
    """
    step = max(1, fam.frame_step)
    offset = max(1, fam.frame_offset)
    requested = max(offset, fam.min_num_frames, num_frames)
    if fam.max_num_frames is not None:
        requested = min(requested, fam.max_num_frames)
    delta = requested - offset
    if fam.snap_frames_up:
        snapped = ((delta + step - 1) // step) * step + offset
    else:
        snapped = (delta // step) * step + offset
    if fam.max_num_frames is not None:
        snapped = min(snapped, fam.max_num_frames)
    return max(offset, fam.min_num_frames, snapped)


def snap_video_size(fam: VideoFamily, width: int, height: int) -> tuple[int, int]:
    """Width/height floored to the family's required multiple (minimum one unit)."""
    multiple = max(1, fam.resolution_multiple)
    snap = lambda v: max(multiple, (max(1, v) // multiple) * multiple)  # noqa: E731
    return snap(width), snap(height)


def format_video_resolution_presets(fam: VideoFamily) -> str:
    """The family's presets as '768x512, 1216x704, ...' for messages and logs."""
    return ", ".join(f"{w}x{h}" for w, h in fam.resolution_presets)


class VideoShapeError(ValueError):
    """A shape this family cannot render. A ValueError subclass so the existing
    ``except ValueError`` callers keep catching it, and a distinct type so the generate
    route can answer 422 (the body is in range, the shape is not supported) without
    widening the 400 it gives every other bad-input ValueError."""


def validate_video_request_shape(
    fam: VideoFamily,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_frames: Optional[int] = None,
) -> None:
    """Raise ``ValueError`` when a request asks for a shape ``fam`` does not support.

    The generate route calls this at the API boundary so HTTP enforces exactly the rules the Desktop
    interface offers: its resolution select lists only ``resolution_presets`` and its duration
    select only lattice frame counts, while the API took anything inside the coarse request bounds
    and then SNAPPED it. The snap is silent and floors, so a 256x256 request survived untouched and
    denoised at a size no checkpoint was ever trained for.

    This is a separate, explicit check rather than a change to ``snap_video_size`` /
    ``snap_num_frames``, which internal callers still need. It stays silent for anything it cannot
    judge (a family that declares no presets keeps the old SIZE snapping), and ``None`` means "use
    the family default". The frame lattice is deliberately NOT part of that escape hatch: every
    family declares a ``frame_step``, so an off-lattice count is always refused.
    """
    presets = tuple((int(w), int(h)) for w, h in fam.resolution_presets)
    # No declared presets: leave sizes to the snap; frame check below still runs.
    if presets and (width is not None or height is not None):
        # Keyed on None (generate() keys on falsiness), so an explicit 0 is judged, not replaced.
        want_w = presets[0][0] if width is None else int(width)
        want_h = presets[0][1] if height is None else int(height)
        if (want_w, want_h) not in presets:
            raise VideoShapeError(
                f"{want_w}x{want_h} is not a supported resolution for {fam.name}. "
                f"Supported resolutions: {format_video_resolution_presets(fam)}."
            )
    if num_frames is not None:
        # Lattice is k*frame_step + frame_offset (H3 is 17k+5); same fields as snap_num_frames.
        step = max(1, fam.frame_step)
        offset = max(1, fam.frame_offset)
        count = int(num_frames)
        # Trained window, also enforced below so out-of-range requests are not silently snapped.
        ceiling = MAX_VIDEO_NUM_FRAMES
        if fam.max_num_frames is not None:
            ceiling = min(ceiling, int(fam.max_num_frames))
        floor = max(offset, int(fam.min_num_frames))
        if count < offset or (count - offset) % step != 0:
            # Computed from the lattice: snap_num_frames floors for some families and ceils for others.
            below = offset + max(0, (count - offset) // step) * step
            above = below + step
            # Only suggest points within the request model's `le` and the family range.
            loadable = [n for n in (below, above) if floor <= n <= ceiling]
            if len(loadable) == 2:
                nearest = f"the nearest supported counts are {loadable[0]} and {loadable[1]}"
            elif len(loadable) == 1:
                nearest = f"the nearest supported count is {loadable[0]}"
            else:
                nearest = f"supported counts run from {floor} to {ceiling}"
            raise VideoShapeError(
                f"{count} is not a supported frame count for {fam.name}. Its VAE compresses time by "
                f"{step}, so a frame count must be k * {step} + {offset}; {nearest} "
                f"(the default is {fam.default_num_frames})."
            )
        # On the lattice but outside the trained window: refuse rather than silently snap.
        if count < floor or count > ceiling:
            raise VideoShapeError(
                f"{count} is not a supported frame count for {fam.name}. "
                f"Supported counts run from {floor} to {ceiling} "
                f"(the default is {fam.default_num_frames})."
            )


def validate_video_keyframe_conditioning(
    fam: VideoFamily, h3_task: Optional[str], *, has_keyframes: bool
) -> None:
    """Raise ``ValueError`` when a checkpoint cannot take the keyframes a request supplies.

    Pure in the family and the MiniMax-H3 partition, which is what lets the generate route judge
    the checkpoint it is about to SWITCH TO by the same rules the backend applies to the loaded
    one. Without that, an auto-switch evicts a working pipeline and spends minutes loading a
    target for a request that was already known to be unservable.
    """
    if not has_keyframes:
        return
    from .video_minimax_h3 import H3_TASK_REFERENCES

    if not fam.supports_keyframes:
        raise ValueError(
            f"{fam.name} generates from the prompt alone; it takes no first or last frame."
        )
    if h3_task == H3_TASK_REFERENCES:
        raise ValueError(
            "The MiniMax-H3 checkpoint is the Ref2VA partition, which conditions on references "
            "rather than keyframes. Load a minimax_h3_fl2va checkpoint to generate from a first "
            "or last frame."
        )


def validate_video_flow_controls(
    fam: VideoFamily,
    flow_shift: Optional[float],
    audio_flow_shift: Optional[float],
    *,
    engine: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` when a request sets a shift the checkpoint cannot honour.

    The backend's flow-shift rules, kept here so the generate route can judge the checkpoint it
    is about to switch TO by the same ones. ``engine`` is optional because a target's engine is
    normally not chosen until the load runs; where it IS determined by the pick, as MiniMax-H3
    GGUFs are, passing it refuses an unservable request before anything is evicted.
    """
    if flow_shift is not None and fam.default_flow_shift is None:
        raise ValueError(f"{fam.name} does not expose a video flow_shift control.")
    if audio_flow_shift is not None and fam.default_audio_flow_shift is None:
        raise ValueError(f"{fam.name} does not expose an audio_flow_shift control.")
    if (
        audio_flow_shift is not None
        and engine == "sd_cpp"
        and audio_flow_shift != fam.default_audio_flow_shift
    ):
        raise ValueError(
            "stable-diffusion.cpp derives the audio schedule against a fixed "
            f"{fam.default_audio_flow_shift:g} shift, so audio_flow_shift needs the "
            "Diffusers engine."
        )


def validate_video_reference_conditioning(
    fam: VideoFamily,
    h3_task: Optional[str],
    *,
    has_references: bool,
    reference_image_size: Optional[str] = None,
    engine: Optional[str] = None,
) -> None:
    """Raise ``ValueError`` when a checkpoint cannot be conditioned on the request's references.

    The absence of references is a rule too: the Ref2VA partition has no text-only denoiser. See
    ``validate_video_keyframe_conditioning`` for why these live here rather than inline.

    ``engine`` is optional for the same reason it is on the flow controls: a target's engine is
    normally unknown before the load, but where the pick decides it, passing it refuses an
    unservable sizing policy before anything is evicted.
    """
    from .video_minimax_h3 import H3_REF_SIZE_MATCH, H3_REF_SIZE_MAX, H3_TASK_REFERENCES

    if not has_references:
        if h3_task == H3_TASK_REFERENCES:
            raise ValueError(
                "The MiniMax-H3 checkpoint is the Ref2VA partition, which generates from "
                "references. Add at least one reference image or video, or load a "
                "minimax_h3_fl2va checkpoint for text-to-video."
            )
        return
    if not fam.supports_references:
        raise ValueError(f"{fam.name} takes no reference images, videos or audio.")
    if h3_task != H3_TASK_REFERENCES:
        raise ValueError(
            "The MiniMax-H3 checkpoint is the FL2VA partition, which conditions on keyframes "
            "rather than references. Load a minimax_h3_ref2va checkpoint to generate from "
            "references."
        )
    policy = (reference_image_size or H3_REF_SIZE_MATCH).strip().lower()
    if policy not in (H3_REF_SIZE_MATCH, H3_REF_SIZE_MAX):
        raise ValueError(
            f"reference_image_size must be '{H3_REF_SIZE_MATCH}' or '{H3_REF_SIZE_MAX}'."
        )
    if policy == H3_REF_SIZE_MAX and engine == "sd_cpp":
        raise ValueError(
            "stable-diffusion.cpp scales every reference to the generation's pixel area, so "
            f"'{H3_REF_SIZE_MAX}' reference sizing needs the Diffusers engine. Use "
            f"'{H3_REF_SIZE_MATCH}' with this checkpoint."
        )


# (steps, guidance) per variant, matched by substring, most specific first.
_VIDEO_GENERATION_DEFAULTS: tuple[tuple[str, int, float], ...] = (
    ("distilled", 8, 1.0),
    ("ltx", 40, 4.0),
    # T2V-A14B before the generic Wan key.
    ("a14b", 20, 3.5),
    ("wan2.2-14b", 20, 3.5),
    ("wan", 20, 5.0),
    ("hunyuanvideo", 20, 6.0),
)


def default_video_generation_params(
    *identifiers: Optional[str], fallback: tuple[int, float] = (40, 4.0)
) -> tuple[int, float]:
    """Default ``(steps, guidance)`` for a loaded video model; the first identifier
    naming a known variant wins, so a GGUF filename ('...distilled...Q4_K_M.gguf')
    beats the family base repo. ``fallback`` is used when no identifier names a variant --
    callers pass the resolved family's own default so a Wan model loaded from an opaque local
    path under an explicit family_override still gets 50/5.0, not the hardcoded LTX 40/4.0."""
    variant = video_generation_variant(*identifiers)
    for key, steps, guidance in _VIDEO_GENERATION_DEFAULTS:
        if key == variant:
            return steps, guidance
    return fallback


def video_generation_variant(*identifiers: Optional[str]) -> Optional[str]:
    """The ``_VIDEO_GENERATION_DEFAULTS`` key the first identifier naming one matches, or None; the precedence
    ``default_video_generation_params`` uses, so every per-variant decision agrees with the defaults."""
    for identifier in identifiers:
        needle = (identifier or "").lower()
        for key, _steps, _guidance in _VIDEO_GENERATION_DEFAULTS:
            # Reject a preceding ASCII letter so "swan-video" does not match "wan".
            if re.search(r"(?<![a-z])" + re.escape(key), needle):
                return key
    return None
