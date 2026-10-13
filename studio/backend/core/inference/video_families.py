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
from .family_name_match import normalize_family_name, token_in_name, token_length
from .family_profile import AutoPrefer, ConvRotSpec, FamilyProfile, UntiledVae

# num_frames ceiling; VideoGenerateRequest imports it for its `le` so gate and bound cannot drift.
MAX_VIDEO_NUM_FRAMES = 1024

# Routes match these EXACTLY to return 409 instead of 500.
VIDEO_NOT_LOADED_MSG = "No video model is loaded."
VIDEO_CANCELLED_MSG = "Video generation was cancelled."
VIDEO_GENERATION_BUSY_MSG = "A video generation is already in progress."
VIDEO_MODEL_CHANGED_MSG = "The requested video model changed before generation was reserved."

# One LTX version token: ltx / ltxv / ltx-video, optional separator, major 2, then ".N" / "_N" (minor, any digits, so
# "ltx-2.10" reads as 2.10), "N" ending the
# token ("ltx25"), or nothing ("ltx-2-19b" = 2.0). Anything else ("ltx-24fps", "ltx-transformer") is no name evidence,
# as is a bare "ltx23" / "ltx-23" (no release uses that spelling, so it cannot outrank the repo).
_LTX_VERSION = re.compile(
    r"(?<![a-z0-9])ltx(?:v|[-_ ]?video)?[-_ ]?2(?:[._]([0-9]+)(?![0-9])|([0-9])(?![0-9a-z])|(?![0-9a-z]|[._][0-9]))"
)


def ltx_name_version(name: Optional[str]) -> Optional[str]:
    for found in _LTX_VERSION.finditer((name or "").lower()):
        if found.group(2) is None or int(found.group(2)) >= 4:
            return f"2.{found.group(1) or found.group(2) or 0}"
    return None


# Family per LTX release: 2.0 / 2.3 run the LTX-2 assembly, 2.5 (diffusers called its pre-release "2.4") its own family.
LTX2_FAMILY = "ltx-2"
LTX25_FAMILY = "ltx-2.5"
LTX_FAMILIES = (LTX2_FAMILY, LTX25_FAMILY)


def ltx_family_for_version(version: Optional[str]) -> Optional[str]:
    """``ltx-2`` for 2.0-2.3, ``ltx-2.5`` for 2.4 / 2.5, None for a release Studio does not know (2.6+)."""
    if not version:
        return None
    minor = int(version[2:])
    if minor <= 3:
        return LTX2_FAMILY
    if minor <= 5:
        return LTX25_FAMILY
    return None


def unsupported_ltx_version(*names: Optional[str]) -> Optional[str]:
    # Releases after 2.5 are unknown: neither LTX family can claim their weights.
    for name in names:
        version = ltx_name_version(name)
        if version is not None and ltx_family_for_version(version) is None:
            return version
    return None


def unsupported_ltx_message(version: str, subject: str) -> str:
    return (
        f"{subject} is LTX-{version}, which Studio does not support yet. Studio runs LTX-2.0 / LTX-2.3 "
        "(Lightricks/LTX-2.3 or unsloth/LTX-2.3-GGUF) and LTX-2.5 (Lightricks/LTX-2.5-Diffusers or "
        "unsloth/LTX-2.5-GGUF)."
    )


def wrong_ltx_family_message(version: str, family: str, subject: str) -> str:
    expected = ltx_family_for_version(version)
    return (
        f"{subject} is LTX-{version}, which loads as the '{expected}' family, not '{family}'. "
        "LTX-2.5 is not a drop-in LTX-2.3 (Gemma 4 text encoder, no video FFN bias, keyframe position "
        "embedding), so one family cannot load the other's weights. Pick the "
        f"'{expected}' family, or leave the family on Auto."
    )


def _ltx_leaf(name: Optional[str]) -> str:
    # Last segment only: a local parent folder must not decide.
    return os.path.basename((name or "").replace("\\", "/").rstrip("/"))


def ltx_pick_version(
    repo_id: Optional[str],
    filename: Optional[str] = None,
    display_repo_id: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """``(version, shown)`` the pick's names declare. A file whose own name has a versioned LTX token decides alone
    (an ltx-2.3 file in an ``ltx-2.5-tests`` folder is 2.3); any other file name (``model``, ``ltx-transformer``)
    defers to the display id, then the repo id."""
    base = _ltx_leaf(filename)
    if ltx_name_version(base) is not None:
        picks: tuple[tuple[str, Optional[str]], ...] = ((base, filename),)
    else:
        picks = (
            (base, filename),
            (_ltx_leaf(display_repo_id), display_repo_id),
            (_ltx_leaf(repo_id), repo_id),
        )
    for name, shown in picks:
        version = ltx_name_version(name)
        if version is not None:
            return version, str(shown)
    return None


def assert_supported_ltx_version(
    repo_id: Optional[str],
    filename: Optional[str] = None,
    display_repo_id: Optional[str] = None,
) -> None:
    """Refuse a pick whose names declare an LTX release no family supports (2.6 and later)."""
    found = ltx_pick_version(repo_id, filename, display_repo_id)
    if found is not None and ltx_family_for_version(found[0]) is None:
        raise ValueError(unsupported_ltx_message(found[0], f"'{found[1]}'"))


def ltx_family_by_name(
    repo_id: Optional[str],
    filename: Optional[str] = None,
    display_repo_id: Optional[str] = None,
) -> Optional[str]:
    """The LTX family the pick's names declare, or None (no versioned LTX token, or an unknown release)."""
    found = ltx_pick_version(repo_id, filename, display_repo_id)
    return ltx_family_for_version(found[0]) if found is not None else None


def assert_ltx_pick_supported(
    family: Optional[str],
    repo_id: Optional[str],
    filename: Optional[str] = None,
    display_repo_id: Optional[str] = None,
) -> None:
    """Name guard for a pick already resolved to ``family``: an LTX family never loads another release's weights
    (LTX-2.5 never reaches the 2.3 assembly, and the reverse), and a release after 2.5 is refused. A non-LTX family
    never trips on an LTX token."""
    if family not in LTX_FAMILIES:
        return
    found = ltx_pick_version(repo_id, filename, display_repo_id)
    if found is None:
        return
    version, shown = found
    expected = ltx_family_for_version(version)
    if expected is None:
        raise ValueError(unsupported_ltx_message(version, f"'{shown}'"))
    if expected != family:
        raise ValueError(wrong_ltx_family_message(version, family, f"'{shown}'"))


def local_ltx_pipeline_family(path: Optional[str]) -> Optional[str]:
    """The LTX family a local diffusers folder's ``model_index.json`` declares: ``ltx-2.5`` for a Gemma 4 encoder or a
    ``diffusion_decoder`` / ``duration_head`` component, ``ltx-2`` for any other LTX2 pipeline; None when the folder
    has no readable LTX2 index."""
    import json
    from pathlib import Path

    try:
        index = Path(str(path or "")).expanduser() / "model_index.json"
        if not path or not index.is_file():
            return None
        spec = json.loads(index.read_text(encoding = "utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(spec, dict) or not str(spec.get("_class_name", "")).startswith("LTX2"):
        return None
    encoder = spec.get("text_encoder")
    encoder_class = encoder[1] if isinstance(encoder, list) and len(encoder) == 2 else ""
    # An empty or null component entry is absent, like a malformed text_encoder above.
    if str(encoder_class).startswith("Gemma4") or any(
        isinstance(spec.get(k), list) and bool(spec.get(k)) and spec.get(k)[0]
        for k in ("diffusion_decoder", "duration_head")
    ):
        return LTX25_FAMILY
    return LTX2_FAMILY


def assert_local_pipeline_family(path: Optional[str], family: Optional[str]) -> None:
    """Refuse a local diffusers LTX folder whose ``model_index.json`` declares the other LTX family."""
    if family not in LTX_FAMILIES:
        return
    declared = local_ltx_pipeline_family(path)
    if declared is not None and declared != family:
        version = "2.5" if declared == LTX25_FAMILY else "2.3"
        raise ValueError(
            wrong_ltx_family_message(version, family, f"'{path}' (by its model_index.json)")
        )


def ltx25_training_message(subject: str) -> str:
    return (
        f"{subject} is LTX-2.5, which Studio can run on the Video page but cannot train yet. "
        "Train on LTX-2 (Lightricks/LTX-2) instead."
    )


def assert_local_pipeline_not_ltx25(path: Optional[str]) -> None:
    """Training guard: refuse a local diffusers LTX-2.5 folder whose name does not say so; unreadable passes."""
    if local_ltx_pipeline_family(path) == LTX25_FAMILY:
        raise ValueError(ltx25_training_message(f"'{path}' (by its model_index.json)"))


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
    # Same spelling as DiffusionFamily: pin the filter only on these (major, minor) archs.
    filter_reduction_configs_archs: tuple[tuple[int, int], ...] = field(default_factory = tuple)
    # Curated GGUF repo for the picker (the DiT as single-file GGUF quants).
    gguf_repo: Optional[str] = None
    # Subdirectory of gguf_repo a bare repo id means; None keeps the root.
    gguf_default_subdir: Optional[str] = None
    # Picker label for root files when they differ from gguf_default_subdir ("UD-Q4_K_XL · dev").
    gguf_root_label: Optional[str] = None
    # Hosted PRE-CAST text-encoder checkpoints as (scheme, component, repo_id); same semantics as
    # DiffusionFamily.te_prequant_repos.
    te_prequant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Hosted PRE-QUANTIZED DENOISER checkpoints as (scheme, repo_id); separate from the text encoder.
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Resident GB of a hosted denoiser when _QUANT_STEADY_FACTOR is wrong (H3 is also pruned).
    prequant_resident_gb: Optional[float] = None
    prequant_resident_gb_by_scheme: tuple[tuple[str, float], ...] = field(default_factory = tuple)
    # (base_repo, scheme, repo_id) keyed on lowercased base id: a prequant only loads on its own base.
    prequant_variant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # (scheme, filename) overrides the derived name, which stays as fallback; task rows get none.
    prequant_filenames: tuple[tuple[str, ...], ...] = field(default_factory = tuple)
    # Tasks with a separate denoiser partition, served ONLY by their own (scheme, task, filename) row.
    prequant_partition_tasks: tuple[str, ...] = field(default_factory = tuple)
    modular_workflow: Optional[str] = None
    default_flow_shift: Optional[float] = None
    comfy_flow_shift: Optional[float] = None
    default_audio_flow_shift: Optional[float] = None
    supports_keyframes: bool = False
    # False for a keyframe family that takes a first frame only (LTX-2.5); MiniMax-H3 takes both.
    accepts_last_frame: bool = True
    # Diffusers class serving a first-frame request over the loaded pipeline's components (video_i2v).
    i2v_pipeline_class: Optional[str] = None
    i2v_pipeline_config: tuple[tuple[str, object], ...] = field(default_factory = tuple)
    # (identifier regex, ((kwarg, value), ...)); first matching row applies.
    i2v_call_kwargs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = field(
        default_factory = tuple
    )
    # (identifier regex, refusal) for checkpoints the name tokens claim but this family cannot run.
    unsupported_checkpoints: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # Omni-reference conditioning. MiniMax-H3 loads this in a separate denoiser partition.
    supports_references: bool = False
    supports_cfg: bool = True
    # True when Unsloth can TRAIN a LoRA on this family (every such name must resolve through get_trainer).
    trainable: bool = False
    # Display label (Train UI); None = the family name.
    label: Optional[str] = None
    # Per-family policy data; see family_profile.FamilyProfile.
    profile: FamilyProfile = field(default_factory = FamilyProfile)


# ComfyUI default 864x480: multiples of 32, inside the trained 1:4..4:1 range.
def _h3_480p_presets() -> tuple[tuple[int, int], ...]:
    flag = os.environ.get("UNSLOTH_VIDEO_H3_480P", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return ()
    return ((864, 480), (480, 864))


# LTX-Video 0.9.x matches ltx-2's "ltx-video"/"ltxv" tokens but is another architecture; names of LTX-2 itself pass.
_LTX_VIDEO_09_RE = (
    r"^(?!.*(?<![a-z0-9])ltx[-_ ]?2(?:$|[-_ /.]))"
    r"(?=.*(?<![a-z0-9])ltx(?:[-_ ]?video|v)(?:$|[-_ /.](?!2(?:$|[-_ /.]|\.\d))))"
)
_LTX_VIDEO_09_REFUSAL = (
    "This is an LTX-Video 0.9.x checkpoint, a different architecture from LTX-2 that Studio does not "
    "run. Load LTX-2 or LTX-2.3 instead (for example unsloth/LTX-2.3-GGUF)."
)
# Last path segment of each identifier only: an owner or parent folder named i2v does not make a checkpoint I2V.
_HV15_I2V_RE = r"(?<![a-z0-9])i2v(?![a-z0-9])(?=[^/\\|]*[/\\]*\s*(?:$|\|))"
_HV15_I2V_REFUSAL = (
    "This is a HunyuanVideo-1.5 image-to-video checkpoint. It needs its own image-conditioned "
    "transformer and SigLIP image encoder, which Studio does not load yet. Load a HunyuanVideo-1.5 "
    "text-to-video checkpoint (hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v or "
    "720p_t2v) instead."
)


# --- Shared FamilyProfile data (see family_profile.FamilyProfile) ---

# int8 small-M exclusions (on top of diffusion_transformer_quant._INT8_EXCLUDE_NAME_TOKENS); baked into the published
# int8 artifacts, so each moves only with a rebuild and republish.
# HunyuanVideo-1.5's attention trim shrinks the text / image streams from their padded lengths to the VALID token
# counts, so every text-stream Linear runs at a tiny M the int8 dynamic path cannot handle. Both failures measured on
# B200. M = 0 (t2v byt5 / image streams trim to zero tokens): torchao returns the input UNPROJECTED (a quantized 1472
# -> 2048 Linear maps [1, 0, 1472] to [1, 0, 1472]), so the 2048-wide cond-type add crashes -- context_embedder_2 /
# image_embedder. M <= 16 (a short prompt, or the empty negative prompt's ~6 tokens): torch._int_mm requires M > 16
# and raises -- the TokenRefiner and every block's context-stream projections. These run at M = text tokens (tens)
# against the video stream's M ~ 32k+, so bf16 here costs nothing measurable and the video-stream linears keep full
# int8 coverage. "context_embedder" also matches "context_embedder_2" (substring check).
_HUNYUAN15_INT8_EXCLUDES = (
    "context_embedder",
    "image_embedder",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "ff_context",
)
# nvfp4 raises on an empty activation; HunyuanVideo-1.5's attention trim hits it every t2v render.
_HUNYUAN15_NVFP4_ZERO_ROW_TOKENS = ("image_embedder", "context_embedder_2")
# LTX-2 is audiovisual: every block carries an audio-side stream (audio_attn1 / audio_attn2 / audio_ff, fed by
# audio_proj_in) plus the a2v / v2a cross attentions. A video-only run feeds a MINIMAL audio stream -- one token for a
# still -- so those linears run at M = 1, under the same _int_mm floor of 16. The audio stream is inert on a
# video-only run (isolated modalities, out of the loss, out of the LoRA targets), so leaving it in bf16 costs nothing
# while the video-stream linears keep full int8 coverage. One token covers every audio-side name, including
# video_to_audio_attn, whose keys/values are that same one-token stream.
# The audio token alone is not enough. LTX-2 also carries LTX2AdaLayerNormSingle projections whose names say nothing
# about audio -- av_cross_attn_video_scale_shift / av_cross_attn_video_a2v_gate (the cross-modality modulation,
# computed unconditionally in forward, isolate_modalities or not) and prompt_adaln / audio_prompt_adaln. Each is a
# Linear over a BATCH-sized input, so M = batch = 1 for a default training run: the same _int_mm floor, reached
# whatever the audio stream does.
_LTX2_INT8_EXCLUDES = ("audio", "av_cross_attn", "adaln")
# Untiled decode peaks (video_vae_untiled), measured fp16 on a B200. Wan: per latent pixel (h x w), frame count does not
# enter (the causal cache decodes one latent frame at a time), plus the (spatial, temporal) RGB output Wan's decode
# holds about twice (per-frame torch.cat, then clamp); 1021 frames at 1280x704 is 5.2 GiB per fp16 copy. A14B and
# Wan2.1 share the Wan2.1 VAE (8x spatial), measured <= 0.58 MiB per latent pixel at 480p / 720p.
_WAN21_VAE_UNTILED = UntiledVae(bytes_per_latent_pixel = 0.6 * 2**20, output_scale = (8, 4))
# LTX-2 / 2.3 holds every frame: peak per output frame x latent pixel, 0.102-0.105 MiB measured bf16 (B200). LTX-2.5
# has the same conv decoder layout (vae/ of the Diffusers pack); unmeasured, priced as 2.3.
_LTX2_VAE_UNTILED = UntiledVae(bytes_per_frame_latent_pixel = 0.11 * 2**20, temporal_scale = 8)

_FAMILIES: tuple[VideoFamily, ...] = (
    VideoFamily(
        name = "minimax-h3",
        trainable = True,
        label = "MiniMax-H3",
        profile = FamilyProfile(
            # Same small-M story as HunyuanVideo-1.5 with one extra trap: the adaLN projection is ``adaln_proj``,
            # which no generic token matches ("norm" does not appear in the name). On the DENSE checkpoint it is
            # Linear(2688 -> 96768), clears min_features = 512, gets quantized, then runs at M = 1 and raises
            # ``self.size(0) needs to be greater than 16, but got 1`` at the first denoise (the offline builder
            # happily bakes it, and torch.compile then dies on that module). On the pruned-modulation form it is
            # Linear(8 -> 96768), under min_features and skipped for free (verified on the fl2va_pruned build: all 51
            # rejected), so the exclusion makes the DENSE path correct and is a no-op on the pruned one.
            int8_exclude_tokens = ("adaln_proj",),
            # H3's OTHER small-M linears (context_embedder + the two token_refiner blocks: 13 Linears, 798 M params,
            # 0.80 GB bf16, 3.9% of the int8 checkpoint) are row-PADDED to the _int_mm floor instead of excluded:
            # padding returns the caller's rows bitwise unchanged (per-row activation scales, asserted in
            # diffusion_quant_pad). Measured on B200, 640x384 x 124 frames (h3_quant/README_int8.md). qwen-image,
            # qwen-image-edit and hunyuanvideo-1.5 have the same small-M shape but each has a PUBLISHED int8
            # checkpoint whose metadata bakes its exclusion set, so they move only with a rebuild and republish.
            int8_pad_tokens = ("context_embedder", "token_refiner"),
            # The packed sequence runs three modalities through one set of linears, so its activation range is not the
            # per-family range the fp8 filter was measured against. The trainer refuses both outright; declaring it
            # here keeps /diffusion/info from advertising a start that is guaranteed to 400.
            train_scheme_deny = frozenset({"fp8", "mxfp8"}),
            auto_static_skip = (("minimaxai/minimax-h3", (("max", 2), ("steps", 30))),),
        ),
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
        # No ref2va row means a ref2va pick is refused, not served the keyframe checkpoint.
        prequant_partition_tasks = ("ref2va",),
        # Both FP8 and INT8 schemes are ~20.3 GB resident against the 66.3 GB dense denoiser.
        prequant_resident_gb = 20.3,
        modular_workflow = "fl2va",
        default_flow_shift = 12.0,
        default_audio_flow_shift = 3.0,
        supports_keyframes = True,
        supports_references = True,
        supports_cfg = False,
    ),
    # LTX-2.5 (diffusers >= 0.40): the 22B LTX-2.3 DiT without the video FFN bias plus a keyframe position embedding,
    # conditioned by a fine-tuned Gemma 4 12B (Gemma4UnifiedForConditionalGeneration, transformers >= 5.10.1). That
    # encoder runs in a spawned worker on the .venv_t5_510 sidecar (video_ltx25_te_worker); everything else runs here.
    # Defaults are the reference two-stage distilled recipe (video_ltx25): 8 distilled sigmas at half resolution, x2
    # latent upsample, 3 sigmas at full resolution, CFG 1. Two stages need a 64 px canvas (32 px per stage), so the
    # reference 1280x720 resolves to the 1280x704 latent canvas it actually renders.
    VideoFamily(
        name = "ltx-2.5",
        profile = FamilyProfile(
            # the 2.3 DiT less the video FFN bias
            int8_exclude_tokens = _LTX2_INT8_EXCLUDES,
            # Hosted rotated int8 build only: rotation baked into the file (every quantized Linear, group 256) and read
            # back from its metadata. No runtime ConvRot spec (group 0), so quantizing at load stays plain int8;
            # default_on only picks the hosted rotated file (the env's ``0`` picks the plain one).
            int8_convrot = ConvRotSpec(
                group = 0,
                suffixes = (),
                variant_repos = ("unsloth/LTX-2.5-FP8",),
                default_on = True,
            ),
            untiled_vae = _LTX2_VAE_UNTILED,
            # 2.5 keeps the 2.3 conv VAE encoder (byte-identical weights), so the same latent space.
            compile_static = True,
            stg_compile_adapter = True,
            preview_factors = "ltx2",
            # The split single-file repo next to the Diffusers pack (safetensors, no remote code).
            trusted_repos = ("lightricks/ltx-2.5",),
        ),
        pipeline_class = "LTX2Pipeline",
        transformer_class = "LTX2VideoTransformer3DModel",
        base_repo = "Lightricks/LTX-2.5-Diffusers",
        aliases = ("ltx2.5", "ltx25", "ltxv-2.5", "ltx-video-2.5"),
        has_audio = True,
        default_steps = 8,
        default_guidance = 1.0,
        default_num_frames = 121,
        default_fps = 24,
        frame_step = 8,
        resolution_multiple = 64,
        resolution_presets = ((1280, 704), (704, 1280), (960, 512), (768, 512)),
        # transformer 42.0 bf16 (both DiT files are 42,018,190,584 bytes); Gemma 4 12B 23.9 (text_encoder/ shards),
        # held by the encoder worker; VAE 1.45 + connectors 6.34 + latent upsampler 1.0 + audio VAE / vocoder 0.37.
        bf16_components_gb = (42.0, 23.9, 9.2),
        # Hosted DiTs baked from the pack's distilled transformer/ (tensor-identical to the single-file distilled DiT).
        # Keyed on the pack: only a pipeline load seeds a hosted denoiser, and the pack never loads its dev DiT. int8
        # resolves LTX-2.5-INT8-ConvRot.safetensors (rotation in its metadata), LTX-2.5-INT8 with
        # UNSLOTH_DIFFUSION_INT8_CONVROT=0; fp8 resolves LTX-2.5-FP8.safetensors.
        prequant_variant_repos = (
            ("lightricks/ltx-2.5-diffusers", "int8", "unsloth/LTX-2.5-FP8"),
            ("lightricks/ltx-2.5-diffusers", "fp8", "unsloth/LTX-2.5-FP8"),
        ),
        # File sizes (decimal GB): int8 keeps the audio stream, AV cross-attention and adaLN in bf16; fp8 quantizes them.
        prequant_resident_gb_by_scheme = (("int8", 25.11), ("fp8", 19.06)),
        # Dev DiT at the root, distilled under distilled/; the pack above supplies VAEs and connectors.
        gguf_repo = "unsloth/LTX-2.5-GGUF",
        # Family defaults (8 + 3 steps, CFG 1) are the distilled DiT's.
        gguf_default_subdir = "distilled",
        gguf_root_label = "dev",
        supports_keyframes = True,
        accepts_last_frame = False,
        # Same DiT layout as LTX-2.3, so the same measured knobs.
        cudnn_benchmark = False,
        filter_reduction_configs = True,
    ),
    # LTX-2 (diffusers >= 0.39): ~19B single-stream video DiT generating synchronized audio + video in one pass. The
    # Gemma3-12B encoder is stored fp32 on the hub (~49 GB download, ~24 GB resident bf16). The base repo carries the
    # dev config (40 steps, CFG 4).
    VideoFamily(
        name = "ltx-2",
        trainable = True,
        label = "LTX-2",
        profile = FamilyProfile(
            int8_exclude_tokens = _LTX2_INT8_EXCLUDES,
            # 2.3 is the same audiovisual DiT, and builders spell it by its version.
            quant_names = ("ltx-2.3",),
            untiled_vae = _LTX2_VAE_UNTILED,
            compile_static = True,
            stg_compile_adapter = True,
            preview_factors = "ltx2",
            trusted_repos = ("lightricks/ltx-2.3-fp8",),
        ),
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
        i2v_pipeline_class = "LTX2ImageToVideoPipeline",
        # LTX-2.3's reference workflow re-compresses the frame at CRF 18; LTX-2.0 keeps diffusers' 33.
        i2v_call_kwargs = ((r"ltx[-_ ]?2[-_.]?3(?![0-9])", (("image_crf", 18),)),),
        unsupported_checkpoints = ((_LTX_VIDEO_09_RE, _LTX_VIDEO_09_REFUSAL),),
    ),
    # Wan2.2-TI2V-5B (diffusers >= 0.35): VAE temporal compression 4 gives frame counts 4k+1.
    VideoFamily(
        name = "wan2.2-ti2v-5b",
        profile = FamilyProfile(
            untiled_vae = UntiledVae(bytes_per_latent_pixel = 2.5 * 2**20, output_scale = (16, 4)),
            preview_factors = "wan22_5b",
            auto_static_skip = (
                ("wan-ai/wan2.2-ti2v-5b-diffusers", (("default", 2), ("max", 3), ("steps", 50))),
            ),
        ),
        comfy_flow_shift = 8.0,  # ComfyUI ModelSamplingSD3 8 (TI2V-5B template)
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        # fp8 / int8 share one repo, files Wan2.2-TI2V-5B-<SCHEME>.pt. Sizes from Hub metadata (2026-08-04).
        prequant_repos = (
            ("nvfp4", "unsloth/Wan2.2-TI2V-5B-NVFP4"),
            ("fp8", "unsloth/Wan2.2-TI2V-5B-FP8"),
            ("int8", "unsloth/Wan2.2-TI2V-5B-FP8"),
        ),
        prequant_filenames = (("nvfp4", "Wan2.2-TI2V-5B-NVFP4.pt"),),
        prequant_resident_gb_by_scheme = (("nvfp4", 2.9), ("fp8", 5.1), ("int8", 5.0)),
        # "wan2.2-5b"/"wan-ti2v" are the picker/GGUF short ids; "wan2.2-ti2v" catches the repo stem
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
        # expand_timesteps on the loaded config keeps the first-frame latent clean.
        i2v_pipeline_class = "WanImageToVideoPipeline",
    ),
    # Wan2.2-T2V-A14B dual-expert MoE: low-noise steps go to transformer_2 (cfg2_kwarg).
    VideoFamily(
        name = "wan2.2-t2v-a14b",
        profile = FamilyProfile(
            auto_prefer = AutoPrefer(
                floor = (10, 0),
                schemes = ("nvfp4", "int8", "fp8", "mxfp8"),
                backend = "flashinfer",
            ),
            untiled_vae = _WAN21_VAE_UNTILED,
        ),
        comfy_flow_shift = 5.0,  # ComfyUI ModelSamplingSD3 5 (T2V-A14B template)
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        # fp8 / int8: expert 2 is <name>-2.pt, not the derived transformer_2 name, so it needs its own row.
        prequant_repos = (
            ("nvfp4", "unsloth/Wan2.2-T2V-A14B-NVFP4"),
            ("fp8", "unsloth/Wan2.2-T2V-A14B-FP8"),
            ("int8", "unsloth/Wan2.2-T2V-A14B-FP8"),
        ),
        prequant_filenames = (
            ("nvfp4", "Wan2.2-T2V-A14B-NVFP4.pt"),
            ("nvfp4", "transformer_2", "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt"),
            ("fp8", "transformer_2", "Wan2.2-T2V-A14B-FP8-2.pt"),
            ("int8", "transformer_2", "Wan2.2-T2V-A14B-INT8-2.pt"),
        ),
        # BOTH experts: the plan subtracts one denoiser term and this family builds two.
        prequant_resident_gb_by_scheme = (("nvfp4", 16.2), ("fp8", 29.2), ("int8", 28.8)),
        # "wan2.2_t2v": ComfyUI's expert files (wan2.2_t2v_high_noise_14B_*.safetensors), paired at load.
        aliases = ("wan2.2-14b", "wan-t2v", "wan2.2-t2v", "wan2.2_t2v", "wan-t2v-a14b", "wan-a14b"),
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
        # no gguf_repo: community GGUFs split the experts; a gguf / single_file pick of either expert loads the pair
        # (video_moe_pair)
    ),
    # Wan2.1-T2V-1.3B: 480P only (upstream calls 720P unstable for 1.3B).
    VideoFamily(
        name = "wan2.1-t2v-1.3b",
        profile = FamilyProfile(untiled_vae = _WAN21_VAE_UNTILED),
        comfy_flow_shift = 8.0,  # reference workflow shift; replaces the repo's flow_shift 3.0
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        aliases = ("wan2.1-1.3b", "wan21-t2v-1.3b", "wan21-1.3b"),
        has_audio = False,
        default_steps = 30,
        default_guidance = 6.0,
        default_num_frames = 33,
        default_fps = 16,
        frame_step = 4,
        resolution_multiple = 16,
        resolution_presets = ((832, 480), (480, 832)),
        bf16_components_gb = (2.8, 11.4, 0.5),
        vae_force_fp32 = True,
        cudnn_benchmark = False,
    ),
    # Wan2.1-T2V-14B: same keys as one Wan2.2 A14B expert, run as a single denoiser.
    VideoFamily(
        name = "wan2.1-t2v-14b",
        profile = FamilyProfile(untiled_vae = _WAN21_VAE_UNTILED),
        comfy_flow_shift = 5.0,  # reference 14B workflow shift (the 1.3B uses 8); replaces the repo's flow_shift 3.0
        pipeline_class = "WanPipeline",
        transformer_class = "WanTransformer3DModel",
        base_repo = "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        aliases = ("wan2.1-14b", "wan21-t2v-14b", "wan21-14b"),
        has_audio = False,
        default_steps = 20,
        default_guidance = 6.0,
        default_num_frames = 33,
        default_fps = 16,
        frame_step = 4,
        resolution_multiple = 16,
        resolution_presets = ((832, 480), (1280, 720), (480, 832), (720, 1280)),
        bf16_components_gb = (28.6, 11.4, 0.5),
        vae_force_fp32 = True,
        cudnn_benchmark = False,
    ),
    # HunyuanVideo-1.5 (diffusers >= 0.39): 8.3B DiT, Qwen2.5-VL + ByT5 encoders. Three quirks: no guidance kwarg (CFG
    # on the ``guider``), no callback_on_step_end (generate() wraps scheduler.step), and no upstream model_index.json,
    # so only the community repacks load.
    VideoFamily(
        name = "hunyuanvideo-1.5",
        profile = FamilyProfile(
            int8_exclude_tokens = _HUNYUAN15_INT8_EXCLUDES,
            nvfp4_zero_row_tokens = _HUNYUAN15_NVFP4_ZERO_ROW_TOKENS,
            preview_factors = "hv15",
            auto_static_skip = (
                (
                    "hunyuanvideo-community/hunyuanvideo-1.5-diffusers-480p_t2v",
                    (("max", 2), ("steps", 50)),
                ),
            ),
        ),
        comfy_flow_shift = 7.0,  # ComfyUI model default and template shift 7
        pipeline_class = "HunyuanVideo15Pipeline",
        transformer_class = "HunyuanVideo15Transformer3DModel",
        base_repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        unsupported_checkpoints = ((_HV15_I2V_RE, _HV15_I2V_REFUSAL),),
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
        profile = FamilyProfile(
            int8_exclude_tokens = _HUNYUAN15_INT8_EXCLUDES,
            nvfp4_zero_row_tokens = _HUNYUAN15_NVFP4_ZERO_ROW_TOKENS,
            preview_factors = "hv15",
        ),
        comfy_flow_shift = 7.0,  # ComfyUI model default and template shift 7
        pipeline_class = "HunyuanVideo15Pipeline",
        transformer_class = "HunyuanVideo15Transformer3DModel",
        base_repo = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        unsupported_checkpoints = ((_HV15_I2V_RE, _HV15_I2V_REFUSAL),),
        # RTN, not GPTQ: the 720p transformer is separately trained; 480p corrections do not apply.
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


# Wan2.1 / FastWan / AnyFlow names no family can serve, each with its refusal. Kept to Wan2.1 / FastWan / AnyFlow
# spellings so no other family's name can match. FastWan, AnyFlow, I2V and conditioning markers count only in the last path segment (an owner or folder
# named "fastwan", "i2v" or "fun" is not such a checkpoint), and "fun" only right after the Wan2.1 token, where every Fun
# release puts it.
_WAN21_TOKEN = r"(?<![a-z0-9])wan-?2-?1(?![0-9])"
_UNSUPPORTED_VIDEO_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    (
        r"(?:^|[/\\])(?=[^/\\]*(?<![a-z0-9])fastwan(?![a-z]))[^/\\]*[/\\]*$",
        "This is a FastVideo FastWan checkpoint. FastWan is distilled for FastVideo's own few-step "
        "sampler and sparse attention (its repos declare WanDMDPipeline, which Diffusers does not "
        "ship), so Studio cannot run it faithfully. Load Wan-AI/Wan2.1-T2V-1.3B-Diffusers or "
        "Wan-AI/Wan2.1-T2V-14B-Diffusers instead.",
    ),
    (
        r"(?:^|[/\\])(?=[^/\\]*(?<![a-z0-9])anyflow(?![a-z]))[^/\\]*[/\\]*$",
        "This is an NVIDIA AnyFlow checkpoint. AnyFlow is a flow-map distillation of Wan2.1 whose "
        "transformer takes a second (delta) timestep and samples through AnyFlowPipeline with "
        "FlowMapEulerDiscreteScheduler, so the Wan text-to-video pipeline Studio loads cannot run it "
        "faithfully. Load Wan-AI/Wan2.1-T2V-1.3B-Diffusers or Wan-AI/Wan2.1-T2V-14B-Diffusers instead.",
    ),
    (
        r"(?:^|[/\\])(?=[^/\\]*"
        + _WAN21_TOKEN
        + r")(?=[^/\\]*(?<![a-z0-9])(?:i2v|flf2v)(?![a-z0-9]))"
        r"[^/\\]*[/\\]*$",
        "This is a Wan2.1 image-to-video checkpoint (I2V-14B 480P / 720P or FLF2V). It generates "
        "from a start frame through its own image-conditioned transformer and a CLIP vision "
        "encoder, which Studio does not load yet; image-to-video support is coming. Load a Wan2.1 "
        "text-to-video checkpoint (Wan-AI/Wan2.1-T2V-1.3B-Diffusers or "
        "Wan-AI/Wan2.1-T2V-14B-Diffusers) instead.",
    ),
    (
        r"(?:^|[/\\])(?=[^/\\]*" + _WAN21_TOKEN + r")(?=[^/\\]*(?:(?<![a-z0-9])"
        r"(?:vace|magref|phantom|multitalk|infinitetalk|controlnet|recammaster)|"
        + _WAN21_TOKEN
        + r"-fun)"
        r"(?![a-z0-9]))[^/\\]*[/\\]*$",
        "This is a Wan2.1 conditioning variant (VACE, Fun, MAGREF, Phantom, talking-head, "
        "ControlNet or ReCamMaster). Its transformer takes extra control inputs the text-to-video "
        "pipeline does not provide, so Studio cannot run it. Load a plain Wan2.1 text-to-video "
        "checkpoint instead.",
    ),
)


def unsupported_video_checkpoint_reason(
    *identifiers: Optional[str], wan21_checkpoint: Optional[str] = None
) -> Optional[str]:
    """Why a Wan 2.1 / FastWan / AnyFlow pick cannot load, or None. ``wan21_checkpoint``: the selected checkpoint of a
    pick that already resolved to a Wan2.1 family, whose last segment may omit the Wan2.1 token
    (``i2v-14b-480p-Q4_K_M.gguf``) and is judged as if it carried it; containers never are."""
    needles = [normalize_family_name(str(identifier)) for identifier in identifiers if identifier]
    if wan21_checkpoint:
        needle = normalize_family_name(str(wan21_checkpoint))
        needles.append(needle)
        name = re.split(r"[/\\]", needle.rstrip("/\\"))[-1]
        # A leading "fun" is too generic to mark a Fun release unless it names one (version, size, InP, Control); the
        # other markers are still judged.
        generic_fun = re.match(r"fun(?![a-z0-9])", name) and not re.match(
            r"fun-(?:v\d|1-3b|14b|inp|control)", name
        )
        needles.append(("wan2-1-t2v-" if generic_fun else "wan2-1-") + name)
    for needle in needles:
        for pattern, message in _UNSUPPORTED_VIDEO_CHECKPOINTS:
            if re.search(pattern, needle):
                return message
    return None


def _token_in_needle(token: str, needle: str) -> bool:
    """Whole path/name segment match, as in diffusion_families (a short alias like
    'ltx' must not match inside an unrelated word)."""
    return token_in_name(token, needle)


def detect_video_family(repo_id: str, override: Optional[str] = None) -> Optional[VideoFamily]:
    """Family from an override (exact name or alias), else the longest whole-segment name in the repo id."""
    if override:
        key = override.strip().lower()
        for fam in _FAMILIES:
            if key == fam.name or key in fam.aliases:
                return fam
        norm = normalize_family_name(key)
        for fam in _FAMILIES:
            if any(normalize_family_name(t) == norm for t in (fam.name, *fam.aliases)):
                return fam
        return None
    # I2V / VACE / FastWan names can carry a T2V token (Wan2.1_T2V_14B_FusionX_VACE): no T2V pipeline over those weights.
    if unsupported_video_checkpoint_reason(repo_id):
        return None
    needle = repo_id.lower()
    best: Optional[tuple[VideoFamily, int]] = None
    for fam in _FAMILIES:
        for token in (fam.name, *fam.aliases):
            if _token_in_needle(token, needle) and (best is None or token_length(token) > best[1]):
                best = (fam, token_length(token))
    if best is None:
        return None
    fam = best[0]
    # HV1.5 resolution tier is baked into the weights (480p vs 720p); route on the tier marker.
    if fam.name == "hunyuanvideo-1.5" and re.search(r"(?:^|[-_./\\])720p", needle):
        for candidate in _FAMILIES:
            if candidate.name == "hunyuanvideo-1.5-720p":
                return candidate
    return fam


def gguf_default_subdir_for_repo(repo_id: Optional[str]) -> Optional[str]:
    """The ``gguf_default_subdir`` of the family whose curated ``gguf_repo`` is *repo_id*, or None."""
    needle = (repo_id or "").strip().casefold()
    if not needle:
        return None
    for fam in _FAMILIES:
        if fam.gguf_repo and fam.gguf_default_subdir and fam.gguf_repo.casefold() == needle:
            return fam.gguf_default_subdir
    return None


def gguf_root_label_for_repo(repo_id: Optional[str]) -> Optional[str]:
    """The ``gguf_root_label`` of the family whose curated ``gguf_repo`` is *repo_id*, or None."""
    needle = (repo_id or "").strip().casefold()
    if not needle:
        return None
    for fam in _FAMILIES:
        if fam.gguf_repo and fam.gguf_root_label and fam.gguf_repo.casefold() == needle:
            return fam.gguf_root_label
    return None


def supported_video_family_names() -> tuple[str, ...]:
    return tuple(fam.name for fam in _FAMILIES)


def pipeline_available_video_families(*, device: Optional[str] = None) -> tuple[VideoFamily, ...]:
    from .diffusion_families import family_selectable
    return tuple(
        fam
        for fam in _FAMILIES
        if family_selectable(fam) and not (device == "mps" and fam.modular_workflow)
    )


def resolve_video_base_repo(
    fam: VideoFamily,
    base_repo: Optional[str],
    repo_id: Optional[str] = None,
) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback. An LTX-2.5 DiT picked from the
    split single-file release (or a local copy of it) defaults to that release, whose companion files sit beside it."""
    base = (base_repo or "").strip()
    if not base and fam.name == LTX25_FAMILY and repo_id:
        from .video_ltx25 import ltx25_default_base
        base = ltx25_default_base(repo_id) or ""
    return base or fam.base_repo


def _prequant_base_key(repo_id: Optional[str]) -> str:
    """Trimmed, lowercased lookup key; kept local so the video and image registries stay decoupled."""
    return (repo_id or "").strip().lower()


def video_family_prequant_repo(
    fam: VideoFamily,
    scheme: str,
    base_repo: Optional[str] = None,
) -> Optional[str]:
    """Hosted pre-quantized denoiser repo for a scheme, base-specific first; malformed rows are skipped."""
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
    """True when a task-specific artifact is named for this scheme, via the load's own resolver."""
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
    """True when a hosted denoiser covers (scheme, task); partition tasks need their own row."""
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
    """Schemes with a hosted denoiser; with task, narrowed so a refusal never suggests one that lacks it."""
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
    """Snap frames down to the k * step + offset lattice, since off-lattice counts break latent shapes."""
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
    """Reject unsupported shapes at the API instead of snapping; the frame lattice is always enforced."""
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


def video_family_accepts_first_frame(fam: VideoFamily) -> bool:
    return bool(
        getattr(fam, "supports_keyframes", False) or getattr(fam, "i2v_pipeline_class", None)
    )


def video_family_accepts_last_frame(fam: VideoFamily) -> bool:
    return bool(getattr(fam, "supports_keyframes", False)) and bool(
        getattr(fam, "accepts_last_frame", True)
    )


def video_family_uses_i2v_pipeline(fam: VideoFamily) -> bool:
    return bool(getattr(fam, "i2v_pipeline_class", None)) and not getattr(
        fam, "supports_keyframes", False
    )


def _identifier_blob(identifiers: tuple[Optional[str], ...]) -> list[str]:
    return [str(i).strip().lower() for i in identifiers if i and str(i).strip()]


def video_checkpoint_refusal(fam: VideoFamily, *identifiers: Optional[str]) -> Optional[str]:
    """Identifiers are joined into one line so an LTX-2.3 filename excuses an ``ltx-video`` folder; the last segment
    of the most specific identifier is also judged alone, so an LTX-2 folder cannot excuse an LTX-Video 0.9 file."""
    names = _identifier_blob(identifiers)
    joined = " | ".join(names)
    pick = os.path.basename(names[-1].replace("\\", "/").rstrip("/")) if names else ""
    for entry in getattr(fam, "unsupported_checkpoints", ()) or ():
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        pattern, message = entry
        try:
            if any(text and re.search(pattern, text) for text in (pick, joined)):
                return str(message)
        except re.error:
            continue
    return None


def video_i2v_call_kwargs(fam: VideoFamily, *identifiers: Optional[str]) -> dict[str, object]:
    names = _identifier_blob(identifiers)
    for entry in getattr(fam, "i2v_call_kwargs", ()) or ():
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            continue
        pattern, pairs = entry
        try:
            if any(re.search(pattern, name) for name in names):
                return {str(k): v for k, v in pairs}
        except (re.error, TypeError, ValueError):
            continue
    return {}


def snap_canvas_within_cap(
    width: float, height: float, multiple: int, max_pixels: int
) -> tuple[int, int]:
    """Snap a "match source" canvas to ``multiple`` without leaving ``max_pixels``: each side rounds to the nearest
    multiple, and when that rounding pushes the area past the cap a side rounds down instead (the largest area under
    the cap, then the aspect nearest the unsnapped one). The frontend's matchedCanvas mirrors this."""
    import math

    def down(value: float) -> int:
        return max(multiple, math.floor(value / multiple) * multiple)

    def near(value: float) -> int:
        return max(multiple, math.floor(value / multiple + 0.5) * multiple)

    w, h = near(width), near(height)
    if w * h <= max_pixels:
        return w, h
    target = math.log(width / height)
    fits = [
        (cw, ch)
        for cw, ch in ((down(width), h), (w, down(height)), (down(width), down(height)))
        if cw * ch <= max_pixels
    ]
    if not fits:
        return down(width), down(height)
    return max(fits, key = lambda wh: (wh[0] * wh[1], -abs(math.log(wh[0] / wh[1]) - target)))


def first_frame_canvas(
    fam: VideoFamily, aspect_width: float, aspect_height: float
) -> tuple[int, int]:
    """Nearest-aspect preset in log space; ties keep table order so the family default wins."""
    import math

    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(
            f"The first frame has no usable aspect ratio ({aspect_width}x{aspect_height})."
        )
    presets = tuple((int(w), int(h)) for w, h in fam.resolution_presets)
    if not presets:
        return snap_video_size(fam, int(aspect_width), int(aspect_height))
    target = math.log(aspect_width / aspect_height)
    return min(presets, key = lambda wh: abs(math.log(wh[0] / wh[1]) - target))


def validate_video_keyframe_conditioning(
    fam: VideoFamily,
    h3_task: Optional[str],
    *,
    has_keyframes: bool,
    has_last_frame: bool = False,
) -> None:
    """Raise ``ValueError`` when a checkpoint cannot take the keyframes a request supplies.

    Pure in the family and the MiniMax-H3 partition, which is what lets the generate route judge
    the checkpoint it is about to SWITCH TO by the same rules the backend applies to the loaded
    one. Without that, an auto-switch evicts a working pipeline and spends minutes loading a
    target for a request that was already known to be unservable.
    """
    if not (has_keyframes or has_last_frame):
        return
    from .video_minimax_h3 import H3_TASK_REFERENCES

    if not video_family_accepts_first_frame(fam):
        raise ValueError(
            f"{fam.name} generates from the prompt alone; it takes no first or last frame."
        )
    if has_last_frame and not video_family_accepts_last_frame(fam):
        raise ValueError(f"{fam.name} animates from a first frame only; it takes no last frame.")
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
    """Reject flow shifts the checkpoint cannot honour; passing engine refuses before any eviction."""
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
    """Reject references the checkpoint cannot take; the ref2va partition has no text-only denoiser."""
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
    # Wan2.1 14B FusionX merge (CausVid / AccVideo distilled): its card asks for CFG 1 and 8-10 steps.
    ("fusionx", 8, 1.0),
    ("ltx", 40, 4.0),
    # T2V-A14B before the generic Wan key.
    ("a14b", 20, 3.5),
    ("wan2.2-14b", 20, 3.5),
    # Before the generic Wan key; the 14B reference workflow runs 20 steps, the 1.3B 30.
    ("wan2.1-t2v-14b", 20, 6.0),
    ("wan2.1-14b", 20, 6.0),
    ("wan21-t2v-14b", 20, 6.0),
    ("wan21-14b", 20, 6.0),
    ("wan2.1", 30, 6.0),
    ("wan21", 30, 6.0),
    ("wan", 20, 5.0),
    ("hunyuanvideo", 20, 6.0),
)
# The Wan and FusionX keys count only in the model / file name, so an owner or folder spelling one cannot outrank the family the
# checkpoint itself names.
_NAME_SEGMENT_DEFAULT_KEYS = frozenset(
    k for k, _s, _g in _VIDEO_GENERATION_DEFAULTS if k.startswith(("wan", "fusionx"))
)


def default_video_generation_params(
    *identifiers: Optional[str],
    fallback: tuple[int, float] = (40, 4.0),
    family: Optional[str] = None,
) -> tuple[int, float]:
    """Default ``(steps, guidance)`` for a loaded video model; the first identifier
    naming a known variant wins, so a GGUF filename ('...distilled...Q4_K_M.gguf')
    beats the family base repo. ``fallback`` is used when no identifier names a variant --
    callers pass the resolved family's own default so a Wan model loaded from an opaque local
    path under an explicit family_override still gets 50/5.0, not the hardcoded LTX 40/4.0."""
    if family == LTX25_FAMILY:
        # The generic table's "ltx" key is the LTX-2 dev recipe, and 2.5's Diffusers pack names no variant at all.
        return LTX25_DEV_DEFAULTS if ltx25_is_dev(*identifiers) else LTX25_DISTILLED_DEFAULTS
    variant = video_generation_variant(*identifiers)
    for key, steps, guidance in _VIDEO_GENERATION_DEFAULTS:
        if key == variant:
            return steps, guidance
    return fallback


# LTX-2.5 defaults: distilled = the reference two-stage recipe's 8 stage-1 sigmas at CFG 1; dev = the diffusers pipeline's
# SFT recipe (30 steps, CFG 3).
LTX25_DISTILLED_DEFAULTS: tuple[int, float] = (8, 1.0)
LTX25_DEV_DEFAULTS: tuple[int, float] = (30, 3.0)
_LTX25_DEV = re.compile(r"(?<![a-z0-9])(?:dev|full|sft)(?![a-z0-9])|transformer_full")
_LTX25_DISTILLED = re.compile(r"(?<![a-z0-9])distill")


def ltx25_is_dev(*identifiers: Optional[str]) -> bool:
    """The first identifier naming a variant decides (selected file before repo); unnamed means distilled, which is
    what the Diffusers pack's default ``transformer/`` and the reference workflow ship."""
    for identifier in identifiers:
        leaf = _ltx_leaf(identifier).lower()
        if _LTX25_DISTILLED.search(leaf):
            return False
        if _LTX25_DEV.search(leaf):
            return True
    return False


def video_generation_variant(*identifiers: Optional[str]) -> Optional[str]:
    """The ``_VIDEO_GENERATION_DEFAULTS`` key the first identifier naming one matches, or None; the precedence
    ``default_video_generation_params`` uses, so every per-variant decision agrees with the defaults."""
    for identifier in identifiers:
        whole = (identifier or "").lower()
        name = whole.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        for key, _steps, _guidance in _VIDEO_GENERATION_DEFAULTS:
            needle = name if key in _NAME_SEGMENT_DEFAULT_KEYS else whole
            # Match the key as a name segment: reject a preceding ASCII letter so "swan-video" does not false-match
            # "wan".
            if re.search(r"(?<![a-z])" + re.escape(key), needle) or re.search(
                r"(?<![a-z])" + re.escape(normalize_family_name(key)), normalize_family_name(needle)
            ):
                return key
    return None
