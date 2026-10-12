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

# The request model's ceiling on num_frames, declared HERE so the shape gate and the bound cannot drift: the gate's
# refusal names the lattice point above the request, and suggesting one the request model would itself reject is a
# dead end. VideoGenerateRequest imports this for its `le`.
MAX_VIDEO_NUM_FRAMES = 1024

# Runtime->route contract: routes match these EXACTLY for a 409 instead of a 500.
VIDEO_NOT_LOADED_MSG = "No video model is loaded."
VIDEO_CANCELLED_MSG = "Video generation was cancelled."
VIDEO_GENERATION_BUSY_MSG = "A video generation is already in progress."
VIDEO_MODEL_CHANGED_MSG = "The requested video model changed before generation was reserved."

# One LTX version token: ltx / ltxv / ltx-video, optional separator, major 2, then ".N" / "_N" (minor), "N" ending the
# token ("ltx25"), or nothing ("ltx-2-19b" = 2.0). Anything else ("ltx-24fps", "ltx-transformer") is no name evidence,
# as is a bare "ltx23" / "ltx-23" (no release uses that spelling, so it cannot outrank the repo).
_LTX_VERSION = re.compile(
    r"(?<![a-z0-9])ltx(?:v|[-_ ]?video)?[-_ ]?2(?:[._]([0-9])(?![0-9])|([0-9])(?![0-9a-z])|(?![0-9a-z]|[._][0-9]))"
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
        "(Lightricks/LTX-2.3 or unsloth/LTX-2.3-GGUF) and LTX-2.5 (Lightricks/LTX-2.5-Diffusers)."
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
    if str(encoder_class).startswith("Gemma4") or any(
        isinstance(spec.get(k), list) and spec.get(k)[0]
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
    # Pipeline kwarg carrying the guidance value.
    cfg_kwarg: str = "guidance_scale"
    # The pipe attribute holding the denoiser (all current video families are DiTs).
    denoiser_attr: str = "transformer"
    # Extra lowercased substrings (besides ``name``) that map a repo id here.
    aliases: tuple[str, ...] = field(default_factory = tuple)
    # True when the pipeline returns synchronized audio (LTX-2): export muxes the track and size estimates count the
    # audio VAE + vocoder.
    has_audio: bool = False
    # Wan2.2-A14B dual-expert MoE: a second DiT handles the low-noise steps with its own guidance kwarg
    transformer2_class: Optional[str] = None
    is_moe: bool = False
    cfg2_kwarg: Optional[str] = None
    # HunyuanVideo-1.5 guidance: __call__ takes NO guidance kwarg; CFG lives on a ``guider`` whose scale is set per
    # request.
    guidance_via_guider: bool = False
    # Generation defaults + shape. A valid frame count is k*frame_step + frame_offset.
    default_steps: int = 40
    default_guidance: float = 4.0
    default_num_frames: int = 121
    default_fps: int = 24
    frame_step: int = 8
    # Offset in the temporal lattice. Existing pipelines use k*step+1; MiniMax-H3 uses 17k+5.
    frame_offset: int = 1
    min_num_frames: int = 1
    max_num_frames: Optional[int] = None
    snap_frames_up: bool = False
    # Width/height must be divisible by this (LTX-2's pipeline rejects non-/32).
    resolution_multiple: int = 32
    # (width, height) UI presets, landscape first; the first is the default.
    resolution_presets: tuple[tuple[int, int], ...] = ((768, 512),)
    # Clip lengths offered by the UI. Most families use short previews; H3 is trained for 5-15 seconds.
    duration_presets: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
    # Component bf16-RESIDENT sizes in decimal GB (denoiser(s), text encoder, VAE + audio): what sits on device after
    # the cast.
    bf16_components_gb: Optional[tuple[float, float, float]] = None
    # true when the DiT compiles cleanly with regional torch.compile (declares _repeated_blocks)
    supports_torch_compile: bool = True
    # Opt-in to the CUDA-graph capture of the denoiser forward (diffusion_cuda_graph.py): only for a family that runs
    # ONE forward per step with no CFG and no step cache, so a capture has one stable input tree to replay.
    supports_cuda_graph: bool = False
    # Opt-in to the capture when the denoiser is offloaded: the replay then records the onloads too (set from an A/B).
    offload_cuda_graph: bool = False
    # Status text when the family declines the capture (the measured reason), instead of "family opts out".
    cuda_graph_decline: Optional[str] = None
    # Video DiTs are bf16-native, so fp16 promotes to float32; defaults True.
    fp16_incompatible: bool = True
    # "native" = measured accurate in plain float16 on fp16-only cards; None = promote (unmeasured).
    fp16_guard: Optional[str] = None
    # Wan VAE decodes in float32 (bf16 causes banding / black frames), so the loader pins it back. Its size term is
    # already fp32.
    vae_force_fp32: bool = False
    # False holds cudnn.benchmark off: its per-process conv pick makes servers decode the same latents differently.
    cudnn_benchmark: bool = True
    # One config per reduction instead of a per-process benchmark that can change the render (diffusion_speed.
    # pin_reduction_configs). Off by default: it can keep a slower config (HunyuanVideo-1.5 ~2% per step).
    filter_reduction_configs: bool = False
    # Curated GGUF repo for the picker (the DiT as single-file GGUF quants).
    gguf_repo: Optional[str] = None
    # Hosted PRE-CAST text-encoder checkpoints as (scheme, component, repo_id); same semantics as
    # DiffusionFamily.te_prequant_repos.
    te_prequant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Hosted PRE-QUANTIZED DENOISER checkpoints as (scheme, repo_id). The DiT, NOT the text encoder that
    # te_prequant_repos above covers: the two are separate artifacts because a load can take one without the other. Same
    # semantics as DiffusionFamily.prequant_repos, so the shared diffusion_prequant resolver reads this table through
    # plain attribute access.
    prequant_repos: tuple[tuple[str, str], ...] = field(default_factory = tuple)
    # RESIDENT size of one of those hosted denoisers, in decimal GB, when the generic _QUANT_STEADY_FACTOR does not
    # describe it. MiniMax-H3's is both quantized AND structurally pruned (the curve-form adaLN, ~40% of the released
    # parameters), so 0.55 x 66.3 GB over-states it by 16 GB and a hard refusal turns away a load that fits. Measured
    # from Hub file metadata (2026-08-09): MiniMax-H3-FP8.pt 20,260,192,855 bytes, MiniMax-H3-INT8.pt 20,253,894,865.
    prequant_resident_gb: Optional[float] = None
    prequant_resident_gb_by_scheme: tuple[tuple[str, float], ...] = field(default_factory = tuple)
    # Per-variant overrides as (base_repo, scheme, repo_id), keyed on the LOWERCASED upstream base id. A pre-quantized
    # checkpoint is baked from ONE base's weights and the loader refuses it for any other base, so a variant that ships
    # its own denoiser needs its own entry; a variant without one falls through to prequant_repos and, if that
    # checkpoint was baked elsewhere, the loader's base_model_id check sends the load back to the dense path.
    prequant_variant_repos: tuple[tuple[str, str, str], ...] = field(default_factory = tuple)
    # Preferred checkpoint FILENAME for a scheme, as (scheme, filename), overriding the ``<Model>-<SCHEME>.pt`` name
    # ``prequant_repo_filename`` derives. The derived name stays on as the fallback, so a repo hosting BOTH an old and a
    # new artifact serves the new one to a build that asks for it by name and the old one to every build that does not.
    # That is what lets a rotated (v2) checkpoint ship without regressing an already-installed Unsloth, which would
    # otherwise refuse the v2 tag and fall all the way back to the dense download. A row may also be (scheme, task,
    # filename), naming the artifact for ONE task; it beats the task-agnostic row and, unlike it, gets no filename
    # fallback (see resolve_prequant_source).
    prequant_filenames: tuple[tuple[str, ...], ...] = field(default_factory = tuple)
    # Tasks whose denoiser is a DIFFERENT checkpoint partition from the one the task-agnostic ``prequant_filenames`` /
    # ``prequant_repos`` rows describe. Such a task is served ONLY by its own (scheme, task, filename) row: the
    # partitions share a base, a class, a config and a key set, so nothing downstream can tell them apart and an unnamed
    # artifact would load cleanly and generate from the wrong partition. Empty for every family with a single denoiser,
    # which is what makes this field free to ignore.
    prequant_partition_tasks: tuple[str, ...] = field(default_factory = tuple)
    # Modular Diffusers workflow to load instead of a conventional DiffusionPipeline. Its components are loaded without
    # pruning the workflow's routing blocks.
    modular_workflow: Optional[str] = None
    # Released video and audio sigma shifts, when configurable.
    default_flow_shift: Optional[float] = None
    # ComfyUI's static sigma shift (None = shipped); unlike default_flow_shift, no user control.
    comfy_flow_shift: Optional[float] = None
    default_audio_flow_shift: Optional[float] = None
    # First/last-frame conditioning: the request may carry keyframe images.
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
    # Guidance-distilled families expose neither CFG nor a negative prompt.
    supports_cfg: bool = True


# MiniMax-H3 at ComfyUI's template default (ResolutionSelector 16:9, 0.4 MP, multiple 32 -> 864x480). Both sides are
# multiples of 32 (VAE spatial compression 16 x patch 2), the only spatial rule the diffusers modular pipeline and
# stable-diffusion.cpp (align_image_size, multiple = vae_scale_factor 16 * down_factor 2) enforce, and 864/480 = 1.8 sits
# inside the trained 1:4..4:1 range. UNSLOTH_VIDEO_H3_480P=0 withdraws the pair.
def _h3_480p_presets() -> tuple[tuple[int, int], ...]:
    flag = os.environ.get("UNSLOTH_VIDEO_H3_480P", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return ()
    return ((864, 480), (480, 864))  # fastest


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
        # Model-card ratios use H3's canvas rule. Keep the legacy 1024 square and cheaper 16:9 tiers for
        # compatibility.
        resolution_presets = (
            (1344, 768),
            (1536, 672),
            (1024, 768),
            (1024, 1024),
            (768, 1024),
            (768, 1344),
            (960, 544),  # faster
            (544, 960),  # faster
        )
        + _h3_480p_presets(),
        duration_presets = (5.0, 10.0, 14.4),
        # Decimal GB resident estimates: transformer, Qwen3-VL conditioner, video+audio VAEs.
        bf16_components_gb = (66.3, 66.8, 11.1),
        # Regionally compilable. The DiT declares _repeated_blocks (MiniMaxH3TransformerBlock +
        # MiniMaxH3TokenRefinerBlock); every block sees (1, S, 5376) plus an (S,) index tensor, where S is the PACKED
        # length (18,870 video + 207 audio rows + the caption's text rows at 960x544x124). The caption moves S by ~2%
        # and S cannot change mid-denoise, so dynamic=True traces once and holds: measured 1.30x over eager with zero
        # recompiles across captions of 19 to 402 tokens. The loader engages this only when the denoiser is RESIDENT;
        # compiling inside a full CPU-offload rotation measured slower than eager.
        supports_torch_compile = True,
        # Measured and declined. The early 1.006x does not survive pairing: at 960x544x124, 30 steps, fp8, the graph
        # is 32.188 s p50 against 32.246 s eager over 5 renders per arm, 1.0018x, inside the 0.19% spread of a
        # never-armed control. The capture works (29 replays/render, 0 eager, 0 fallbacks, bit-identical); there is
        # nothing to win. The GPU is 92.7% busy either way, so removing ALL 25.3 s of host launch time (0.873 s/step
        # to 0.1 ms) moved the wall 0.058 s: the host was never the wall. At 4 cores instead of 192 that 25.3 s is
        # unchanged, so no weaker host turns H3 launch-bound. The cost is real: 3.93 GB held plus 4.03 GB at the
        # capture peak, on a family already needing 87.5 GB, and 31.1 s on the first render (break-even ~530 videos).
        # This is the CHEAPEST grid H3 ships, so the larger presets can only be more GPU-bound.
        supports_cuda_graph = False,
        # Streamed, H3's prequant groups onload through diffusers' stream path, which waits on the host per group: a
        # capture cannot hold them (arm_after_placement refuses it by name when the family is forced).
        cuda_graph_decline = (
            "measured GPU-bound: 1.0018x resident for 3.93 GB held; streamed, its group onloads wait on the host "
            "(27 per step at 16 GB), which a graph cannot record"
        ),
        gguf_repo = "unsloth/MiniMax-H3-GGUF",
        # Hosted pre-quantized FL2VA denoisers. The modular workflow builds each component through its own
        # from_pretrained, so there is no dense module to quantise in place: these are the ONLY way to run the 66.3 GB
        # transformer quantized, and seeding one also stops that download. Both schemes live in ONE repo, at the root,
        # named <Model>-<SCHEME>.pt, the layout every image-side prequant repo uses and the one prequant_repo_filename
        # builds without help.
        prequant_repos = (("int8", "unsloth/MiniMax-H3-FP8"), ("fp8", "unsloth/MiniMax-H3-FP8")),
        # The INT8 denoiser is ConvRot-rotated (see diffusion_convrot): its weights live in a Hadamard-rotated basis and
        # are wrong unless the loader rotates the activations to match, so it carries the v2 format tag an Unsloth
        # predating that code refuses. Shipping it under its own name rather than over MiniMax-H3-INT8.pt keeps both
        # true at once: this build gets the rotated artifact, and an older install still resolves the plain one instead
        # of refusing the v2 tag and falling back to the 66.3 GB dense download. The reference (ref2va) denoiser is a
        # SECOND partition in the same base repo, so it gets its own artifact under its own name for both schemes. The
        # two partitions are otherwise indistinguishable to the loader -- same class, same config, same 635 keys, same
        # base_model_id -- so the task, not any later check, is the only thing that keeps a reference load off the
        # keyframe weights. Keyframe (fl2va, which also covers text-only) keeps resolving exactly what it resolved
        # before: the rotated INT8 by name, FP8 by the derived MiniMax-H3-FP8.pt.
        prequant_filenames = (
            ("int8", "MiniMax-H3-INT8-ConvRot.pt"),
            ("int8", "ref2va", "MiniMax-H3-Ref2VA-INT8-ConvRot.pt"),
            ("fp8", "ref2va", "MiniMax-H3-Ref2VA-FP8.pt"),
        ),
        # Keeps the two partitions honest: without a ref2va row above, a ref2va prequant pick is refused rather than
        # served the keyframe checkpoint. Equal to H3_TASK_REFERENCES; the literal avoids importing the H3 helper
        # module into the registry.
        prequant_partition_tasks = ("ref2va",),
        # Both schemes are ~20.3 GB resident against the 66.3 GB dense denoiser; see the field.
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
        # transformer 37.8 bf16; Gemma3-12B TE ~24.4 RESIDENT (the hub stores it fp32 but the pipeline loads bf16); VAE
        # 2.4 + connectors 2.9 + audio 0.2. The old 50.4 figure double-counted the fp32 store.
        bf16_components_gb = (37.8, 24.4, 5.5),
        gguf_repo = "unsloth/LTX-2.3-GGUF",
        # pre-cast Gemma3-12B TE (fp32 ~49 GB on the hub, pre-cast ~13.2 GB): the biggest download win
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
    # Wan2.2-TI2V-5B (diffusers >= 0.35, verified on 0.39): ~5B single-stream DiT (UMT5 encoder), no audio. Its VAE's
    # temporal compression 4 gives valid frame counts 4k+1. Defaults 20 steps / CFG 5 (ComfyUI's template).
    VideoFamily(
        name = "wan2.2-ti2v-5b",
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
        # TI2V-5B is a 720P-only checkpoint: upstream SUPPORTED_SIZES is exactly ('704*1280', '1280*704') and
        # generate.py asserts membership, so nothing else is offered. First is the default the loader plans against.
        resolution_presets = ((1280, 704), (704, 1280)),
        # bf16-RESIDENT. transformer + VAE ship FP32 on disk (20.0 GB = 5B x 4), so bf16 transformer ~10.0; UMT5 TE bf16
        # (11.4); VAE fp32 (2.8).
        bf16_components_gb = (10.0, 11.4, 2.8),
        vae_force_fp32 = True,
        cudnn_benchmark = False,
        # Streamed: faster than eager with bit-identical latents; two replays per step (CFG), one graph per shape.
        offload_cuda_graph = True,
        # UMT5 keeps its overflowing `wo` in fp32 itself; the VAE stays fp32 (vae_force_fp32).
        fp16_guard = "native",
        # Byte-identical mirror of QuantStack/Wan2.2-TI2V-5B-GGUF (13 quants + companion VAE).
        gguf_repo = "unsloth/Wan2.2-TI2V-5B-GGUF",
        # expand_timesteps on the loaded config keeps the first-frame latent clean.
        i2v_pipeline_class = "WanImageToVideoPipeline",
    ),
    # Wan2.2-T2V-A14B (diffusers >= 0.35, verified on 0.39): the dual-expert MoE. Both transformers are
    # WanTransformer3DModel with boundary_ratio 0.875; high-noise steps route through transformer, low-noise through
    # transformer_2, so cfg2_kwarg is threaded only here.
    VideoFamily(
        name = "wan2.2-t2v-a14b",
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
        # is_moe drives the dual-DiT optimisation layers; cfg2_kwarg names the pipeline kwarg for transformer_2's
        # guidance.
        transformer2_class = "WanTransformer3DModel",
        is_moe = True,
        cfg2_kwarg = "guidance_scale_2",
        default_steps = 20,
        default_guidance = 3.5,
        # 81 frames at 16 fps ~5s (81 = 4*20 + 1), the A14B card's default clip.
        default_num_frames = 81,
        default_fps = 16,  # A14B runs at 16 fps (vs TI2V-5B's 24)
        frame_step = 4,
        resolution_multiple = 16,
        # 480p + 720p presets. A14B's VAE is 8x so multiple 16 renders 720 exactly (TI2V-5B's 16x VAE floors it to 704).
        resolution_presets = ((1280, 720), (832, 480), (480, 832), (720, 1280)),
        # bf16-RESIDENT. Each expert ships FP32 (57.15 GB = 14.3B x 4), so ~28.6 bf16 each and ~57.2 for BOTH, not the
        # 114.3 fp32 sum. UMT5 TE bf16 (11.4); VAE fp32 (0.5).
        bf16_components_gb = (57.2, 11.4, 0.5),
        vae_force_fp32 = True,
        # same VAE as TI2V-5B
        cudnn_benchmark = False,
        # no gguf_repo: community GGUFs split the experts; a gguf / single_file pick of either expert loads the pair
        # (video_moe_pair)
    ),
    # Wan2.1-T2V-1.3B: 480P only (upstream calls 720P unstable for 1.3B).
    VideoFamily(
        name = "wan2.1-t2v-1.3b",
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
        # HV15 VAE compresses 16x spatial / 4x temporal, patch-1, so sizes snap /16, frames 4k+1
        frame_step = 4,
        resolution_multiple = 16,
        # 480p-class presets (the base is the 480p variant): landscape, vertical, square. Every entry is a real bucket
        # of this tier (generate_crop_size_list(base_size=640)); 624x624 was not one, so the square option snapped
        # off-tier at 1521 tokens against the trained 1600.
        resolution_presets = ((832, 480), (480, 832), (640, 640)),
        # DiT fp32 on disk (32.0 to 16.6 bf16); VAE 4.7 to 2.4; Qwen2.5-VL TE bf16 14.0 + ByT5 0.8
        bf16_components_gb = (16.6, 14.8, 2.4),
        fp16_guard = "native",
        cudnn_benchmark = False,
        # Needs the capture-safe forward (diffusion_capture_safe) and the memoized trim (diffusion_attention).
        offload_cuda_graph = True,
    ),
    # The 720p t2v repack: same architecture and footprint as the 480p entry, only the trained resolution differs. Its
    # own family so a 720p load defaults to 720p sizes; the full-path alias outranks the generic token.
    VideoFamily(
        name = "hunyuanvideo-1.5-720p",
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
        # 720p-class presets: landscape, vertical, square (all /16).
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
    # HunyuanVideo-1.5's resolution tier is baked into the weights: the 480p and 720p repacks ship transformer
    # target_size 640 vs 960 and scheduler shift 5.0 vs 9.0, and their bucket lists are disjoint. Only the literal
    # 720p_t2v path was aliased, so 720p_i2v and every GGUF repack fell through to the generic "hunyuanvideo-1.5" token
    # and inherited the 480p base repo -- which is also where the VAE and text encoder come from. Re-route on the tier
    # marker instead.
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


def resolve_video_base_repo(
    fam: VideoFamily, base_repo: Optional[str], repo_id: Optional[str] = None
) -> str:
    """The companion diffusers repo: caller-supplied if given, else the family fallback. An LTX-2.5 DiT picked from the
    split single-file release (or a local copy of it) defaults to that release, whose companion files sit beside it."""
    base = (base_repo or "").strip()
    if not base and fam.name == LTX25_FAMILY and repo_id:
        from .video_ltx25 import ltx25_default_base

        base = ltx25_default_base(repo_id) or ""
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
    # Normalised to int pairs so membership holds however a family spelled its presets (the status payload hands them
    # out as lists, and a round-trip through it must not silently stop matching).
    presets = tuple((int(w), int(h)) for w, h in fam.resolution_presets)
    # No declared presets: no table to judge a SIZE against, so leave that to the snap (unusual/custom families). The
    # frame check below still runs either way -- frame_step is always declared.
    if presets and (width is not None or height is not None):
        # Resolve a half-specified request against the default preset first, as generate() does, then judge the pair.
        # Keyed on None, where generate() keys on falsiness: a 0 is judged as 0 here rather than replaced by the
        # default. The route cannot deliver one (the request model bounds it at 32), and refusing an explicit 0 beats
        # silently rendering something else, so the two agree on every value that can arrive.
        want_w = presets[0][0] if width is None else int(width)
        want_h = presets[0][1] if height is None else int(height)
        if (want_w, want_h) not in presets:
            raise VideoShapeError(
                f"{want_w}x{want_h} is not a supported resolution for {fam.name}. "
                f"Supported resolutions: {format_video_resolution_presets(fam)}."
            )
    if num_frames is not None:
        # The lattice is k * frame_step + frame_offset, NOT a hardcoded k * step + 1: most families do sit at offset 1,
        # but MiniMax-H3 is 17k + 5, and judging it against 17k + 1 would refuse every count its own duration select
        # offers, starting with its default of 124. Read the two fields snap_num_frames reads, so the gate and the snap
        # can never disagree about what is valid.
        step = max(1, fam.frame_step)
        offset = max(1, fam.frame_offset)
        count = int(num_frames)
        # The window the family declares it was trained for. Hoisted out of the lattice branch because it is also
        # enforced below: it used to exist only to WORD the lattice error ("supported counts run from 124 to 345") while
        # a request outside it was accepted and silently snapped, so num_frames=5 rendered 124 frames and num_frames=872
        # rendered 345.
        ceiling = MAX_VIDEO_NUM_FRAMES
        if fam.max_num_frames is not None:
            ceiling = min(ceiling, int(fam.max_num_frames))
        floor = max(offset, int(fam.min_num_frames))
        if count < offset or (count - offset) % step != 0:
            # The two lattice points straddling the request say more than a prefix of the lattice would, and stay short.
            # Computed from the lattice rather than via snap_num_frames, which floors for some families and CEILS for
            # others (snap_frames_up) and so cannot be relied on for "below".
            below = offset + max(0, (count - offset) // step) * step
            above = below + step
            # Only name a point the caller could actually load: past the request model's own `le`, or outside this
            # family's declared range, it answers with a second, differently-shaped 422.
            loadable = [n for n in (below, above) if floor <= n <= ceiling]
            if len(loadable) == 2:
                nearest = f"the nearest supported counts are {loadable[0]} and {loadable[1]}"
            elif len(loadable) == 1:
                nearest = f"the nearest supported count is {loadable[0]}"
            else:
                # The request is outside the family's whole range; point at the range instead.
                nearest = f"supported counts run from {floor} to {ceiling}"
            raise VideoShapeError(
                f"{count} is not a supported frame count for {fam.name}. Its VAE compresses time by "
                f"{step}, so a frame count must be k * {step} + {offset}; {nearest} "
                f"(the default is {fam.default_num_frames})."
            )
        # On the lattice but outside the trained window. The request model bounds num_frames at 1..1024, so a count well
        # under the floor or well over the ceiling arrives here and used to be snapped in silence, which on the native
        # path is a 25x compute surprise.
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


# Default (steps, guidance) per checkpoint variant, matched by substring (picked id then base repo), most specific
# first.
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
