# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure command builder for the ``sd-cli`` (stable-diffusion.cpp) native engine.

No torch / diffusers / subprocess here: everything is a pure function of its
arguments so the whole argv construction can be unit-tested without a GPU, the
binary, or any model files. The engine (``sd_cpp_engine.py``) calls
``build_sd_cpp_command`` and runs the result.

stable-diffusion.cpp consumes the *same* split GGUF assets Studio already
curates for the diffusers path: a transformer (``--diffusion-model``), a VAE
(``--vae``), and one or more text encoders, wired to the family-specific flag
(Z-Image's Qwen3 -> ``--llm``, Qwen-Image's Qwen2-VL -> ``--qwen2vl``,
FLUX.1's CLIP-L + T5 -> ``--clip_l`` / ``--t5xxl``). The memory policy the
diffusers planner picks (``none`` / ``group`` / ``model`` / ``sequential``)
maps to sd.cpp's own offload flags here, so one user-facing knob drives both
engines.

Ref (flag names): stable-diffusion.cpp ``examples/cli`` and ``docs/z_image.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
)

# Per-family text-encoder wiring: ordered (sd-cli flag, human label) pairs in the
# order the text encoders are supplied. Z-Image uses a single Qwen3 LLM; FLUX.1
# pairs CLIP-L with a T5; Qwen-Image uses Qwen2-VL; FLUX.2-klein uses a Qwen3 LLM.
# Keyed by ``DiffusionFamily.name`` so the engine stays data-driven and the
# family registry (diffusion_families.py) need not import sd.cpp specifics.
_TE_FLAGS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "z-image": ("--llm",),
    "flux.2-klein": ("--llm",),
    "qwen-image": ("--qwen2vl",),
    "flux.1": ("--clip_l", "--t5xxl"),
}


def text_encoder_flags_for_family(family_name: str) -> tuple[str, ...]:
    """sd-cli text-encoder flags for ``family_name`` (empty if unknown)."""
    return _TE_FLAGS_BY_FAMILY.get(family_name, ())


# sd-cli's image-generation mode. Current stable-diffusion.cpp uses ``img_gen``
# (the older spelling was ``txt2img``); img2img is the same mode with --init-img,
# so this one token covers both text- and image-conditioned generation.
DEFAULT_MODE = "img_gen"


@dataclass(frozen = True)
class SdCppModelFiles:
    """Resolved on-disk paths for one diffusion checkpoint's components.

    ``diffusion_model`` (the transformer) is required; the rest are optional so a
    single-file checkpoint that bundles the VAE / encoders still builds a valid
    command. Any field left None is simply omitted from the argv.
    """

    diffusion_model: str
    vae: Optional[str] = None
    clip_l: Optional[str] = None
    clip_g: Optional[str] = None
    t5xxl: Optional[str] = None
    llm: Optional[str] = None
    qwen2vl: Optional[str] = None


@dataclass(frozen = True)
class SdCppGenParams:
    """Generation parameters, mapped 1:1 onto sd-cli's sampling flags.

    The image-conditioning fields cover the img_gen variants: ``init_img`` +
    ``strength`` make it img2img, adding ``mask`` makes it inpaint, and
    ``ref_images`` drives FLUX Kontext / Qwen-Image-Edit style editing. ``lora_dir``
    points sd-cli at a LoRA directory; the LoRAs themselves are selected with
    ``<lora:name:weight>`` tags inside ``prompt`` (sd.cpp's own syntax).
    """

    prompt: str
    negative_prompt: Optional[str] = None
    # None = "unset": an image-conditioned run (img2img/inpaint/edit) then lets
    # sd.cpp derive the size from the input image instead of forcing a resize; a
    # plain txt2img run with unset dims falls back to 1024x1024 (see the builder).
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    sampling_method: Optional[str] = None
    batch_count: int = 1
    # image-to-image / inpaint / edit
    init_img: Optional[str] = None
    strength: Optional[float] = None
    mask: Optional[str] = None
    ref_images: tuple[str, ...] = ()
    # LoRA
    lora_dir: Optional[str] = None
    lora_apply_mode: Optional[str] = None


@dataclass(frozen = True)
class SdCppUpscaleParams:
    """Inputs for sd-cli's ESRGAN upscale mode (a separate run mode)."""

    input_image: str
    upscale_model: str
    repeats: int = 1
    tile_size: Optional[int] = None


def offload_flags(
    policy: str,
    *,
    vae_tiling: bool = False,
    diffusion_fa: bool = False,
) -> list[str]:
    """Translate a diffusers memory *policy* into sd-cli offload flags.

    - ``none``: weights stay resident, no offload flags.
    - ``group`` (balanced): stream the model through VRAM (``--offload-to-cpu``);
      flash attention shrinks the activation peak.
    - ``model`` / ``sequential`` (low-VRAM): offload everything, also push the
      CLIP/VAE to CPU and tile the VAE so the decode fits a tiny budget.

    ``vae_tiling`` / ``diffusion_fa`` force those flags on regardless of policy.
    """
    flags: list[str] = []
    fa = diffusion_fa
    tile = vae_tiling
    if policy in (OFFLOAD_GROUP, OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL):
        flags.append("--offload-to-cpu")
        fa = True
    if policy in (OFFLOAD_MODEL, OFFLOAD_SEQUENTIAL):
        flags.append("--clip-on-cpu")
        flags.append("--vae-on-cpu")
        tile = True
    if fa:
        flags.append("--diffusion-fa")
    if tile:
        flags.append("--vae-tiling")
    # Stable order, de-duplicated (a forced flag can coincide with a policy one).
    seen: set[str] = set()
    out: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def build_sd_cpp_command(
    binary: str,
    files: SdCppModelFiles,
    params: SdCppGenParams,
    *,
    output_path: str,
    offload: Optional[list[str]] = None,
    threads: Optional[int] = None,
    verbose: bool = False,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the full ``sd-cli`` argv for one text-to-image generation.

    Required model/IO flags come first (so a failure points at the obvious
    place), then sampling params (only those that are set), then offload flags,
    then any caller ``extra_args`` last so sd.cpp's last-wins parser lets a power
    user override anything Studio set.
    """
    if not files.diffusion_model:
        raise ValueError("diffusion_model path is required")
    if not str(params.prompt).strip():
        raise ValueError("prompt is required")

    cmd: list[str] = [binary, "--mode", DEFAULT_MODE, "--diffusion-model", files.diffusion_model]
    for flag, value in (
        ("--vae", files.vae),
        ("--clip_l", files.clip_l),
        ("--clip_g", files.clip_g),
        ("--t5xxl", files.t5xxl),
        ("--llm", files.llm),
        ("--qwen2vl", files.qwen2vl),
    ):
        if value:
            cmd += [flag, value]

    cmd += ["--prompt", params.prompt]
    if params.negative_prompt:
        cmd += ["--negative-prompt", params.negative_prompt]
    # img2img / inpaint / edit conditioning (img_gen mode with an input image).
    if params.init_img:
        cmd += ["--init-img", params.init_img]
    if params.strength is not None:
        cmd += ["--strength", _fmt_float(params.strength)]
    if params.mask:
        cmd += ["--mask", params.mask]
    for ref in params.ref_images:
        cmd += ["--ref-image", ref]
    # LoRA: the directory to scan; individual LoRAs are <lora:name:w> tags in prompt.
    if params.lora_dir:
        cmd += ["--lora-model-dir", params.lora_dir]
    if params.lora_apply_mode:
        cmd += ["--lora-apply-mode", params.lora_apply_mode]
    # Emit explicit dims when given. For an image-conditioned run (img2img /
    # inpaint / edit) that leaves them unset, omit the flags so sd.cpp derives the
    # size from the input image (set_width_and_height_if_unset) rather than forcing
    # a 1024x1024 resize/crop of the source. A plain txt2img run with unset dims
    # keeps the prior 1024 default.
    if params.width is not None or params.height is not None:
        w = int(params.width) if params.width is not None else 1024
        h = int(params.height) if params.height is not None else 1024
        cmd += ["--width", str(w), "--height", str(h)]
    elif not (params.init_img or params.ref_images):
        cmd += ["--width", "1024", "--height", "1024"]
    if params.steps is not None:
        cmd += ["--steps", str(int(params.steps))]
    if params.cfg_scale is not None:
        cmd += ["--cfg-scale", _fmt_float(params.cfg_scale)]
    if params.guidance is not None:
        cmd += ["--guidance", _fmt_float(params.guidance)]
    if params.sampling_method:
        cmd += ["--sampling-method", str(params.sampling_method)]
    if params.seed is not None:
        cmd += ["--seed", str(int(params.seed))]
    if params.batch_count and params.batch_count != 1:
        cmd += ["--batch-count", str(int(params.batch_count))]

    cmd += ["--output", output_path]
    if threads is not None:
        cmd += ["--threads", str(int(threads))]
    if offload:
        cmd += list(offload)
    if verbose:
        cmd += ["-v"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def build_sd_cpp_upscale_command(
    binary: str,
    params: SdCppUpscaleParams,
    *,
    output_path: str,
    verbose: bool = False,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the ``sd-cli --mode upscale`` argv (ESRGAN super-resolution).

    Upscale is a distinct run mode: it takes an input image and an ESRGAN model,
    no prompt or text encoders. ``repeats`` runs the upscaler N times (each pass
    is a fixed scale factor for the model).
    """
    if not params.input_image:
        raise ValueError("input_image is required for upscale")
    if not params.upscale_model:
        raise ValueError("upscale_model is required for upscale")
    # A truthiness guard below would silently swallow repeats=0 and fall back to
    # sd-cli's default of one pass, turning an explicit no-op into a real upscale.
    # Reject it (and negatives) so the caller's intent isn't quietly changed.
    if params.repeats < 1:
        raise ValueError("repeats must be >= 1 for upscale")
    cmd: list[str] = [
        binary,
        "--mode",
        "upscale",
        "--init-img",
        params.input_image,
        "--upscale-model",
        params.upscale_model,
    ]
    if params.repeats != 1:
        cmd += ["--upscale-repeats", str(int(params.repeats))]
    if params.tile_size is not None:
        cmd += ["--upscale-tile-size", str(int(params.tile_size))]
    cmd += ["--output", output_path]
    if verbose:
        cmd += ["-v"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def _fmt_float(value: float) -> str:
    """Compact float -> str: drop a trailing ``.0`` so ``1.0`` -> ``1`` (sd-cli
    accepts both, but the tidy form keeps logged commands readable)."""
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)
