# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure command builder for the ``sd-cli`` (stable-diffusion.cpp) native engine.

No torch / diffusers / subprocess: a pure function of its arguments, so argv construction
unit-tests without a GPU, binary, or model files. sd.cpp consumes the SAME split GGUF assets as
the diffusers path -- a transformer (``--diffusion-model``), a VAE (``--vae``), and text encoders
wired to the family flag (Z-Image Qwen3 -> ``--llm``, Qwen-Image Qwen2-VL -> ``--qwen2vl``, FLUX.1
-> ``--clip_l`` / ``--t5xxl``). The diffusers memory policy maps to sd.cpp's offload flags here, so
one knob drives both engines. Ref: sd.cpp ``examples/cli`` and ``docs/z_image.md``.
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

# Per-family text-encoder flags, in supply order. Keyed by ``DiffusionFamily.name`` so the engine
# stays data-driven and the family registry need not import sd.cpp specifics.
_TE_FLAGS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "z-image": ("--llm",),
    "flux.2-klein": ("--llm",),
    "flux.2-dev": ("--llm",),
    "qwen-image": ("--qwen2vl",),
    "flux.1": ("--clip_l", "--t5xxl"),
}


def text_encoder_flags_for_family(family_name: str) -> tuple[str, ...]:
    """sd-cli text-encoder flags for ``family_name`` (empty if unknown)."""
    return _TE_FLAGS_BY_FAMILY.get(family_name, ())


# sd-cli's image-gen mode (``img_gen``; older ``txt2img``). img2img is the same mode with
# --init-img, so this one token covers both text- and image-conditioned generation.
DEFAULT_MODE = "img_gen"


@dataclass(frozen = True)
class SdCppModelFiles:
    """Resolved on-disk paths for one checkpoint's components. ``diffusion_model`` is required; the
    rest are optional (a bundled single-file checkpoint still builds). None fields are omitted."""

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

    Image-conditioning fields cover the img_gen variants: ``init_img`` + ``strength`` = img2img,
    + ``mask`` = inpaint, ``ref_images`` = FLUX Kontext / Qwen-Image-Edit editing. ``lora_dir`` is
    the LoRA directory; individual LoRAs are ``<lora:name:weight>`` tags in ``prompt``.
    """

    prompt: str
    negative_prompt: Optional[str] = None
    # None = unset: an image-conditioned run lets sd.cpp derive the size from the input; a plain
    # txt2img run with unset dims falls back to 1024x1024 (see the builder).
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


# Native (sd.cpp) speed profiles (engine-side analogue of diffusion_speed). off: nothing. default:
# --diffusion-fa (flash attention) + --diffusion-conv-direct (numerically exact). On the CPU tier
# this serves, direct conv measured z-image Q8_0 sampling 56.1 -> 51.3s (~9%) with decode/RSS
# unchanged, so it's in the default profile. max keeps it (profiles are a superset chain).
NATIVE_SPEED_OFF = "off"
NATIVE_SPEED_DEFAULT = "default"
NATIVE_SPEED_MAX = "max"
NATIVE_SPEED_MODES = (NATIVE_SPEED_OFF, NATIVE_SPEED_DEFAULT, NATIVE_SPEED_MAX)


def native_speed_flags(speed_mode: Optional[str]) -> list[str]:
    """sd-cli speed flags for a native speed mode (empty for off / None). Separate from offload:
    ``--diffusion-fa`` is a win in its own right. De-duplicated against offload at the call site."""
    mode = (speed_mode or NATIVE_SPEED_OFF).strip().lower()
    if mode in ("", NATIVE_SPEED_OFF):
        return []
    if mode in (NATIVE_SPEED_DEFAULT, NATIVE_SPEED_MAX):
        return ["--diffusion-fa", "--diffusion-conv-direct"]
    raise ValueError(f"native speed_mode must be one of {NATIVE_SPEED_MODES}, got '{speed_mode}'")


def offload_flags(
    policy: str,
    *,
    vae_tiling: bool = False,
    diffusion_fa: bool = False,
) -> list[str]:
    """Translate a diffusers memory policy into sd-cli offload flags.

    ``none``: resident, no flags. ``group``: stream the model (``--offload-to-cpu``) + flash
    attention. ``model`` / ``sequential``: offload everything, also CLIP/VAE to CPU + VAE tiling.
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

    Required model/IO flags first, then set sampling params, then offload flags, then caller
    ``extra_args`` last (sd.cpp's parser is last-wins, so a power user can override anything).
    """
    if not files.diffusion_model:
        raise ValueError("diffusion_model path is required")
    # ``(prompt or "")`` so a None prompt is rejected here, not passed as literal "None".
    if not (params.prompt or "").strip():
        raise ValueError("prompt is required")
    # sd-cli inpaint needs the source image: a --mask with no --init-img is invalid, so reject it
    # here rather than fail deep in sd-cli.
    if params.mask and not params.init_img:
        raise ValueError("init_img is required when mask is set (inpaint needs a source image)")

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
    # img2img / inpaint / edit conditioning.
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
    # Emit explicit dims when given. An image-conditioned run that leaves them unset omits the
    # flags so sd.cpp derives the size from the input; a plain txt2img keeps the 1024 default.
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
        # sd-cli names extra batch images itself (output_2.png, ...) but the runner collects only
        # --output, so a CLI batch drops all but the first. Batches use the sdcpp server API.
        raise ValueError(
            "sd-cli runs are single-image; use the sdcpp server API for batch generation."
        )

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
    """Build the ``sd-cli --mode upscale`` argv (ESRGAN). A distinct run mode: input image + ESRGAN
    model, no prompt or text encoders. ``repeats`` runs the upscaler N times (each a fixed scale)."""
    if not params.input_image:
        raise ValueError("input_image is required for upscale")
    if not params.upscale_model:
        raise ValueError("upscale_model is required for upscale")
    # Reject repeats < 1 explicitly: a truthiness guard would swallow repeats=0 into sd-cli's
    # one-pass default, quietly changing the caller's intent.
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


def build_sd_cpp_server_command(
    binary: str,
    files: SdCppModelFiles,
    *,
    host: str,
    port: int,
    vae_format: Optional[str] = None,
    offload: Optional[list[str]] = None,
    native_speed: Optional[str] = None,
    threads: Optional[int] = None,
    scratch_dir: Optional[str] = None,
    verbose: bool = False,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """Build the ``sd-server`` argv: model + hardware/server flags only.

    sd-server loads the model once at spawn from the SAME flags as sd-cli, plus ``--listen-ip`` /
    ``--listen-port``. Per-generation params go in each ``/sdcpp/v1/img_gen`` request, so one
    resident process serves many generations without reloading. ``offload`` / ``native_speed`` map
    to the same sd.cpp flags. ``scratch_dir`` is pointed at by the LoRA / upscaler / embeddings dir
    flags (sd-server iterates those and fails on a missing dir, so give it a real empty one).
    ``extra_args`` last (last-wins).
    """
    if not files.diffusion_model:
        raise ValueError("diffusion_model path is required")

    cmd: list[str] = [binary, "--diffusion-model", files.diffusion_model]
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
    if vae_format:
        cmd += ["--vae-format", vae_format]
    cmd += ["--listen-ip", str(host), "--listen-port", str(int(port))]
    if scratch_dir:
        cmd += [
            "--lora-model-dir",
            scratch_dir,
            "--hires-upscalers-dir",
            scratch_dir,
            "--embd-dir",
            scratch_dir,
        ]
    if threads is not None:
        cmd += ["--threads", str(int(threads))]

    offload = list(offload or [])
    if offload:
        cmd += offload
    # De-dup speed flags against offload (may already include --diffusion-fa).
    cmd += [f for f in native_speed_flags(native_speed) if f not in offload]
    if verbose:
        cmd += ["-v"]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def build_img_gen_request(
    *,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    batch_count: int = 1,
    sample_method: Optional[str] = None,
    flow_shift: Optional[float] = None,
    cfg_scale: Optional[float] = None,
    distilled_guidance: Optional[float] = None,
    output_format: str = "png",
    lora: Optional[list[dict]] = None,
) -> dict:
    """Build the ``POST /sdcpp/v1/img_gen`` JSON body for one text-to-image request.

    The API takes the whole batch in one request, reusing the resident model. Sampling lives under
    ``sample_params``; guidance is split like the one-shot engine (a FLUX distilled value ->
    ``guidance.distilled_guidance``, a real CFG scale -> ``guidance.txt_cfg``). Only set keys are emitted.
    """
    if not str(prompt).strip():
        raise ValueError("prompt is required")

    guidance: dict = {}
    if cfg_scale is not None:
        guidance["txt_cfg"] = float(cfg_scale)
    if distilled_guidance is not None:
        guidance["distilled_guidance"] = float(distilled_guidance)

    sample_params: dict = {}
    if steps is not None:
        sample_params["sample_steps"] = int(steps)
    if sample_method:
        sample_params["sample_method"] = str(sample_method)
    if flow_shift is not None:
        sample_params["flow_shift"] = float(flow_shift)
    if guidance:
        sample_params["guidance"] = guidance

    req: dict = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "width": int(width),
        "height": int(height),
        "batch_count": max(1, int(batch_count)),
        "output_format": output_format,
    }
    if seed is not None:
        req["seed"] = int(seed)
    if sample_params:
        req["sample_params"] = sample_params
    # Structured LoRA list: the API resolves each ``path`` against the server's ``--lora-model-dir``
    # (prompt-embedded ``<lora:>`` tags are unsupported server-side), so LoRAs are staged and named here.
    if lora:
        req["lora"] = lora
    return req


def _fmt_float(value: float) -> str:
    """Compact float -> str: drop a trailing ``.0`` (``1.0`` -> ``1``) to keep logged commands tidy."""
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)
