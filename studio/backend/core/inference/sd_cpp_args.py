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
    "flux.2-dev": ("--llm",),
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


# Native (sd.cpp) speed profiles, the engine-side analogue of diffusion_speed's
# modes. off: nothing (default). default: --diffusion-fa (flash attention;
# near-lossless attention speed/memory win) + --diffusion-conv-direct. Direct conv
# is numerically exact (no quality tradeoff) and, on the CPU tier this engine
# actually serves (Studio routes to sd.cpp only on no-GPU hosts), the A/B on the
# master-741-484baa4 linux build measured z-image Q8_0 sampling 56.1s -> 51.3s
# (~9%) with decode and peak RSS unchanged, so it belongs in the default profile
# rather than opt-in. max keeps it too (the profiles stay a superset chain).
NATIVE_SPEED_OFF = "off"
NATIVE_SPEED_DEFAULT = "default"
NATIVE_SPEED_MAX = "max"
NATIVE_SPEED_MODES = (NATIVE_SPEED_OFF, NATIVE_SPEED_DEFAULT, NATIVE_SPEED_MAX)


def native_speed_flags(speed_mode: Optional[str]) -> list[str]:
    """sd-cli speed flags for a native speed mode (empty for off / None).

    These are separate from the offload flags: ``--diffusion-fa`` is a speed/memory
    win in its own right, not tied to whether weights are offloaded. De-duplicated
    against offload flags at the call site (offload already adds ``--diffusion-fa``).
    """
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
    # ``(prompt or "")`` so a None prompt is rejected here rather than slipping past
    # ``str(None)`` == "None" (truthy) and landing in argv as a literal "None".
    if not (params.prompt or "").strip():
        raise ValueError("prompt is required")
    # sd-cli inpaint needs the source image too: a --mask with no --init-img is an
    # invalid invocation (sd-cli has nothing to inpaint into), so reject it here with a
    # clear error instead of emitting a command that fails deep in sd-cli.
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
        # sd-cli names the extra batch images itself (output_2.png, ...) and the runner
        # collects only the literal --output path, so a CLI batch would silently drop
        # every image after the first. Batches go through the sdcpp server API instead.
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

    ``sd-server`` (stable-diffusion.cpp ``examples/server``) loads the model once at
    spawn from the SAME flags ``sd-cli`` takes (``--diffusion-model`` / ``--vae`` /
    the text encoders / ``--vae-format`` / offload + speed), and adds ``--listen-ip``
    / ``--listen-port``. Per-generation parameters (prompt, size, steps, seed, cfg,
    sampler, batch) are NOT here -- they go in each ``/sdcpp/v1/img_gen`` request, so
    one resident process serves many generations without reloading the weights.

    ``offload`` / ``native_speed`` map to the exact same sd.cpp flags as the one-shot
    engine (``--offload-to-cpu`` / ``--diffusion-fa`` / ...), verified to be accepted
    by ``sd-server --help``. ``scratch_dir`` (if given) is pointed at by the LoRA /
    hires-upscaler / embeddings directory flags: sd-server's img_gen handler recursively
    iterates those dirs, and an unset / missing dir makes it fail the request, so we give
    it a real (empty) directory. ``extra_args`` is appended last so a power user can
    override anything (sd.cpp's parser is last-wins).
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
    # De-dup speed flags against offload (offload may already include --diffusion-fa).
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

    The native ``sdcpp`` API takes the whole batch in one request (``batch_count``),
    so a batch reuses the resident model with no reload. Sampling lives under
    ``sample_params``; guidance is split exactly like the one-shot engine's
    ``_map_guidance``: a FLUX distilled value goes to ``guidance.distilled_guidance``,
    a real classifier-free scale goes to ``guidance.txt_cfg``. Only set keys are
    emitted so the server applies its own defaults for the rest.
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
    # Structured LoRA list: the sdcpp API resolves each ``path`` against the server's
    # scanned ``--lora-model-dir`` (prompt-embedded ``<lora:>`` tags are intentionally
    # unsupported server-side), so LoRAs must be staged there and named here.
    if lora:
        req["lora"] = lora
    return req


def _fmt_float(value: float) -> str:
    """Compact float -> str: drop a trailing ``.0`` so ``1.0`` -> ``1`` (sd-cli
    accepts both, but the tidy form keeps logged commands readable)."""
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)
