# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hires fix: txt2img, upscale, then refine (img2img at a denoise) in one request.

Defaults are ComfyUI's two-pass example (comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/, the
``hiresfix_latent_workflow.png`` graph): LatentUpscale nearest-exact 768 -> 1152 (1.5x), second KSampler denoise 0.5,
14 steps. ComfyUI's KSampler runs exactly ``steps`` steps at a denoise (``comfy/samplers.py``:
``new_steps = int(steps / denoise)``, last ``steps`` sigmas), and A1111 does the same for its hires pass
(``modules/processing.py`` ``sample_img2img(..., steps=self.hr_second_pass_steps or self.steps)`` ->
``sd_samplers_common.setup_img2img_steps``), so "Hires steps" here is the exact count the refine pass runs, and
None means the base pass's count.

On the diffusers engine the refine pass is the family's img2img pipeline (the same one the Upscale workflow uses),
so it exists exactly where ``img2img_pipeline_class`` is set. ``latent`` (SDXL only) re-encodes the base image and
interpolates the latents nearest-exact before the refine pass; ``lanczos`` and model upscalers work in pixels. The
sd.cpp engine runs its own ``hires`` request block (sd-server ``/sdcpp/v1/img_gen``, examples/common/common.cpp).

Graph policy: the base pass is the unchanged txt2img call, so its compiled graphs and CUDA graphs are the ones every
plain render uses. The refine pass runs the img2img pipeline (sharing the denoiser) at the hires size, a new input
shape: a CUDA-graph wrapper records one more graph for it beside the base shape's (``diffusion_cuda_graph``: one
graph per input shape, ``MAX_GRAPHS_PER_MODULE``), and a compiled denoiser follows its usual shape policy. The next
hires render at the same size replays both.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import math
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from loggers import get_logger

from . import diffusion_upscalers as upscalers

logger = get_logger(__name__)

DEFAULT_SCALE = 1.5
DEFAULT_DENOISE = 0.5
MIN_SCALE = 1.0
MAX_SCALE = 4.0
MAX_SIDE = 2048
# The latent upscaler needs a 4-channel SDXL latent the img2img pipe accepts as is
# (StableDiffusionXLImg2ImgPipeline.prepare_latents: ``if image.shape[1] == 4: init_latents = image``).
LATENT_FAMILIES = frozenset({"sdxl"})
SDXL_LATENT_FACTOR = 8

# Rules for how an img2img pipeline turns (num_inference_steps, strength) into steps run.
RULE_FLOOR = "floor"  # SDXL: init_timestep = min(int(N * s), N)
RULE_FLOW = "flow"  # FLUX / Qwen-Image / Z-Image: t_start = int(max(N - min(N * s, N), 0)); runs N - t_start


@dataclass(frozen = True)
class HiresParams:
    scale: float = DEFAULT_SCALE
    denoise: float = DEFAULT_DENOISE
    # None = the base pass's step count (A1111 ``hr_second_pass_steps or steps``).
    steps: Optional[int] = None
    # None = the family default (latent on SDXL, Lanczos elsewhere).
    upscaler: Optional[str] = None
    # None = the base guidance (Forge ``hr_cfg``, infotext "Hires CFG Scale").
    guidance: Optional[float] = None


@dataclass
class HiresPass:
    """Private context one hires request hands its two inner ``generate`` calls."""

    cancel: threading.Event
    # Base pass: called with the activation guard kwargs to refuse a refine pass that cannot fit, before denoising.
    preflight: Optional[Callable[[dict], None]] = None
    # Refine pass: the upscaled source (PIL at the hires size) and, for the latent upscaler, the base image to encode.
    source: Any = None
    latent_from: Any = None
    # The base pass leaves a pending background compile for after the refine pass (else the refine pass waits on it).
    defer_bg_kick: bool = False
    refine: bool = False
    stats: dict = field(default_factory = dict)


def default_upscaler(family_name: Optional[str]) -> str:
    return upscalers.LATENT if family_name in LATENT_FAMILIES else upscalers.LANCZOS


def resolve_params(hires: HiresParams, family_name: Optional[str]) -> HiresParams:
    """Fill the family default upscaler and validate the request against the family (ValueError -> 400)."""
    upscaler = hires.upscaler or default_upscaler(family_name)
    upscalers.check_upscaler_id(upscaler)
    if upscaler == upscalers.LATENT and family_name not in LATENT_FAMILIES:
        raise ValueError(
            "The latent upscaler is available on SDXL only; pick Lanczos or a model upscaler for this model."
        )
    if not (MIN_SCALE <= float(hires.scale) <= MAX_SCALE):
        raise ValueError(f"Hires upscale must be between {MIN_SCALE:g} and {MAX_SCALE:g}.")
    if not (0.0 <= float(hires.denoise) <= 1.0):
        raise ValueError("Hires denoising strength must be between 0 and 1.")
    if upscaler == upscalers.LATENT and float(hires.denoise) <= 0.0:
        raise ValueError(
            "The latent upscaler needs a refine pass: set a denoising strength above 0."
        )
    return replace(hires, upscaler = upscaler)


def hires_target_size(
    width: int,
    height: int,
    scale: float,
    *,
    multiple: int = 16,
    max_side: int = MAX_SIDE,
    max_pixels: Optional[int] = None,
) -> tuple[int, int]:
    """The refine size: ``round(side * scale)`` to a multiple of 16, scaled down so the longest side fits ``max_side``
    (and the area ``max_pixels``). The same rounding and 2048 cap as the Upscale workflow."""
    multiple = max(1, int(multiple))
    tw_f, th_f = float(width) * float(scale), float(height) * float(scale)
    fit = min(1.0, float(max_side) / max(tw_f, th_f))
    if max_pixels:
        fit = min(fit, math.sqrt(float(max_pixels) / (tw_f * th_f)))
    tw = max(multiple, int(round(tw_f * fit / multiple)) * multiple)
    th = max(multiple, int(round(th_f * fit / multiple)) * multiple)
    if max_pixels:
        while tw * th > max_pixels and (tw > multiple or th > multiple):
            if tw >= th:
                tw -= multiple
            else:
                th -= multiple
    return tw, th


def steps_run(num_inference_steps: int, strength: float, rule: str) -> int:
    """Steps an img2img pipeline actually denoises for ``num_inference_steps`` at ``strength``."""
    n = int(num_inference_steps)
    s = float(strength)
    if rule == RULE_FLOOR:
        return min(int(n * s), n)
    t_start = int(max(n - min(n * s, n), 0))
    return n - t_start


def refine_inference_steps(steps: int, denoise: float, rule: str) -> int:
    """The ``num_inference_steps`` that makes the refine pass run EXACTLY ``steps`` steps at ``denoise``."""
    steps = max(1, int(steps))
    denoise = float(denoise)
    if denoise >= 1.0:
        return steps
    # ComfyUI's schedule first (``new_steps = int(steps / denoise)``), then the nearest count that also works.
    guess = int(steps / denoise)
    for n in (guess, guess + 1, guess - 1, guess + 2, guess - 2, guess + 3):
        if n >= steps and steps_run(n, denoise, rule) == steps:
            return n
    # Float edge: fall back to the smallest count that reaches it.
    n = steps
    while steps_run(n, denoise, rule) < steps and n < 100000:
        n += 1
    return n


def timestep_rule(pipe: Any) -> str:
    """Which rounding ``pipe.get_timesteps`` applies; FLOOR for SDXL-style ``int(N * strength)``."""
    fn = getattr(type(pipe), "get_timesteps", None)
    try:
        src = inspect.getsource(fn) if fn is not None else ""
    except (OSError, TypeError):
        src = ""
    return RULE_FLOOR if "int(num_inference_steps * strength)" in src else RULE_FLOW


# ---------------------------------------------------------------------------------------------------------------------
# Capability (status)


def _upscaler_rows(family_name: Optional[str], model_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    if family_name in LATENT_FAMILIES:
        rows.append(
            {
                "id": upscalers.LATENT,
                "label": upscalers.BUILTIN_LABELS[upscalers.LATENT],
                "kind": "latent",
            }
        )
    rows.append(
        {
            "id": upscalers.LANCZOS,
            "label": upscalers.BUILTIN_LABELS[upscalers.LANCZOS],
            "kind": "pixel",
        }
    )
    rows.extend({"id": r["id"], "label": r["label"], "kind": "model"} for r in model_rows)
    return rows


def capability(
    family: Any,
    *,
    model_rows: Optional[list[dict]] = None,
    max_side: int = MAX_SIDE,
) -> Optional[dict]:
    """Status ``hires`` block for a family with an img2img path; None hides the control."""
    if family is None or getattr(family, "edit", False) or getattr(family, "layer_count", 0):
        return None
    if not getattr(family, "img2img_pipeline_class", None):
        return None
    name = getattr(family, "name", None)
    side = min(int(max_side), int(getattr(family, "max_output_side", MAX_SIDE) or MAX_SIDE))
    return {
        "upscalers": _upscaler_rows(name, list(model_rows or [])),
        "default_upscaler": default_upscaler(name),
        "defaults": {"scale": DEFAULT_SCALE, "denoise": DEFAULT_DENOISE},
        "max_side": side,
    }


def unavailable_reason(family: Any) -> Optional[str]:
    if family is None or getattr(family, "edit", False):
        return None
    if not getattr(family, "img2img_pipeline_class", None):
        return f"Hires fix needs an image-to-image path, which {getattr(family, 'name', 'this model')} does not have yet."
    return None


# ---------------------------------------------------------------------------------------------------------------------
# Metadata


def record_fields(result_hires: Optional[dict]) -> dict:
    """Gallery-record keys for a hires result block (empty when off)."""
    if not result_hires:
        return {}
    return {
        "hires_scale": result_hires.get("scale"),
        "hires_denoise": result_hires.get("denoise"),
        "hires_steps": result_hires.get("steps"),
        "hires_upscaler": result_hires.get("upscaler"),
        "hires_guidance": result_hires.get("guidance"),
        "hires_base_width": result_hires.get("base_width"),
        "hires_base_height": result_hires.get("base_height"),
    }


def infotext_meta(meta: dict) -> dict:
    """``meta`` with ``width``/``height`` set to the base pass for a Hires fix image (A1111 ``Size`` is the first
    pass, ``modules/processing.py`` create_infotext ``"Size": f"{p.width}x{p.height}"``); unchanged otherwise."""
    if (
        meta.get("hires_scale") is None
        or not meta.get("hires_base_width")
        or not meta.get("hires_base_height")
    ):
        return meta
    return {**meta, "width": meta["hires_base_width"], "height": meta["hires_base_height"]}


def infotext_fields(meta: dict) -> str:
    """A1111 infotext for a hires image, in A1111's own key names (``modules/processing.py``): ``Denoising
    strength``, ``Hires upscale``, ``Hires steps`` (only when set, as A1111 omits 0), ``Hires upscaler``, and Forge's
    ``Hires CFG Scale``. Empty for any other image."""
    if meta.get("hires_scale") is None:
        return ""
    parts = [
        f"Denoising strength: {meta.get('hires_denoise')}",
        f"Hires upscale: {meta.get('hires_scale')}",
    ]
    if meta.get("hires_steps"):
        parts.append(f"Hires steps: {meta['hires_steps']}")
    label = upscalers.model_label(meta.get("hires_upscaler"))
    if label:
        parts.append(f"Hires upscaler: {label}")
    if meta.get("hires_guidance") is not None:
        parts.append(f"Hires CFG Scale: {meta['hires_guidance']}")
    return ", " + ", ".join(parts)


# ---------------------------------------------------------------------------------------------------------------------
# sd.cpp


# sd-server's img_gen ``hires`` object (examples/common/common.cpp, SDGenerationParams::from_json) and the --help flag
# that marks a build carrying it (``--hires`` landed with the same parser, upstream PR #1451).
SD_CPP_HELP_MARKER = "--hires-denoising-strength"


_sd_cpp_help_cache: dict[tuple[str, int], bool] = {}


def sd_cpp_supports_hires(binary: Optional[str]) -> bool:
    """Whether ``binary``'s ``--help`` lists the hires options. Fails closed: an older sd-server ignores an unknown
    ``hires`` JSON key and would render the base pass alone. Cached per (path, mtime)."""
    import os

    if not binary:
        return False
    try:
        key = (str(binary), int(os.stat(binary).st_mtime_ns))
    except OSError:
        return False
    cached = _sd_cpp_help_cache.get(key)
    if cached is None:
        from core.inference.sd_cpp_backend import _sd_cpp_probe_output

        text = _sd_cpp_probe_output(str(binary), "--help")
        cached = text is not None and SD_CPP_HELP_MARKER in text
        _sd_cpp_help_cache[key] = cached
    return cached


def sd_cpp_capability(
    family: Any, *, mode: Optional[str], binary: Optional[str]
) -> tuple[Optional[dict], Optional[str]]:
    """``(hires, hires_unavailable_reason)`` status values for the sd.cpp engine."""
    base = capability(family, model_rows = upscalers.native_model_upscalers())
    if base is None:
        # No image-to-image path: hidden like the sampler picker, no reason line.
        return None, None
    if mode != "server":
        return None, "Hires fix runs on the resident native server; this load uses one-shot sd-cli."
    if not sd_cpp_supports_hires(binary):
        return (
            None,
            "This stable-diffusion.cpp build has no Hires fix support; update the native engine.",
        )
    return base, None


@dataclass(frozen = True)
class SdCppHiresPlan:
    params: HiresParams
    base_width: int
    base_height: int
    target: tuple[int, int]
    refine_steps: int


def plan_sd_cpp(
    family: Any,
    hires: HiresParams,
    *,
    mode: Optional[str],
    binary: Optional[str],
    width: int,
    height: int,
    steps: int,
) -> SdCppHiresPlan:
    """Validate a Hires fix request for the native engine (ValueError -> 400) and size it like the diffusers engine."""
    cap, reason = sd_cpp_capability(family, mode = mode, binary = binary)
    if cap is None:
        raise ValueError(
            reason
            or f"Hires fix is not supported for '{getattr(family, 'name', 'this')}' on the native engine."
        )
    params = resolve_params(hires, getattr(family, "name", None))
    if float(params.denoise) <= 0.0:
        raise ValueError(
            "Upscale only (denoise 0) is not available on the native engine; set a denoise above 0."
        )
    if params.guidance is not None:
        raise ValueError(
            "A separate Hires CFG is not available on the native engine; leave it unset."
        )
    target = hires_target_size(
        int(width),
        int(height),
        params.scale,
        multiple = int(getattr(family, "dimension_multiple", 16) or 16),
        max_side = min(MAX_SIDE, int(getattr(family, "max_output_side", MAX_SIDE) or MAX_SIDE)),
        max_pixels = int(getattr(family, "max_output_pixels", 0) or 0) or None,
    )
    from core.inference.diffusion_conditioning import check_output_size

    check_output_size(family, *target)
    return SdCppHiresPlan(
        params = params,
        base_width = int(width),
        base_height = int(height),
        target = target,
        refine_steps = int(params.steps or steps),
    )


def stage_sd_cpp_upscaler(
    upscaler_id: str,
    scratch_dir: str,
    *,
    cancel: Any = None,
) -> tuple[str, Any]:
    """Link a model upscaler into sd-server's --hires-upscalers-dir; returns (name without extension, staged path)."""
    import os
    import shutil
    from pathlib import Path

    source = upscalers.resolve_model_path(upscaler_id, cancel = cancel)
    name = (
        "hires_"
        + "".join(ch if ch.isalnum() else "_" for ch in upscaler_id)[:80]
        + "_"
        + os.urandom(3).hex()
    )
    dest = Path(scratch_dir) / (name + source.suffix.lower())
    try:
        os.symlink(source, dest)
    except OSError:
        try:
            os.link(source, dest)
        except OSError:
            shutil.copyfile(source, dest)
    return name, dest


def result_block(plan: "SdCppHiresPlan") -> dict:
    p = plan.params
    return {
        "scale": float(p.scale),
        "denoise": float(p.denoise),
        "steps": p.steps,
        "upscaler": p.upscaler,
        "guidance": p.guidance,
        "base_width": plan.base_width,
        "base_height": plan.base_height,
        "width": plan.target[0],
        "height": plan.target[1],
        "refine_steps": plan.refine_steps,
    }


def sd_cpp_upscaler_name(upscaler_id: str, staged_name: Optional[str] = None) -> str:
    """The ``hires.upscaler`` string sd.cpp resolves: its built-in names, else a model file stem in
    --hires-upscalers-dir."""
    if upscaler_id == upscalers.LATENT:
        return "Latent (nearest-exact)"
    if upscaler_id == upscalers.LANCZOS:
        return "Lanczos"
    if not staged_name:
        raise ValueError(f"Upscaler '{upscaler_id}' was not staged for the native engine.")
    return staged_name


def sd_cpp_payload(
    hires: HiresParams, base_steps: int, target: tuple[int, int], upscaler_name: str
) -> dict:
    """The ``hires`` request object for sd-server. ``steps`` is the exact refine count (sd.cpp: ``hires.steps > 0 ?
    hires.steps : default_steps``, scheduler steps = steps / denoise, src/stable-diffusion.cpp), the target size is
    passed explicitly so both engines produce the same size."""
    return {
        "enabled": True,
        "upscaler": upscaler_name,
        "scale": float(hires.scale),
        "target_width": int(target[0]),
        "target_height": int(target[1]),
        "steps": int(hires.steps or base_steps),
        "denoising_strength": float(hires.denoise),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Diffusers orchestration


def encode_upscaled_latents(pipe: Any, base_image: Any, target: tuple[int, int]) -> Any:
    """SDXL latent upscaler: VAE-encode the base image exactly as StableDiffusionXLImg2ImgPipeline.prepare_latents does
    (upcast when the VAE asks, latents_mean/std, scaling_factor), then interpolate nearest-exact to the hires size."""
    import torch
    import torch.nn.functional as F

    vae = pipe.vae
    param = next(vae.parameters())
    device, dtype = param.device, param.dtype
    x = pipe.image_processor.preprocess(base_image).to(device = device, dtype = dtype)
    upcast = bool(getattr(vae.config, "force_upcast", False)) and dtype == torch.float16
    with torch.inference_mode():
        if upcast:
            vae.to(dtype = torch.float32)
            x = x.float()
        try:
            lat = vae.encode(x).latent_dist.mode()
        finally:
            if upcast:
                vae.to(dtype = dtype)
        lat = lat.to(dtype)
        mean = getattr(vae.config, "latents_mean", None)
        std = getattr(vae.config, "latents_std", None)
        sf = float(vae.config.scaling_factor)
        if mean is not None and std is not None:
            m = torch.tensor(mean, device = device, dtype = dtype).view(1, -1, 1, 1)
            s = torch.tensor(std, device = device, dtype = dtype).view(1, -1, 1, 1)
            lat = (lat - m) * sf / s
        else:
            lat = lat * sf
        tw, th = target
        factor = int(getattr(pipe, "vae_scale_factor", SDXL_LATENT_FACTOR) or SDXL_LATENT_FACTOR)
        lat = F.interpolate(lat, size = (th // factor, tw // factor), mode = "nearest-exact")
    return lat


def _rearm_cancel(backend: Any, cancel: threading.Event) -> None:
    """The base pass deregisters the generation's cancel event as it returns; register it again for the refine pass so
    Stop still lands."""
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
    from utils.account_context import current_account_id

    with backend._generation_cancel_lock:
        if cancel.is_set():
            raise RuntimeError(DIFFUSION_CANCELLED_MSG)
        backend._active_generate_cancel = cancel
        backend._active_generate_account = current_account_id()


def _publish_upscale_phase(backend: Any, total_steps: int) -> None:
    """Keep progress ACTIVE between the passes (the base pass clears it as it returns)."""
    from core.inference.diffusion import _GenState
    with backend._lock:
        backend._gen = _GenState(total_steps = max(1, int(total_steps)), phase = "encode")


@contextlib.contextmanager
def _progress_cleared_on_exit(backend: Any):
    """The phase published between the passes must not outlive the request (an upscale-only run has no refine pass
    to clear it, and a failed upscale none either)."""
    try:
        yield
    finally:
        with backend._lock:
            backend._gen = None


def _check_request(kwargs: dict) -> None:
    for name, label in (
        ("init_image", "a source image"),
        ("mask_image", "a mask"),
        ("reference_images", "reference images"),
        ("workflow", "an explicit workflow"),
        ("localized_edit", "a localized edit"),
    ):
        if kwargs.get(name):
            raise ValueError(
                f"Hires fix runs on text to image only; it cannot be combined with {label}."
            )
    if kwargs.get("upscale_model"):
        raise ValueError("Hires fix picks its upscaler in hires.upscaler, not upscale_model.")
    if kwargs.get("upscale") is not None and float(kwargs["upscale"]) > 1.0:
        raise ValueError("Hires fix runs on text to image only; use either Hires fix or Upscale.")
    controlnet = kwargs.get("controlnet")
    if controlnet is not None and controlnet[3] not in (None, 0, 0.0):
        raise ValueError("Hires fix cannot be combined with ControlNet yet.")


def run_hires(
    backend: Any, generate: Callable[..., dict], hires: HiresParams, kwargs: dict
) -> dict:
    """Base pass, upscale, refine pass, all inside ONE generation slot (no other render can run between them)."""
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG, DIFFUSION_NOT_LOADED_MSG
    from core.inference.diffusion_memory import (
        ImageActivationShortfallError,
        raise_on_image_activation_shortfall,
    )
    from core.inference.generation_ticket import bind_generation_cancel
    from core.inference.diffusion_conditioning import check_output_size
    from PIL import Image

    _check_request(kwargs)
    state = getattr(backend, "_state", None)
    if state is None:
        raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
    fam = state.family
    if capability(fam) is None:
        raise ValueError(
            unavailable_reason(fam)
            or f"Hires fix is not supported for the '{fam.name}' model family."
        )
    params = resolve_params(hires, fam.name)
    width = int(kwargs.get("width") or 1024)
    height = int(kwargs.get("height") or 1024)
    base_steps = int(kwargs.get("steps") or 9)
    refine_steps = int(params.steps or base_steps)
    target = hires_target_size(
        width,
        height,
        params.scale,
        multiple = int(getattr(fam, "dimension_multiple", 16) or 16),
        max_side = min(MAX_SIDE, int(getattr(fam, "max_output_side", MAX_SIDE) or MAX_SIDE)),
        max_pixels = int(getattr(fam, "max_output_pixels", 0) or 0) or None,
    )
    check_output_size(fam, *target)
    if upscalers.is_model_upscaler(params.upscaler) and not upscalers.spandrel_available():
        raise ValueError(
            "Model upscalers need the spandrel package, which is not installed. Pick Lanczos instead."
        )
    refine = float(params.denoise) > 0.0

    def _preflight(guard_kwargs: dict) -> None:
        if not refine:
            return
        try:
            raise_on_image_activation_shortfall(
                **{
                    **guard_kwargs,
                    "width": target[0],
                    "height": target[1],
                    "batch_size": 1,
                    "source_driven": False,
                    "condition_pixels": 0,
                    "controlnet": False,
                }
            )
        except ImageActivationShortfallError as exc:
            raise ImageActivationShortfallError(
                f"Hires fix: the refine pass at {target[0]}x{target[1]} does not fit where the base pass at "
                f"{width}x{height} does. {exc}"
            ) from exc

    cancel = threading.Event()
    bind_generation_cancel(cancel)
    t0 = time.perf_counter()
    with backend._generation_slot(cancel), _progress_cleared_on_exit(backend):
        ctx = HiresPass(cancel = cancel, preflight = _preflight, defer_bg_kick = refine)
        base = generate(backend, **kwargs, _hires_pass = ctx)
        t_base = time.perf_counter() - t0
        images = list(base["images"])
        seeds = list(base.get("seeds") or [base["seed"]] * len(images))
        prompts = kwargs.get("prompts") or None
        refined: list[Any] = []
        upscale_s = 0.0
        refine_s = 0.0
        upscale_info: dict = {}
        state = getattr(backend, "_state", None)
        keep_resident = state is not None and getattr(state, "offload_policy", "none") in (
            None,
            "none",
        )
        device = getattr(state, "device", "cpu") if state is not None else "cpu"
        for index, image in enumerate(images):
            if cancel.is_set():
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            _rearm_cancel(backend, cancel)
            _publish_upscale_phase(backend, refine_steps)
            t_up = time.perf_counter()
            latent_from = None
            if params.upscaler == upscalers.LATENT:
                # Placeholder at the hires size: the refine call reads only its size; the latents replace it.
                source = Image.new("RGB", target)
                latent_from = image
            elif params.upscaler == upscalers.LANCZOS:
                source = image.convert("RGB").resize(target, Image.LANCZOS)
            else:
                outcome = upscalers.upscale_with_model(
                    image,
                    params.upscaler,
                    target,
                    device = device,
                    # Small (RealESRGAN x4plus is 33 MB in fp16): kept for the next request unless the load
                    # offloads, where it leaves right after (the unload drops it too).
                    keep_resident = keep_resident,
                    cancel = cancel,
                )
                source = outcome.image
                upscale_info = {
                    "model_scale": outcome.model_scale,
                    "tiles": outcome.tiles,
                    "dtype": outcome.dtype,
                    "released": outcome.released,
                }
            upscale_s += time.perf_counter() - t_up
            if not refine:
                refined.append(source)
                continue
            t_ref = time.perf_counter()
            pass_kwargs = dict(kwargs)
            for drop in ("prompts", "seeds", "batch_size", "width", "height", "upscale"):
                pass_kwargs.pop(drop, None)
            pass_kwargs["prompt"] = (
                prompts[index] if prompts and index < len(prompts) else kwargs.get("prompt")
            )
            pass_kwargs["seed"] = int(seeds[index])
            pass_kwargs["batch_size"] = 1
            pass_kwargs["strength"] = float(params.denoise)
            # Placeholder only; the refine pass sizes from the source image.
            pass_kwargs["width"], pass_kwargs["height"] = target
            if params.guidance is not None:
                pass_kwargs["guidance"] = float(params.guidance)
            pass_ctx = HiresPass(cancel = cancel, source = source, latent_from = latent_from, refine = True)
            pass_ctx.stats["refine_steps"] = refine_steps
            pass_kwargs["steps"] = refine_steps  # converted to num_inference_steps inside the call
            try:
                from core.inference import diffusion_bg_compile
                with diffusion_bg_compile.no_record():
                    out = generate(backend, **pass_kwargs, _hires_pass = pass_ctx)
            except ImageActivationShortfallError as exc:
                raise ImageActivationShortfallError(
                    f"Hires fix refine pass at {target[0]}x{target[1]}: {exc}"
                ) from exc
            refined.extend(out["images"])
            refine_s += time.perf_counter() - t_ref
    result = dict(base)
    result["images"] = refined
    result["workflow"] = "txt2img"
    result["hires"] = {
        "scale": float(params.scale),
        "denoise": float(params.denoise),
        "steps": params.steps,
        "upscaler": params.upscaler,
        "guidance": params.guidance,
        "base_width": width,
        "base_height": height,
        "width": target[0],
        "height": target[1],
        "refine_steps": refine_steps if refine else 0,
        "timing": {
            "base_s": round(t_base, 3),
            "upscale_s": round(upscale_s, 3),
            "refine_s": round(refine_s, 3),
            "total_s": round(time.perf_counter() - t0, 3),
        },
        **({"model": upscale_info} if upscale_info else {}),
    }
    logger.info(
        "diffusion.hires: %dx%d -> %dx%d via %s, denoise %.2f, %d refine steps, %d image(s), %.2fs base + %.2fs "
        "upscale + %.2fs refine",
        width,
        height,
        target[0],
        target[1],
        params.upscaler,
        params.denoise,
        refine_steps if refine else 0,
        len(refined),
        t_base,
        upscale_s,
        refine_s,
    )
    return result


def with_hires(generate: Callable[..., dict]) -> Callable[..., dict]:
    """Add a keyword-only ``hires`` to an engine's ``generate``. ``hires=None`` calls ``generate`` unchanged."""

    @functools.wraps(generate)
    def wrapper(
        self: Any,
        *args: Any,
        hires: Optional[HiresParams] = None,
        **kwargs: Any,
    ) -> dict:
        if hires is None:
            return generate(self, *args, **kwargs)
        if args:
            raise TypeError("generate(hires=...) takes keyword arguments only")
        return run_hires(self, generate, hires, kwargs)

    sig = inspect.signature(generate)
    params = [p for p in sig.parameters.values() if p.name != "_hires_pass"]
    params.append(
        inspect.Parameter(
            "hires", inspect.Parameter.KEYWORD_ONLY, default = None, annotation = Optional[HiresParams]
        )
    )
    wrapper.__signature__ = sig.replace(parameters = params)  # type: ignore[attr-defined]
    return wrapper
