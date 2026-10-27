# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hires fix: txt2img, upscale, then refine (img2img at a denoise) in one request.

Defaults follow ComfyUI's 2_pass_txt2img example (1.5x, denoise 0.5, 14 steps). Hires steps is the exact refine
count, as in ComfyUI (``new_steps = int(steps / denoise)``) and A1111 (``hr_second_pass_steps or steps``).
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
# The smallest non-zero denoise: the 0.01 step of A1111's "Denoising strength" slider and ComfyUI's KSampler denoise.
# Below it ``steps / denoise`` grows without bound.
MIN_DENOISE = 0.01
MIN_SCALE = 1.0
MAX_SCALE = 4.0
MAX_SIDE = 2048
# StableDiffusionXLImg2ImgPipeline.prepare_latents takes a 4-channel latent as is.
LATENT_FAMILIES = frozenset({"sdxl"})
SDXL_LATENT_FACTOR = 8

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
    cancel: threading.Event
    preflight: Optional[Callable[[dict], None]] = None
    source: Any = None
    # Base pass: return the sampler's latents (``output_type="latent"``), not decoded images (latent upscaler).
    latent_output: bool = False
    # Refine pass: this image's base latents, ``(1, C, h, w)``, upscaled in latent space as the img2img input.
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
    if 0.0 < float(hires.denoise) < MIN_DENOISE:
        raise ValueError(
            f"Hires denoising strength must be 0 (upscale only) or at least {MIN_DENOISE:g}."
        )
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
    """``round(side * scale)`` to a multiple of 16, scaled down to fit ``max_side`` and ``max_pixels``."""
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
    n = steps
    # denoise >= MIN_DENOISE bounds this at about steps / MIN_DENOISE.
    limit = int(math.ceil(steps / max(denoise, MIN_DENOISE))) + 4
    while steps_run(n, denoise, rule) < steps:
        if n >= limit:
            raise ValueError(
                f"Hires fix cannot schedule {steps} refine steps at denoise {denoise:g}."
            )
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
    native: bool = False,
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
        # sd.cpp takes no separate refine CFG and no upscale-only pass (plan_sd_cpp refuses both).
        "supports_guidance": not native,
        "min_denoise": MIN_DENOISE if native else 0.0,
    }


def unavailable_reason(family: Any) -> Optional[str]:
    if family is None or getattr(family, "edit", False):
        return None
    if not getattr(family, "img2img_pipeline_class", None):
        return f"Hires fix needs an image-to-image path, which {getattr(family, 'name', 'this model')} does not have yet."
    return None


def record_fields(result_hires: Optional[dict]) -> dict:
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
    """A1111 ``Size`` is the base pass, so a Hires fix image records the base ``width``/``height``."""
    if (
        meta.get("hires_scale") is None
        or not meta.get("hires_base_width")
        or not meta.get("hires_base_height")
    ):
        return meta
    return {
        **meta,
        "width": meta["hires_base_width"],
        "height": meta["hires_base_height"],
        # The rendered size, for infotext_fields' Hires resize.
        "hires_width": meta.get("width"),
        "hires_height": meta.get("height"),
    }


def infotext_fields(meta: dict) -> str:
    """A1111 infotext for a hires image in A1111 / Forge key names; empty for any other image."""
    if meta.get("hires_scale") is None:
        return ""
    parts = [f"Denoising strength: {meta.get('hires_denoise')}"]
    out_w, out_h = meta.get("hires_width"), meta.get("hires_height")
    base_w, base_h = meta.get("hires_base_width"), meta.get("hires_base_height")
    scale = float(meta["hires_scale"])
    if (
        out_w
        and out_h
        and base_w
        and base_h
        and (int(out_w), int(out_h)) != (int(int(base_w) * scale), int(int(base_h) * scale))
    ):
        # A1111 sizes ``Hires upscale`` as int(side * scale); a capped or 16-rounded target is written the way A1111
        # writes an explicit size (processing.py: "Hires resize" in place of "Hires upscale"), so it reproduces.
        parts.append(f"Hires resize: {int(out_w)}x{int(out_h)}")
    else:
        parts.append(f"Hires upscale: {meta.get('hires_scale')}")
    if meta.get("hires_steps"):
        parts.append(f"Hires steps: {meta['hires_steps']}")
    label = upscalers.model_label(meta.get("hires_upscaler"))
    if label:
        parts.append(f"Hires upscaler: {label}")
    if meta.get("hires_guidance") is not None:
        parts.append(f"Hires CFG Scale: {meta['hires_guidance']}")
    return ", " + ", ".join(parts)


# sd-server img_gen ``hires`` object (examples/common/common.cpp), upstream PR #1451.
SD_CPP_HELP_MARKER = "--hires-denoising-strength"


_sd_cpp_help_cache: dict[tuple[str, int], bool] = {}


def sd_cpp_supports_hires(binary: Optional[str]) -> bool:
    """Whether ``--help`` lists hires. Fails closed: an older sd-server ignores ``hires`` and renders the base pass."""
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
    base = capability(family, model_rows = upscalers.native_model_upscalers(), native = True)
    if base is None:
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
    # Steps the Hires pass actually samples (sd.cpp derives them from the sent ``steps`` and the strength).
    refine_steps: int
    # The ``steps`` sent so sd.cpp samples the requested count (A1111 runs exactly ``Hires steps``); None = requested.
    request_steps: Optional[int] = None


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
    request_steps = sd_cpp_hires_request_steps(int(params.steps or steps), float(params.denoise))
    return SdCppHiresPlan(
        params = params,
        base_width = int(width),
        base_height = int(height),
        target = target,
        refine_steps = sd_cpp_hires_sampled_steps(request_steps, float(params.denoise)),
        request_steps = request_steps,
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
    if upscaler_id == upscalers.LATENT:
        return "Latent (nearest-exact)"
    if upscaler_id == upscalers.LANCZOS:
        return "Lanczos"
    if not staged_name:
        raise ValueError(f"Upscaler '{upscaler_id}' was not staged for the native engine.")
    return staged_name


def sd_cpp_hires_sampled_steps(steps: int, denoise: float) -> int:
    """Steps sd.cpp's Hires pass samples (and reports progress over), from make_hires_sigma_schedule in
    src/pipeline/request.cpp: ``scheduler_steps = int(steps / strength)``, ``t_enc = int(scheduler_steps * strength)``
    capped at ``scheduler_steps - 1``, in C ``float`` arithmetic."""
    import numpy as np

    strength = np.float32(denoise)
    effective = max(1, int(steps))
    scheduler_steps = max(1, int(np.float32(effective) / strength))
    t_enc = int(np.float32(scheduler_steps) * strength)
    if t_enc >= scheduler_steps:
        t_enc = scheduler_steps - 1
    return max(1, t_enc)


def sd_cpp_hires_request_steps(steps: int, denoise: float) -> int:
    """The ``steps`` to send so sd.cpp samples exactly ``steps`` (A1111 setup_img2img_steps: ``t_enc = requested - 1``,
    i.e. the requested count). sd.cpp truncates twice, so 20 at 0.35 samples 19; the nearest sent count that samples
    the request is used, else the request itself (some counts are unreachable, e.g. 20 at 0.35)."""
    steps = max(1, int(steps))
    for sent in range(steps, steps + 4):
        if sd_cpp_hires_sampled_steps(sent, denoise) == steps:
            return sent
    return steps


def sd_cpp_payload(
    hires: HiresParams,
    base_steps: int,
    target: tuple[int, int],
    upscaler_name: str,
    request_steps: Optional[int] = None,
) -> dict:
    """``request_steps`` (plan_sd_cpp) makes sd.cpp sample the requested refine count; the size is explicit so both
    engines match."""
    return {
        "enabled": True,
        "upscaler": upscaler_name,
        "scale": float(hires.scale),
        "target_width": int(target[0]),
        "target_height": int(target[1]),
        "steps": int(request_steps or hires.steps or base_steps),
        "denoising_strength": float(hires.denoise),
    }


def upscale_latents(pipe: Any, latents: Any, target: tuple[int, int]) -> Any:
    """ComfyUI ``LatentUpscale``: ``common_upscale(samples, w // 8, h // 8, "nearest-exact", crop="disabled")`` on the
    sampler's own latents. No VAE round trip: nearest-exact commutes with the per-channel latent scaling."""
    import torch
    import torch.nn.functional as F

    if not isinstance(latents, torch.Tensor) or latents.ndim != 4:
        raise TypeError("The latent upscaler needs the base pass's latents, (1, C, h, w).")
    tw, th = target
    factor = int(getattr(pipe, "vae_scale_factor", SDXL_LATENT_FACTOR) or SDXL_LATENT_FACTOR)
    with torch.inference_mode():
        return F.interpolate(latents, size = (th // factor, tw // factor), mode = "nearest-exact")


def _rearm_cancel(backend: Any, cancel: threading.Event) -> None:
    """The base pass deregisters the cancel event; re-register it so Stop still lands in the refine pass."""
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
    """Clear the between-passes phase even when no refine pass runs (upscale-only or failed upscale)."""
    try:
        yield
    finally:
        with backend._lock:
            backend._gen = None


def _kick_deferred_bg_compile(backend: Any) -> None:
    state = getattr(backend, "_state", None)
    bg = getattr(state, "bg_compile", None) if state is not None else None
    if bg is not None:
        try:
            bg.kick()  # idempotent; a no-op unless a compile is pending
        except Exception as exc:  # noqa: BLE001 - never mask the pass's own error
            logger.debug("diffusion.hires: deferred compile kick failed: %s", exc)


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
    width = int(kwargs.get("width") or 1024)
    height = int(kwargs.get("height") or 1024)
    base_steps = int(kwargs.get("steps") or 9)
    # Filled from the loaded family once the generation slot is held (a load cannot swap the model under the plan).
    params = hires
    refine_steps = 0
    target = (width, height)
    refine = False

    def _plan() -> None:
        nonlocal params, refine_steps, target, refine
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
    with (
        backend._generation_slot(cancel),
        _progress_cleared_on_exit(backend),
        contextlib.ExitStack() as after_base,
    ):
        _plan()
        # A model upscaler also runs on the GPU after the base pass, so a background compile waits for it too.
        defer_kick = refine or upscalers.is_model_upscaler(params.upscaler)
        ctx = HiresPass(
            cancel = cancel,
            preflight = _preflight,
            defer_bg_kick = defer_kick,
            # The latent upscaler needs no pixels: the base pass skips its VAE decode and the refine its encode.
            latent_output = refine and params.upscaler == upscalers.LATENT,
        )
        base = generate(backend, **kwargs, _hires_pass = ctx)
        if defer_kick:
            # The base pass left a compile pending: start it once the passes after it end, also when one fails or is
            # stopped (the last refine kicks it on success; a repeat kick is a no-op).
            after_base.callback(_kick_deferred_bg_compile, backend)
        t_base = time.perf_counter() - t0
        images = list(base["images"])
        seeds = list(base.get("seeds") or [base["seed"]] * len(images))
        prompts = kwargs.get("prompts") or None
        refined: list[Any] = []
        upscale_s = 0.0
        refine_s = 0.0
        upscale_info: dict = {}
        refine_negative: Optional[str] = None
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
                # Placeholder at the hires size: the refine call reads only its size; the base latents replace it.
                source = Image.new("RGB", target)
                latent_from = image  # this image's (1, C, h, w) base latents
            elif params.upscaler == upscalers.LANCZOS:
                source = image.convert("RGB").resize(target, Image.LANCZOS)
            else:
                outcome = upscalers.upscale_with_model(
                    image,
                    params.upscaler,
                    target,
                    device = device,
                    # Small (33 MB fp16): kept resident unless the load offloads.
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
                # Stop may land during the resize itself; an accepted cancel never returns an image.
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
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
            pass_kwargs["width"], pass_kwargs["height"] = target
            if params.guidance is not None:
                pass_kwargs["guidance"] = float(params.guidance)
            controls = pass_kwargs.get("sd_controls")
            if controls is not None and getattr(controls, "variation_strength", None):
                # A variation seed shapes the base pass's noise; the refine pass takes its noise from the source.
                pass_kwargs["sd_controls"] = replace(
                    controls, variation_seed = None, variation_strength = None
                )
            pass_ctx = HiresPass(
                cancel = cancel,
                source = source,
                latent_from = latent_from,
                # Kick a deferred background compile only after the LAST refine (else the next image waits on it).
                defer_bg_kick = index < len(images) - 1,
                refine = True,
            )
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
            if out.get("negative_prompt") and not refine_negative:
                refine_negative = out["negative_prompt"]
    result = dict(base)
    result["images"] = refined
    if refine_negative and not result.get("negative_prompt"):
        # Hires CFG engaged a negative the base pass's guidance left off; the recipe must still carry it.
        result["negative_prompt"] = refine_negative
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
