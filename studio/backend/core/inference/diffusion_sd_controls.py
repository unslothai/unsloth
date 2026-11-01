# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A1111 CLIP skip N = diffusers N-1 on SD 1.x, N-2 on SDXL; variation seeds follow webui rng; base
noise still drawn in prepare_latents, so generator advances as before"""

from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

from .diffusion_prompt_weighting import (
    encode_weighted_prompts,
    exceeds_one_chunk,
    has_prompt_syntax,
)

# A1111 CLIP skip at or below this = the family's default layer (SD 1.x would be 1).
_CLIP_SKIP_BASE: dict[str, int] = {"sdxl": 2}
CLIP_SKIP_MAX = 12
GUIDANCE_RESCALE_MAX = 1.0
# arXiv 2305.08891 sec 3.4; a recommendation only, never a load default.
GUIDANCE_RESCALE_VPRED_HINT = 0.7
VARY_STRENGTH = 0.15


@dataclass(frozen = True)
class SdControls:
    """Request values. ``None`` everywhere = today's behaviour."""

    prompt_weighting: Optional[bool] = None
    clip_skip: Optional[int] = None
    cfg_rescale: Optional[float] = None
    variation_seed: Optional[int] = None
    variation_strength: Optional[float] = None

    def requested(self) -> bool:
        return (
            self.prompt_weighting is True
            or (self.clip_skip is not None and self.clip_skip > 1)
            or bool(self.cfg_rescale)
            or bool(self.variation_strength)
        )


def supports_sd_controls(family: Any, pipe: Any) -> bool:
    """The loaded pipeline is a CLIP-conditioned U-Net family these controls are defined for."""
    name = getattr(family, "name", None)
    if name not in _CLIP_SKIP_BASE or pipe is None:
        return False
    return (
        getattr(pipe, "text_encoder", None) is not None
        and getattr(pipe, "tokenizer", None) is not None
    )


def diffusers_clip_skip(family_name: str, a1111: Optional[int]) -> Optional[int]:
    """A1111 CLIP skip (1 = off) -> diffusers ``clip_skip`` (None = the pipeline's layer)."""
    if a1111 is None:
        return None
    skip = int(a1111) - _CLIP_SKIP_BASE.get(family_name, 1)
    return skip if skip > 0 else None


def _min_text_layers(pipe: Any) -> Optional[int]:
    layers = []
    for attr in ("text_encoder", "text_encoder_2"):
        config = getattr(getattr(pipe, attr, None), "config", None)
        value = getattr(config, "num_hidden_layers", None)
        if isinstance(value, int):
            layers.append(value)
    return min(layers) if layers else None


def clip_skip_max(family_name: str, pipe: Any) -> int:
    """Largest A1111 value whose layer exists in every text encoder (the embedding output excluded)."""
    layers = _min_text_layers(pipe)
    if layers is None:
        return CLIP_SKIP_MAX
    return max(1, min(CLIP_SKIP_MAX, layers))


def _prediction_type(pipe: Any) -> Optional[str]:
    config = getattr(getattr(pipe, "scheduler", None), "config", None) or {}
    try:
        return config.get("prediction_type")
    except AttributeError:
        return None


def status_fields(family: Any, pipe: Any) -> Optional[dict[str, Any]]:
    """``sd_controls`` for the status poll: None when the loaded model has none of these controls."""
    if not supports_sd_controls(family, pipe):
        return None
    prediction = _prediction_type(pipe)
    return {
        "prompt_weighting": True,
        "clip_skip": {
            "min": 1,
            "max": clip_skip_max(family.name, pipe),
            "default": 1,
            "same_as_default_through": _CLIP_SKIP_BASE.get(family.name, 1),
        },
        "cfg_rescale": {
            "min": 0.0,
            "max": GUIDANCE_RESCALE_MAX,
            "default": 0.0,
            "recommended": GUIDANCE_RESCALE_VPRED_HINT if prediction == "v_prediction" else None,
        },
        "variation_seed": True,
        "vary_strength": VARY_STRENGTH,
        "prediction_type": prediction,
    }


def validate(
    controls: Optional[SdControls], family: Any, pipe: Any, call_params: Any, workflow: str
) -> None:
    """Refuse (ValueError -> 400) a control the loaded model or workflow cannot apply, before any GPU work."""
    if controls is None or not controls.requested():
        return
    label = getattr(family, "label", None) or getattr(family, "name", "this model")
    if not supports_sd_controls(family, pipe):
        names = [
            name
            for name, on in (
                ("prompt weighting", controls.prompt_weighting is True),
                ("CLIP skip", controls.clip_skip is not None and controls.clip_skip > 1),
                ("CFG rescale", bool(controls.cfg_rescale)),
                ("variation seed", bool(controls.variation_strength)),
            )
            if on
        ]
        raise ValueError(
            f"{', '.join(names)} applies to SDXL models only; the loaded model is {label}. "
            "Leave these unset for this model."
        )
    if controls.clip_skip is not None and controls.clip_skip > clip_skip_max(family.name, pipe):
        raise ValueError(
            f"CLIP skip {controls.clip_skip} exceeds this model's text encoder depth "
            f"(at most {clip_skip_max(family.name, pipe)})."
        )
    if controls.cfg_rescale and "guidance_rescale" not in call_params:
        raise ValueError(f"CFG rescale is not available for the {workflow} workflow of {label}.")
    if (controls.prompt_weighting is True or _clip_skip_engaged(controls, family)) and (
        "prompt_embeds" not in call_params
    ):
        raise ValueError(
            f"Prompt weighting and CLIP skip are not available for the {workflow} workflow of {label}."
        )
    if controls.variation_strength and (
        workflow not in ("txt2img", "controlnet") or not _latents_hookable(pipe)
    ):
        raise ValueError(
            "Variation seed applies to text-to-image (and ControlNet) generation only; the "
            f"{workflow} workflow derives its noise from the source image."
        )


def _clip_skip_engaged(controls: SdControls, family: Any) -> bool:
    return diffusers_clip_skip(getattr(family, "name", ""), controls.clip_skip) is not None


def weighting_engaged(
    controls: Optional[SdControls], family: Any, pipe: Any, texts: Sequence[Optional[str]]
) -> bool:
    """Encodes here for weight syntax, prompts past 75 tokens, or a CLIP skip past the model layer;
    else the pipeline encodes, bit-identically."""
    if not supports_sd_controls(family, pipe):
        return False
    controls = controls or SdControls()
    if _clip_skip_engaged(controls, family):
        return True
    if controls.prompt_weighting is not False and any(has_prompt_syntax(t) for t in texts):
        return True
    return any(exceeds_one_chunk(pipe, t) for t in texts)


def apply_prompt_controls(
    pipe: Any,
    family: Any,
    controls: Optional[SdControls],
    call_kwargs: dict[str, Any],
    *,
    guidance: float,
) -> bool:
    """Swap ``prompt`` / ``negative_prompt`` for weighted embeddings in ``call_kwargs`` when engaged.
    True when weight syntax was applied. Runs inside the render thread and grad mode."""
    prompt = call_kwargs.get("prompt")
    negative = call_kwargs.get("negative_prompt")
    prompts = [prompt] if isinstance(prompt, str) else list(prompt or [])
    negatives = None
    if negative is not None:
        negatives = [negative] * len(prompts) if isinstance(negative, str) else list(negative)
    if not prompts or not weighting_engaged(controls, family, pipe, prompts + (negatives or [])):
        return False
    controls = controls or SdControls()
    unet_config = getattr(getattr(pipe, "unet", None), "config", None)
    # The pipeline's do_classifier_free_guidance: guidance > 1 and no guidance-embedding U-Net (LCM).
    guidance_engaged = (
        float(guidance) > 1.0 and getattr(unet_config, "time_cond_proj_dim", None) is None
    )
    embeds = encode_weighted_prompts(
        pipe,
        prompts,
        negatives,
        guidance_engaged = guidance_engaged,
        weighted = controls.prompt_weighting is not False,
        clip_skip = diffusers_clip_skip(family.name, controls.clip_skip),
    )
    call_kwargs.pop("prompt", None)
    call_kwargs.pop("negative_prompt", None)
    call_kwargs.update(embeds)
    return controls.prompt_weighting is not False and any(
        has_prompt_syntax(t) for t in prompts + (negatives or [])
    )


def guidance_rescale_kwargs(controls: Optional[SdControls], call_params: Any) -> dict[str, Any]:
    if controls is None or not controls.cfg_rescale or "guidance_rescale" not in call_params:
        return {}
    return {"guidance_rescale": float(controls.cfg_rescale)}


def slerp(val: float, low: Any, high: Any) -> Any:
    """webui ``modules/rng.py`` ``slerp`` on one image's ``[C, H, W]`` noise (norms over dim 1, as webui)."""
    import torch

    low_norm = low / torch.norm(low, dim = 1, keepdim = True)
    high_norm = high / torch.norm(high, dim = 1, keepdim = True)
    dot = (low_norm * high_norm).sum(1)
    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high


def variation_noise(base: Any, variation: Any, strength: float) -> Any:
    """Per-image slerp of ``[B, C, H, W]`` latents in float32, cast back to the latents' dtype."""
    import torch

    if not strength:
        return base
    rows = [
        slerp(float(strength), b.float(), v.float())
        for b, v in zip(base.unbind(0), variation.unbind(0))
    ]
    return torch.stack(rows).to(base.dtype)


def _latents_hookable(pipe: Any) -> bool:
    fn = getattr(pipe, "prepare_latents", None)
    if not callable(fn):
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "num_channels_latents" in params and "generator" in params and "latents" in params


def variation_seeds(variation_seed: int, count: int) -> list[int]:
    """One variation seed per image, ``variation_seed + index`` (webui ``all_subseeds``)."""
    return [int(variation_seed) + i for i in range(count)]


@contextlib.contextmanager
def variation_latents(
    pipe: Any, seeds: Optional[Sequence[int]], strength: Optional[float]
) -> Iterator[None]:
    """For one pipeline call, slerp the initial latents toward ``seeds``' noise by ``strength``. No-op when off."""
    if not strength or not seeds:
        yield
        return
    from diffusers.utils.torch_utils import randn_tensor
    import torch

    original = pipe.prepare_latents
    signature = inspect.signature(original)
    had_own = "prepare_latents" in getattr(pipe, "__dict__", {})

    def prepare_latents(*args: Any, **kwargs: Any) -> Any:
        latents = original(*args, **kwargs)
        if kwargs.get("latents") is not None:
            return latents
        bound = signature.bind(*args, **kwargs)
        if bound.arguments.get("latents") is not None:
            return latents
        generator = bound.arguments.get("generator")
        gens = generator if isinstance(generator, list) else [generator]
        if len(gens) != latents.shape[0] and len(gens) == 1:
            gens = gens * latents.shape[0]
        devices = [g.device if g is not None else latents.device for g in gens]
        var_gens = [
            torch.Generator(device = d).manual_seed(int(s))
            for d, s in zip(devices, list(seeds)[: len(devices)])
        ]
        if len(var_gens) != latents.shape[0]:
            raise ValueError("variation seeds do not match the batch")
        var = randn_tensor(
            tuple(latents.shape),
            generator = var_gens if len(var_gens) > 1 else var_gens[0],
            device = latents.device,
            dtype = latents.dtype,
        )
        # Same scaling prepare_latents applied to the base noise, so strength 1 is exactly the variation seed's noise.
        var = var * pipe.scheduler.init_noise_sigma
        return variation_noise(latents, var, float(strength))

    pipe.prepare_latents = prepare_latents
    try:
        yield
    finally:
        if had_own:
            pipe.prepare_latents = original
        else:
            try:
                del pipe.prepare_latents
            except AttributeError:
                pass


def record_fields(
    controls: Optional[SdControls],
    *,
    weighting_used: bool,
    variation_seed_list: Optional[Sequence[int]],
) -> dict[str, Any]:
    """Result / recipe fields for what actually ran; None = off (prompt_weighting: False = turned off)."""
    controls = controls or SdControls()
    varied = bool(controls.variation_strength) and bool(variation_seed_list)
    return {
        "prompt_weighting": True
        if weighting_used
        else (False if controls.prompt_weighting is False else None),
        "clip_skip": int(controls.clip_skip)
        if controls.clip_skip and controls.clip_skip > 1
        else None,
        "cfg_rescale": float(controls.cfg_rescale) if controls.cfg_rescale else None,
        "variation_seed": int(variation_seed_list[0]) if varied else None,
        "variation_seeds": [int(s) for s in variation_seed_list] if varied else None,
        "variation_strength": float(controls.variation_strength) if varied else None,
    }
