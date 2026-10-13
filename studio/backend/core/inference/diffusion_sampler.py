# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sampler (scheduler) choice for the diffusers image engine, plus the v-prediction markers a
single-file SDXL checkpoint carries. Names follow AUTOMATIC1111; the mapping is the diffusers
scheduler overview table. No torch/diffusers imports at module level."""

from __future__ import annotations

from functools import lru_cache, wraps
from typing import Any, Optional

DEFAULT_SAMPLER = "default"
AUTOMATIC_SCHEDULE = "automatic"

# id -> (label, diffusers class, constructor overrides)
SAMPLERS: dict[str, tuple[str, Optional[str], dict[str, Any]]] = {
    DEFAULT_SAMPLER: ("Model default", None, {}),
    "euler": ("Euler", "EulerDiscreteScheduler", {}),
    "euler_a": ("Euler a", "EulerAncestralDiscreteScheduler", {}),
    "dpmpp_2m": (
        "DPM++ 2M",
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "dpmsolver++", "solver_order": 2},
    ),
    "dpmpp_2m_sde": (
        "DPM++ 2M SDE",
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "solver_order": 2},
    ),
}

# id -> (label, sigma flag set True; None = every sigma flag off)
SCHEDULE_TYPES: dict[str, tuple[str, Optional[str]]] = {
    AUTOMATIC_SCHEDULE: ("Automatic", None),
    "uniform": ("Uniform", None),
    "karras": ("Karras", "use_karras_sigmas"),
    "exponential": ("Exponential", "use_exponential_sigmas"),
}
_SIGMA_FLAGS = ("use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas", "use_lu_lambdas")

# AUTOMATIC1111's per-sampler default schedule (modules/sd_samplers_kdiffusion.py); absent = keep the config's.
_AUTOMATIC_FOR = {"dpmpp_2m": "karras", "dpmpp_2m_sde": "exponential"}


def _init_params(cls: type) -> frozenset[str]:
    import inspect
    try:
        return frozenset(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return frozenset()


def _class_for(sampler: str, resident: Any) -> Optional[type]:
    cls_name = SAMPLERS[sampler][1]
    if cls_name is None:
        return type(resident)
    import diffusers

    return getattr(diffusers, cls_name, None)


def _schedule_types_for(cls: type) -> list[str]:
    params = _init_params(cls)
    return [sid for sid, (_label, flag) in SCHEDULE_TYPES.items() if flag is None or flag in params]


@lru_cache(maxsize = 32)
def _options_for(scheduler_cls: type) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    compatible = {
        c if isinstance(c, str) else getattr(c, "__name__", "")
        for c in (getattr(scheduler_cls, "_compatibles", None) or ())
    }
    if not compatible:
        return ()
    import diffusers

    options = []
    for sid, (label, cls_name, _overrides) in SAMPLERS.items():
        if cls_name is not None and cls_name not in compatible:
            continue
        cls = scheduler_cls if cls_name is None else getattr(diffusers, cls_name, None)
        if cls is None:
            continue
        options.append((sid, label, tuple(_schedule_types_for(cls))))
    return tuple(options) if len(options) > 1 else ()


def sampler_options(scheduler: Any) -> list[dict[str, Any]]:
    """Samplers the resident scheduler can be swapped for, each with the schedule types it takes.

    Empty when the scheduler lists no compatible classes (every flow-matching model), so the
    control is absent there rather than offering choices that cannot run. Read on every status poll,
    hence cached per class."""
    config = getattr(scheduler, "config", None) or {}
    # A DPM / UniPC scheduler driving a flow model lists the epsilon samplers as compatible too.
    if (
        scheduler is None
        or config.get("use_flow_sigmas")
        or config.get("prediction_type") == "flow_prediction"
    ):
        return []
    return [
        {"id": sid, "label": label, "schedule_types": list(types)}
        for sid, label, types in _options_for(type(scheduler))
    ]


def resolve_schedule(
    sampler: Optional[str],
    schedule_type: Optional[str],
    config: Any = None,
) -> Optional[str]:
    """The schedule a request runs: ``automatic`` resolves to the sampler's A1111 default, None = config's."""
    schedule_type = schedule_type or AUTOMATIC_SCHEDULE
    if schedule_type != AUTOMATIC_SCHEDULE:
        return schedule_type
    resolved = _AUTOMATIC_FOR.get(sampler or DEFAULT_SAMPLER)
    # Exponential spacing down from a zero-SNR sigma_max (~4096) washes out detail (NoobAI-XL V-Pred 1.0).
    if resolved == "exponential" and (config or {}).get("rescale_betas_zero_snr"):
        return "karras"
    return resolved


def build_scheduler(
    resident: Any, sampler: Optional[str], schedule_type: Optional[str]
) -> tuple[Optional[Any], Optional[str]]:
    """``(scheduler, resolved schedule)`` for a request; scheduler None keeps ``resident`` as is.

    Built from the resident config, so prediction type and zero-SNR settings carry over. Raises
    ValueError for a choice this model does not offer."""
    sampler = sampler or DEFAULT_SAMPLER
    schedule_type = schedule_type or AUTOMATIC_SCHEDULE
    if sampler not in SAMPLERS:
        raise ValueError(f"Unknown sampler '{sampler}'.")
    if schedule_type not in SCHEDULE_TYPES:
        raise ValueError(f"Unknown schedule type '{schedule_type}'.")
    if sampler == DEFAULT_SAMPLER and schedule_type == AUTOMATIC_SCHEDULE:
        return None, None
    options = {o["id"]: o for o in sampler_options(resident)}
    if sampler not in options:
        raise ValueError(
            f"The {SAMPLERS[sampler][0]} sampler is not available for the loaded model."
        )
    if schedule_type not in options[sampler]["schedule_types"]:
        raise ValueError(
            f"The {SCHEDULE_TYPES[schedule_type][0]} schedule type does not apply to the "
            f"{SAMPLERS[sampler][0]} sampler."
        )
    cls = _class_for(sampler, resident)
    params = _init_params(cls)
    overrides = {k: v for k, v in SAMPLERS[sampler][2].items() if k in params}
    resolved = resolve_schedule(sampler, schedule_type, resident.config)
    if resolved is not None:
        flag = SCHEDULE_TYPES[resolved][1]
        for name in _SIGMA_FLAGS:
            if name in params:
                overrides[name] = name == flag
    built = cls.from_config(resident.config, **overrides)
    set_begin_index = getattr(built, "set_begin_index", None)
    if callable(set_begin_index):
        # Zero-SNR Karras grids repeat 999, so DPM++ without a begin index (SDXL ControlNet sets none) overruns
        # its sigmas. img2img / inpaint set their own start after set_timesteps.
        set_timesteps = built.set_timesteps

        @wraps(set_timesteps)
        def _set_timesteps(*args: Any, **kwargs: Any) -> Any:
            out = set_timesteps(*args, **kwargs)
            set_begin_index(0)
            return out

        built.set_timesteps = _set_timesteps
    return built, resolved


def sampler_label(sampler: Optional[str]) -> Optional[str]:
    entry = SAMPLERS.get(sampler or "")
    return entry[0] if entry and sampler != DEFAULT_SAMPLER else None


def schedule_label(schedule_type: Optional[str]) -> Optional[str]:
    entry = SCHEDULE_TYPES.get(schedule_type or "")
    return entry[0] if entry and schedule_type != AUTOMATIC_SCHEDULE else None


def _safetensors_top_level_keys(path: str) -> frozenset[str]:
    import json
    import struct

    with open(path, "rb") as fh:
        (size,) = struct.unpack("<Q", fh.read(8))
        # A header past 100 MB is not a safetensors file.
        if size <= 0 or size > 100_000_000:
            return frozenset()
        header = json.loads(fh.read(size))
    return frozenset(k for k in header if "." not in k)


def checkpoint_prediction_overrides(path: Optional[str]) -> dict[str, Any]:
    """Scheduler overrides a single-file checkpoint asks for through its ``v_pred`` / ``ztsnr`` marker
    tensors (ComfyUI's detection, comfy/supported_models.py SDXL.model_type). Empty when unmarked."""
    if not path or not str(path).endswith(".safetensors"):
        return {}
    try:
        keys = _safetensors_top_level_keys(str(path))
    except Exception:  # noqa: BLE001 - an unreadable header keeps the shipped scheduler
        return {}
    if "v_pred" not in keys:
        return {}
    overrides: dict[str, Any] = {"prediction_type": "v_prediction"}
    if "ztsnr" in keys:
        # Zero terminal SNR samples from the last timestep (arXiv 2305.08891, section 3.3).
        overrides.update(rescale_betas_zero_snr = True, timestep_spacing = "trailing")
    return overrides


def apply_checkpoint_prediction_type(
    pipe: Any,
    path: Optional[str],
    logger: Any = None,
) -> bool:
    """Rebuild ``pipe.scheduler`` for a v-prediction checkpoint. True when it changed; idempotent."""
    overrides = checkpoint_prediction_overrides(path)
    scheduler = getattr(pipe, "scheduler", None)
    config = getattr(scheduler, "config", None)
    if not overrides or config is None:
        return False
    params = _init_params(type(scheduler))
    overrides = {k: v for k, v in overrides.items() if k in params}
    if all(config.get(k) == v for k, v in overrides.items()):
        return False
    pipe.scheduler = type(scheduler).from_config(config, **overrides)
    if logger is not None:
        logger.info("scheduler set from checkpoint markers: %s", overrides)
    return True
