# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refuse, from config.json alone, an architecture neither installed mlx-vlm nor mlx-lm can build."""

from __future__ import annotations

import re
from importlib.util import find_spec
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)

# mlx-vlm appends "No module named 'mlx_vlm.speculative.drafters.<type>'": a registry search, not a drafter load (#18972).
_UPSTREAM_UNSUPPORTED = re.compile(r"Model type ([A-Za-z0-9_.\-]+) not supported")

_FAMILY_HINTS = {
    "inkling2_loop": (
        "Inkling 2 Mini cannot run on MLX in Unsloth yet: it needs a newer mlx-vlm than the one installed. "
        "Use the GGUF build instead: unsloth/Inkling-2-Mini-GGUF."
    ),
}
# Kept clear of routes.inference._NOT_SUPPORTED_HINTS, which would wrap this message in a generic one.
_GENERIC_HINT = (
    "Update Unsloth once a newer mlx-vlm or mlx-lm adds it, or use a GGUF build of this model."
)


def _version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001 -- a version string only decorates the message
        return "not installed"


def _vlm_builds(config: dict) -> bool:
    import mlx_vlm.utils as vlm_utils

    resolve = getattr(vlm_utils, "get_model_type", None)
    if resolve is not None:
        model_type = resolve(config)
    else:
        raw = str(config["model_type"]).lower()
        model_type = vlm_utils.MODEL_REMAPPING.get(raw, raw)
    if not model_type:
        raise KeyError("model_type")
    return any(
        find_spec(f"{package}.{model_type}") is not None
        for package in ("mlx_vlm.models", "mlx_vlm.speculative.drafters")
    )


def _lm_builds(config: dict) -> bool:
    import mlx_lm.utils as lm_utils

    resolve = getattr(lm_utils, "model_package", None)
    if resolve is not None:
        package = resolve(config)
    else:
        model_type = config["model_type"]
        package = lm_utils.MODEL_REMAPPING.get(model_type, model_type)
    return find_spec(f"mlx_lm.models.{package}") is not None


def unsupported_model_type(config) -> Optional[str]:
    """The ``model_type`` neither installed mlx-vlm nor mlx-lm has a module for, else None (also if inconclusive)."""
    if not isinstance(config, dict) or config.get("model_file"):
        return None
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        return None
    for builds in (_vlm_builds, _lm_builds):
        try:
            if builds(config):
                return None
        except Exception as exc:  # noqa: BLE001 -- an inconclusive check never refuses a load
            logger.debug("MLX support check for %s inconclusive: %s", model_type, exc)
            return None
    return model_type


def unsupported_model_type_message(model_type: str, model_name: Optional[str] = None) -> str:
    who = f"{model_name} uses" if model_name else "This model uses"
    hint = _FAMILY_HINTS.get(str(model_type).lower(), _GENERIC_HINT)
    return (
        f"{who} the '{model_type}' architecture, which the MLX runtime in this Unsloth install "
        f"(mlx-vlm {_version('mlx-vlm')}, mlx-lm {_version('mlx-lm')}) cannot load. {hint}"
    )


def upstream_unsupported_message(msg: str) -> Optional[str]:
    """The readable refusal for mlx-vlm's / mlx-lm's own "Model type X not supported" error, or None."""
    match = _UPSTREAM_UNSUPPORTED.search(msg or "")
    return unsupported_model_type_message(match.group(1)) if match else None
