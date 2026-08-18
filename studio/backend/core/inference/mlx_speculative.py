# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discovery, recommendation, and preflight policy for MLX speculative decoding."""

from functools import lru_cache
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Optional


MLX_SPECULATIVE_METHODS = frozenset({"mtp", "dflash", "eagle3"})
MLX_SPECULATIVE_MODES = MLX_SPECULATIVE_METHODS | {"auto"}

# Each method joins this set with the load path that can run it, so a request for a
# method the worker cannot execute is refused before the active model is torn down.
ENABLED_MLX_SPECULATIVE_METHODS: frozenset[str] = frozenset()

# Refusals reach the client as prose, while the codes stay the vocabulary the response
# schema will use to say why a resolved method differs from the one requested.
MLX_SPECULATIVE_REFUSALS: dict[str, str] = {
    "method_not_integrated": "This build cannot run the requested MLX speculative decoding method.",
}

# A code with no entry of its own still has to reach the client as a 400 rather than
# as the KeyError a subscript would raise.
MLX_SPECULATIVE_GENERIC_REFUSAL = "This model cannot use the requested speculative decoding method."


def mlx_speculative_refusal_text(reason: str) -> str:
    """The sentence for a refusal, as an error detail. Unknown reasons read generically."""
    return MLX_SPECULATIVE_REFUSALS.get(reason, MLX_SPECULATIVE_GENERIC_REFUSAL)


def normalize_mlx_speculative_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MLX_SPECULATIVE_MODES else "off"


@lru_cache(maxsize = 1)
def mlx_speculative_runtime_capabilities() -> dict[str, Any]:
    result: dict[str, Any] = {
        "common": False,
        "methods": {method: False for method in sorted(MLX_SPECULATIVE_METHODS)},
        "reason": "runtime_unavailable",
    }
    if sys.platform != "darwin":
        result["reason"] = "mlx_requires_apple_silicon"
        return result
    try:
        drafters = importlib.import_module("mlx_vlm.speculative.drafters")
        ar = importlib.import_module("mlx_vlm.generate.ar")
        utils = importlib.import_module("mlx_vlm.speculative.utils")
        return _runtime_capabilities_from_modules(drafters, ar, utils)
    except Exception:
        return result


def _runtime_capabilities_from_modules(drafters: Any, ar: Any, utils: Any) -> dict[str, Any]:
    signature = inspect.signature(ar.generate_step)
    common = (
        callable(getattr(drafters, "load_drafter", None))
        and callable(getattr(utils, "run_speculative_rounds", None))
        and {"draft_model", "draft_kind", "draft_block_size"}.issubset(signature.parameters)
    )
    known = set(getattr(drafters, "KNOWN_DRAFTER_KINDS", ()))
    methods = {}
    for method in sorted(MLX_SPECULATIVE_METHODS):
        try:
            dispatch = utils.get_speculative_rounds_batch(method)
        except Exception:
            dispatch = None
        methods[method] = bool(common and method in known and callable(dispatch))
    result = {"common": common, "methods": methods}
    result["reason"] = None if common else "runtime_missing_speculative_api"
    return result


def _public_target_model_id(target_id: str) -> str:
    from core.inference.model_ids import public_model_id

    try:
        local = Path(target_id).expanduser()
    except RuntimeError:
        # A "~unknown-user" prefix has no home to expand; it names no repository either.
        return "local-model"
    if local.is_absolute() or local.exists() or target_id.startswith("."):
        return local.name or "local-model"
    return public_model_id(target_id) or "local-model"


def mlx_speculative_options(target_id: str) -> dict[str, Any]:
    """Speculative drafters usable with ``target_id``, with local paths redacted.

    The candidate list is empty until this backend can enumerate drafters; a runtime
    without speculative support answers the same way, and callers must handle both.
    """
    capabilities = mlx_speculative_runtime_capabilities()
    return {
        "target_model": _public_target_model_id(target_id),
        "experimental": True,
        "runtime_supported": bool(capabilities["common"]),
        "runtime_reason": capabilities["reason"],
        "candidates": [],
    }


def mlx_speculative_request_reason(method: Any) -> Optional[str]:
    """Why an MLX speculative request cannot be served, or None when it can.

    Off and Auto always resolve: Auto falls back to ordinary MLX generation when no
    drafter can run. An explicit method is refused unless its execution path is
    available, so an unsupported request never reaches model teardown.
    """
    mode = normalize_mlx_speculative_mode(method)
    if mode in {"off", "auto"}:
        return None
    if mode not in ENABLED_MLX_SPECULATIVE_METHODS:
        return "method_not_integrated"
    return None
