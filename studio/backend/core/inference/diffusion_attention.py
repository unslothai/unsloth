# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusers adapter for Studio's logical attention backend policy."""

from __future__ import annotations

import contextlib
from typing import Any, Optional

from .attention_policy import attention_fallback_order, normalize_attention_backend


DIFFUSERS_ATTENTION_BACKEND_CANDIDATES: dict[str, tuple[str, ...]] = {
    "flash": ("_flash_3_hub", "flash", "_native_flash"),
    "sdpa": ("native",),
    "flex": ("flex",),
    "xformers": ("xformers",),
}


def _target_name(name: str, target: Any) -> str:
    class_name = target.__class__.__name__
    return f"{name}:{class_name}" if name else class_name


def _attention_targets(pipe: Any) -> list[tuple[str, Any]]:
    targets: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for name in ("transformer", "unet"):
        target = getattr(pipe, name, None)
        if target is None:
            continue
        if not hasattr(target, "set_attention_backend") and not hasattr(
            target,
            "reset_attention_backend",
        ):
            continue
        ident = id(target)
        if ident in seen:
            continue
        seen.add(ident)
        targets.append((_target_name(name, target), target))
    if hasattr(pipe, "set_attention_backend") or hasattr(
        pipe, "reset_attention_backend"
    ):
        ident = id(pipe)
        if ident not in seen:
            targets.append((_target_name("pipeline", pipe), pipe))
    return targets


def _reset_attention_targets(targets: list[tuple[str, Any]]) -> None:
    for _, target in targets:
        reset = getattr(target, "reset_attention_backend", None)
        if callable(reset):
            with contextlib.suppress(Exception):
                reset()
    with contextlib.suppress(Exception):
        from diffusers.models.attention_dispatch import (
            AttentionBackendName,
            _AttentionBackendRegistry,
        )

        _AttentionBackendRegistry.set_active_backend(AttentionBackendName.NATIVE)


def _try_pipeline_xformers(pipe: Any) -> bool:
    enable = getattr(pipe, "enable_xformers_memory_efficient_attention", None)
    if not callable(enable):
        return False
    enable()
    return True


def apply_diffusers_attention_backend(
    pipe: Any,
    requested: Optional[str],
    *,
    prefer_default_for_auto: bool = False,
) -> dict[str, Any]:
    """Apply the best Diffusers attention backend for a Studio request.

    Diffusers backend names differ by version and model, so this function
    tries each mapped candidate and falls back to the next logical Studio
    backend. It never raises for an unavailable attention backend; callers
    get a status payload describing the effective choice.
    """

    normalized = normalize_attention_backend(requested)
    logical_order = attention_fallback_order(normalized)
    targets = _attention_targets(pipe)
    result: dict[str, Any] = {
        "requested": normalized,
        "effective": None,
        "diffusers_backend": None,
        "fallback_order": list(logical_order),
        "targets": [name for name, _ in targets],
        "applied": False,
        "warnings": [],
        "errors": [],
    }

    if normalized == "auto" and prefer_default_for_auto:
        _reset_attention_targets(targets)
        result["effective"] = "default"
        result["diffusers_backend"] = "default"
        result["applied"] = False
        result["warnings"].append(
            "Using the pipeline default attention backend for this model family."
        )
        return result

    def add_error(logical: str, candidate: str, exc: BaseException) -> None:
        result["errors"].append(
            {
                "logical_backend": logical,
                "diffusers_backend": candidate,
                "error": str(exc)[:500],
            }
        )

    for logical_backend in logical_order:
        if logical_backend == "default":
            _reset_attention_targets(targets)
            result["effective"] = "default"
            result["diffusers_backend"] = "default"
            result["applied"] = False
            if normalized != "auto":
                result["warnings"].append(
                    f"Requested attention backend {normalized!r} was unavailable; "
                    "using the pipeline default."
                )
            return result

        if logical_backend == "xformers" and not targets:
            try:
                if _try_pipeline_xformers(pipe):
                    result["effective"] = "xformers"
                    result["diffusers_backend"] = (
                        "enable_xformers_memory_efficient_attention"
                    )
                    result["applied"] = True
                    return result
            except Exception as exc:
                add_error(
                    logical_backend, "enable_xformers_memory_efficient_attention", exc
                )

        if not targets:
            continue

        for candidate in DIFFUSERS_ATTENTION_BACKEND_CANDIDATES.get(
            logical_backend, ()
        ):
            changed: list[tuple[str, Any]] = []
            try:
                for _, target in targets:
                    setter = getattr(target, "set_attention_backend", None)
                    if not callable(setter):
                        continue
                    setter(candidate)
                    changed.append(("", target))
                if changed:
                    result["effective"] = logical_backend
                    result["diffusers_backend"] = candidate
                    result["applied"] = True
                    return result
            except Exception as exc:
                add_error(logical_backend, candidate, exc)
                _reset_attention_targets(changed or targets)

        if logical_backend == "xformers":
            try:
                if _try_pipeline_xformers(pipe):
                    result["effective"] = "xformers"
                    result["diffusers_backend"] = (
                        "enable_xformers_memory_efficient_attention"
                    )
                    result["applied"] = True
                    return result
            except Exception as exc:
                add_error(
                    logical_backend, "enable_xformers_memory_efficient_attention", exc
                )

    _reset_attention_targets(targets)
    result["effective"] = "default"
    result["diffusers_backend"] = "default"
    result["applied"] = False
    if normalized != "auto":
        result["warnings"].append(
            f"Requested attention backend {normalized!r} was unavailable; "
            "using the pipeline default."
        )
    return result
