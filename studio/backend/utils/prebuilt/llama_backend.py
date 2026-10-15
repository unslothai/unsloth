# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Canonical llama.cpp backend vocabulary, precedence, and marker readers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

# The standalone installer imports this map, so both share one vocabulary.
INSTALL_KIND_BACKENDS: dict[str, str] = {
    "linux-cuda": "cuda",
    "linux-arm64-cuda": "cuda",
    "windows-cuda": "cuda",
    # Distinct from windows-cuda: the arm64 bundle ships its own ARM64 cudart/cublas.
    "windows-arm64-cuda": "cuda",
    "linux-rocm": "rocm",
    "windows-hip": "rocm",
    "windows-rocm": "rocm",
    "linux-vulkan": "vulkan",
    "windows-vulkan": "vulkan",
    "linux-cpu": "cpu",
    "linux-arm64": "cpu",
    "windows-cpu": "cpu",
    "windows-arm64": "cpu",
    "macos-arm64": "metal",
    "macos-x64": "metal",
}

# metal is absent: it is the only macOS build, so there is nothing to choose.
REQUESTABLE_BACKENDS = ("auto", "cpu", "cuda", "rocm", "vulkan")

# Longest-token-first, so "cuda13-older" cannot read as something else.
_ASSET_BACKEND_TOKENS = (
    ("vulkan", "vulkan"),
    ("cuda", "cuda"),
    ("rocm", "rocm"),
    ("hip", "rocm"),
    ("macos", "metal"),
    ("cpu", "cpu"),
)


def normalize_backend(value: Any) -> Optional[str]:
    """Return a canonical backend name, or None for an unknown value."""
    if not isinstance(value, str) or not value.strip():
        return None
    backend = value.strip().lower()
    if backend == "hip":
        return "rocm"
    if backend in REQUESTABLE_BACKENDS or backend == "metal":
        return backend
    return None


def normalize_backend_request(value: Any) -> Optional[str]:
    """Return a canonical user-selectable backend, excluding actual-only Metal."""
    backend = normalize_backend(value)
    return backend if backend in REQUESTABLE_BACKENDS else None


def is_requestable_backend(value: Any) -> bool:
    """Whether this build knows how to install ``value``."""
    return normalize_backend_request(value) is not None


def backend_for_install_kind(install_kind: Any) -> Optional[str]:
    if not isinstance(install_kind, str):
        return None
    return INSTALL_KIND_BACKENDS.get(install_kind.strip())


def install_kinds_for_backend(backend: Any) -> frozenset[str]:
    """Return every install kind that satisfies ``backend``."""
    normalized = normalize_backend(backend)
    if normalized is None:
        return frozenset()
    return frozenset(
        kind for kind, kind_backend in INSTALL_KIND_BACKENDS.items() if kind_backend == normalized
    )


def environment_backend_override(primary: Any, legacy_vulkan: Any) -> Optional[str]:
    """Public selector wins over the legacy Vulkan boolean when its value is recognized, auto included."""
    backend = normalize_backend_request(primary)
    if backend is not None:
        return backend
    if isinstance(legacy_vulkan, str) and legacy_vulkan.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return "vulkan"
    return None


def backend_from_asset_name(asset: Any) -> Optional[str]:
    """Infer the backend for a marker that predates the backend field."""
    if not isinstance(asset, str) or not asset:
        return None
    name = asset.lower()
    for token, backend in _ASSET_BACKEND_TOKENS:
        if token in name:
            return backend
    return None


def marker_backend(marker: Optional[Mapping[str, Any]]) -> Optional[str]:
    """What the install described by ``marker`` runs on, or None if unknowable."""
    if not marker:
        return None
    recorded = normalize_backend(marker.get("backend"))
    if recorded is not None:
        return recorded
    from_kind = backend_for_install_kind(marker.get("install_kind"))
    return from_kind if from_kind is not None else backend_from_asset_name(marker.get("asset"))


def marker_backend_request(marker: Optional[Mapping[str, Any]]) -> str:
    """Never None: auto means detect, and a value this build does not know is returned verbatim."""
    if not marker:
        return "auto"
    recorded = marker.get("backend_request")
    if isinstance(recorded, str) and recorded.strip():
        return normalize_backend_request(recorded) or recorded.strip().lower()
    if bool(marker.get("force_cpu")):
        return "cpu"
    if "llama_backend" not in marker and "vulkan" in str(marker.get("asset") or "").lower():
        # Old markers cannot distinguish chosen Vulkan from automatic Vulkan.
        return "vulkan"
    legacy = marker.get("llama_backend")
    if not isinstance(legacy, str) or legacy in ("", "auto"):
        return "auto"
    return normalize_backend_request(legacy) or legacy.strip().lower()


def marker_backend_was_chosen(marker: Optional[Mapping[str, Any]]) -> bool:
    """Chosen unless plainly automatic or its request went unsatisfied; not simply request != auto."""
    if not marker:
        return False
    if marker.get("backend_request_unsatisfied"):
        return False
    recorded = marker.get("backend_request")
    if isinstance(recorded, str) and recorded.strip():
        return recorded.strip().lower() != "auto"
    if bool(marker.get("force_cpu")):
        return True
    legacy = marker.get("llama_backend")
    return legacy is not None and legacy not in ("", "auto")
