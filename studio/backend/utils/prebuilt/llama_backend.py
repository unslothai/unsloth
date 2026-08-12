# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read llama.cpp backend fields from an install marker."""

from __future__ import annotations

from typing import Any, Mapping, Optional

# install_kind -> accelerator. Mirrors install_llama_prebuilt.INSTALL_KIND_BACKENDS.
INSTALL_KIND_BACKENDS: dict[str, str] = {
    "linux-cuda": "cuda",
    "linux-arm64-cuda": "cuda",
    "windows-cuda": "cuda",
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

# Backends a user may ask for. "metal" is absent on purpose: it is the only macOS
# build, so there is nothing to choose.
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


def backend_for_install_kind(install_kind: Any) -> Optional[str]:
    if not isinstance(install_kind, str):
        return None
    return INSTALL_KIND_BACKENDS.get(install_kind.strip())


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


def marker_backend_request(marker: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Return the recorded choice, with legacy marker compatibility.

    None means the marker contains an unknown choice, not automatic detection.
    """
    if not marker:
        return "auto"
    recorded = normalize_backend(marker.get("backend_request"))
    if recorded is not None:
        return recorded
    if marker.get("backend_request") is not None:
        return None
    if bool(marker.get("force_cpu")):
        return "cpu"
    if "llama_backend" not in marker and "vulkan" in str(marker.get("asset") or "").lower():
        # Old markers cannot distinguish chosen Vulkan from automatic Vulkan.
        return "vulkan"
    legacy = marker.get("llama_backend")
    if legacy in (None, "", "auto"):
        return "auto"
    if legacy == "vulkan":
        return "vulkan"
    return None


def marker_backend_was_chosen(marker: Optional[Mapping[str, Any]]) -> bool:
    """Return whether the backend was chosen instead of detected.

    Legacy Vulkan markers count as detected. Unknown explicit choices count as
    chosen so this build does not undo them.
    """
    if not marker:
        return False
    if marker.get("backend_request") is not None:
        return normalize_backend(marker.get("backend_request")) not in ("auto",)
    if bool(marker.get("force_cpu")):
        return True
    legacy = marker.get("llama_backend")
    if legacy in (None, "", "auto"):
        return False
    return True
