# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Canonical llama.cpp backend vocabulary, precedence, and marker readers."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional

# The standalone installer imports this map, so planning and marker reads share
# one vocabulary instead of relying on mirrored definitions.
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


def normalize_backend_request(value: Any) -> Optional[str]:
    """Return a canonical user-selectable backend, excluding actual-only Metal."""
    backend = normalize_backend(value)
    return backend if backend in REQUESTABLE_BACKENDS else None


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
    """Resolve the public selector over the legacy Vulkan boolean.

    A recognized public value is authoritative, including ``auto``. Unknown or
    absent public values leave the legacy flag in effect for compatibility.
    """
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


def marker_satisfies_backend_option(
    marker: Optional[Mapping[str, Any]], backend_request: Any, option: Optional[Mapping[str, Any]]
) -> bool:
    """Whether a marker is one of the installable outcomes for an option."""
    request = normalize_backend_request(backend_request)
    if not marker or request is None or not option or not option.get("available"):
        return False
    resolved_backend = normalize_backend(option.get("resolved_backend"))
    if resolved_backend is None:
        return False
    if marker_backend_request(marker) != request or marker_backend(marker) != resolved_backend:
        return False

    option_assets = option.get("acceptable_assets")
    if not isinstance(option_assets, list):
        option_assets = []
    acceptable_assets = {asset for asset in option_assets if isinstance(asset, str) and asset}
    asset = option.get("asset")
    if isinstance(asset, str) and asset:
        acceptable_assets.add(asset)
    return marker.get("asset") in acceptable_assets


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
    recorded = normalize_backend_request(marker.get("backend_request"))
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
        return normalize_backend_request(marker.get("backend_request")) not in ("auto",)
    if bool(marker.get("force_cpu")):
        return True
    legacy = marker.get("llama_backend")
    if legacy in (None, "", "auto"):
        return False
    return True


def marker_install_identity(marker: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Stable identity for the exact installed llama.cpp runtime bundle."""
    if not marker:
        return None
    fingerprint = marker.get("install_fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        return f"fingerprint:{fingerprint}"
    asset = marker.get("asset")
    if not isinstance(asset, str) or not asset:
        return None
    fields = (
        "published_repo",
        "release_tag",
        "asset",
        "asset_sha256",
        "binary_repo",
        "binary_release_tag",
        "source_asset",
        "source_sha256",
        "runtime_line",
        "bundle_profile",
        "coverage_class",
        "ggml_tree",
        "backend",
    )
    payload = {field: marker.get(field) for field in fields}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys = True, separators = (",", ":")).encode("utf-8")
    ).hexdigest()
    return f"legacy:{digest}"
