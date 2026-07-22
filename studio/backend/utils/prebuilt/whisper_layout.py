# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Canonical whisper.cpp install-root and marker lookup helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MARKER_NAME = "UNSLOTH_WHISPER_PREBUILT_INFO.json"


@dataclass(frozen=True)
class MarkerLookup:
    marker: Optional[dict]
    root: Optional[Path]
    authoritative: bool
    invalid: bool = False
    slim_collision: bool = False


def canonical_install_root(binary_path: Optional[str]) -> Optional[Path]:
    """Return the managed root for the two supported CMake binary layouts."""
    if not binary_path:
        return None
    parent = Path(binary_path).parent
    if parent.name == "bin" and parent.parent.name == "build":
        return parent.parent.parent
    if (
        parent.name == "Release"
        and parent.parent.name == "bin"
        and parent.parent.parent.name == "build"
    ):
        return parent.parent.parent.parent
    return None


def _parse_marker(path: Path) -> tuple[Optional[dict], bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, True
    return (payload, False) if isinstance(payload, dict) else (None, True)


def lookup_marker(binary_path: Optional[str]) -> MarkerLookup:
    """Prefer the install-root marker over archive metadata beside the binary."""
    if not binary_path:
        return MarkerLookup(None, None, False)
    binary = Path(binary_path)
    root = canonical_install_root(binary_path)
    inner_marker = binary.parent / MARKER_NAME
    inner_payload, _ = _parse_marker(inner_marker) if inner_marker.is_file() else (None, False)
    slim_collision = bool(
        inner_payload
        and (inner_payload.get("install_kind") == "slim" or inner_payload.get("backend") == "slim")
    )
    if root is not None:
        root_marker = root / MARKER_NAME
        if root_marker.is_file():
            marker, invalid = _parse_marker(root_marker)
            return MarkerLookup(marker, root, True, invalid, slim_collision)
        if slim_collision:
            return MarkerLookup(None, root, False, True, True)

    for parent in binary.parents[:5]:
        candidate = parent / MARKER_NAME
        if candidate.is_file():
            marker, invalid = _parse_marker(candidate)
            return MarkerLookup(marker, parent, False, invalid, slim_collision)
    return MarkerLookup(None, None, False)
