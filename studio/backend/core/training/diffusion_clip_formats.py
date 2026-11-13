# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video container extensions for clips, kept import-free so the trainer and upload route share it."""

from __future__ import annotations

# Lowercase, with the leading dot, so a Path.suffix.lower() can be tested against the set directly.
CLIP_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"})

__all__ = ["CLIP_EXTS"]
