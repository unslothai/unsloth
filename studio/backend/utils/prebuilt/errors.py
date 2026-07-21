# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared exceptions for the prebuilt installers."""

from __future__ import annotations


class PrebuiltFallback(RuntimeError):
    """Recoverable failure -- caller should fall back to a source build (exit 2)."""


class BusyInstallConflict(RuntimeError):
    """Another process holds the install lock (exit 3)."""
