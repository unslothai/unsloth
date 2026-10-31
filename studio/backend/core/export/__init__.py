# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""get_export_backend() returns ExportOrchestrator; the real ExportBackend runs in a subprocess."""

from .orchestrator import ExportOrchestrator, get_export_backend

# Expose ExportOrchestrator as ExportBackend for backward compat
ExportBackend = ExportOrchestrator

__all__ = [
    "ExportBackend",
    "ExportOrchestrator",
    "get_export_backend",
]
