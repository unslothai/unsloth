# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Export submodule - Model export operations

The default get_export_backend() returns an ExportOrchestrator that
delegates to a subprocess. The original ExportBackend runs inside
the subprocess and can be imported directly from .export when needed.
"""

from .orchestrator import ExportOrchestrator, get_export_backend

# Expose ExportOrchestrator as ExportBackend for backward compat
ExportBackend = ExportOrchestrator

__all__ = [
    "ExportBackend",
    "ExportOrchestrator",
    "get_export_backend",
]
