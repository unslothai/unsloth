# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared bootstrap for tests/studio: puts studio/backend on sys.path so the
regression tests can import the backend tree (core.*, routes.*, utils.*) from
the repo root. Mirrors studio/backend/tests/conftest.py."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# tests/studio/load_freeze/test_load_orchestrator.py seeds a bare ``structlog``
# stub via ``sys.modules.setdefault`` so its imports survive without the
# optional dep. That bare module has no ``get_logger``; when a later test
# imports the route chain (routes.inference -> core.inference.external_provider)
# the module-level ``structlog.get_logger(__name__)`` then raises
# AttributeError. This conftest loads before the load_freeze subpackage is
# collected, so seed a working ``structlog`` first (the real package when
# installed, otherwise a minimal logging-backed shim). The later setdefault
# becomes a no-op and the route imports resolve a usable ``get_logger``.
try:
    import structlog  # noqa: F401

    _structlog_usable = hasattr(structlog, "get_logger")
except Exception:
    _structlog_usable = False

if not _structlog_usable:
    import logging
    import types

    _structlog_stub = types.ModuleType("structlog")
    _structlog_stub.get_logger = lambda *args, **kwargs: logging.getLogger("structlog")
    sys.modules["structlog"] = _structlog_stub
