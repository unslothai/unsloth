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
