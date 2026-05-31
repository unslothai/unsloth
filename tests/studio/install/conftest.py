# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pytest configuration for studio/install tests.

install_python_stack.py does ``from backend.utils.wheel_utils import ...``
which requires the ``studio/`` directory to be on sys.path.  When tests are
run from the repo root (the normal case), the studio package is not
automatically importable, so we add it here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# <repo-root>/studio  →  makes `backend` importable as a package
_STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))
