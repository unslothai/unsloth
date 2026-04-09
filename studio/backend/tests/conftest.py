# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Shared pytest configuration for the backend test suite.
Ensures that the backend root is on sys.path so that
`import utils.utils` (and similar flat imports) resolve correctly.
"""

import sys
from pathlib import Path

# Add backend root to sys.path (mirrors how the app itself is launched)
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))
