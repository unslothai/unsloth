#!/usr/bin/env python3
"""Emit JWT tokens for Playwright E2E auth injection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from auth import storage  # noqa: E402
from auth.authentication import create_access_token, create_refresh_token  # noqa: E402

storage.ensure_default_admin()
subject = storage.DEFAULT_ADMIN_USERNAME
print(
    json.dumps(
        {
            "access_token": create_access_token(subject),
            "refresh_token": create_refresh_token(subject),
        }
    )
)
