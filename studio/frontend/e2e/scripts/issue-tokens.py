#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
# ensure_default_admin seeds the account with must_change_password, and
# get_current_subject answers 403 "Password change required" for any token
# without the desktop claim, so a plain token cannot reach a single protected
# route on a fresh E2E install. Mint with the desktop exemption instead.
print(
    json.dumps(
        {
            "access_token": create_access_token(subject, desktop = True),
            "refresh_token": create_refresh_token(subject, desktop = True),
        }
    )
)
