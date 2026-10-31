# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Revealing a chat's sandbox folder drives the host's file manager, so only the owner may."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import routes.inference as inf  # noqa: E402
import utils.paths.path_utils as path_utils  # noqa: E402
from hub.services.models import account_access  # noqa: E402


def _wire(monkeypatch, tmp_path, managed):
    launched = []

    async def _auth(request, token):
        return "user"

    monkeypatch.setattr(inf, "_authenticate_header_or_query", _auth)
    monkeypatch.setattr(account_access, "managed_account", lambda: managed)
    monkeypatch.setattr(inf, "_sandbox_dir_for", lambda s, create = False: str(tmp_path))
    monkeypatch.setattr(path_utils, "reveal_in_file_manager", lambda p, **k: launched.append(p))
    return launched


def test_a_managed_account_cannot_open_the_folder_on_the_owners_desktop(tmp_path, monkeypatch):
    launched = _wire(monkeypatch, tmp_path, managed = True)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(inf.reveal_sandbox_dir("s1", request = None))
    assert refused.value.status_code == 403
    assert launched == []


def test_the_owner_still_opens_the_folder(tmp_path, monkeypatch):
    launched = _wire(monkeypatch, tmp_path, managed = False)
    assert asyncio.run(inf.reveal_sandbox_dir("s1", request = None))["status"] == "ok"
    assert len(launched) == 1
