# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account converts to Q4NX only from GGUFs inside its own workspace."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from models import ConvertQ4NXRequest
from routes import export as export_routes
from utils.account_context import OWNER, AccountContext, arun_as
from utils.paths import workspace_root

ALICE = AccountContext("alice-q4nx-account", "alice")
BOB = AccountContext("bob-q4nx-account", "bob")


def _convert(account, gguf_path):
    request = ConvertQ4NXRequest(
        save_directory = "q4nx",
        gguf_path = str(gguf_path),
        base_model = "unsloth/Qwen3-0.6B",
        hf_token = "own-token",
    )
    return asyncio.run(
        arun_as(account, export_routes.convert_q4nx(request, "subject", allow_ambient = False))
    )


@pytest.fixture
def converted(monkeypatch):
    calls = []

    async def to_thread(fn, *args, **kwargs):
        calls.append(fn)
        raise RuntimeError("stop before converting")

    monkeypatch.setattr(export_routes.asyncio, "to_thread", to_thread)
    return calls


def test_managed_account_cannot_convert_another_accounts_gguf(converted):
    from utils.account_context import run_as

    foreign = run_as(BOB, workspace_root) / "exports" / "private.Q4_1.gguf"
    with pytest.raises(HTTPException) as exc:
        _convert(ALICE, foreign)
    assert exc.value.status_code == 403
    assert converted == []


def test_managed_account_and_owner_reach_the_converter_for_allowed_paths(converted):
    from utils.account_context import run_as

    own = run_as(ALICE, workspace_root) / "exports" / "mine.Q4_1.gguf"
    for account, path in ((ALICE, own), (OWNER, "/anywhere/model.Q4_1.gguf")):
        with pytest.raises(HTTPException) as exc:
            _convert(account, path)
        assert exc.value.status_code == 400 and "stop before converting" in exc.value.detail
    assert len(converted) == 2
