# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The auto-switch resolver index is one account's answer: its roots (that account's scan folders
and HF cache home) are private, so a process-global snapshot hands the next account the previous
one's checkpoint path and the load then refuses it as foreign."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.inference as inference
from auth import policy
from core.inference import local_model_resolver as resolver
from hub.services.models import account_access as access
from storage import studio_db
from utils import openai_auto_switch_settings as switch_settings
from utils.account_context import OWNER, AccountContext, arun_as, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(inference, "_managed_catalogs", {})
    monkeypatch.setattr(switch_settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(switch_settings, "get_openai_auto_download_enabled", lambda: False)
    monkeypatch.setattr(switch_settings, "get_model_override", lambda *a, **k: None)
    resolver.invalidate_index()
    yield tmp_path
    resolver.invalidate_index()


def _private_gguf(
    account,
    home,
    filename = "shared-model.gguf",
):
    """A checkpoint in a scan folder registered in *account*'s own database."""
    root = home / "accounts" / account.account_id / "private_models"
    root.mkdir(parents = True, exist_ok = True)
    model = root / filename
    model.write_bytes(b"GGUF" + b"\0" * 64)

    def _register():
        connection = studio_db.get_connection()
        with connection:
            connection.execute(
                "INSERT INTO scan_folders (path, created_at) VALUES (?, datetime('now'))",
                (str(root),),
            )
        connection.close()

    run_as(account, _register)
    return model


def _request():
    return SimpleNamespace(
        scope = {},
        state = SimpleNamespace(),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        headers = {},
    )


def test_a_warm_resolver_index_does_not_answer_for_the_next_account(home):
    alice_model = _private_gguf(ALICE, home)
    bob_model = _private_gguf(BOB, home)

    assert run_as(ALICE, resolver.resolve_local_gguf, "shared-model")[0] == str(alice_model)
    assert run_as(BOB, resolver.resolve_local_gguf, "shared-model")[0] == str(bob_model)
    # And back again: neither account's snapshot is displaced by the other's.
    assert run_as(ALICE, resolver.resolve_local_gguf, "shared-model")[0] == str(alice_model)


def test_an_alias_another_account_warmed_still_switches_to_the_callers_own_checkpoint(
    home, monkeypatch
):
    alice_model = _private_gguf(ALICE, home)
    bob_model = _private_gguf(BOB, home)
    loaded = []

    async def _load(request, *args, **kwargs):
        # The real impl refuses a path the caller cannot see, which is the 404 under test.
        access.require_model_access(request.model_path)
        loaded.append(request.model_path)
        return {}

    monkeypatch.setattr(inference, "_load_model_impl", _load)

    asyncio.run(
        arun_as(ALICE, inference._maybe_auto_switch_model("shared-model", _request(), "alice"))
    )
    assert loaded == [str(alice_model)]
    asyncio.run(arun_as(BOB, inference._maybe_auto_switch_model("shared-model", _request(), "bob")))
    assert loaded[-1] == str(bob_model)


def test_the_owner_keeps_one_index_on_a_single_user_install(home, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    owner_model = _private_gguf(OWNER, home, "owner-model.gguf")
    scans = []
    real = resolver._build_index
    monkeypatch.setattr(resolver, "_build_index", lambda: (scans.append(1), real())[1])

    for _ in range(3):
        assert resolver.resolve_local_gguf("owner-model")[0] == str(owner_model)
    assert len(scans) == 1
