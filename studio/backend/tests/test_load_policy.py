# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from pathlib import Path

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from auth import authentication, storage
from core.inference import load_policy as lp
from utils.account_context import AccountContext, run_as

from .mcp_harness import isolated_auth, owner_auth  # noqa: F401  (fixtures)

CHAT_KEY = storage.STUDIO_CHAT_TOOLS_KEY_NAME


# A user's own key cannot opt into, or out of, the chat's policy by its name.
@pytest.mark.parametrize(
    "name, internal, policy",
    [
        (CHAT_KEY, True, "beside_only"),
        (storage.STUDIO_CHAT_TOOLS_API_KEY_NAME, True, "beside_only"),
        (CHAT_KEY, False, "any"),
        ("Data recipe", True, "any"),
    ],
)
@pytest.mark.usefixtures("owner_auth")
def test_the_auth_layer_takes_the_policy_from_the_key(name, internal, policy):
    key, _ = storage.create_api_key(username = "unsloth", name = name, internal = internal)
    bearer = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = key)

    async def admit():
        await authentication._get_current_credential(bearer, allow_password_change = False)
        return lp.current_load_policy().value

    assert asyncio.run(admit()) == policy


# Only the owner's own UI chat loads beside without the switch; an unknown value reads as never.
@pytest.mark.parametrize(
    "role, value, ui_chat, beside, read",
    [
        ("owner", "beside_only", True, True, "beside_only"),
        ("owner", "beside_only", False, False, "beside_only"),
        ("user", "beside_only", True, False, "beside_only"),
        ("owner", "bogus", True, False, "never"),
    ],
)
def test_owner_chat_loads_beside(role, value, ui_chat, beside, read):
    account = AccountContext("a", "a", role, load_policy = value, ui_chat = ui_chat)
    assert run_as(account, lp.owner_chat_loads_beside) is beside
    assert run_as(account, lp.current_load_policy).value == read


def test_routes_and_media_switch_do_not_read_the_mcp_caller():
    for name in ("routes/inference.py", "routes/video.py", "core/inference/media_auto_switch.py"):
        assert "studio_mcp" not in (Path(__file__).parents[1] / name).read_text(
            encoding = "utf-8"
        ), name


# A chat an sk-unsloth key started mints the API-named key, so it never gets the owner's pass.
@pytest.mark.usefixtures("owner_auth")
def test_a_chat_started_by_an_api_key_needs_the_switch():
    from core.inference import studio_tools_mcp
    from utils.account_context import current_account

    async def chat_tools_pass(outer_key):
        await authentication._get_current_credential(
            HTTPAuthorizationCredentials(scheme = "Bearer", credentials = outer_key),
            allow_password_change = False,
        )
        inner_key, _ = studio_tools_mcp._mint(current_account(), 30)
        await authentication._get_current_credential(
            HTTPAuthorizationCredentials(scheme = "Bearer", credentials = inner_key),
            allow_password_change = False,
        )
        return lp.current_load_policy().value, lp.owner_chat_loads_beside()

    api_key, _ = storage.create_api_key(username = "unsloth", name = "my client")
    assert asyncio.run(chat_tools_pass(api_key)) == ("beside_only", False)
