# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account boundaries shared by model services and inference routes."""

from __future__ import annotations

import re

from utils.account_context import OWNER, AccountContext, current_account, is_owner_context

_ACCOUNT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def media_link_target(media_id: str) -> str:
    """Bind a bearer media link to its workspace without changing owner links."""
    if is_owner_context():
        return media_id
    return f"{current_account().account_id}:{media_id}"


def media_link_account(target: str | None, media_id: str) -> AccountContext | None:
    """Resolve an already signature-verified target, never an unsigned account selector."""
    if target == media_id:
        return OWNER
    if not target:
        return None
    account_id, sep, signed_id = target.partition(":")
    if not sep or signed_id != media_id or not _ACCOUNT_ID.fullmatch(account_id):
        return None
    if account_id == OWNER.account_id:
        return OWNER
    return AccountContext(account_id, "", "user")
