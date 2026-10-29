# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which model loads a request may cause, decided by the credential that authenticated it."""

from __future__ import annotations

from enum import Enum


class LoadPolicy(str, Enum):
    # Any load "Switch model by request" allows.
    ANY = "any"
    # Only a load that cannot unload the model serving chat (one beside it).
    BESIDE_ONLY = "beside_only"
    NEVER = "never"


def policy_for_credential(*, internal: bool, key_name: object) -> LoadPolicy:
    """The chat's own tools key acts inside a chat, so it must never unload the model answering it."""
    from auth.storage import STUDIO_CHAT_TOOLS_KEY_NAME

    if internal and key_name == STUDIO_CHAT_TOOLS_KEY_NAME:
        return LoadPolicy.BESIDE_ONLY
    return LoadPolicy.ANY


def current_load_policy() -> LoadPolicy:
    from utils.account_context import current_account
    try:
        return LoadPolicy(current_account().load_policy)
    except ValueError:
        # An unknown value is the strictest reading, never a wider one.
        return LoadPolicy.NEVER
