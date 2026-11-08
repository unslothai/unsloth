# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which model loads a request may cause, decided by the credential that authenticated it."""

from __future__ import annotations

from enum import Enum


class LoadPolicy(str, Enum):
    ANY = "any"
    # Only a load that cannot unload the model serving chat (one beside it).
    BESIDE_ONLY = "beside_only"
    NEVER = "never"


def policy_for_credential(*, internal: bool, key_name: object) -> LoadPolicy:
    """The chat's own tools key acts inside a chat, so it must never unload the model answering it."""
    from auth.storage import STUDIO_CHAT_TOOLS_API_KEY_NAME, STUDIO_CHAT_TOOLS_KEY_NAME

    if internal and key_name in (STUDIO_CHAT_TOOLS_KEY_NAME, STUDIO_CHAT_TOOLS_API_KEY_NAME):
        return LoadPolicy.BESIDE_ONLY
    return LoadPolicy.ANY


def owner_chat_loads_beside() -> bool:
    """The owner's own chat may load beside its model without "Switch model by request": this key
    can never unload the chat model, and only a signed-in owner session (not an API key) mints it."""
    from utils.account_context import current_account, is_owner_context
    return (
        current_load_policy() is LoadPolicy.BESIDE_ONLY
        and current_account().ui_chat
        and is_owner_context()
    )


# Chat tools keys whose calling chat is in the keep-warm count, so a switch may discount exactly that chat.
_counted_parents: set[int] = set()


def note_counted_parent(key_id: int) -> None:
    _counted_parents.add(key_id)


def forget_counted_parent(key_id: int) -> None:
    _counted_parents.discard(key_id)


def parent_chat_counted() -> bool:
    from utils.account_context import current_account
    account = current_account()
    return account.key_id in _counted_parents and current_load_policy() is LoadPolicy.BESIDE_ONLY


def current_load_policy() -> LoadPolicy:
    from utils.account_context import current_account
    try:
        return LoadPolicy(current_account().load_policy)
    except ValueError:
        # An unknown value is the strictest reading, never a wider one.
        return LoadPolicy.NEVER
