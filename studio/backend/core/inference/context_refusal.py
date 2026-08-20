# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Why a prompt did not fit, carried from the context fit to the message the user reads.

The fit knows the SHAPE of a refusal: how much of the prompt is the single turn just
sent, and how much is the floor that eviction could not reduce. The message the user
finally sees is built much later, in `_friendly_error`, out of llama-server's own error
text, which knows only a total. So a two-message thread whose one turn is oversized is
told to "shorten the conversation", which is not advice it can act on.

Threading the diagnosis through `_friendly_error`'s forty-odd call sites would be a
worse cure than the disease, so it rides the request instead. A ContextVar is per-task
and asyncio copies the context per request, so one request's refusal cannot describe
another's.
"""

from contextvars import ContextVar
from typing import Optional

__all__ = [
    "record_fit",
    "clear",
    "latest_refusal",
    "describe_oversize",
]


_LATEST_REFUSAL: ContextVar[Optional[dict]] = ContextVar(
    "unsloth_context_refusal", default = None
)

# What share of the irreducible prompt the latest turn has to be before the turn, rather
# than the conversation, is named as the problem. Never all of it: the system prompt and
# the template wrapper are in the floor too, and on a small window they are not a
# rounding error. Two thirds is comfortably above a normal turn's share of a long thread
# and comfortably below a thread whose single turn IS the thread.
_TURN_DOMINATES = 0.66


def record_fit(truncation) -> None:
    """Remember a fit that refused, and forget one that succeeded.

    Called on every `context_truncated` event, not only the refusals, so that a tool loop
    whose later iteration fits does not leave the earlier refusal behind to explain an
    unrelated error.
    """
    if not isinstance(truncation, dict):
        return
    if truncation.get("fits"):
        _LATEST_REFUSAL.set(None)
        return
    _LATEST_REFUSAL.set(dict(truncation))


def clear() -> None:
    _LATEST_REFUSAL.set(None)


def latest_refusal() -> Optional[dict]:
    """The most recent fit on this request that could not fit, if there was one."""
    return _LATEST_REFUSAL.get()


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _oversized_turn_role(context_tokens: int) -> Optional[str]:
    """The role of the turn that is too big on its own, or None if the history is.

    `None` also covers "no diagnosis was recorded" and "the diagnosis describes a
    different window than the one the server just refused" -- both mean fall back to the
    generic advice rather than guess.
    """
    refusal = latest_refusal()
    if not refusal:
        return None
    recorded_context = _int(refusal.get("context_length"))
    if context_tokens and recorded_context and recorded_context != context_tokens:
        # A different load, or a different backend. It cannot describe this refusal.
        return None
    irreducible = _int(refusal.get("irreducible_tokens"))
    latest_turn = _int(refusal.get("latest_turn_tokens"))
    if irreducible <= 0 or latest_turn <= 0:
        return None
    if latest_turn < _TURN_DOMINATES * irreducible:
        return None
    return str(refusal.get("latest_turn_role") or "") or "user"


def describe_oversize(request_tokens: int, context_tokens: int) -> str:
    """The user-facing message for a prompt that exceeds the loaded context window.

    Three shapes, because there are three different things the user can do:

    * the conversation is long -- shorten it, or raise the window.
    * the message just sent is too big by itself -- shortening the chat does nothing.
    * a tool returned more than the window holds -- the user did not write it and cannot
      shorten it, so the only lever is the window (or a narrower tool call).
    """
    head = (
        f"Message too long: {request_tokens} tokens exceeds the "
        f"{context_tokens}-token context window. "
    )
    role = _oversized_turn_role(context_tokens)
    if role in ("tool", "function"):
        return (
            head + "A tool returned more than this context window can hold, so shortening "
            "the conversation will not help. Increase the Context Length in Model "
            "settings, or ask for a smaller slice of the file or page."
        )
    if role is not None:
        return (
            head + "The message just sent does not fit on its own, so shortening the "
            "conversation will not help. Increase the Context Length in Model settings, "
            "or send it in smaller pieces."
        )
    return (
        head + "Try increasing the Context Length in Model settings, or shorten the "
        "conversation."
    )
