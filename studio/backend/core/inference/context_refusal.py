# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Why a prompt did not fit, carried from the context fit to the message the user reads.

The fit knows the SHAPE of a refusal (how much is the turn just sent vs the floor
eviction could not reduce); `_friendly_error` builds the message much later from
llama-server's text, which knows only a total, and so tells a two-message thread to
"shorten the conversation". Threading the diagnosis through `_friendly_error`'s
forty-odd call sites would be worse than the disease, so it rides the request in a
ContextVar: per-task, and asyncio copies the context per request, so one request's
refusal cannot describe another's.
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

# Share of the irreducible prompt the latest turn must be before the turn, not the
# conversation, is named as the problem. Never all of it: the system prompt and template
# wrapper are in the floor too. Two thirds sits above a normal turn's share of a long
# thread and below a thread whose single turn IS the thread.
_TURN_DOMINATES = 0.66


def record_fit(truncation) -> None:
    """Remember a fit that refused, and forget one that succeeded.

    Called on every `context_truncated` event, not just refusals, so a tool loop whose
    later iteration fits does not leave a stale refusal behind to explain another error.
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
    """Role of the turn that is too big on its own, or None if the history is.

    None also covers no diagnosis recorded, and a diagnosis describing a different
    window than the one just refused: both fall back to generic advice rather than guess.
    """
    refusal = latest_refusal()
    if not refusal:
        return None
    recorded_context = _int(refusal.get("context_length"))
    if context_tokens and recorded_context and recorded_context != context_tokens:
        # A different load or backend: it cannot describe this refusal.
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

    * long conversation -- shorten it, or raise the window.
    * the message just sent is too big alone -- shortening the chat does nothing.
    * a tool returned more than the window holds -- the user did not write it, so the
      only lever is the window (or a narrower tool call).
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
