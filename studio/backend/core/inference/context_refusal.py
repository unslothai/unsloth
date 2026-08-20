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
    "open_slot",
]


# A one-key box, not the refusal itself. Both `asyncio.create_task` and
# `asyncio.to_thread` copy the context, and a `.set()` in a copy is invisible to the
# original -- which is where `_friendly_error` runs. Copies share VALUES, though, so a
# box installed before the copies is the same object in all of them. See `open_slot`.
_REFUSAL_SLOT: ContextVar[Optional[dict]] = ContextVar("unsloth_context_refusal", default = None)

# Share of the irreducible prompt the latest turn must be before the turn, not the
# conversation, is named as the problem. Never all of it: the system prompt and template
# wrapper are in the floor too. Two thirds sits above a normal turn's share of a long
# thread and below a thread whose single turn IS the thread.
#
# Dominating is NOT the same as not fitting. A 3400-token turn under a 4096-token prompt
# budget is two thirds of the floor and still fits on its own; what pushed the request
# over was the system prompt beside it. So dominance only earns the softer "most of this
# prompt is ..." wording, and the flat claim that the turn does not fit is made only when
# the turn alone exceeds the budget.
_TURN_DOMINATES = 0.66


def open_slot() -> None:
    """Install a slot here that a worker thread or child task can record into.

    Call it in the request's own context, before spawning anything, on any path that
    diagnoses the fit somewhere other than where the error is formatted. The
    non-streaming GGUF drains are that case twice over: `asyncio.create_task` copies the
    context and so does `asyncio.to_thread`, and on the path that matters the drain
    records the refusal and then raises the oversize error it explains, so there is no
    return value to carry it back in either.
    """
    _REFUSAL_SLOT.set({"refusal": None})


def _slot(*, create: bool = False) -> Optional[dict]:
    slot = _REFUSAL_SLOT.get()
    if slot is None and create:
        # No one opened one, so this context is where the message is built too.
        slot = {"refusal": None}
        _REFUSAL_SLOT.set(slot)
    return slot


def record_fit(truncation) -> None:
    """Remember a fit that refused, and forget one that succeeded.

    Called on every `context_truncated` event, not just refusals, so a tool loop whose
    later iteration fits does not leave a stale refusal behind to explain another error.
    """
    if not isinstance(truncation, dict):
        return
    slot = _slot(create = True)
    slot["refusal"] = None if truncation.get("fits") else dict(truncation)


def clear() -> None:
    slot = _slot()
    if slot is not None:
        # Empty the shared slot as well as dropping it, so anything already holding a
        # reference to it (a worker mid-flight) does not read a stale refusal back.
        slot["refusal"] = None
    _REFUSAL_SLOT.set(None)


def latest_refusal() -> Optional[dict]:
    """The most recent fit on this request that could not fit, if there was one."""
    slot = _slot()
    return slot["refusal"] if slot else None


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _blame_latest_turn(context_tokens: int):
    """`(role, fits_alone)` for the turn worth naming, or None if the history is to blame.

    None also covers no diagnosis recorded, and a diagnosis describing a different
    window than the one just refused: both fall back to generic advice rather than guess.

    `fits_alone` is False only when the turn's own rendered size is over the budget the
    prompt had to fit in, which is the only evidence that it cannot be sent at all.
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
    # The prompt's real budget is the window minus the reply reserved out of it. Fall
    # back to the window itself, which is the same test one reservation looser.
    budget = _int(refusal.get("prompt_target")) or recorded_context or context_tokens
    role = str(refusal.get("latest_turn_role") or "") or "user"
    return role, not (budget and latest_turn > budget)


def describe_oversize(request_tokens: int, context_tokens: int) -> str:
    """The user-facing message for a prompt that exceeds the loaded context window.

    The advice splits on the only two things that change what the user can do: whose
    turn is the bulk of the prompt (the user can rewrite their own message; a tool
    result they cannot), and whether that turn is merely most of the prompt or actually
    too big to send at all.
    """
    head = (
        f"Message too long: {request_tokens} tokens exceeds the "
        f"{context_tokens}-token context window. "
    )
    blamed = _blame_latest_turn(context_tokens)
    if blamed is None:
        return (
            head + "Try increasing the Context Length in Model settings, or shorten the "
            "conversation."
        )
    role, fits_alone = blamed
    if role in ("tool", "function"):
        cause = (
            "Most of this prompt is a single tool result"
            if fits_alone
            else "A tool returned more than this context window can hold"
        )
        lever = "ask for a smaller slice of the file or page"
    else:
        cause = (
            "Most of this prompt is the message just sent"
            if fits_alone
            else "The message just sent does not fit on its own"
        )
        lever = "send it in smaller pieces"
    hedge = "will not help much" if fits_alone else "will not help"
    return (
        f"{head}{cause}, so shortening the conversation {hedge}. Increase the Context "
        f"Length in Model settings, or {lever}."
    )
