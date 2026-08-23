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

    `fits_alone` is False only when the turn's own rendered size is at or over the
    CONTEXT WINDOW, which is the only evidence that it cannot be sent at all.
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
    # Both numbers price a whole rendered PROMPT, so both carry the same floor: the
    # template's wrapper and, on a tool-enabled request, the entire tool catalogue
    # rendered into the system turn. Left in, that floor swamps the comparison -- a
    # 6,000-token MCP catalogue makes a 20-token "hi" 97% of the irreducible prompt, and
    # this would tell the user to send that "hi" in smaller pieces when the only thing
    # that would help is advertising fewer tools. Off BOTH sides, so what is compared is
    # what the turn contributed against what the rest of the conversation contributed.
    shared = _int(refusal.get("shared_prompt_tokens"))
    shared = max(0, min(shared, latest_turn - 1, irreducible - 1))
    latest_turn -= shared
    irreducible -= shared
    if latest_turn < _TURN_DOMINATES * irreducible:
        return None
    # The WINDOW, not the fit's `prompt_target`. The hard wordings below say the turn is
    # more than this context window can hold, and `prompt_target` is the window minus the
    # reply reserved out of it (`prompt_budget`, up to a quarter of the window), so a turn
    # between the two is refused by the fit and would still have been SERVED on its own:
    # llama-server admits a prompt on its size alone, with nothing reserved for the reply
    # ("if (slot.task->n_tokens() >= slot.n_ctx)" in tools/server/server-context.cpp, the
    # same check whose text this message rewrites), and stops the reply at the wall
    # instead. Claiming the window cannot hold such a turn would be false, so that band
    # keeps the softer "most of this prompt is ..." wording, which is true of it.
    #
    # `>=`, not `>`, to match that check: a turn exactly the size of the window is
    # refused by llama-server too.
    #
    # The turn WITHOUT the shared floor, against the whole window: the hard wording is a
    # claim about the turn's own size, and a turn that clears the window only once a tool
    # catalogue is standing beside it is not a turn the window cannot hold. Halving such a
    # turn does make it fit, so it keeps the soft wording, whose lever still works.
    window = recorded_context or context_tokens
    # Not defaulted to "user": a role we cannot name is a turn we cannot give advice
    # about, and `describe_oversize` falls back to the generic wording for it.
    role = str(refusal.get("latest_turn_role") or "")
    return role, not (window and latest_turn >= window)


# Per role: what to call the oversized turn when it merely dominates the prompt, what to
# call it when it does not fit at all, and the one lever besides the window that is worth
# offering. The lever is the whole point of splitting by role -- "send it in smaller
# pieces" is sound advice for a pasted message and useless for the other three, which the
# user did not type and mostly cannot shorten by splitting.
_ROLE_ADVICE = {
    "user": (
        "Most of this prompt is the message just sent",
        "The message just sent does not fit on its own",
        "send it in smaller pieces",
    ),
    "tool": (
        "Most of this prompt is a single tool result",
        "A tool returned more than this context window can hold",
        "ask for a smaller slice of the file or page",
    ),
    # The reply being resumed after it hit Max Tokens. Splitting it is not a thing the
    # user can do, and there is nothing to shorten: the partial is what it is.
    "assistant": (
        "Most of this prompt is the reply being continued",
        "The reply being continued is already too long for this window",
        "start a new reply",
    ),
    # System and developer turns survive eviction by construction, so splitting one
    # across several messages preserves the total and changes nothing.
    "system": (
        "Most of this prompt is the system instructions",
        "The system instructions do not fit on their own",
        "shorten the system prompt",
    ),
}
_ROLE_ADVICE["function"] = _ROLE_ADVICE["tool"]
_ROLE_ADVICE["developer"] = _ROLE_ADVICE["system"]


def describe_oversize(request_tokens: int, context_tokens: int) -> str:
    """The user-facing message for a prompt that exceeds the loaded context window.

    The advice splits on the only two things that change what the user can do: whose
    turn is the bulk of the prompt, and whether that turn is merely most of the prompt
    or actually too big to send at all. An unrecognised role falls back to the generic
    wording rather than blaming a turn it cannot describe.
    """
    head = (
        f"Message too long: {request_tokens} tokens exceeds the "
        f"{context_tokens}-token context window. "
    )
    blamed = _blame_latest_turn(context_tokens)
    advice = _ROLE_ADVICE.get(blamed[0]) if blamed else None
    if advice is None:
        return (
            head + "Try increasing the Context Length in Model settings, or shorten the "
            "conversation."
        )
    dominant_cause, oversize_cause, lever = advice
    fits_alone = blamed[1]
    cause = dominant_cause if fits_alone else oversize_cause
    hedge = "will not help much" if fits_alone else "will not help"
    return (
        f"{head}{cause}, so shortening the conversation {hedge}. Increase the Context "
        f"Length in Model settings, or {lever}."
    )
