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
    "ContextBudgetExceeded",
]


# Copies of the context share VALUES, so the slot is a mutable box. See `open_slot`.
_REFUSAL_SLOT: ContextVar[Optional[dict]] = ContextVar("unsloth_context_refusal", default = None)

# Share of the irreducible prompt the latest turn must reach to be blamed (soft wording only).
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

    `fits_alone` is False only when the turn's own COUNTED rendered size is at or over
    the CONTEXT WINDOW, which is the only evidence that it cannot be sent at all.
    """
    refusal = latest_refusal()
    if not refusal:
        return None
    recorded_context = _int(refusal.get("context_length"))
    if context_tokens and recorded_context and recorded_context != context_tokens:
        return None
    irreducible = _int(refusal.get("irreducible_tokens"))
    latest_turn = _int(refusal.get("latest_turn_tokens"))
    if irreducible <= 0 or latest_turn <= 0:
        return None
    # Only a COUNTED turn compares with `irreducible_tokens`: the char estimate can be off 15x either way.
    # Absent flag means an older producer, which always counted.
    exact = bool(refusal.get("latest_turn_exact", True))
    if not exact:
        return None
    # Subtract the shared floor (template wrapper, tool catalogue) from both sides, or it swamps the ratio.
    shared = _int(refusal.get("shared_prompt_tokens"))
    shared = max(0, min(shared, latest_turn - 1, irreducible - 1))
    latest_turn -= shared
    irreducible -= shared
    if latest_turn < _TURN_DOMINATES * irreducible:
        return None
    # The WINDOW, not `prompt_target`: llama-server admits on "n_tokens() >= n_ctx" alone.
    window = recorded_context or context_tokens
    role = str(refusal.get("latest_turn_role") or "")
    return role, not (window and latest_turn >= window)


def _history_cannot_help(context_tokens: int) -> bool:
    """True when the prompt is over the window with every evictable turn already gone.

    `irreducible_tokens` is not "the prompt": it is what the fit measured AFTER dropping
    every group `truncate_oldest_messages` is willing to drop, and a refusal is only ever
    recorded once that evictor returned zero (the fit's loop exits on `dropped == 0`, and
    any other exit means the prompt fits). So it prices the floor eviction cannot go
    below: the template wrapper, the tool catalogue, every system/developer turn, the
    latest user turn and the final group. Deleting ordinary history changes none of those,
    which is why this number is invariant under the one action the generic advice asks for.

    Against the WINDOW for the same reason `_blame_latest_turn` uses it: llama-server
    admits a prompt on size alone ("n_tokens() >= n_ctx"), so at or over it the request is
    refused no matter how short the conversation gets. Below it, shortening really can
    work -- the fit refuses at `prompt_target` but passes the untrimmed messages on, and
    llama-server serves anything under `n_ctx` -- so that case keeps the generic advice.
    """
    refusal = latest_refusal()
    if not refusal:
        return False
    recorded_context = _int(refusal.get("context_length"))
    if context_tokens and recorded_context and recorded_context != context_tokens:
        return False
    irreducible = _int(refusal.get("irreducible_tokens"))
    window = recorded_context or context_tokens
    return irreducible > 0 and window > 0 and irreducible >= window


# Split by role because the lever differs: the user cannot split turns they did not type.
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
    # `edit_file` with empty `old_string` is whole-file creation, so the content IS the argument.
    "assistant_tool_call": (
        "Most of this prompt is the file the model passed to a tool",
        "The file the model passed to a tool does not fit on its own",
        "ask for a smaller file, or raise the Context Length before retrying",
    ),
    "assistant_tool_payload": (
        "Most of this prompt is what the model passed to a tool",
        "What the model passed to a tool does not fit on its own",
        "ask for less in one call, or raise the Context Length before retrying",
    ),
    "assistant": (
        "Most of this prompt is the reply being continued",
        "The reply being continued is already too long for this window",
        "start a new reply",
    ),
    "system": (
        "Most of this prompt is the system instructions",
        "The system instructions do not fit on their own",
        "shorten the system prompt",
    ),
}
_ROLE_ADVICE["function"] = _ROLE_ADVICE["tool"]
_ROLE_ADVICE["developer"] = _ROLE_ADVICE["system"]


def oversize_advice(context_tokens: int) -> str:
    """The remedy half of an oversize refusal: what the user can actually do.

    Split out from :func:`describe_oversize` so a surface that must keep its own head
    wording -- the Anthropic passthrough sends Anthropic's "Prompt is too long: N
    tokens > M maximum", which is what its clients key on -- can still pair it with
    this diagnosis instead of prescribing compaction for a prompt no compaction fits.
    """
    blamed = _blame_latest_turn(context_tokens)
    advice = _ROLE_ADVICE.get(blamed[0]) if blamed else None
    if advice is None:
        if _history_cannot_help(context_tokens):
            # What survives eviction is already over the window, so "shorten the conversation" cannot work.
            return (
                "Even with every earlier turn dropped, this prompt would still be "
                "too long, so shortening the conversation will not help. Increase the "
                "Context Length in Model settings, or reduce what every request carries: "
                "the system prompt and any tools that are enabled."
            )
        return "Try increasing the Context Length in Model settings, or shorten the conversation."
    dominant_cause, oversize_cause, lever = advice
    fits_alone = blamed[1]
    cause = dominant_cause if fits_alone else oversize_cause
    hedge = "will not help much" if fits_alone else "will not help"
    return (
        f"{cause}, so shortening the conversation {hedge}. Increase the Context "
        f"Length in Model settings, or {lever}."
    )


def describe_oversize(request_tokens: int, context_tokens: int) -> str:
    """The user-facing message for a prompt that exceeds the loaded context window.

    The advice splits on the only two things that change what the user can do: whose
    turn is the bulk of the prompt, and whether that turn is merely most of the prompt
    or actually too big to send at all. An unrecognised role falls back to the generic
    wording rather than blaming a turn it cannot describe.
    """
    return (
        f"Message too long: {request_tokens} tokens exceeds the "
        f"{context_tokens}-token context window. "
    ) + oversize_advice(context_tokens)


_TOOL_LEVERS = {
    "edit_file": "ask for a smaller file",
    "python": "run a shorter program",
    "terminal": "run a shorter command",
    "render_html": "render a smaller page",
    "web_search": "ask a narrower question",
    "search_knowledge_base": "ask a narrower question",
    "search_conversation": "ask a narrower question",
}


def describe_unservable_tool_call(
    tool_name: str,
    request_tokens: int,
    context_tokens: int,
    *,
    compacted_calls: int = 0,
) -> str:
    """The message for a tool call refused BEFORE it ran, because its turn cannot be served.

    `describe_oversize` reconstructs blame from a recorded diagnosis, because by the time it
    speaks the request has already been rejected and the cause has to be inferred. This one
    is said by the loop that is holding the call, so it names the tool outright instead of
    guessing at a role, and it is the only refusal on this path that can promise nothing was
    written -- which is the fact the user most needs and the 400 could never offer.

    ``compacted_calls`` is reported when history was already spent trying to make room, so
    "increase the Context Length" does not read as advice nobody tried.
    """
    # The bar is window minus a reply floor, so explain the gap or the numbers look contradictory.
    head = (
        f"Not enough context left to run {tool_name}: the next request would be about "
        f"{request_tokens} tokens of a {context_tokens}-token window, leaving no room to "
        "reply. "
    )
    tried = ""
    if compacted_calls > 0:
        calls = "call" if compacted_calls == 1 else "calls"
        tried = (
            f"Arguments from {compacted_calls} earlier tool {calls} were already compacted "
            "to make room. "
        )
    # The gate runs for every tool, so non-file tools get neutral advice.
    lever = _TOOL_LEVERS.get(tool_name, "ask for less in one call")
    return (
        head + tried + "Nothing was written. Increase the Context Length in Model settings, "
        f"or {lever}, then try again."
    )


class ContextBudgetExceeded(ValueError):
    """Prompt refused before generation; carries the counts so nothing parses the message."""

    def __init__(self, request_tokens: int, context_tokens: int):
        self.request_tokens = int(request_tokens)
        self.context_tokens = int(context_tokens)
        super().__init__(describe_oversize(self.request_tokens, self.context_tokens))
