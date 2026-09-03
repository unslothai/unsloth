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


# A one-key box, not the refusal itself: `.set()` in a context copy is invisible to the
# original, where `_friendly_error` runs, but copies share VALUES. See `open_slot`.
_REFUSAL_SLOT: ContextVar[Optional[dict]] = ContextVar("unsloth_context_refusal", default = None)

# Share of the irreducible prompt the latest turn must reach before the turn, not the
# conversation, is blamed. Never all of it: the system prompt and template wrapper are in
# the floor too. Dominating is NOT the same as not fitting, so it only earns the softer
# "most of this prompt is ..." wording; the flat "does not fit" needs the turn alone to
# exceed the budget.
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
        # Empty it as well as dropping it, so a worker mid-flight holding a reference
        # cannot read a stale refusal back.
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
        # A different load or backend: it cannot describe this refusal.
        return None
    irreducible = _int(refusal.get("irreducible_tokens"))
    latest_turn = _int(refusal.get("latest_turn_tokens"))
    if irreducible <= 0 or latest_turn <= 0:
        return None
    # Only a COUNTED turn is comparable to `irreducible_tokens`. That is a tokenizer count
    # of the rendered prompt; the fallback `latest_turn_tokens` is the message's JSON at
    # four characters a token, so weighing them against each other compares a guess with a
    # truth rather than two sides of one. Measured on the bundled gemma-4 template with a
    # real Gemma tokenizer: 16,400 characters of newlines estimate 8,207 tokens against
    # 557 rendered, 14.8x, which alone clears this ratio against a 8,629-token prompt the
    # turn is 6.5% of -- next to a system prompt that is 93% of it. The user was then told
    # "Most of this prompt is a single tool result" and to fetch a smaller slice of a file
    # that was not the problem. Escaped JSON runs the other way at 0.86x, so the error is
    # not even one-directional and cannot be corrected for.
    #
    # The producer now prices such a turn by difference against the prompt it measured
    # (`turn_diagnosis`), so this flag is False only when nothing could be counted at all.
    # There, no turn is named: a lost diagnosis costs the user a specific lever, a false
    # one sends them after the wrong one. Absent flag means a producer that predates it,
    # which was always a count.
    exact = bool(refusal.get("latest_turn_exact", True))
    if not exact:
        return None
    # Both numbers price a whole rendered PROMPT, so both carry the same floor (template
    # wrapper plus any tool catalogue). Left in, it swamps the comparison: a 6,000-token
    # MCP catalogue makes a 20-token "hi" 97% of the irreducible prompt. Off BOTH sides,
    # so the turn's contribution is compared against the rest of the conversation's.
    shared = _int(refusal.get("shared_prompt_tokens"))
    shared = max(0, min(shared, latest_turn - 1, irreducible - 1))
    latest_turn -= shared
    irreducible -= shared
    if latest_turn < _TURN_DOMINATES * irreducible:
        return None
    # The WINDOW, not the fit's `prompt_target` (the window minus reserved reply room):
    # llama-server admits a prompt on its size alone ("n_tokens() >= n_ctx" in
    # tools/server/server-context.cpp, the check whose text this rewrites), so a turn
    # between the two really would have been served and only earns the soft wording.
    # `>=` to match that check. Compared without the shared floor, since the hard wording
    # is a claim about the turn's own size.
    window = recorded_context or context_tokens
    # Reached only on a counted turn, per the gate above, so this is a claim about a size
    # that was measured. A turn the template renders as nothing on its own is counted by
    # difference, which is why every Gemma tool result can earn this wording again rather
    # than being hedged down for being a guess.
    # Not defaulted to "user": `describe_oversize` gives an unnameable role generic advice.
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
        # A different load or backend: it cannot describe this refusal.
        return False
    irreducible = _int(refusal.get("irreducible_tokens"))
    window = recorded_context or context_tokens
    return irreducible > 0 and window > 0 and irreducible >= window


# Per role: what to call the turn when it merely dominates, what to call it when it does
# not fit at all, and the lever worth offering. The lever is why this splits by role --
# "send it in smaller pieces" is useless for turns the user did not type.
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
    # The model passed a file-sized argument to a tool. The user did not type it and
    # cannot split it, and the tool cannot be asked for less: `edit_file` with an empty
    # `old_string` is whole-file creation, so the content IS the argument. The only levers
    # are the window itself and not asking for a file this size in a window this small.
    "assistant_tool_call": (
        "Most of this prompt is the file the model passed to a tool",
        "The file the model passed to a tool does not fit on its own",
        "ask for a smaller file, or raise the Context Length before retrying",
    ),
    # The same shape with no file in it: an oversized program, command, query or MCP
    # payload. "Ask for a smaller file" names the wrong thing and cannot be acted on, so
    # this one says what is actually true of every tool.
    "assistant_tool_payload": (
        "Most of this prompt is what the model passed to a tool",
        "What the model passed to a tool does not fit on its own",
        "ask for less in one call, or raise the Context Length before retrying",
    ),
    # The reply resumed after it hit Max Tokens: the user cannot split or shorten it.
    "assistant": (
        "Most of this prompt is the reply being continued",
        "The reply being continued is already too long for this window",
        "start a new reply",
    ),
    # These survive eviction, so splitting one preserves the total and changes nothing.
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
        if _history_cannot_help(context_tokens):
            # No turn to name, and yet "shorten the conversation" is not merely vague
            # here, it is an action that provably cannot work: what survives eviction is
            # already at or over the window. Named levers rather than a role, because the
            # bulk is spread across the parts eviction never touches, and the recorded
            # fields cannot say which of them it is -- `shared_prompt_tokens` bundles the
            # template wrapper with the catalogue, so a large one does not prove there
            # are tools. Both levers are offered, and neither is claimed to be the cause.
            return (
                head + "Even with every earlier turn dropped, this prompt would still be "
                "too long, so shortening the conversation will not help. Increase the "
                "Context Length in Model settings, or reduce what every request carries: "
                "the system prompt and any tools that are enabled."
            )
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


# What the user can actually shorten, per tool. Anything absent gets the neutral line:
# an MCP tool's payload is not a file and not a program, and guessing at it is worse
# than saying the one thing that is true of every tool.
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
    # Says "leaving no room to reply" rather than only quoting the two numbers. The bar is
    # the window minus a small reply floor, so a refusal at 3,740 against 4,096 reads as a
    # contradiction unless the message accounts for the gap it is refusing over.
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
    # The gate runs for EVERY enabled tool, so the file wording was reaching an oversized
    # `python`, `terminal`, web or MCP call and telling the user to ask for a smaller
    # file when no file was involved -- advice that cannot make the actual program,
    # command or payload any smaller. `edit_file` keeps the line it was written for.
    lever = _TOOL_LEVERS.get(tool_name, "ask for less in one call")
    return (
        head + tried + "Nothing was written. Increase the Context Length in Model settings, "
        f"or {lever}, then try again."
    )
