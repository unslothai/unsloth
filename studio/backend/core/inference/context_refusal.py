# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Carries the context fit's refusal diagnosis to the user-facing message via a ContextVar;
_friendly_error sees only llama-server's total, so a two-message thread would be told to shorten
the conversation; asyncio copies the context per request, so one refusal cannot describe another."""

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
    """Call in the request's own context before spawning; create_task and to_thread copy that context."""
    _REFUSAL_SLOT.set({"refusal": None})


def _slot(*, create: bool = False) -> Optional[dict]:
    slot = _REFUSAL_SLOT.get()
    if slot is None and create:
        # No one opened one, so this context is where the message is built too.
        slot = {"refusal": None}
        _REFUSAL_SLOT.set(slot)
    return slot


def record_fit(truncation) -> None:
    """Called on every fit, not only refusals, so a later fitting iteration clears a stale refusal."""
    if not isinstance(truncation, dict):
        return
    slot = _slot(create = True)
    slot["refusal"] = None if truncation.get("fits") else dict(truncation)


def clear() -> None:
    slot = _slot()
    if slot is not None:
        # empty it as well as dropping it, so a worker mid-flight holding a reference cannot read a stale refusal back
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
    """Turn to name as (role, fits_alone); fits_alone is False only at or over the counted window."""
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
    # Only a counted turn compares with irreducible_tokens: the char estimate is 14.8x high or 0.86x low.
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
    # Use the WINDOW (not prompt_target) and the turn without shared floor; llama-server admits on size.
    window = recorded_context or context_tokens
    # Reached only on a counted turn, per the gate above, so this is a claim about a size that was measured. A turn
    # the template renders as nothing on its own is counted by difference, so every Gemma tool result can earn this
    # wording again rather than being hedged down for being a guess. Not defaulted to "user": `describe_oversize`
    # gives an unnameable role generic advice.
    role = str(refusal.get("latest_turn_role") or "")
    return role, not (window and latest_turn >= window)


def _history_cannot_help(context_tokens: int) -> bool:
    """True when the unevictable floor alone still reaches the window, so shortening history cannot help."""
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
    # Per-tool levers the user can shorten; absent tools get the neutral line, since an MCP payload is neither a file nor a program
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


def oversize_advice(context_tokens: int) -> str:
    """Split from describe_oversize so the Anthropic passthrough can keep its own head wording."""
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
    """Advice turns on whose turn is the bulk and whether it can be sent; unknown roles get generic text."""
    return (
        f"Message too long: {request_tokens} tokens exceeds the "
        f"{context_tokens}-token context window. "
    ) + oversize_advice(context_tokens)


# A refused call skips the general floor, since its refusal enters a prompt that already does not fit; the final size check still bounds the receipt
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
    """Refused before the tool runs, so it can promise nothing was written, which a 400 never could."""
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
