# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeps standing instructions past filler turns; opt-in via ROLLING_INSTRUCTION_PIN_GROUPS (0 = off)"""

from __future__ import annotations

import os
import re

from core.inference.context_window import estimate_messages_tokens_dense, group_turns
from utils.current_date_prompt_settings import strip_current_date_update_note

# 80 chars: a typed paragraph is an instruction. No keyword heuristics users trip by accident.
INSTRUCTION_MIN_CHARS = int(os.environ.get("ROLLING_INSTRUCTION_MIN_CHARS", "80"))
PIN_GROUPS = int(os.environ.get("ROLLING_INSTRUCTION_PIN_GROUPS", "0"))
PIN_MAX_TOKENS = int(os.environ.get("ROLLING_INSTRUCTION_PIN_MAX_TOKENS", "1024"))
# Cap the pin at this share of the prompt budget, so it stays a minority on small models too.
PIN_MAX_FRACTION = float(os.environ.get("ROLLING_INSTRUCTION_PIN_MAX_FRACTION", "0.10"))

# A pure REJECT list: it can only stop something being treated as an instruction, never promote one.
_CONTINUATIONS = frozenset(
    {
        "continue",
        "continue please",
        "carry on",
        "go on",
        "go ahead",
        "keep going",
        "proceed",
        "next",
        "more",
        "yes",
        "y",
        "yeah",
        "yep",
        "ok",
        "okay",
        "k",
        "sure",
        "no",
        "n",
        "nope",
        "thanks",
        "thank you",
        "ta",
        "done",
        "good",
        "great",
        "fine",
        "please continue",
        "please carry on",
        "resume",
        "and",
        "then",
    }
)
# Keyboards autocorrect "..." to U+2026, so match the ellipsis characters too.
_PUNCTUATION = re.compile(r"[\s\.,!\?;:\-–—\u2025\u2026]+")

# Words that cannot name a request's subject. Negation left out, as in store._ARCHIVE_STOPWORDS.
_FUNCTION_WORDS = frozenset(
    """
a about all also am an and another any anything are as at be been being both but by can
could did do does doing each either else even ever for from get give had has have he her
here him his how i if in into is it its just like me mine my of on once one only or other
our ours out over please same she should so some someone something still such than that
the their theirs them then there these they thing things this those to too us was we were
what when where which while who whom whose why will with would you your yours
""".split()
)


def _text_of(message: dict) -> str:
    # without Studio's date note: it pads a short first turn past the bar and is no instruction.
    content = message.get("content")
    if isinstance(content, str):
        return strip_current_date_update_note(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return strip_current_date_update_note("\n".join(parts))
    return ""


def _has_non_text_part(message: dict) -> bool:
    """An upload is never filler. A one-word message with an image attached is a real
    request, and treating it as a continuation would pin the wrong turn."""
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict) and part.get("type") not in (None, "text") for part in content
    )


def is_substantive(message: dict, *, min_chars: int = INSTRUCTION_MIN_CHARS) -> bool:
    """Whether a user message is an instruction rather than a nudge to keep going."""
    if message.get("role") != "user":
        return False
    if _has_non_text_part(message):
        return True
    text = _text_of(message).strip()
    if len(text) < min_chars:
        return False
    normalised = _PUNCTUATION.sub(" ", text.lower()).strip()
    return normalised not in _CONTINUATIONS


def last_substantive_instruction(
    messages: list[dict],
    *,
    min_chars: int = INSTRUCTION_MIN_CHARS,
    skip_latest: bool = True,
) -> str | None:
    """skip_latest exists because the newest user message is too thin to search with."""
    users = [m for m in messages if m.get("role") == "user"]
    if skip_latest and users:
        users = users[:-1]
    for message in reversed(users):
        if is_substantive(message, min_chars = min_chars):
            text = _text_of(message).strip()
            if text:
                return text
    return None


def is_thin_query(text: str, *, min_chars: int = INSTRUCTION_MIN_CHARS) -> bool:
    """Thin = names nothing searchable, not short; word count spends the lone top_k 1 slot on an anchor."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) >= min_chars:
        return False
    normalised = _PUNCTUATION.sub(" ", stripped.lower()).strip()
    if normalised in _CONTINUATIONS:
        return True
    # Short AND anaphoric: "what is ZQXVARA123?" names something and stays the query.
    words = normalised.split()
    if not words:
        return True
    return all(word in _FUNCTION_WORDS or word in _CONTINUATIONS for word in words)


def _protected_cost(turns: list[list[dict]], index: int) -> int:
    """Charges the pinned group and its reply, since that is what is kept; not a trailing tool exchange."""
    return estimate_messages_tokens_dense(turns[index])


def pinned_instruction_ids(
    messages: list[dict],
    *,
    groups: int = PIN_GROUPS,
    min_chars: int = INSTRUCTION_MIN_CHARS,
    max_tokens: int = PIN_MAX_TOKENS,
    prompt_target: int | None = None,
) -> set[int]:
    """Oversized instructions (over max_tokens) are skipped whole, so they cannot starve the window."""
    if groups <= 0 or not messages:
        return set()
    ceiling = max_tokens
    if prompt_target:
        ceiling = min(ceiling, int(prompt_target * PIN_MAX_FRACTION))
    if ceiling <= 0:
        return set()

    turns = group_turns(messages)
    # The window already protects the newest user group, and recall replaces that dict anyway.
    newest_user = next(
        (
            index
            for index in range(len(turns) - 1, -1, -1)
            if any(m.get("role") == "user" for m in turns[index])
        ),
        None,
    )

    pinned: set[int] = set()
    spent = 0
    taken = 0
    for index in range(len(turns) - 1, -1, -1):
        if taken >= groups:
            break
        if index == newest_user:
            continue
        group = turns[index]
        head = group[0]
        if not is_substantive(head, min_chars = min_chars):
            continue
        cost = _protected_cost(turns, index)
        if spent + cost > ceiling:
            continue
        pinned.add(id(head))
        spent += cost
        taken += 1
    return pinned
