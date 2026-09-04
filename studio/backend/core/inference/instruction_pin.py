# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the user's standing instructions when the rolling window evicts everything else.

The defect this exists for. `context_window.truncate_oldest_messages` protects system and
developer groups, the final group, and the newest USER group. So "do task B, and always
report results as a table" is safe only while it IS the newest user turn. After a few
agent turns and a short follow-up -- "continue", "yes", "keep going" -- the newest user
group is that filler, and the instruction becomes an ordinary eviction candidate: the
oldest one, so the first to go.

Then the archive cannot rescue it either, because the forced recall on the compaction
turn uses the latest user message as its query. The archive gets searched for the word
"continue".

That follow-up is not a contrived case. OpenCode writes a synthetic "Continue if you have
next steps" turn after every auto compaction, and Zed emits "Continue where you left
off", so in both, the newest user turn straight after a compaction is filler by
construction. Everyone else's answer is to ask a summarizer to remember the instruction;
this is the deterministic version, and Zed's 80 KB verbatim replay of recent user
messages is the closest existing thing to it.

Two knobs, both off by default so the change ships inert:

    ROLLING_INSTRUCTION_PIN_GROUPS      how many instruction groups to hold (0 = today)
    ROLLING_INSTRUCTION_PIN_MAX_TOKENS  absolute ceiling on what they may cost

The pin is applied through `truncate_oldest_messages`'s existing `protected_message_ids`
parameter, so nothing in the rolling-window layer changes.
"""

from __future__ import annotations

import os
import re

from core.inference.context_window import estimate_messages_tokens_dense, group_turns

# 80 characters: someone who typed a paragraph wrote an instruction. Nothing inspects meaning or keywords, which are the
# heuristics a user trips by accident.
INSTRUCTION_MIN_CHARS = int(os.environ.get("ROLLING_INSTRUCTION_MIN_CHARS", "80"))
PIN_GROUPS = int(os.environ.get("ROLLING_INSTRUCTION_PIN_GROUPS", "0"))
PIN_MAX_TOKENS = int(os.environ.get("ROLLING_INSTRUCTION_PIN_MAX_TOKENS", "1024"))
# ... and never more than this share of the prompt budget, so the pin stays a minority of the window on a small model as
# well as a large one.
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
# keyboards autocorrect "..." to U+2026, so `continue...` matched nothing and recall searched for "continue"
# U+2026 and U+2025 as well as the ASCII spellings: keyboards autocorrect "..." to one ellipsis character, so
# `continue…` matched nothing and recall searched for "continue".
_PUNCTUATION = re.compile(r"[\s\.,!\?;:\-–—\u2025\u2026]+")

# A closed list, not a word count: words that cannot name a request's subject "Anaphoric" as a closed list rather than a
# word count: words that cannot name the subject of a request. "what about it" has nothing to search for; "review
# billing" names its own subject and keeps its retrieval slots. Negation is left out, as in `store._ARCHIVE_STOPWORDS`:
# a missed anchor is cheaper than a wrong one.
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
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
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
    """The most recent real instruction, for use as a recall query.

    ``skip_latest`` skips the newest user message, which is the one that was too thin to
    search with in the first place.
    """
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
    """Whether searching an archive for this text is worth a retrieval slot.

    Thin means the message NAMES NOTHING TO SEARCH FOR: every word of it is a function
    word, an anaphor or a continuation. It deliberately does not mean "short". Counting
    words instead swept in every self-contained two-word request -- "review billing",
    "restart nginx", "ZQXVARA123?" -- and the anchor a thin query earns is spent ahead of
    the user's own words in `conversation_archive.recall`. At the top_k of 1 that both the
    over-budget backoff (4 -> 2 -> 1) and a small window (`_recall_top_k` is
    `budget // CHUNK_TOKENS`) reach, the anchor takes the only slot: measured on a
    nine-turn archive, "review billing" at top_k=1 recalled the standing instruction and
    NOT the billing turn, which the same recall returns without an anchor.
    """
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
    """What pinning the head of ``turns[index]`` actually costs the window.

    `truncate_oldest_messages` protects by GROUP, not by message, and `group_turns` puts
    an assistant reply that carries no tool calls in the SAME group as the user message it
    answers. So pinning a one-line instruction also holds that reply, and charging only the
    instruction let a pin exceed the ceiling by an arbitrary amount: measured at 28 tokens
    charged against 20037 actually held. The budget has to count what is really kept, or it
    is not a budget.

    It must not count more than that either. A trailing tool exchange is NOT held: an
    assistant message with tool calls opens its own group, and `truncate_oldest_messages`
    skips a protected group BEFORE the `starts_user_turn` expansion that would otherwise
    absorb the groups behind it, so that tool group stays its own eviction unit and is
    evicted independently of the pin. Charging it would let one large tool result cost a
    small instruction its pin over tokens the pin never keeps -- which is the case the pin
    exists for, since an agent run is exactly where the filler follow-up appears.
    """
    # 4 chars per token undercharges CJK and emoji ~2x; over-charging only refuses the pin
    # Dense: 4 chars per token undercharges CJK and emoji ~2x, so a 1056-token turn was charged 276 and cleared a 1024
    # ceiling. Over-charging only refuses the pin.
    return estimate_messages_tokens_dense(turns[index])


def pinned_instruction_ids(
    messages: list[dict],
    *,
    groups: int = PIN_GROUPS,
    min_chars: int = INSTRUCTION_MIN_CHARS,
    max_tokens: int = PIN_MAX_TOKENS,
    prompt_target: int | None = None,
) -> set[int]:
    """`id()`s of the messages in the most recent instruction groups worth protecting.

    Bounded twice over. At most ``groups`` of them, and never more than ``max_tokens``
    summed across everything the pin actually holds: only the USER message is named, but
    the window protects by group, so the reply in that group is held with it and is charged
    with it -- and a trailing tool exchange, which is its own group and stays independently
    evictable, is not (see `_protected_cost`). Anything larger
    than the ceiling is not pinned AT ALL rather than partially: the single enormous
    instruction is precisely the thing that could starve the window, so it is the thing
    excluded.

    ``groups`` of 0 returns an empty set, which makes every downstream byte identical to
    today.
    """
    if groups <= 0 or not messages:
        return set()
    ceiling = max_tokens
    if prompt_target:
        ceiling = min(ceiling, int(prompt_target * PIN_MAX_FRACTION))
    if ceiling <= 0:
        return set()

    turns = group_turns(messages)
    # the newest user group is already window-protected
    # The newest user group is already protected by the window, and the inline recall path replaces that message with a
    # new dict, so its id would go stale anyway.
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
