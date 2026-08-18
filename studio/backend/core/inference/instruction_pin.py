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

from core.inference.context_window import estimate_message_tokens, group_turns

# 80 characters. Objective, and there is no way to trip it accidentally in either
# direction that a reasonable person would argue with: someone who typed a paragraph
# wrote an instruction. Nothing here inspects meaning, mood or keywords, because those
# are exactly the heuristics a user trips by accident and cannot predict.
INSTRUCTION_MIN_CHARS = int(os.environ.get("ROLLING_INSTRUCTION_MIN_CHARS", "80"))
PIN_GROUPS = int(os.environ.get("ROLLING_INSTRUCTION_PIN_GROUPS", "0"))
PIN_MAX_TOKENS = int(os.environ.get("ROLLING_INSTRUCTION_PIN_MAX_TOKENS", "1024"))
# ... and never more than this share of the prompt budget, so the floor stays a minority
# of the window on a small model as well as a large one.
PIN_MAX_FRACTION = float(os.environ.get("ROLLING_INSTRUCTION_PIN_MAX_FRACTION", "0.10"))

# A pure REJECT list: it can only stop something being treated as an instruction, never
# promote one. Deleting it changes nothing except which side of the line a long-winded
# "okay, please keep going with that" falls on.
_CONTINUATIONS = frozenset({
    "continue", "continue please", "carry on", "go on", "go ahead", "keep going",
    "proceed", "next", "more", "yes", "y", "yeah", "yep", "ok", "okay", "k", "sure",
    "no", "n", "nope", "thanks", "thank you", "ta", "done", "good", "great", "fine",
    "please continue", "please carry on", "resume", "and", "then",
})
_PUNCTUATION = re.compile(r"[\s\.,!\?;:\-–—]+")


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
        isinstance(part, dict) and part.get("type") not in (None, "text")
        for part in content
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


def last_substantive_instruction(messages: list[dict], *,
                                 min_chars: int = INSTRUCTION_MIN_CHARS,
                                 skip_latest: bool = True) -> str | None:
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
    """Whether searching an archive for this text is worth a retrieval slot."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) >= min_chars:
        return False
    normalised = _PUNCTUATION.sub(" ", stripped.lower()).strip()
    # Short AND anaphoric. A short question that names something ("what is ZQXVARA123?")
    # is a perfectly good query and must not be replaced by an older instruction.
    return normalised in _CONTINUATIONS or len(normalised.split()) <= 2


def pinned_instruction_ids(messages: list[dict], *, groups: int = PIN_GROUPS,
                           min_chars: int = INSTRUCTION_MIN_CHARS,
                           max_tokens: int = PIN_MAX_TOKENS,
                           prompt_target: int | None = None) -> set[int]:
    """`id()`s of the messages in the most recent instruction groups worth protecting.

    Bounded twice over. At most ``groups`` of them, and never more than ``max_tokens``
    summed across the pinned USER messages -- their replies are not pinned, because the
    instruction is the thing that has to survive, not the answer to it. Anything larger
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
    # The newest user group is already protected by the window itself, and pinning it
    # would be worse than pointless: the inline recall path REPLACES that message with a
    # new dict, so its id would go stale and silently protect nothing.
    newest_user = next(
        (index for index in range(len(turns) - 1, -1, -1)
         if any(m.get("role") == "user" for m in turns[index])),
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
        cost = estimate_message_tokens(head)
        if spent + cost > ceiling:
            continue
        pinned.add(id(head))
        spent += cost
        taken += 1
    return pinned
