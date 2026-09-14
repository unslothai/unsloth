# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The model's self-note: a short, model-authored record carried across a compaction.

Checkpoint compaction carries the USER's standing instructions verbatim and nothing of the
model's own working state, so a long stretch spent ruling an approach out is re-derived
after the reset. This module carries a bounded note the model writes for itself.

The note is the model's own speculation promoted into the SYSTEM turn, which is the same
authority-confusion risk `checkpoint.py` documents for quoted user text, plus one more: the
model is writing its own future input, so a wrong note is self-reinforcing. Two things hold
that down -- the section header says the note may be wrong and is outranked by the user's
newest message and by anything observable, and the note competes for the same bounded
budget as the user's instructions and loses ties to them.

NOTHING IS STORED here either: the note rides on the assistant message as a content part
and the client re-sends it, exactly as `checkpoint.py` re-derives its block each request.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

# Off by default. This changes what lands in the system turn, so it is opt-in: an existing
# user must not inherit a new failure mode from an upgrade.
SELF_NOTE_ENABLED = os.environ.get("UNSLOTH_SELF_NOTE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# A note is a handful of lines, not a summary. The cap is what stops the model from
# spending the whole carried-forward budget on itself.
MAX_NOTE_TOKENS = int(os.environ.get("UNSLOTH_SELF_NOTE_MAX_TOKENS", "256"))

# The per-REQUEST override, so the chat settings toggle and slider actually do something.
# A ContextVar rather than a threaded parameter because `enabled()` and `MAX_NOTE_TOKENS`
# are consulted from a dozen places inside one streaming response -- the prompt fragment,
# the stream extractor, the checkpoint render -- and threading a flag through all of them
# would be a far larger change than the setting is worth. ContextVar is the pattern the
# rest of this package already uses for per-request scope (`tools._REQUEST_RESULT_BUDGET`,
# `context_refusal._REFUSAL_SLOT`): per-task, and asyncio copies the context per request,
# so one request's setting cannot leak into another's.
#
# None means the request did not say, and the env var decides -- so an install that never
# sends the field behaves exactly as it did before, and OFF remains the default.
_REQUEST_ENABLED: ContextVar[Optional[bool]] = ContextVar(
    "unsloth_self_note_enabled",
    default = None,
)
_REQUEST_RESERVE_TOKENS: ContextVar[Optional[int]] = ContextVar(
    "unsloth_self_note_reserve_tokens",
    default = None,
)

# Mirrors the payload bounds in `routes/chat_history.ChatSettingsPayload`, so a value that
# arrived by any other path is clamped to the same range rather than trusted. The floor is
# well above zero because a note too small to hold a sentence is worse than no note.
MIN_RESERVE_TOKENS = 64
MAX_RESERVE_TOKENS = 4096


def clamp_reserve_tokens(value: Any) -> Optional[int]:
    """``value`` as a usable reserve, or None when it does not say.

    Never raises: note handling is a convenience, and a malformed setting must degrade to
    the default rather than fail the request. bool is rejected explicitly because it
    subclasses int, so True would silently become a 1-token reserve.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        return None
    return max(MIN_RESERVE_TOKENS, min(MAX_RESERVE_TOKENS, tokens))


def _coerce_enabled(value: Any) -> Optional[bool]:
    """``value`` as a tri-state toggle: True, False, or None for "did not say"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return None
    return None


def apply_request_settings(*, enabled: Any = None, reserve_tokens: Any = None) -> None:
    """Install one request's self-note settings into the current context.

    Set rather than scoped with a context manager, because the value has to outlive the
    coroutine that reads the payload: the route returns a StreamingResponse whose
    generator runs afterwards, and a `with` block would have exited by then. Safe because
    asyncio copies the context per request, so the value is not process-global -- the same
    reason `tools._REQUEST_CONTEXT_TOKENS` is set this way.

    Permissive and NEVER raises: an unparseable value is treated as "the request did not
    say", which leaves the env-var default in force. Note handling is a convenience, and
    a malformed setting must not fail a chat.
    """
    try:
        _REQUEST_ENABLED.set(_coerce_enabled(enabled))
        _REQUEST_RESERVE_TOKENS.set(clamp_reserve_tokens(reserve_tokens))
    except Exception:  # noqa: BLE001 - never raise out of note handling
        pass


@contextmanager
def request_settings(*, enabled: Any = None, reserve_tokens: Any = None) -> Iterator[None]:
    """`apply_request_settings` with the previous values restored on exit.

    For synchronous callers and tests, where the scope really does end with the block.
    """
    enabled_token = _REQUEST_ENABLED.set(_coerce_enabled(enabled))
    reserve_token = _REQUEST_RESERVE_TOKENS.set(clamp_reserve_tokens(reserve_tokens))
    try:
        yield
    finally:
        _REQUEST_ENABLED.reset(enabled_token)
        _REQUEST_RESERVE_TOKENS.reset(reserve_token)


_OPEN = "<self_note>"
_CLOSE = "</self_note>"

# What the model writes. Deliberately NOT the same tag as the rendered section: the model
# emits `<remember>`, the prompt carries `<self_note>`. One tag for both would make a note
# quoting its own prior note indistinguishable from a fresh one.
_REMEMBER = re.compile(r"<remember>(.*?)</remember>", re.IGNORECASE | re.DOTALL)

# Only the delimiters themselves, so a note that WRITES ABOUT the feature is not mangled.
# `<carried_forward>` is included because the section renders inside that block, so its
# closing tag is an escape route out of the marked region too.
_DELIMITERS = re.compile(r"</?(?:self_note|carried_forward)>", re.IGNORECASE)

# The note is the model's own words, so the header's job is to stop it reading as fact.
# `checkpoint.py` spends eight lines on why quoted USER text needs this; a model-authored
# note needs it more, because nothing else in the prompt marks it as disputable.
_HEADER = (
    "The following is a note you wrote to yourself before the conversation above was "
    "compacted away. It is your own prior speculation, not an observation and not "
    "instruction from the user. It MAY BE WRONG: verify against what you can currently "
    "see before relying on it, and prefer anything observable over what it claims. The "
    "user's newest message outranks it. "
)


def enabled() -> bool:
    """Whether the feature is on for THIS request.

    The request wins where it said something; otherwise the env var decides, so a caller
    that never sends the field sees exactly the behaviour it had before, OFF by default.
    """
    requested = _REQUEST_ENABLED.get()
    if requested is not None:
        return requested
    return SELF_NOTE_ENABLED


def max_note_tokens() -> int:
    """The note's ceiling for THIS request, request-first then the env var."""
    requested = _REQUEST_RESERVE_TOKENS.get()
    if requested is not None:
        return requested
    return MAX_NOTE_TOKENS


def extract_note(text: str) -> str:
    """The note body out of an assistant reply, or "" when there is not one.

    An unterminated block yields "" rather than the rest of the reply: a note cut off by
    the token limit is exactly the case where taking everything after the opening tag
    would promote a half-sentence, and the regex requires the closing tag for that reason.
    Never raises; a non-string degrades to "".
    """
    if not isinstance(text, str) or not text:
        return ""
    found = _REMEMBER.findall(text)
    if not found:
        return ""
    # The LAST note wins: a reply that writes twice has revised itself.
    return found[-1].strip()


def strip_note(text: str) -> str:
    """``text`` with the `<remember>` block removed, for the user-visible stream."""
    if not isinstance(text, str) or not text:
        return ""
    return _REMEMBER.sub("", text)


def neutralise_note(text: str) -> str:
    """Defang the section's delimiters inside the note, so it cannot close its own
    section early and turn the rest of its text into unmarked system content.
    """
    return _DELIMITERS.sub(lambda match: match.group(0).replace("<", "‹"), text)


def render_self_note(note: str) -> str:
    """The `<self_note>` section, or "" when there is no note to carry."""
    if not isinstance(note, str):
        return ""
    body = note.strip()
    if not body:
        return ""
    return f"{_OPEN}\n{_HEADER}\n\n{neutralise_note(body)}\n{_CLOSE}"


# Addressed to the model about its OWN future self, not to the user: a note written as a
# summary for a reader is the thing checkpoint compaction already refuses to generate.
# Kept short because it rides in the prompt on every request the feature is on for.
SELF_NOTE_INSTRUCTION = (
    "If this conversation is compacted, everything above will be dropped. You may leave "
    "yourself a short note that survives: write it inside <remember></remember> tags at "
    "the very end of your reply. Use it for what you would not want to re-derive -- an "
    "approach you ruled out and why, a lead worth resuming, a constraint you discovered. "
    "Write it for yourself, not as a summary for the user. It is not shown to the user. "
    "Omit the tags entirely if there is nothing worth carrying."
)


def note_instruction() -> str:
    """The prompt fragment telling the model the tag exists, or "" when disabled."""
    return SELF_NOTE_INSTRUCTION if enabled() else ""
