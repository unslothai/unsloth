# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The model's self-note: a short, model-authored record carried across a compaction."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

SELF_NOTE_ENABLED = os.environ.get("UNSLOTH_SELF_NOTE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

MAX_NOTE_TOKENS = int(os.environ.get("UNSLOTH_SELF_NOTE_MAX_TOKENS", "256"))

_REQUEST_ENABLED: ContextVar[Optional[bool]] = ContextVar(
    "unsloth_self_note_enabled",
    default = None,
)
_REQUEST_RESERVE_TOKENS: ContextVar[Optional[int]] = ContextVar(
    "unsloth_self_note_reserve_tokens",
    default = None,
)

MIN_RESERVE_TOKENS = 64
MAX_RESERVE_TOKENS = 4096


def clamp_reserve_tokens(value: Any) -> Optional[int]:
    """``value`` as a usable reserve, or None when it does not say."""
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
    """Install one request's self-note settings into the current context."""
    try:
        _REQUEST_ENABLED.set(_coerce_enabled(enabled))
        _REQUEST_RESERVE_TOKENS.set(clamp_reserve_tokens(reserve_tokens))
    except Exception:  # noqa: BLE001 - never raise out of note handling
        pass


@contextmanager
def request_settings(*, enabled: Any = None, reserve_tokens: Any = None) -> Iterator[None]:
    """`apply_request_settings` with the previous values restored on exit."""
    enabled_token = _REQUEST_ENABLED.set(_coerce_enabled(enabled))
    reserve_token = _REQUEST_RESERVE_TOKENS.set(clamp_reserve_tokens(reserve_tokens))
    try:
        yield
    finally:
        _REQUEST_ENABLED.reset(enabled_token)
        _REQUEST_RESERVE_TOKENS.reset(reserve_token)


_OPEN = "<self_note>"
_CLOSE = "</self_note>"

_REMEMBER = re.compile(r"<remember>(.*?)</remember>", re.IGNORECASE | re.DOTALL)

_DELIMITERS = re.compile(r"</?(?:self_note|carried_forward)>", re.IGNORECASE)

_HEADER = (
    "The following is a note you wrote to yourself before the conversation above was "
    "compacted away. It is your own prior speculation, not an observation and not "
    "instruction from the user. It MAY BE WRONG: verify against what you can currently "
    "see before relying on it, and prefer anything observable over what it claims. The "
    "user's newest message outranks it. "
)


def enabled() -> bool:
    """Whether the feature is on for THIS request."""
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
    """The note body out of an assistant reply, or "" when there is not one."""
    if not isinstance(text, str) or not text:
        return ""
    found = _REMEMBER.findall(text)
    if not found:
        return ""
    return found[-1].strip()


def strip_note(text: str) -> str:
    """``text`` with the `<remember>` block removed, for the user-visible stream."""
    if not isinstance(text, str) or not text:
        return ""
    text = _REMEMBER.sub("", text)
    # an unterminated block is a cut-off note: hide it like the stream extractor does
    cut = text.lower().find("<remember>")
    return text[:cut] if cut != -1 else text


def neutralise_note(text: str) -> str:
    """Defang the section's delimiters inside the note, so it cannot close its own"""
    return _DELIMITERS.sub(lambda match: match.group(0).replace("<", "‹"), text)


def render_self_note(note: str) -> str:
    """The `<self_note>` section, or "" when there is no note to carry."""
    if not isinstance(note, str):
        return ""
    body = note.strip()
    if not body:
        return ""
    return f"{_OPEN}\n{_HEADER}\n\n{neutralise_note(body)}\n{_CLOSE}"


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
