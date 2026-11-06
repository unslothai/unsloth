# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat error banner shows the whole error text."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
THREAD = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"


def _error_message_classes() -> set[str]:
    source = THREAD.read_text(encoding = "utf-8")
    at = source.index("<ErrorPrimitive.Message")
    tag = source[at : source.index("/>", at)]
    start = tag.index('className="') + len('className="')
    return set(tag[start : tag.index('"', start)].split())


def test_chat_error_message_is_not_clamped():
    # A clamp hid the context-length advice that follows the first two lines.
    classes = _error_message_classes()
    assert not any(c.startswith("line-clamp") for c in classes), classes
    assert not any(c in {"truncate", "overflow-hidden"} for c in classes), classes


def test_chat_error_message_wraps_long_text():
    classes = _error_message_classes()
    assert {"whitespace-pre-wrap", "break-words", "min-w-0"} <= classes, classes
