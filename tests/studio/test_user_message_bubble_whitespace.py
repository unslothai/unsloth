# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
THREAD = REPO / "studio/frontend/src/components/assistant-ui/thread.tsx"


def _bubble_classes() -> set[str]:
    source = THREAD.read_text(encoding = "utf-8")
    start = source.index('className="aui-user-message-content ') + len('className="')
    return set(source[start : source.index('"', start)].split())


def test_user_message_bubble_keeps_indentation():
    # assistant-ui sets `pre-line` inline on the text, so only an important rule can override it.
    assert "[&_p]:whitespace-pre-wrap!" in _bubble_classes()
