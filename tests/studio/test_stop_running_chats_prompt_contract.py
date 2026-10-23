# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The stop-running-chats dialog counts conversations, not generation handles, and says what it does."""

from __future__ import annotations

from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[2]
FRONTEND = WORKDIR / "studio" / "frontend" / "src"


def _read(rel: str) -> str:
    path = FRONTEND / rel
    assert path.exists(), f"missing source file: {path}"
    return path.read_text(encoding = "utf-8")


def test_the_prompt_counts_conversations_not_generation_handles():
    # One chat can hold several handles during a tool continuation, so count deduplicated threads.
    src = _read("features/chat/utils/confirm-stop-running-chats.ts")
    assert "entry.thread_id" in src, "the unnamed entries have to be counted separately"
    primary = src.index("running.length + unnamed")
    fallback = src.index("Math.max(active.count")
    assert primary < fallback, "the handle count must be the fallback, not the primary"


def test_an_unload_is_not_described_as_a_reload():
    # Confirming ejectModel unloads and leaves no model loaded, so copy must not promise a reload.
    dialog = _read("features/chat/components/stop-running-chats-dialog.tsx")
    assert "Stop and unload" in dialog and "Stop and reload" in dialog
    assert "leaves no model loaded" in dialog

    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    eject = runtime.index('"Unloading the model"')
    assert (
        '"unload"' in runtime[eject : eject + 120]
    ), "the eject path must ask for the unload wording"


def test_the_tts_request_names_its_thread():
    src = _read("features/chat/api/chat-adapter.ts")
    call = src.index("const result = await generateAudio(")
    assert (
        "thread_id: resolvedThreadId" in src[call : call + 600]
    ), "the TTS payload must carry the resolved thread id"
