# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source contracts for the "stop running chats" confirmation.

The dialog is what a user reads before losing in-flight work, so two things have
to hold: it counts conversations rather than generation handles, and it describes
what confirming actually does. There is no frontend test runner in this repo, so
these read the source the way the other frontend contracts here do.
"""

from __future__ import annotations

from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[2]
FRONTEND = WORKDIR / "studio" / "frontend" / "src"


def _read(rel: str) -> str:
    path = FRONTEND / rel
    assert path.exists(), f"missing source file: {path}"
    return path.read_text(encoding = "utf-8")


def test_the_prompt_counts_conversations_not_generation_handles():
    # One chat holds several handles while a tool continuation registers its next leg before the previous unwinds
    # (active_generations.ActiveGeneration mints one per __enter__), so active.count exceeds the deduplicated thread_ids
    # and the dialog offered to stop two chats while listing one title.
    src = _read("features/chat/utils/confirm-stop-running-chats.ts")
    assert "entry.thread_id" in src, "the unnamed entries have to be counted separately"
    # The raw handle count survives only for a backend too old to send the entries.
    primary = src.index("running.length + unnamed")
    fallback = src.index("Math.max(active.count")
    assert primary < fallback, "the handle count must be the fallback, not the primary"


def test_an_unload_is_not_described_as_a_reload():
    # ejectModel confirms through the same dialog, but confirming calls /unload and leaves no model loaded:
    dialog = _read("features/chat/components/stop-running-chats-dialog.tsx")
    assert "Stop and unload" in dialog and "Stop and reload" in dialog
    assert "leaves no model loaded" in dialog

    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    eject = runtime.index('"Unloading the model"')
    assert (
        '"unload"' in runtime[eject : eject + 120]
    ), "the eject path must ask for the unload wording"


def test_the_tts_request_names_its_thread():
    # The audio branch registers its run locally under the thread key, and the backend tracker reads payload.thread_id.
    src = _read("features/chat/api/chat-adapter.ts")
    call = src.index("const result = await generateAudio(")
    assert (
        "thread_id: resolvedThreadId" in src[call : call + 600]
    ), "the TTS payload must carry the resolved thread id"
