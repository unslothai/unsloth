# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"


def test_chat_run_reuses_one_thread_metadata_read() -> None:
    source = ADAPTER.read_text(encoding = "utf-8")
    run = source[source.index("export function createOpenAIStreamAdapter(") :]

    assert run.count("getStoredChatThread(resolvedThreadId)") == 1
    assert run.count("getStoredChatThreadReadResult(resolvedThreadId)") == 1
    assert "const sharedThreadRecordRead = resolvedThreadId" in run
    assert "(result) => result.cacheable" in run
    assert "createRetryableSharedRead" in run
    assert "const thread = await getStoredChatThread(resolvedThreadId);" in run
    assert "getStoredChatThread(resolvedThreadId).catch(() => undefined)" not in run

    for call in (
        "resolveProjectId(\n          resolvedThreadId,\n          readThreadRecord,",
        "resolveSandboxSessionId(\n        resolvedThreadId,\n        readThreadRecord,",
        "resolveChatInstructions(\n        resolvedThreadId,",
        "resolveUseAdapter(\n        resolvedThreadId,\n        options,\n        readThreadRecord,",
    ):
        assert call in run
