# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ADAPTER = REPO / "studio/frontend/src/features/chat/api/chat-adapter.ts"


def squash(text: str) -> str:
    """Drop whitespace so reformatting chat-adapter.ts cannot break these checks."""
    return "".join(text.split())


def test_chat_run_reuses_one_thread_metadata_read() -> None:
    source = ADAPTER.read_text(encoding = "utf-8")
    run = squash(source[source.index("export function createOpenAIStreamAdapter(") :])

    assert run.count("getStoredChatThread(resolvedThreadId)") == 1
    assert run.count("getStoredChatThreadReadResult(resolvedThreadId)") == 1
    assert squash("const sharedThreadRecordRead = resolvedThreadId") in run
    assert squash("(result) => result.cacheable") in run
    assert "createRetryableSharedRead" in run
    assert squash("const thread = await getStoredChatThread(resolvedThreadId);") in run
    assert squash("getStoredChatThread(resolvedThreadId).catch(() => undefined)") not in run

    # The first shared read must sit after the model-ready wait, so a chat moved
    # to another project mid-load is still seen by the reads that follow.
    model_ready_boundary = run.index(squash("// Re-read store after auto-load / model-ready wait."))
    first_shared_read = run.index(squash("const sandboxSessionId = await resolveSandboxSessionId("))
    assert first_shared_read > model_ready_boundary

    # Prefixes, so a trailing comma or a one-line call both match.
    for call in (
        "resolveProjectId(resolvedThreadId,readThreadRecord",
        "resolveSandboxSessionId(resolvedThreadId,readThreadRecord",
        "resolveChatInstructions(resolvedThreadId,params.systemPrompt,params.systemVariables,readThreadRecord",
        "resolveUseAdapter(resolvedThreadId,options,readThreadRecord",
    ):
        assert call in run
