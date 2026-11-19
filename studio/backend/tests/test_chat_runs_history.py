# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


from routes.chat_generation_runs import CreateChatGenerationRun, _sanitize_request

FULL = "Paris is the capital. It sits on the Seine. Its museums are famous."
HEARD = "Paris is the capital."


def _run(messages):
    return CreateChatGenerationRun(
        runId = "run-1",
        threadId = "thread-1",
        userMessageId = "user-2",
        assistantMessageId = "assistant-2",
        requestPayload = {"model": "local", "messages": messages},
    )


def test_the_run_keeps_the_cut_reply_the_request_sent():
    messages = [
        {"role": "user", "content": "Tell me about Paris."},
        {"role": "assistant", "content": HEARD},
        {"role": "user", "content": "And the weather?"},
    ]
    sanitized = _sanitize_request(_run(messages))
    assert [m["content"] for m in sanitized["messages"]] == [
        "Tell me about Paris.",
        HEARD,
        "And the weather?",
    ]
    assert FULL not in str(sanitized)


def test_the_run_does_not_add_messages_the_request_left_out():
    # The client prunes an unheard reply with its prompt; the run must not restore it.
    messages = [{"role": "user", "content": "And the weather?"}]
    sanitized = _sanitize_request(_run(messages))
    assert sanitized["messages"] == messages
