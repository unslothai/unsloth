# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A mid-stream llama-server error must reach the user with its cause intact.

The two failures these cover were both observed live against a model loaded at a 2048
token context:

- Two chats generating at once starved the shared unified KV cache. llama.cpp killed both
  tasks with "Context size has been exceeded". The chat loop ignored the error chunk (it
  carries no ``choices``) and the reply ended mid-code with no finish_reason, so no
  continue bar rendered and auto-continue never fired.
- Deep Research sent a 2358 token request into that 2048 token window. The server said so
  precisely, naming both counts, and ``research_runs`` replaced it with "Local model
  stream failed".
"""

from core.inference.stream_errors import (
    KV_STARVATION_MESSAGE,
    describe_stream_error,
    error_message_from_chunk,
    is_context_oversize,
    is_kv_starvation,
)


class TestErrorMessageFromChunk:
    def test_a_chunk_without_an_error_key_is_not_an_error(self):
        assert error_message_from_chunk({"choices": [{"delta": {"content": "hi"}}]}) is None

    def test_a_non_dict_chunk_is_not_an_error(self):
        assert error_message_from_chunk("[DONE]") is None
        assert error_message_from_chunk(None) is None

    def test_the_nested_shape_yields_the_server_message(self):
        chunk = {"error": {"message": "Context size has been exceeded.", "code": 500}}
        assert error_message_from_chunk(chunk) == "Context size has been exceeded."

    def test_the_bare_string_shape_yields_the_server_message(self):
        assert error_message_from_chunk({"error": "boom"}) == "boom"

    def test_an_unrecognised_error_shape_is_still_an_error(self):
        # Empty string, not None: the caller must still fail the stream. Returning None
        # here would restore the silent truncation this module exists to remove.
        assert error_message_from_chunk({"error": {"code": 500}}) == ""


class TestClassification:
    def test_the_starvation_wordings_are_recognised(self):
        for text in (
            "Context size has been exceeded.",
            "srv decode: failed to find free space in the KV cache, retrying",
            "decode: failed to find a memory slot for batch of size 2",
        ):
            assert is_kv_starvation(text), text
            assert not is_context_oversize(text), text

    def test_an_oversize_refusal_is_not_read_as_starvation(self):
        text = "request (2358 tokens) exceeds the available context size (2048 tokens), try increasing it"
        assert is_context_oversize(text)
        assert not is_kv_starvation(text)

    def test_an_unrelated_error_is_neither(self):
        assert not is_kv_starvation("tokenizer failed")
        assert not is_context_oversize("tokenizer failed")

    def test_empty_input_is_neither(self):
        assert not is_kv_starvation(None) and not is_kv_starvation("")
        assert not is_context_oversize(None) and not is_context_oversize("")


class TestDescribeStreamError:
    def test_starvation_explains_concurrency_rather_than_repeating_the_server(self):
        described = describe_stream_error("Context size has been exceeded.")
        assert described == KV_STARVATION_MESSAGE
        # The server's own wording sends the user off to shorten a conversation that was
        # never too long, so it must not be what they read.
        assert "Context size has been exceeded" not in described
        assert "at the same time" in described

    def test_an_oversize_refusal_keeps_both_token_counts_and_gains_a_remedy(self):
        described = describe_stream_error(
            "request (2358 tokens) exceeds the available context size (2048 tokens), try increasing it"
        )
        assert "2358" in described and "2048" in described
        assert "Context Length in Model settings" in described

    def test_an_unrelated_error_is_passed_through_verbatim(self):
        assert describe_stream_error("tokenizer failed") == "tokenizer failed"

    def test_an_empty_error_still_says_something(self):
        described = describe_stream_error("")
        assert described and "stopped generating early" in described

    def test_the_prefix_names_the_caller(self):
        described = describe_stream_error("tokenizer failed", prefix = "Deep Research")
        assert described == "Deep Research: tokenizer failed"

    def test_no_outcome_is_the_old_fixed_string(self):
        # The regression guard: whatever the input, the user must never be handed a
        # message that discards the cause.
        for text in ("Context size has been exceeded.", "tokenizer failed", ""):
            assert "Local model stream failed" not in describe_stream_error(text)
