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
    LlamaStreamError,
    describe_stream_error,
    error_message_from_chunk,
    is_context_oversize,
    is_kv_starvation,
    stream_error_from_chunk,
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


class TestSurvivesTheRouteLayer:
    """Raising the right message is not enough: `routes/inference.py` rewrites it.

    Both defects here were live. `_friendly_error` ends with a catch-all that
    replaced any unrecognised exception with "An internal error occurred", so the
    cause survived the stream loop and then died one layer up. And
    `_classify_llama_generation_error` flags an overflow by finding "context" beside
    "window", which the starvation text says while explaining that the window is
    SHARED, so it was labelled `context_length_exceeded` and set the client
    compacting a conversation that was never too long.
    """

    @staticmethod
    def _routes():
        import routes.inference as routes_inference
        return routes_inference

    def _error(self, message):
        return stream_error_from_chunk({"error": {"message": message}})

    def test_starvation_reaches_the_user_instead_of_an_internal_error(self):
        routes = self._routes()
        described = routes._friendly_error(self._error("Context size has been exceeded."))
        assert described == KV_STARVATION_MESSAGE
        assert "An internal error occurred" not in described

    def test_starvation_is_not_classified_as_a_context_overflow(self):
        # True would set the client compacting. False would emit a 400 and tell the
        # client its own request was at fault, discouraging the retry that is the right
        # response to server capacity exhaustion. None keeps it a 500.
        routes = self._routes()
        assert (
            routes._classify_llama_generation_error(self._error("Context size has been exceeded."))
            is None
        )

    def test_an_unrelated_failure_is_not_downgraded_to_a_client_error(self):
        routes = self._routes()
        assert routes._classify_llama_generation_error(self._error("tokenizer failed")) is None

    def test_an_oversize_refusal_keeps_the_established_wording_and_triggers_compaction(self):
        routes = self._routes()
        error = self._error(
            "request (2358 tokens) exceeds the available context size (2048 tokens), try increasing it"
        )
        described = routes._friendly_error(error)
        assert described.startswith("Message too long: 2358 tokens")
        assert "2048-token context window" in described
        # An overflow genuinely is one, so the client should compact here.
        assert routes._classify_llama_generation_error(error) is True

    def test_an_unrelated_error_survives_verbatim(self):
        routes = self._routes()
        assert routes._friendly_error(self._error("tokenizer failed")) == "tokenizer failed"

    def test_deep_research_shows_the_friendly_text_not_the_server_text(self):
        """`_safe_error` reads str(exc), which is deliberately the server's own wording.
        Reading it here showed the raw "Context size has been exceeded." on the very
        path this exception was introduced to explain."""
        from core.research_runs import _safe_error

        assert _safe_error(self._error("Context size has been exceeded.")) == KV_STARVATION_MESSAGE
        oversize = _safe_error(
            self._error("request (2358 tokens) exceeds the available context size (2048 tokens)")
        )
        assert "2358" in oversize and "Context Length in Model settings" in oversize
        # A plain exception still reads from str().
        assert _safe_error(RuntimeError("plain")) == "plain"

    def test_str_of_the_error_stays_the_server_text(self):
        # What lets the existing token-count regex in _friendly_error still match.
        error = self._error("request (10 tokens) exceeds the available context size (5 tokens)")
        assert str(error).startswith("request (10 tokens)")

    def test_the_typed_error_is_a_runtimeerror(self):
        # Callers that already catch RuntimeError around the stream keep working.
        assert isinstance(self._error("boom"), RuntimeError)
        assert isinstance(self._error("boom"), LlamaStreamError)

    def test_a_non_error_chunk_yields_no_exception(self):
        assert stream_error_from_chunk({"choices": [{"delta": {"content": "hi"}}]}) is None


class TestTheNonStreamingPathAlsoReportsTheCause:
    """`stream=false` routes the same exception through `safe_error_detail`.

    That helper exists to stop raw `str(error)` leaking paths, so it returns a fixed
    fallback for anything it does not recognise. A curated `friendly` is written to be
    shown, so it is exempt: without that, streaming clients got the cause and
    non-streaming clients got "An internal error occurred", which is the same defect
    this PR fixes, one layer further out.
    """

    def _error(self, message):
        return stream_error_from_chunk({"error": {"message": message}})

    def test_starvation_reaches_a_non_streaming_client(self):
        from utils.utils import safe_error_detail
        assert (
            safe_error_detail(self._error("Context size has been exceeded."))
            == KV_STARVATION_MESSAGE
        )

    def test_an_oversize_refusal_keeps_both_counts(self):
        from utils.utils import safe_error_detail
        detail = safe_error_detail(
            self._error("request (2358 tokens) exceeds the available context size (2048 tokens)")
        )
        assert "2358" in detail and "2048" in detail

    def test_an_ordinary_exception_is_still_generalised(self):
        """The leak guard must keep working: only the curated message is exempt."""
        from utils.utils import safe_error_detail

        assert (
            safe_error_detail(RuntimeError("/srv/secret/path blew up"))
            == "An internal error occurred"
        )
        assert "/srv/secret" not in safe_error_detail(RuntimeError("/srv/secret/path blew up"))


class TestTheStarvationTextDoesNotReadAsAContextLimitOnTheClient:
    """The chat client re-classifies by substring, so the wording is load bearing.

    `studio/frontend/src/features/chat/api/chat-adapter.ts::isContextLimitError` decides
    which toast a failed generation gets from the error message alone: the backend's
    `code` never reaches it, because `chat-api.ts` turns an in-band error chunk into
    `new Error(parsed.error.message)` and throws only the text. Any of the substrings
    below wins the "Context limit reached" toast, whose advice is "The conversation has
    filled the model's context window ... or start a new chat".

    That is the wrong remedy for starvation, and it is the exact claim this message was
    written to deny: nothing about the conversation was too long, so starting a new chat
    fails identically while the other generation is still running. Asserted here rather
    than in the frontend because the message lives here and the wording is what breaks.
    """

    # Mirrors isContextLimitError. Keep in step with chat-adapter.ts.
    CLIENT_CONTEXT_LIMIT_MARKERS = (
        "context size",
        "context shift",
        "exceeds the available context",
        "message too long",
        "context window",
    )

    def test_the_curated_starvation_message_avoids_the_client_heuristic(self):
        lowered = KV_STARVATION_MESSAGE.lower()
        assert [m for m in self.CLIENT_CONTEXT_LIMIT_MARKERS if m in lowered] == []

    def test_the_message_still_names_the_cause_and_both_remedies(self):
        lowered = KV_STARVATION_MESSAGE.lower()
        assert "same time" in lowered
        assert "fewer running at once" in lowered
        assert "context length" in lowered

    def test_the_server_wording_it_replaces_would_have_hit_the_heuristic(self):
        """Guards the test itself: the raw text is what the rewrite exists to avoid."""
        raw = "Context size has been exceeded."
        assert any(m in raw.lower() for m in self.CLIENT_CONTEXT_LIMIT_MARKERS)
        assert describe_stream_error(raw) == KV_STARVATION_MESSAGE
