# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Estimated turns must not claim prompt dominance; a respawn refit refusal must reach the diagnosis."""

import contextlib

import httpx
import pytest

from core.inference import context_refusal
from core.inference.context_window import fit_rolling_context


# Measured with gemma-4.jinja and the unsloth/gemma-3-270m-it tokenizer:
#   empty prompt 16, system prompt 8009, 16,400-newline tool result 557 (8207 estimated),
#   whole conversation 8629 tokens
_FLOOR = 16
_SYSTEM = 8009
_USER = 30
_CALL = 17
_TOOL = 557


def _gemma_like_counter(messages: list[dict]) -> int:
    """A counter that renders like gemma-4: a LONE tool message renders as nothing."""
    if len(messages) == 1 and messages[0].get("role") == "tool":
        return _FLOOR
    total = _FLOOR
    for message in messages:
        total += {
            "system": _SYSTEM,
            "user": _USER,
            "assistant": _CALL,
            "tool": _TOOL,
        }[message["role"]]
    return total


def _sparse_tool_conversation() -> list[dict]:
    return [
        {"role": "system", "content": "s" * 32000},
        {"role": "user", "content": "Read the log and tell me what broke."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "\n" * 16400},
    ]


def test_an_estimated_turn_never_claims_to_be_most_of_the_prompt():
    """Dominance ratio must not compare an estimated turn size against a tokenizer count of the prompt."""
    messages = _sparse_tool_conversation()
    context_length = 8192

    _fitted, truncation = fit_rolling_context(
        messages,
        context_length = context_length,
        max_tokens = 512,
        count_tokens = _gemma_like_counter,
    )
    assert truncation is not None and truncation["fits"] is False
    assert truncation["irreducible_tokens"] == 8629, "a real count of the rendered prompt"
    assert truncation["latest_turn_exact"] is True
    assert truncation["latest_turn_tokens"] - truncation["shared_prompt_tokens"] == 557

    context_refusal.open_slot()
    try:
        context_refusal.record_fit(truncation)
        message = context_refusal.describe_oversize(8629, context_length)
    finally:
        context_refusal.clear()

    assert "Most of this prompt is a single tool result" not in message, message
    assert "ask for a smaller slice" not in message, message
    assert "shortening the conversation will not help" in message, message


def test_a_dominant_tool_result_still_gets_the_tool_advice():
    """An estimated turn that is nearly the whole prompt must keep the tool advice, not generic wording."""
    messages = [message for message in _sparse_tool_conversation() if message["role"] != "system"]
    context_length = 512

    _fitted, truncation = fit_rolling_context(
        messages,
        context_length = context_length,
        max_tokens = 64,
        count_tokens = _gemma_like_counter,
    )
    assert truncation is not None and truncation["fits"] is False

    context_refusal.open_slot()
    try:
        context_refusal.record_fit(truncation)
        message = context_refusal.describe_oversize(
            truncation["irreducible_tokens"], context_length
        )
    finally:
        context_refusal.clear()

    assert "A tool returned more than this context window can hold" in message, message
    assert "ask for a smaller slice of the file or page" in message, message


def test_a_respawn_refit_that_refuses_is_not_lost_when_the_retry_is_refused():
    """A respawn refit refusal is drained only inside the stream, so a retry refused at the door
    loses it."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = object.__new__(LlamaCppBackend)
    backend._port = 1
    backend._maybe_recover_from_mtp_crash = lambda _exc: False
    backend._respawn_if_dead = lambda: True

    attempts = {"n": 0}

    @contextlib.contextmanager
    def _open_stream(_url, _payload, _cancel_event):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ReadError("llama-server died mid-request")
        raise RuntimeError(
            'llama-server returned 400: {"error":{"code":400,"message":"the request '
            'exceeds the available context size...","type":"exceed_context_size_error",'
            '"n_prompt_tokens":6000,"n_ctx":4096}}'
        )

    backend._open_stream = _open_stream

    refit_ran = {"value": False}
    forwarded = {"value": False}

    def _on_respawn() -> None:
        # Recording is the only carrier: the forwarding loop lives in a stream that is never opened.
        refit_ran["value"] = True
        context_refusal.record_fit({"fits": False, "context_length": 4096})

    context_refusal.open_slot()
    try:
        with pytest.raises(RuntimeError, match = "exceed_context_size_error"):
            with backend._open_chat_stream_with_respawn_retry({}, None, on_respawn = _on_respawn):
                forwarded["value"] = True

        assert attempts["n"] == 2
        assert refit_ran["value"] is True
        assert forwarded["value"] is False, "the forwarding loop never runs on this path"

        refusal = context_refusal.latest_refusal()
        assert refusal is not None, "the respawn refit's refusal has to survive the retry"
        assert refusal.get("fits") is False
    finally:
        context_refusal.clear()


def test_the_respawn_refits_record_the_refusal_rather_than_only_forwarding_it():
    """Source pin: refit callbacks sit in generate_chat_completion_with_tools, unreachable by behaviour."""
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    body = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
    for callback in ("_refit_iteration_after_respawn", "_refit_final_after_respawn"):
        start = body.index(f"def {callback}(")
        chunk = body[start : start + 4000]
        assert (
            "context_refusal.record_fit(truncation)" in chunk
        ), f"{callback} must record the fit itself; forwarding is gated on `fits`"
