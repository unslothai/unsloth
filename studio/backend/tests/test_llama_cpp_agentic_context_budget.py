# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the agentic-loop context-budget guard added for issue #5562.

The guard's contract:
  * a cheap char-based token estimate over a JSON-serialized conversation
  * a constant ratio (default 0.85) of effective context length
  * once estimated tokens exceed ratio * n_ctx, the agentic loop breaks
    early so the post-loop synthesis pass can produce a final answer
    instead of llama-server returning "request exceeds context size"
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


# Optional-dependency stubs.  Use `setdefault`-on-failed-import instead
# of an unconditional `sys.modules.setdefault(...)` so we never shadow a
# real module that just has not been imported yet -- otherwise later
# tests in the same pytest session that do `import httpx` (etc.) would
# receive our stub and fail with AttributeError on real-API access.
def _install_stub_if_missing(name: str, factory) -> None:
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = factory()


def _build_loggers_stub() -> _types.ModuleType:
    mod = _types.ModuleType("loggers")
    mod.get_logger = lambda name: __import__("logging").getLogger(name)
    return mod


def _build_structlog_stub() -> _types.ModuleType:
    mod = _types.ModuleType("structlog")
    mod.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return mod


def _build_httpx_stub() -> _types.ModuleType:
    mod = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
    ):
        setattr(mod, _exc, type(_exc, (Exception,), {}))
    mod.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    mod.Client = type(
        "C",
        (),
        {
            "__init__": lambda s, **kw: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    return mod


_install_stub_if_missing("loggers", _build_loggers_stub)
_install_stub_if_missing("structlog", _build_structlog_stub)
_install_stub_if_missing("httpx", _build_httpx_stub)

import pytest

from core.inference.llama_cpp import (
    _CHARS_PER_TOKEN_ESTIMATE,
    _CONTEXT_BUDGET_RATIO,
    _estimate_conversation_tokens,
)


# Constants -- pin the budget ratio so a silent change to "be safe and
# leave 50% headroom" doesn't ship without an explicit decision.


def test_context_budget_ratio_leaves_headroom():
    # Must leave enough room for a real final answer (>= 10% of ctx),
    # but not so much that small conversations trigger it (<= 30%).
    assert 0.70 < _CONTEXT_BUDGET_RATIO < 0.95


def test_chars_per_token_estimate_in_reasonable_range():
    # 2.0 is too aggressive (over-counts); 5.0 under-counts and lets
    # llama-server error out. 3-4 is the safe band for English + JSON.
    assert 2.5 < _CHARS_PER_TOKEN_ESTIMATE < 5.0


# Helper -- handles the JSON-serializable common case.


def test_estimate_tokens_empty_conversation():
    assert _estimate_conversation_tokens([]) == 0


def test_estimate_tokens_small_conversation_is_small():
    msgs = [{"role": "user", "content": "hi"}]
    est = _estimate_conversation_tokens(msgs)
    # Sanity: well under any realistic context window.
    assert 0 < est < 100


def test_estimate_tokens_scales_with_content_length():
    short = [{"role": "user", "content": "hello world"}]
    long = [{"role": "user", "content": "hello world " * 1000}]
    short_est = _estimate_conversation_tokens(short)
    long_est = _estimate_conversation_tokens(long)
    # The long message should produce a meaningfully larger estimate.
    assert long_est > short_est * 100


def test_estimate_tokens_includes_tool_definitions():
    msgs = [{"role": "user", "content": "search for cats"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for a query. " * 50,
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    bare = _estimate_conversation_tokens(msgs)
    with_tools = _estimate_conversation_tokens(msgs, tools)
    assert with_tools > bare


def test_estimate_tokens_handles_tool_role_messages():
    # The realistic case we are guarding against: many tool round-trips
    # with large search payloads accumulating in the conversation.
    msgs = [{"role": "user", "content": "list 2015 Billboard #3 songs"}]
    for i in range(10):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q": "billboard hot 100 2015 number 3"}',
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "name": "web_search",
                "tool_call_id": f"call_{i}",
                "content": "Result snippet about a song that charted. " * 200,
            }
        )
    est = _estimate_conversation_tokens(msgs)
    # 10 tool results of ~200 reps of a ~40-char snippet ≈ 80k chars / 3.5
    # ≈ 20k tokens.  Well above a 16k context budget, which is exactly
    # the case the loop guard catches.
    assert est > 15000


def test_estimate_tokens_falls_back_on_non_serializable():
    # Pre-existing conversation state should never crash the helper --
    # the loop relies on it on every iteration.  Custom objects without
    # a JSON representation must degrade to a length-based estimate.
    class _Opaque:
        def __str__(self):
            return "x" * 100

    msgs = [{"role": "user", "obj": _Opaque()}]
    est = _estimate_conversation_tokens(msgs)
    # Some non-zero estimate must be returned, not an exception.
    assert est > 0


# Budget threshold semantics -- the exact arithmetic the loop applies.


@pytest.mark.parametrize(
    "ctx_length,prompt_chars,expect_trip",
    [
        # 16k ctx, small prompt: never trips.
        (16384, 1000, False),
        # 16k ctx, prompt at ~60% of budget: does not trip.
        (16384, int(0.60 * 16384 * _CHARS_PER_TOKEN_ESTIMATE), False),
        # 16k ctx, prompt at ~90% of budget: trips.
        (16384, int(0.90 * 16384 * _CHARS_PER_TOKEN_ESTIMATE), True),
        # 4k ctx, near-full prompt: trips.
        (4096, int(0.95 * 4096 * _CHARS_PER_TOKEN_ESTIMATE), True),
    ],
)
def test_budget_threshold_matches_loop_arithmetic(
    ctx_length, prompt_chars, expect_trip
):
    msgs = [{"role": "user", "content": "x" * prompt_chars}]
    projected = _estimate_conversation_tokens(msgs)
    budget = int(_CONTEXT_BUDGET_RATIO * ctx_length)
    assert (projected > budget) is expect_trip
