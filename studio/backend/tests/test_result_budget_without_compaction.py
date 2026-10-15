# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""With auto-compaction off, a tool result may use the window up to the reply floor (#13784).

Reported at a 160K window with Max output tokens: from 120K (75%) on, every MCP result came
back as `(output omitted: N chars, no context room left)`. The result budget reserved
`prompt_budget`'s quarter of the window for the reply, which is the fit compaction trims to;
with compaction off nothing trims, llama-server admits up to the window, and the
pre-execution gate already admits up to the reply floor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference import tools as tools_module  # noqa: E402
from core.inference.context_window import _reply_floor  # noqa: E402
from test_no_progress_tool_results import (  # noqa: E402
    _WEB_SEARCH_TOOL,
    _call,
    _done,
    _make_backend,
    _sse,
)

_CTX = 160_000
_SPENT = 120_500
_MCP_TOOL = "mcp__2e0d1d7fd32e4799__execute_code"
_OUTPUT = "print(1)\n" * 170  # 1530 chars, the size from the report


def _tool(name):
    return {**_WEB_SEARCH_TOOL, "function": {**_WEB_SEARCH_TOOL["function"], "name": name}}


def _named_call(name):
    return _call("run it").replace('"web_search"', json.dumps(name))


def _run_one_call(
    monkeypatch,
    *,
    context_overflow,
    max_tokens = _CTX,
    messages = None,
    tool_name = "web_search",
    vision = False,
    count = lambda *_a, **_k: _SPENT,
):
    received: list[object] = []
    results: list[str] = []

    def _execute(_name, _args, **kwargs):
        budget = kwargs.get("result_budget_tokens")
        received.append(budget)
        token = tools_module._REQUEST_RESULT_BUDGET.set(budget)
        try:
            return tools_module._fit_result_to_room(_OUTPUT, _MCP_TOOL)
        finally:
            tools_module._REQUEST_RESULT_BUDGET.reset(token)

    backend = _make_backend(
        monkeypatch,
        [[_named_call(tool_name), _done()], [_sse({"content": "Done."}), _done()]],
        [],
    )
    backend._effective_context_length = _CTX
    backend._is_vision = vision
    backend._mmproj_accepts_image = vision
    monkeypatch.setattr(backend, "count_chat_tokens", count)
    monkeypatch.setattr("core.inference.tools.execute_tool", _execute)

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = messages or [{"role": "user", "content": "Run the script"}],
            tools = [_WEB_SEARCH_TOOL, _tool(_MCP_TOOL)],
            max_tool_iterations = 4,
            max_tokens = max_tokens,
            context_overflow = context_overflow,
        )
    )
    results = [e.get("result", "") for e in events if e.get("type") == "tool_end"]
    assert received, "the tool never ran"
    return received[0], results


# None is what the route sends with the switch off; "error" is the frontend's own field.
@pytest.mark.parametrize("context_overflow", [None, "error"])
def test_compaction_off_prices_the_result_to_the_reply_floor(monkeypatch, context_overflow):
    budget, results = _run_one_call(monkeypatch, context_overflow = context_overflow)

    assert budget == int((_CTX - _reply_floor(_CTX)) * 0.99) - _SPENT
    assert results and "no context room left" not in results[0]
    assert "No room in the window" not in results[0]
    assert _OUTPUT in results[0]


def test_compaction_off_still_honours_a_smaller_output_cap(monkeypatch):
    budget, _ = _run_one_call(monkeypatch, context_overflow = None, max_tokens = 4096)

    assert budget == int((_CTX - 4096) * 0.99) - _SPENT


def test_compaction_on_keeps_the_fit_target(monkeypatch):
    budget, _ = _run_one_call(monkeypatch, context_overflow = "truncate_oldest")

    # 0.99 * 120K - 120.5K: the fit compaction trims to, unchanged.
    assert budget == 0


def test_compaction_off_keeps_the_reserve_while_media_is_unpriced(monkeypatch):
    # The count strips images, so only the quarter reserve stands between the result and them.
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
    budget, _ = _run_one_call(
        monkeypatch,
        context_overflow = None,
        messages = [{"role": "user", "content": [{"type": "text", "text": "Run it"}, image]}],
    )

    assert budget == 0


def test_compaction_off_keeps_the_reserve_without_an_exact_count(monkeypatch):
    # The fallback estimate undercounts dense text; the quarter reserve absorbs that.
    def _fails(*_a, **_k):
        raise RuntimeError("tokenizer unavailable")

    off, _ = _run_one_call(monkeypatch, context_overflow = None, count = _fails)
    on, _ = _run_one_call(monkeypatch, context_overflow = "truncate_oldest", count = _fails)

    assert off == on


def test_compaction_off_reserves_the_images_an_mcp_result_can_add(monkeypatch):
    from core.inference.llama_cpp import _FALLBACK_IMAGE_TOKENS
    from core.inference.mcp_images import MAX_MODEL_IMAGES

    reserve = _reply_floor(_CTX) + MAX_MODEL_IMAGES * _FALLBACK_IMAGE_TOKENS
    vision, _ = _run_one_call(monkeypatch, context_overflow = None, tool_name = _MCP_TOOL, vision = True)
    text_only, _ = _run_one_call(monkeypatch, context_overflow = None, tool_name = _MCP_TOOL)

    assert vision == int((_CTX - reserve) * 0.99) - _SPENT
    assert text_only == int((_CTX - _reply_floor(_CTX)) * 0.99) - _SPENT
