# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.inference import tools
from test_llama_cpp_tool_loop import _make_backend, _tool_call_sse, _done, _sse, _finish


def test_optional_headroom_never_discards_a_current_result_that_still_fits(monkeypatch):
    payloads = []
    backend = _make_backend(monkeypatch, [
        [_tool_call_sse("python", {"code": "print('facts')"}, "facts"), _finish("tool_calls"), _done()],
        [_tool_call_sse("python", {"code": "print('summary')"}, "summary"), _finish("tool_calls"), _done()],
        [_sse({"content": "Done."}), _done()],
    ], payloads)
    backend._effective_context_length = 8192
    backend._supports_tools = True
    def count(messages, *args, **kwargs):
        return 500 + sum(len(json.dumps(message)) // 4 for message in messages)
    monkeypatch.setattr(backend, "count_chat_tokens", count)
    facts = "EVIDENCE " * 2000
    calls = []
    def execute(name, args, result_budget_tokens=None, **kwargs):
        calls.append((args["code"], result_budget_tokens))
        return facts if len(calls) == 1 else "Summary saved."
    monkeypatch.setattr(tools, "execute_tool", execute)
    list(backend.generate_chat_completion_with_tools(
        messages=[{"role": "user", "content": "Analyze every finding."}],
        tools=[tools.PYTHON_TOOL], max_tokens=None, max_tool_iterations=2,
        context_overflow="truncate_oldest",
    ))
    assert len(calls) == 2
    assert calls[1][1] > 0
    assert any(message.get("content") == facts for message in payloads[-1]["messages"])
    assert count(payloads[-1]["messages"]) < 6144
