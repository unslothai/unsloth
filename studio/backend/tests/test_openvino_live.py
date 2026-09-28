# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live integration tests: a running Studio serving an OpenVINO model, driven over HTTP.

    UNSLOTH_E2E_OPENVINO=1 UNSLOTH_E2E_BASE_URL=http://127.0.0.1:8000 UNSLOTH_E2E_API_KEY=sk-... \\
        pytest tests/test_openvino_live.py
    # plus UNSLOTH_E2E_OPENVINO_LONG=1 for the context-limit cases (several minutes of prefill)
"""

import json
import os
import re
import threading
import time

import httpx
import pytest

BASE = (os.environ.get("UNSLOTH_E2E_BASE_URL") or "").rstrip("/")
KEY = os.environ.get("UNSLOTH_E2E_API_KEY") or ""
URL = f"{BASE}/v1/chat/completions"
AUTH = {"Authorization": f"Bearer {KEY}"}
LONG = os.environ.get("UNSLOTH_E2E_OPENVINO_LONG") == "1"

pytestmark = pytest.mark.skipif(
    os.environ.get("UNSLOTH_E2E_OPENVINO") != "1" or not BASE,
    reason = "needs UNSLOTH_E2E_OPENVINO=1 and a Studio serving an OpenVINO model",
)

READ = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["path"],
        },
    },
}
WEATHER = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
ASK_READ = [{"role": "user", "content": "Read the file a.txt"}]
FRANCE = [{"role": "user", "content": "Capital of France? One word."}]
MARKUP = ("<think>", "</think>", "<tool_call>")


def post(body: dict, timeout: float = 300, headers = AUTH) -> httpx.Response:
    return httpx.post(URL, json = {"model": "x", "max_tokens": 800, **body}, headers = headers, timeout = timeout)


def chat(body: dict, timeout: float = 300) -> dict:
    """One reply, streamed or not, folded into {content, reasoning, calls, finish, usage}."""
    r = post(body, timeout)
    assert r.status_code == 200, r.text[:500]
    if not body.get("stream"):
        choice = r.json()["choices"][0]
        m = choice["message"]
        return {
            "content": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or "",
            "calls": m.get("tool_calls") or [],
            "finish": choice["finish_reason"],
            "usage": r.json().get("usage"),
        }
    lines = [l[6:] for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines and lines[-1] == "[DONE]", lines[-3:]
    out = {"content": "", "reasoning": "", "calls": {}, "finish": None, "usage": None, "finishes": 0}
    for raw in lines[:-1]:
        event = json.loads(raw)
        out["usage"] = event.get("usage") or out["usage"]
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            out["content"] += delta.get("content") or ""
            out["reasoning"] += delta.get("reasoning_content") or ""
            for tc in delta.get("tool_calls") or []:
                slot = out["calls"].setdefault(tc.get("index", 0), {"function": {"name": "", "arguments": ""}})
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                out["finish"] = choice["finish_reason"]
                out["finishes"] += 1
    assert out["finishes"] == 1, "exactly one finish_reason per stream"
    out["calls"] = list(out["calls"].values())
    return out


def no_markup(r: dict) -> None:
    for tag in MARKUP:
        assert tag not in r["content"], f"{tag} leaked into content: {r['content'][:200]!r}"


def args_of(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


# --- tool calling -------------------------------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [True, False])
def test_tool_call(stream, thinking):
    r = chat({"messages": ASK_READ, "tools": [READ], "stream": stream, "enable_thinking": thinking})
    no_markup(r)
    assert r["finish"] == "tool_calls"
    assert [c["function"]["name"] for c in r["calls"]] == ["read"]
    assert args_of(r["calls"][0])["path"] == "a.txt"


def test_opencode_shaped_request():
    """Two system messages and content as parts, as opencode sends them."""
    msgs = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
        {"role": "user", "content": [{"type": "text", "text": "Read the file a.txt"}]},
    ]
    r = chat({"messages": msgs, "tools": [READ], "stream": True})
    assert r["finish"] == "tool_calls" and args_of(r["calls"][0])["path"] == "a.txt"


def test_parameter_types_follow_schema():
    r = chat({"messages": [{"role": "user", "content": "Read the first 5 lines of a.txt (use limit)"}], "tools": [READ]})
    a = args_of(r["calls"][0])
    assert a["limit"] == 5 and isinstance(a["path"], str)


def test_non_ascii_argument_preserved():
    r = chat({"messages": [{"role": "user", "content": "Прочитай файл отчёт_2026.txt"}], "tools": [READ]})
    assert args_of(r["calls"][0])["path"] == "отчёт_2026.txt"


def test_picks_the_right_tool():
    r = chat({"messages": [{"role": "user", "content": "What's the weather in Berlin?"}], "tools": [READ, WEATHER], "stream": True})
    assert [c["function"]["name"] for c in r["calls"]] == ["weather"]


def test_parallel_calls_have_distinct_ids():
    msgs = [{"role": "user", "content": "Get the weather in Berlin and in Paris. Call the tool for both cities at once."}]
    r = post({"messages": msgs, "tools": [WEATHER]}).json()["choices"][0]["message"]
    cities = sorted(args_of(c)["city"] for c in r["tool_calls"])
    assert cities == ["Berlin", "Paris"]
    assert len({c["id"] for c in r["tool_calls"]}) == 2


@pytest.mark.parametrize("stream", [False, True])
def test_answer_uses_tool_result(stream):
    msgs = ASK_READ + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a.txt"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "The secret code is 7429."},
    ]
    r = chat({"messages": msgs, "tools": [READ], "stream": stream})
    no_markup(r)
    assert r["finish"] == "stop" and "7429" in r["content"] and not r["calls"]


def test_no_call_when_not_needed():
    r = chat({"messages": [{"role": "user", "content": "Say hello. Do not use tools."}], "tools": [READ], "stream": True})
    no_markup(r)
    assert not r["calls"] and r["content"].strip()


def test_tool_choice_none_never_calls():
    r = chat({"messages": ASK_READ, "tools": [READ], "tool_choice": "none", "max_tokens": 3000})
    no_markup(r)
    assert not r["calls"] and r["finish"] in ("stop", "length")


def test_forced_tool_choice_object_is_accepted():
    r = chat({"messages": ASK_READ, "tools": [READ], "tool_choice": {"type": "function", "function": {"name": "read"}}})
    assert r["calls"] and r["calls"][0]["function"]["name"] == "read"


def test_empty_tools_list_is_plain_chat():
    r = chat({"messages": FRANCE, "tools": [], "enable_thinking": False})
    assert not r["calls"] and "Paris" in r["content"]


def test_null_parameters_schema():
    tool = {"type": "function", "function": {"name": "read", "parameters": None}}
    assert post({"messages": ASK_READ, "tools": [tool]}).status_code == 200


# --- history the template has to digest ----------------------------------------------------------


def test_broken_arguments_in_history():
    msgs = ASK_READ + [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "read", "arguments": "{broken"}}
        ]},
        {"role": "tool", "tool_call_id": "c", "content": "ok"},
    ]
    assert post({"messages": msgs, "tools": [READ]}).status_code == 200


def test_orphan_tool_message_is_not_a_server_error():
    r = post({"messages": ASK_READ + [{"role": "tool", "tool_call_id": "nope", "content": "x"}], "tools": [READ]})
    assert r.status_code < 500, r.text[:300]


def test_assistant_content_as_parts():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": "bye"},
    ]
    assert post({"messages": msgs}).status_code == 200


# --- reasoning ---------------------------------------------------------------------------------


@pytest.mark.parametrize("thinking", [True, False])
def test_reasoning_split_from_content(thinking):
    r = chat({"messages": FRANCE, "enable_thinking": thinking, "stream": True})
    no_markup(r)
    assert "Paris" in r["content"]
    assert "Paris" not in r["reasoning"] or r["reasoning"] != r["content"]


# --- request validation ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, status",
    [
        ({"model": "x"}, 400),  # no messages
        ({"messages": []}, 400),
        ({"messages": [{"role": "assistant", "content": "only me"}]}, 400),  # template refuses
        ({"messages": ASK_READ, "tools": [{"type": "function"}]}, 400),
        ({"messages": ASK_READ, "max_tokens": 0}, 400),
        ({"messages": ASK_READ, "max_tokens": -5}, 400),
        ({"messages": ASK_READ, "temperature": -1}, 400),
        ({"messages": ASK_READ, "top_p": 1.5}, 400),
        (
            {"messages": [{"role": "user", "content": [
                {"type": "text", "text": "what"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
            ]}]},
            400,
        ),
    ],
)
def test_rejected_requests(body, status):
    r = httpx.post(URL, json = body, headers = AUTH, timeout = 60)
    assert r.status_code == status, r.text[:300]
    assert "error" in r.json()


def test_malformed_json_body():
    r = httpx.post(URL, content = b"{nope", headers = {**AUTH, "Content-Type": "application/json"}, timeout = 60)
    assert r.status_code == 400


def test_missing_key():
    assert post({"messages": FRANCE}, headers = {}).status_code == 401


# --- tokens: budget, usage, truncation ----------------------------------------------------------


@pytest.mark.parametrize("stream", [False, True])
def test_usage_and_length_finish(stream):
    r = chat({"messages": FRANCE, "max_tokens": 20, "stream": stream})
    assert r["finish"] == "length"
    u = r["usage"]
    assert u["completion_tokens"] == 20 and u["prompt_tokens"] > 0
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def test_max_tokens_one():
    r = chat({"messages": ASK_READ, "tools": [READ], "max_tokens": 1})
    assert r["finish"] == "length" and not r["calls"] and r["usage"]["completion_tokens"] == 1


def test_call_cut_by_budget_leaves_no_markup():
    r = chat({"messages": ASK_READ, "tools": [READ], "max_tokens": 60, "enable_thinking": False})
    no_markup(r)


def test_huge_max_tokens_is_clamped():
    r = chat({"messages": FRANCE, "max_tokens": 99_999_999_999, "enable_thinking": False})
    assert "Paris" in r["content"]


def test_prompt_over_context_is_rejected_fast():
    t = time.monotonic()
    r = post({"messages": [{"role": "user", "content": "hello " * 300_000}], "max_tokens": 10}, timeout = 120)
    assert r.status_code == 400 and "maximum context length" in r.text
    assert time.monotonic() - t < 30, "rejected before prefill"


def _context_limit() -> int:
    r = post({"messages": [{"role": "user", "content": "hello " * 300_000}], "max_tokens": 10}, timeout = 120)
    return int(re.search(r"maximum context length is (\d+)", r.text).group(1))


def _prompt_tokens(n_hello: int) -> int:
    r = post({"messages": [{"role": "user", "content": "hello " * n_hello}], "max_tokens": 1}, timeout = 1800)
    return r.json()["usage"]["prompt_tokens"] if r.status_code == 200 else -1


@pytest.mark.skipif(not LONG, reason = "set UNSLOTH_E2E_OPENVINO_LONG=1 (minutes of prefill)")
def test_context_boundary():
    """At the advertised limit: limit-1 still answers with real tokens, the limit itself is a 400."""
    limit = _context_limit()
    overhead = _prompt_tokens(1000) - 1000  # template tokens around the user text
    below = limit - 50 - overhead
    r = post({"messages": [{"role": "user", "content": "hello " * below + "\nSay OK."}], "max_tokens": 500, "enable_thinking": False}, timeout = 1800)
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["usage"]["completion_tokens"] > 0, f"nothing generated at {body['usage']}"
    assert body["usage"]["total_tokens"] <= limit
    at = post({"messages": [{"role": "user", "content": "hello " * (limit - overhead)}], "max_tokens": 10}, timeout = 120)
    assert at.status_code == 400


# --- robustness --------------------------------------------------------------------------------


def test_concurrent_requests_both_answer():
    out = [None, None]

    def go(i):
        out[i] = post({"messages": [{"role": "user", "content": f"Say the number {i + 1}."}], "enable_thinking": False, "max_tokens": 200})

    threads = [threading.Thread(target = go, args = (i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [o.status_code for o in out] == [200, 200]


def test_client_disconnect_does_not_wedge_the_model():
    body = {"model": "x", "stream": True, "max_tokens": 400, "messages": [{"role": "user", "content": "Write a long story."}]}
    with httpx.stream("POST", URL, json = body, headers = AUTH, timeout = 60) as r:
        for i, _ in enumerate(r.iter_lines()):
            if i > 5:
                break
    r = chat({"messages": FRANCE, "enable_thinking": False, "max_tokens": 300}, timeout = 180)
    assert "Paris" in r["content"]
