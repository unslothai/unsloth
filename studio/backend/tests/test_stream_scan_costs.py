# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-chunk scans in the streaming tool loops: same answers as the plain walks, without the rescans."""

from __future__ import annotations

import random
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


from core import tool_healing
from core.inference import llama_cpp, safetensors_agentic
from core.inference.tool_call_parser import held_bare_gemma_tail_len

_TOOLS = [
    {"type": "function", "function": {"name": name}}
    for name in ("web_search", "terminal", "python", "mcp__srv__tool", " padded ")
] + [{"type": "function", "function": {"name": 7}}, {"type": "function", "function": "bad"}]


def _walk_tail(text: str) -> str:
    i = len(text)
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return text[i:]


def _gguf_hold_reference(text: str, tools: list) -> int:
    tail = _walk_tail(text)
    held = len(tail) if tail and llama_cpp._is_rehearsal_prefix(tail, tools) else 0
    names = lambda: set(llama_cpp._gguf_active_tool_names(tools))  # noqa: E731
    return max(held, held_bare_gemma_tail_len(text, names))


def _st_hold_reference(text: str, tools: list, unrestricted: bool) -> int:
    tail = _walk_tail(text)
    held = (
        len(tail)
        if tail and safetensors_agentic._is_rehearsal_prefix(tail, tools, unrestricted = unrestricted)
        else 0
    )
    names = lambda: None if unrestricted else safetensors_agentic._active_tool_names(tools)  # noqa: E731
    return max(held, held_bare_gemma_tail_len(text, names))


def _brace_end_reference(content: str, brace_start: int, gemma_quotes: bool) -> int | None:
    quote = tool_healing._GEMMA_QUOTE
    if brace_start >= len(content) or content[brace_start] != "{":
        return None
    depth, i, in_string, in_gemma = 0, brace_start, False, False
    while i < len(content):
        if gemma_quotes and not in_string and content.startswith(quote, i):
            in_gemma = not in_gemma
            i += len(quote)
            continue
        ch = content[i]
        if in_gemma:
            i += 1
            continue
        if in_string:
            if ch == "\\" and i + 1 < len(content):
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def test_held_tail_matches_the_unbounded_walk():
    rng = random.Random(0)
    pieces = ["web_search", "web_sea", "[ARGS]", "[AR", " ", "\n", "　", "\xa0", "x", "中文"]
    pieces += ["call", "call:", "call:web_search{", "}", "{", "terminal", "python[ARGS]", "padded"]
    for _ in range(1500):
        tools = rng.sample(_TOOLS, rng.randint(0, len(_TOOLS)))
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 24)))
        assert llama_cpp._held_rehearsal_tail_len(text, tools) == _gguf_hold_reference(
            text, tools
        ), (text, tools)
        for unrestricted in (False, True):
            got = safetensors_agentic._held_rehearsal_tail_len(
                text, tools, unrestricted = unrestricted
            )
            assert got == _st_hold_reference(text, tools, unrestricted), (text, unrestricted)


def test_held_tail_does_not_rewalk_spaceless_text():
    """CJK prose has no spaces, so walking back to the last one per chunk made a reply quadratic."""
    tools = _TOOLS[:3]

    # A count of indexed reads, not a timing, so it is the same on every runner.
    class CountingStr(str):
        reads = 0

        def __getitem__(self, key):
            CountingStr.reads += 1
            return str.__getitem__(self, key)

    for module in (llama_cpp, safetensors_agentic):
        reads = []
        for n in (100, 10_000):
            CountingStr.reads = 0
            assert (
                module._held_rehearsal_tail_len(CountingStr("这是没有空格的中文文本" * n), tools)
                == 0
            )
            reads.append(CountingStr.reads)
        assert reads[0] == reads[1] < 100, (module.__name__, reads)
        held = CountingStr("这是没有空格的中文文本" * 10_000 + " web_sea")
        assert module._held_rehearsal_tail_len(held, tools) == 7


def test_balanced_brace_end_matches_the_character_walk():
    rng = random.Random(1)
    pieces = ["{", "}", '"', "\\", '\\"', "\\\\", '<|"|>', "<|", '"|>', "<", "a", " ", "中", "\n"]
    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 20)))
        for start in range(len(text) + 1):
            for gemma in (False, True):
                assert tool_healing._balanced_brace_end(
                    text, start, gemma_quotes = gemma
                ) == _brace_end_reference(text, start, gemma), (text, start, gemma)


def test_passthrough_normalize_hands_back_the_decoded_line():
    """The relay reuses this decode instead of parsing the line again, so it must equal one."""
    import json

    from routes.inference import (
        _SSE_LINE_UNPARSED,
        _normalize_openai_passthrough_sse_line,
        _normalize_openai_passthrough_sse_line_and_data,
    )

    calls = '[{"index":0,"function":{"name":"a"}},{"index":1,"function":{"name":"b"}}]'
    lines = [
        'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
        'data: {"choices":[{"index":0,"delta":{"reasoning_content":"t é 🙂"}}],"x":1e400}',
        'data:{"choices":[{"index":0,"delta":{"reasoning_content":"t","content":"c"}}]}',
        'data: {"choices":[{"delta":{"tool_calls":' + calls + "}}]}",
        'data: ["reasoning_content"]',
        'data: "reasoning_content"',
        'data: {"reasoning_content": broken',
        "data: [DONE]",
        ": keepalive",
    ]
    for line in lines:
        for cap in (False, True):
            out, data = _normalize_openai_passthrough_sse_line_and_data(
                line, cap_parallel_tool_calls = cap
            )
            assert out == _normalize_openai_passthrough_sse_line(line, cap_parallel_tool_calls = cap)
            if data is _SSE_LINE_UNPARSED:
                continue
            assert data == json.loads(out[len("data:") :].strip()), (line, cap)


def test_literal_anchored_search_matches_the_regex_search():
    from core.inference import tool_call_parser as tcp

    pieces = [
        "call",
        "call:",
        "call :",
        "call:web_search{",
        "xcall:a{",
        "_call:a{",
        "recall:b{",
        "{",
        "}",
    ]
    pieces += [
        " ",
        "\n",
        "a",
        "b:",
        "<tool_call>",
        "</tool_call>",
        "中",
        "call:\n web_search {",
        '<|"|>',
    ]
    rng = random.Random(0)
    for _ in range(20000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 30)))
        pos = rng.randint(0, len(text))
        want = tcp._GEMMA_BARE_TC_RE.search(text, pos)
        got = tcp._search_from_literal(tcp._GEMMA_BARE_TC_RE, "call", text, pos)
        assert (want and (want.span(), want.groups())) == (got and (got.span(), got.groups())), (
            text,
            pos,
        )
