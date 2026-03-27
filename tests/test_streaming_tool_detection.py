"""
Exhaustive tests for the speculative-buffer streaming tool detection in
generate_chat_completion_with_tools().

We mock the HTTP layer so llama-server is not required. Each test constructs
the exact SSE byte stream that llama-server would emit, feeds it through the
real method, and asserts on the yielded events.
"""

import json
import threading
import types
import contextlib
from unittest.mock import MagicMock, patch, PropertyMock
import sys, os

# ── helpers ──────────────────────────────────────────────────────────────


def _sse_line(data: dict) -> str:
    """One SSE data line (no trailing blank line -- we add those in the stream)."""
    return f"data: {json.dumps(data)}"


def _sse_done() -> str:
    return "data: [DONE]"


def _make_chunk(delta: dict, finish_reason = None, usage = None, timings = None):
    """Build a chat-completions streaming chunk."""
    choice = {"index": 0, "delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    chunk = {"choices": [choice]}
    if usage:
        chunk["usage"] = usage
    if timings:
        chunk["timings"] = timings
    return chunk


def _build_sse_stream(chunks: list[dict], final_usage = None, final_timings = None) -> str:
    """
    Build a complete SSE text stream from a list of chunk dicts.
    Includes the role chunk, content/tool chunks, and [DONE].
    """
    lines = []
    for c in chunks:
        lines.append(_sse_line(c))
        lines.append("")  # blank line separator
    # Final usage chunk (if provided)
    if final_usage or final_timings:
        meta = {}
        if final_usage:
            meta["usage"] = final_usage
        if final_timings:
            meta["timings"] = final_timings
        meta["choices"] = []
        lines.append(_sse_line(meta))
        lines.append("")
    lines.append(_sse_done())
    lines.append("")
    return "\n".join(lines)


class FakeResponse:
    """Mimics httpx.Response for streaming."""

    def __init__(self, text: str, status_code: int = 200):
        self._text = text
        self.status_code = status_code
        self._closed = False

    def iter_text(self):
        # Yield the whole thing in one shot (simplest case)
        yield self._text

    def read(self):
        return self._text.encode()

    def close(self):
        self._closed = True


class FakeClient:
    """Mimics httpx.Client context manager."""

    def __init__(self, response: FakeResponse):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @contextlib.contextmanager
    def stream(self, method, url, json = None, timeout = None, headers = None):
        yield self._response


# ── Build a minimal LlamaCppBackend for testing ─────────────────────────


def _make_backend():
    """Create a minimal mock backend with just enough to run the method."""
    # We need the real class but only care about generate_chat_completion_with_tools
    # Import the real module
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "unsloth_studio_src")
    )

    # Instead of importing the full module (which has other deps), we'll
    # build a lightweight object that has the method and its dependencies.
    from studio.backend.core.inference.llama_cpp import LlamaCppBackend

    backend = object.__new__(LlamaCppBackend)
    backend._process = True  # is_loaded checks _process is not None
    backend._healthy = True  # is_loaded checks _healthy
    backend._port = 9999  # base_url property reads _port
    backend._api_key = None
    backend._supports_reasoning = False
    return backend


def _synthesis_sse():
    """Build a simple text SSE response for post-tool synthesis."""
    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({"content": "Done."}),
        _make_chunk({}, finish_reason = "stop"),
    ]
    usage = {"prompt_tokens": 20, "completion_tokens": 1}
    return _build_sse_stream(chunks, final_usage = usage)


def _collect_events(backend, sse_text, tools = None, **kwargs):
    """
    Run generate_chat_completion_with_tools with a fake SSE stream
    and collect all yielded events.

    After the first iteration (tool detection), subsequent iterations
    return a plain text synthesis response so the agentic loop terminates.
    """
    if tools is None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]

    call_count = [0]
    synth_sse = _synthesis_sse()

    @contextlib.contextmanager
    def fake_stream_with_retry(client, url, payload, cancel_event, headers = None):
        idx = call_count[0]
        call_count[0] += 1
        # First call: use the provided SSE. Subsequent: plain text synthesis.
        text = sse_text if idx == 0 else synth_sse
        yield FakeResponse(text)

    # Patch execute_tool to return a dummy result
    def fake_execute_tool(
        tool_name, arguments, cancel_event = None, timeout = None, session_id = None
    ):
        return f"Tool {tool_name} result: OK"

    original_stream = backend._stream_with_retry
    backend._stream_with_retry = fake_stream_with_retry

    events = []
    with patch("core.inference.tools.execute_tool", fake_execute_tool, create = True):
        try:
            for event in backend.generate_chat_completion_with_tools(
                messages = [{"role": "user", "content": "Hello"}],
                tools = tools,
                **kwargs,
            ):
                events.append(event)
        except Exception as e:
            import traceback

            traceback.print_exc()
            events.append({"type": "error", "error": str(e)})

    backend._stream_with_retry = original_stream
    return events


# ── The actual tests ─────────────────────────────────────────────────────


def test_no_tool_call_plain_text():
    """90% case: model responds with plain text, no tool call.
    Should stream content immediately without delay."""
    backend = _make_backend()

    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({"content": "Hello"}),
        _make_chunk({"content": " there"}),
        _make_chunk({"content": "!"}, finish_reason = "stop"),
    ]
    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    timings = {"predicted_ms": 100, "predicted_n": 3, "predicted_per_second": 30.0}
    sse = _build_sse_stream(chunks, final_usage = usage, final_timings = timings)

    events = _collect_events(backend, sse)

    # Should have content events with cumulative text
    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) >= 1, f"Expected content events, got: {events}"

    # Final content should contain the full text
    final_content = content_events[-1]["text"]
    assert "Hello there!" in final_content, f"Missing text in: {final_content}"

    # Should have metadata
    meta_events = [e for e in events if e["type"] == "metadata"]
    assert len(meta_events) == 1, f"Expected 1 metadata event, got: {meta_events}"

    # Should have status clear
    status_events = [e for e in events if e["type"] == "status"]
    assert any(e["text"] == "" for e in status_events), "Missing status clear"

    print("PASS: test_no_tool_call_plain_text")


def test_structured_tool_calls():
    """Model emits structured delta.tool_calls (the standard path).
    Should detect instantly and execute."""
    backend = _make_backend()

    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_0",
                        "function": {"name": "web_search", "arguments": ""},
                    }
                ]
            }
        ),
        _make_chunk(
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"query":'}}]}
        ),
        _make_chunk(
            {"tool_calls": [{"index": 0, "function": {"arguments": ' "test"}'}}]}
        ),
        _make_chunk({}, finish_reason = "tool_calls"),
    ]
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    sse = _build_sse_stream(chunks, final_usage = usage)

    events = _collect_events(backend, sse)

    # Should have status update for tool execution
    status_events = [
        e for e in events if e["type"] == "status" and "Searching" in e.get("text", "")
    ]
    assert len(status_events) >= 1, f"Expected search status, got: {events}"

    # Should have tool_start event
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 1, f"Expected 1 tool_start, got: {tool_starts}"
    assert tool_starts[0]["tool_name"] == "web_search"
    assert tool_starts[0]["arguments"] == {"query": "test"}

    # Should have tool_end event
    tool_ends = [e for e in events if e["type"] == "tool_end"]
    assert len(tool_ends) == 1, f"Expected 1 tool_end, got: {tool_ends}"

    print("PASS: test_structured_tool_calls")


def test_xml_tool_call_at_start():
    """Model emits <tool_call>JSON</tool_call> instead of structured tool_calls.
    Buffer should detect <tool_call> prefix and drain."""
    backend = _make_backend()

    tc_json = json.dumps({"name": "web_search", "arguments": {"query": "hello"}})
    content = f"<tool_call>{tc_json}</tool_call>"

    # Stream the XML content token by token to simulate real streaming
    chunks = [_make_chunk({"role": "assistant"})]
    for char in content:
        chunks.append(_make_chunk({"content": char}))
    chunks.append(_make_chunk({}, finish_reason = "stop"))

    usage = {
        "prompt_tokens": 10,
        "completion_tokens": len(content),
        "total_tokens": 10 + len(content),
    }
    sse = _build_sse_stream(chunks, final_usage = usage)

    events = _collect_events(backend, sse)

    # Should detect tool call and execute it
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 1, f"Expected 1 tool_start, got: {events}"
    assert tool_starts[0]["tool_name"] == "web_search"

    tool_ends = [e for e in events if e["type"] == "tool_end"]
    assert len(tool_ends) == 1, f"Expected 1 tool_end"

    print("PASS: test_xml_tool_call_at_start")


def test_xml_function_tag_at_start():
    """Model emits <function=web_search> tag.
    Buffer should detect <function= prefix and drain."""
    backend = _make_backend()

    content = "<function=web_search><parameter=query>hello world</parameter></function>"

    chunks = [_make_chunk({"role": "assistant"})]
    for char in content:
        chunks.append(_make_chunk({"content": char}))
    chunks.append(_make_chunk({}, finish_reason = "stop"))

    usage = {"prompt_tokens": 10, "completion_tokens": len(content)}
    sse = _build_sse_stream(chunks, final_usage = usage)

    events = _collect_events(backend, sse)

    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 1, f"Expected 1 tool_start, got: {events}"
    assert tool_starts[0]["tool_name"] == "web_search"

    print("PASS: test_xml_function_tag_at_start")


def test_whitespace_before_tool_xml():
    """Model emits whitespace then <tool_call>. Buffer should strip
    leading whitespace before prefix check."""
    backend = _make_backend()

    tc_json = json.dumps({"name": "web_search", "arguments": {"query": "test"}})
    content = f"  \n  <tool_call>{tc_json}</tool_call>"

    chunks = [_make_chunk({"role": "assistant"})]
    # Send whitespace as one chunk, then the rest
    chunks.append(_make_chunk({"content": "  \n  "}))
    rest = f"<tool_call>{tc_json}</tool_call>"
    for char in rest:
        chunks.append(_make_chunk({"content": char}))
    chunks.append(_make_chunk({}, finish_reason = "stop"))

    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse)

    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert (
        len(tool_starts) == 1
    ), f"Expected 1 tool_start after whitespace, got: {events}"

    print("PASS: test_whitespace_before_tool_xml")


def test_content_then_tool_xml_safety_net():
    """Rare case: model emits normal content first, then tool XML later.
    Safety net at [DONE] should catch the tool call."""
    backend = _make_backend()

    tc_json = json.dumps({"name": "web_search", "arguments": {"query": "q"}})
    # Start with normal text (triggers STREAMING), then tool XML
    # Send as separate content chunks
    chunks = [_make_chunk({"role": "assistant"})]
    chunks.append(_make_chunk({"content": "Let me search for that. "}))
    chunks.append(_make_chunk({"content": f"<tool_call>{tc_json}</tool_call>"}))
    chunks.append(_make_chunk({}, finish_reason = "stop"))

    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse)

    # The safety net should catch the tool call
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) >= 1, f"Safety net should catch tool, got: {events}"
    assert tool_starts[0]["tool_name"] == "web_search"

    print("PASS: test_content_then_tool_xml_safety_net")


def test_multiple_structured_tool_calls():
    """Model calls two tools in one response (parallel tool calls)."""
    backend = _make_backend()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "python",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                },
            },
        },
    ]

    chunks = [
        _make_chunk({"role": "assistant"}),
        # Two tool calls streamed with different indices
        _make_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_0",
                        "function": {"name": "web_search", "arguments": ""},
                    },
                    {
                        "index": 1,
                        "id": "call_1",
                        "function": {"name": "python", "arguments": ""},
                    },
                ]
            }
        ),
        _make_chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": '{"query": "test"}'}},
                ]
            }
        ),
        _make_chunk(
            {
                "tool_calls": [
                    {"index": 1, "function": {"arguments": '{"code": "print(1)"}'}},
                ]
            }
        ),
        _make_chunk({}, finish_reason = "tool_calls"),
    ]
    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse, tools = tools)

    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 2, f"Expected 2 tool_start events, got: {tool_starts}"

    names = {ts["tool_name"] for ts in tool_starts}
    assert names == {"web_search", "python"}, f"Wrong tool names: {names}"

    print("PASS: test_multiple_structured_tool_calls")


def test_reasoning_tokens_stream_immediately():
    """Thinking model: reasoning_content tokens should stream to user
    immediately, even during BUFFERING state."""
    backend = _make_backend()
    backend._supports_reasoning = True

    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({"reasoning_content": "Let me think..."}),
        _make_chunk({"reasoning_content": " about this."}),
        _make_chunk({"content": "The answer is 42."}),
        _make_chunk({}, finish_reason = "stop"),
    ]
    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse, enable_thinking = True)

    content_events = [e for e in events if e["type"] == "content"]
    assert (
        len(content_events) >= 3
    ), f"Expected at least 3 content events (2 reasoning + 1 content), got: {content_events}"

    # First content events should contain <think> tag
    assert (
        "<think>" in content_events[0]["text"]
    ), "First content should have <think> tag"
    # Last content should have the actual answer
    final = content_events[-1]["text"]
    assert "42" in final, f"Final content should have answer: {final}"

    print("PASS: test_reasoning_tokens_stream_immediately")


def test_reasoning_then_tool_call():
    """Thinking model that reasons then calls a tool.
    Reasoning should stream, then tool detected and executed."""
    backend = _make_backend()
    backend._supports_reasoning = True

    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({"reasoning_content": "I need to search for this."}),
        _make_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_0",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ]
            }
        ),
        _make_chunk({}, finish_reason = "tool_calls"),
    ]
    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse, enable_thinking = True)

    # Reasoning should have been yielded
    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) >= 1, "Reasoning should be yielded"
    assert "<think>" in content_events[0]["text"]
    assert "I need to search" in content_events[0]["text"]

    # Tool should be executed
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 1, f"Expected tool_start: {events}"
    assert tool_starts[0]["tool_name"] == "web_search"

    print("PASS: test_reasoning_then_tool_call")


def test_empty_response():
    """Model returns empty stream (just role + [DONE]). Should not crash."""
    backend = _make_backend()

    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({}, finish_reason = "stop"),
    ]
    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse)

    # Should not crash, just return with no content
    error_events = [e for e in events if e.get("type") == "error"]
    assert len(error_events) == 0, f"Should not error: {error_events}"

    print("PASS: test_empty_response")


def test_buffer_prefix_timeout():
    """Content starts with '<' but is not a tool call (e.g., '<p>Hello</p>').
    Buffer should hold briefly then flush when no prefix match at 32 chars."""
    backend = _make_backend()

    content = "<p>This is a paragraph of HTML content that is not a tool call</p>"
    chunks = [_make_chunk({"role": "assistant"})]
    # Stream char by char
    for char in content:
        chunks.append(_make_chunk({"content": char}))
    chunks.append(_make_chunk({}, finish_reason = "stop"))

    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse)

    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) >= 1, f"Should have content events: {events}"

    # No tool calls should be detected
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 0, f"Should not detect tools in HTML: {tool_starts}"

    # Final content should contain the HTML
    final = content_events[-1]["text"]
    assert "<p>" in final, f"HTML content should pass through: {final}"

    print("PASS: test_buffer_prefix_timeout")


def test_buffer_resolves_to_streaming_on_non_xml_first_char():
    """First content char is not '<' and not whitespace.
    Should immediately transition to STREAMING."""
    backend = _make_backend()

    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({"content": "H"}),  # 'H' is not '<', instant STREAMING
        _make_chunk({"content": "ello"}),
    ]
    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse)

    content_events = [e for e in events if e["type"] == "content"]
    # First content event should appear immediately with just "H"
    assert len(content_events) >= 1
    assert "H" in content_events[0]["text"]

    print("PASS: test_buffer_resolves_to_streaming_on_non_xml_first_char")


def test_draining_false_positive():
    """Buffer detects '<tool' prefix but stream ends before completing
    the tag (e.g., '<tool' then EOF). Should yield content as-is."""
    backend = _make_backend()

    # Content that starts like a tool tag but isn't
    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({"content": "<tool"}),
        _make_chunk({"content": "_tip>Use a screwdriver</tool_tip>"}),
        _make_chunk({}, finish_reason = "stop"),
    ]
    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse)

    # "<tool" is a prefix of "<tool_call>" so it enters BUFFERING.
    # Then "_tip>" does NOT match "<tool_call>" since the buffer becomes
    # "<tool_tip>..." which doesn't start with "<tool_call>" or "<function=".
    # But at >32 chars the buffer should flush.
    # No tool should be executed.
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 0, f"Should not detect tool in <tool_tip>: {tool_starts}"

    print("PASS: test_draining_false_positive")


def test_structured_tool_args_json_parsing():
    """Verify that arguments streamed across multiple chunks get reassembled
    and parsed correctly as JSON."""
    backend = _make_backend()

    # Arguments split across 4 chunks
    arg_parts = ['{"qu', 'ery":', ' "wha', 't is python?"}']

    chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_abc",
                        "function": {"name": "web_search", "arguments": ""},
                    }
                ]
            }
        ),
    ]
    for part in arg_parts:
        chunks.append(
            _make_chunk({"tool_calls": [{"index": 0, "function": {"arguments": part}}]})
        )
    chunks.append(_make_chunk({}, finish_reason = "tool_calls"))

    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse)

    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert len(tool_starts) == 1
    assert tool_starts[0]["arguments"] == {
        "query": "what is python?"
    }, f"Arguments not reassembled correctly: {tool_starts[0]['arguments']}"
    assert tool_starts[0]["tool_call_id"] == "call_abc"

    print("PASS: test_structured_tool_args_json_parsing")


def test_auto_heal_disabled():
    """When auto_heal_tool_calls=False, XML tool calls in content should NOT
    be parsed -- only structured tool_calls are honored."""
    backend = _make_backend()

    tc_json = json.dumps({"name": "web_search", "arguments": {"query": "test"}})
    content = f"<tool_call>{tc_json}</tool_call>"

    chunks = [_make_chunk({"role": "assistant"})]
    # Send as one big content chunk
    chunks.append(_make_chunk({"content": content}))
    chunks.append(_make_chunk({}, finish_reason = "stop"))

    sse = _build_sse_stream(chunks)
    events = _collect_events(backend, sse, auto_heal_tool_calls = False)

    # With auto_heal disabled, the XML should NOT be parsed as a tool call
    tool_starts = [e for e in events if e["type"] == "tool_start"]
    assert (
        len(tool_starts) == 0
    ), f"auto_heal_tool_calls=False should not parse XML tools: {tool_starts}"

    print("PASS: test_auto_heal_disabled")


def test_metrics_accumulation_across_tool_iterations():
    """When tools are called, metrics from the tool iteration should be
    accumulated and included in the final metadata."""
    backend = _make_backend()

    # First iteration: tool call
    tool_chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_0",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query": "test"}',
                        },
                    }
                ]
            }
        ),
        _make_chunk({}, finish_reason = "tool_calls"),
    ]
    tool_usage = {"prompt_tokens": 10, "completion_tokens": 5}
    tool_timings = {"predicted_ms": 50, "predicted_n": 5}
    tool_sse = _build_sse_stream(
        tool_chunks, final_usage = tool_usage, final_timings = tool_timings
    )

    # Second iteration: plain text response (synthesis)
    synth_chunks = [
        _make_chunk({"role": "assistant"}),
        _make_chunk({"content": "Based on my search, the answer is X."}),
        _make_chunk({}, finish_reason = "stop"),
    ]
    synth_usage = {"prompt_tokens": 20, "completion_tokens": 8}
    synth_timings = {"predicted_ms": 100, "predicted_n": 8}
    synth_sse = _build_sse_stream(
        synth_chunks, final_usage = synth_usage, final_timings = synth_timings
    )

    # We need to return different SSE streams for each iteration
    call_count = [0]
    original_sse = [tool_sse, synth_sse]

    fake_responses = [FakeResponse(tool_sse), FakeResponse(synth_sse)]

    @contextlib.contextmanager
    def fake_stream_with_retry(client, url, payload, cancel_event, headers = None):
        idx = min(call_count[0], len(fake_responses) - 1)
        call_count[0] += 1
        yield fake_responses[idx]

    def fake_execute_tool(
        tool_name, arguments, cancel_event = None, timeout = None, session_id = None
    ):
        return "Search result: success"

    backend._stream_with_retry = fake_stream_with_retry

    events = []
    with patch("core.inference.tools.execute_tool", fake_execute_tool, create = True):
        for event in backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Search for test"}],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
        ):
            events.append(event)

    meta_events = [e for e in events if e["type"] == "metadata"]
    assert (
        len(meta_events) == 1
    ), f"Expected exactly 1 metadata event, got: {meta_events}"

    meta = meta_events[0]
    # completion_tokens should be accumulated: 5 (tool iter) + 8 (synthesis) = 13
    assert (
        meta["usage"]["completion_tokens"] == 13
    ), f"Expected 13 total completion tokens, got: {meta['usage']['completion_tokens']}"

    # predicted_ms and predicted_n should also accumulate
    assert (
        meta["timings"]["predicted_n"] == 13
    ), f"Expected 13 predicted_n, got: {meta['timings']['predicted_n']}"

    print("PASS: test_metrics_accumulation_across_tool_iterations")


# ── Run all tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_no_tool_call_plain_text,
        test_structured_tool_calls,
        test_xml_tool_call_at_start,
        test_xml_function_tag_at_start,
        test_whitespace_before_tool_xml,
        test_content_then_tool_xml_safety_net,
        test_multiple_structured_tool_calls,
        test_reasoning_tokens_stream_immediately,
        test_reasoning_then_tool_call,
        test_empty_response,
        test_buffer_prefix_timeout,
        test_buffer_resolves_to_streaming_on_non_xml_first_char,
        test_draining_false_positive,
        test_structured_tool_args_json_parsing,
        test_auto_heal_disabled,
        test_metrics_accumulation_across_tool_iterations,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            import traceback

            print(f"FAIL: {test_fn.__name__}: {e}")
            traceback.print_exc()
            print()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if errors:
        print(f"\nFailed tests:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    print(f"{'='*60}")
