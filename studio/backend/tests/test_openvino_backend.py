# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
import textwrap
import types

import pytest

from core.inference import openvino_backend as ovb


def _ir(path, marker = "openvino_model.xml"):
    path.mkdir(parents = True, exist_ok = True)
    (path / marker).write_text("<net/>")
    return path


def test_resolves_local_dir_prefix_and_hub_cache(tmp_path, monkeypatch):
    local = _ir(tmp_path / "m-ov_int4", "openvino_language_model.xml")
    assert ovb.resolve_openvino_dir(str(local)) == local.resolve()
    assert ovb.resolve_openvino_dir(f"openvino:{local}") == local.resolve()

    hub = tmp_path / "hub"
    repo = hub / "models--org--m-ov_int4"
    _ir(repo / "snapshots" / "old")
    _ir(repo / "snapshots" / "local_export")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("local_export")
    monkeypatch.setattr(ovb, "_hub_cache", lambda: hub)
    assert ovb.resolve_openvino_dir("org/m-ov_int4") == repo / "snapshots" / "local_export"


def test_non_openvino_paths_are_not_claimed(tmp_path, monkeypatch):
    monkeypatch.setattr(ovb, "_hub_cache", lambda: tmp_path)
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "model.safetensors").write_text("")
    for path in [None, "", str(tmp_path / "plain"), "org/missing", "unsloth/Qwen3-GGUF:Q4_K_M"]:
        assert not ovb.is_openvino_model_path(path)


def test_sidecar_python_needs_openvino_genai(monkeypatch, tmp_path):
    monkeypatch.delenv(ovb.PYTHON_ENV, raising = False)
    monkeypatch.setattr(ovb.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ovb.OpenVinoError, match = "pip install openvino-genai"):
        ovb.sidecar_python()
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setenv(ovb.PYTHON_ENV, str(py))
    assert ovb.sidecar_python() == str(py)


def test_think_splitter_routes_reasoning_even_across_split_tags(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino_genai", types.ModuleType("openvino_genai"))
    from core.inference.openvino_sidecar import ThinkSplitter

    s = ThinkSplitter(thinking = True)
    out = []
    for piece in ["plan ", "it</th", "ink>\n\nPar", "is"]:
        out += s.feed(piece)
    out += s.flush()
    joined = {k: "".join(t for kk, t in out if kk == k) for k in ("reasoning", "content")}
    assert joined == {"reasoning": "plan it", "content": "Paris"}

    s = ThinkSplitter(thinking = False)
    assert s.feed("hi") + s.flush() == [("content", "hi")]

    # Reasoning despite enable_thinking=false ends with a bare </think>.
    s = ThinkSplitter(thinking = False)
    out = s.feed("user wants X\n</th") + s.feed("ink>\n\nX") + s.feed("!") + s.flush()
    assert out == [("reasoning", "user wants X"), ("content", "X"), ("content", "!")]


def test_tool_splitter_turns_tool_call_markup_into_tool_calls(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino_genai", types.ModuleType("openvino_genai"))
    from core.inference.openvino_sidecar import ToolSplitter, history_message

    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}
                },
            },
        }
    ]
    s = ToolSplitter(active = True)
    text = "Reading.\n<tool_call>\n<function=read>\n<parameter=path>\n42\n</parameter>\n"
    text += "<parameter=limit>\n10\n</parameter>\n</function>\n</tool_call>"
    out = "".join(s.feed(text[i : i + 3]) for i in range(0, len(text), 3))
    rest, calls = s.finish(tools)
    assert (out + rest).strip() == "Reading."
    assert [c["function"] for c in calls] == [
        {"name": "read", "arguments": '{"path": "42", "limit": 10}'}
    ]

    s = ToolSplitter(active = True)
    s.feed('<tool_call>{"name": "read", "arguments": {"path": "a"}}</tool_call>')
    assert s.finish(tools)[1][0]["function"]["arguments"] == '{"path": "a"}'

    s = ToolSplitter(active = True)
    assert s.feed("<tool_call>junk") == "" and s.finish(tools) == ("<tool_call>junk", [])

    msg = history_message(
        {"role": "assistant", "tool_calls": [{"function": {"name": "read", "arguments": '{"a": 1}'}}]}
    )
    assert msg["tool_calls"][0]["function"]["arguments"] == {"a": 1} and msg["content"] == ""


FAKE_SIDECAR = textwrap.dedent(
    """
    import argparse, http.server
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--model-id"); ap.add_argument("--port", type=int)
    a = ap.parse_args()
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
        def log_message(self, *args): pass
    http.server.HTTPServer(("127.0.0.1", a.port), H).serve_forever()
    """
)


def test_load_serves_upstream_and_unload_stops_the_sidecar(tmp_path, monkeypatch):
    model = _ir(tmp_path / "m-ov_int4")
    fake = tmp_path / "sidecar.py"
    fake.write_text(FAKE_SIDECAR)
    monkeypatch.setattr(ovb, "_SIDECAR", fake)
    monkeypatch.setenv(ovb.PYTHON_ENV, sys.executable)

    backend = ovb.OpenVinoBackend()
    resident = backend.load(str(model), 8192)
    proc = backend._process
    try:
        assert backend.is_loaded and resident.context_length == 8192
        up = backend.upstream()
        assert up.base_url == f"{resident.base_url}/v1" and up.public_model == str(model)
        assert backend.unload() == str(model)
        assert not backend.is_loaded
        assert proc.wait(timeout = 10) is not None
        with pytest.raises(ovb.OpenVinoError):
            backend.upstream()
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_crashing_sidecar_fails_the_load_with_its_output(tmp_path, monkeypatch):
    model = _ir(tmp_path / "m")
    fake = tmp_path / "sidecar.py"
    fake.write_text("print('no GPU found'); raise SystemExit(3)")
    monkeypatch.setattr(ovb, "_SIDECAR", fake)
    monkeypatch.setenv(ovb.PYTHON_ENV, sys.executable)
    backend = ovb.OpenVinoBackend()
    with pytest.raises(ovb.OpenVinoError, match = "no GPU found"):
        backend.load(str(model))
    assert not backend.is_loaded and backend.loading_model is None



def test_chat_history_joins_system_messages(monkeypatch):
    monkeypatch.setitem(sys.modules, "openvino_genai", types.ModuleType("openvino_genai"))
    from core.inference.openvino_sidecar import chat_history

    joined = chat_history(
        [{"role": "system", "content": "a"}, {"role": "user", "content": "q"}, {"role": "system", "content": "b"}]
    )
    assert joined == [{"role": "system", "content": "a\n\nb"}, {"role": "user", "content": "q"}]


def test_finish_reason_reports_length(monkeypatch):
    fake = types.ModuleType("openvino_genai")
    fake.GenerationFinishReason = types.SimpleNamespace(STOP = 1, LENGTH = 2)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake)
    monkeypatch.delitem(sys.modules, "core.inference.openvino_sidecar", raising = False)
    from core.inference.openvino_sidecar import finish_reason

    res = lambda *r: types.SimpleNamespace(finish_reasons = list(r))
    assert finish_reason(res(2), []) == "length"
    assert finish_reason(res(1), []) == "stop"
    assert finish_reason(None, []) == "stop"
    assert finish_reason(res(2), [{"id": "c"}]) == "tool_calls"


def _sidecar(monkeypatch):
    fake = types.ModuleType("openvino_genai")
    fake.GenerationFinishReason = types.SimpleNamespace(STOP = 1, LENGTH = 2)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake)
    monkeypatch.delitem(sys.modules, "core.inference.openvino_sidecar", raising = False)
    from core.inference import openvino_sidecar

    return openvino_sidecar


READ_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {"properties": {"path": {"type": "string"}, "n": {"type": "integer"}}},
        },
    }
]


def _split(sc, text, tools = READ_TOOL, step = 1):
    s = sc.ToolSplitter(active = bool(tools))
    out = "".join(s.feed(text[i : i + step]) for i in range(0, len(text), step))
    rest, calls = s.finish(tools)
    return out + rest, calls


def test_tool_parsing_negative_and_edge_cases(monkeypatch):
    sc = _sidecar(monkeypatch)
    args = lambda calls: [json.loads(c["function"]["arguments"]) for c in calls]

    # Malformed bodies are dropped; when nothing parses the markup is returned as text.
    assert _split(sc, "<tool_call>{not json}</tool_call>") == ("<tool_call>{not json}</tool_call>", [])
    assert _split(sc, '<tool_call>{"arguments": {}}</tool_call>')[1] == []  # no name
    assert _split(sc, "<tool_call></tool_call>")[1] == []
    # One bad block does not sink the good one.
    text = "<tool_call>{bad}</tool_call><tool_call>\n<function=read>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>"
    content, calls = _split(sc, text)
    assert content == "" and args(calls) == [{"path": "x"}] and calls[0]["index"] == 0
    # Cut off by max_tokens mid-call: what arrived is still parsed.
    assert args(_split(sc, "<tool_call>\n<function=read>\n<parameter=path>\nab")[1]) == [{"path": "ab"}]
    # Non-string parameter that is not valid JSON stays a string; string params never get coerced.
    body = "<tool_call><function=read><parameter=n>ten</parameter><parameter=path>007</parameter></function></tool_call>"
    assert args(_split(sc, body)[1]) == [{"n": "ten", "path": "007"}]
    # Multi-line value keeps inner newlines, trims the wrapping ones.
    body = "<tool_call><function=read><parameter=path>\na\nb\n</parameter></function></tool_call>"
    assert args(_split(sc, body)[1]) == [{"path": "a\nb"}]
    # JSON form with arguments already a string; tool unknown to the schema still parses.
    body = '<tool_call>{"name": "other", "arguments": "{\\"q\\": 1}"}</tool_call>'
    assert [c["function"] for c in _split(sc, body)[1]] == [{"name": "other", "arguments": '{"q": 1}'}]
    # Ids are unique across calls.
    two = "<tool_call><function=read></function></tool_call>" * 2
    calls = _split(sc, two)[1]
    assert len(calls) == 2 and calls[0]["id"] != calls[1]["id"] and [c["index"] for c in calls] == [0, 1]
    # Look-alike text is not a tag, including a partial "<tool" left at the very end.
    assert _split(sc, "use <tools> or <tool_calls x") == ("use <tools> or <tool_calls x", [])
    assert _split(sc, "ends with <tool_ca") == ("ends with <tool_ca", [])
    # No tools offered: markup passes through untouched.
    assert _split(sc, "<tool_call>{}</tool_call>", tools = []) == ("<tool_call>{}</tool_call>", [])
    # Garbage tool definitions do not crash the parser.
    junk = [None, "x", {"type": "function"}, {"function": {"parameters": None}}, {"function": {"name": "read"}}]
    assert args(_split(sc, "<tool_call><function=read><parameter=n>3</parameter></function></tool_call>", tools = junk)[1]) == [{"n": 3}]


def test_history_and_content_edge_cases(monkeypatch):
    sc = _sidecar(monkeypatch)
    assert sc.flatten_content(None) == ""
    assert sc.flatten_content([{"type": "image_url", "image_url": {}}, {"type": "text", "text": "a"}, 5]) == "a"
    assert sc.history_message({"content": "q"}) == {"role": "user", "content": "q"}
    # Bad or empty arguments become {} instead of breaking the template.
    msg = sc.history_message(
        {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "a", "arguments": "{oops"}},
            {"function": {"name": "b", "arguments": ""}},
            {"function": {"name": "c", "arguments": {"k": 1}}},
            {"id": "x"},
        ]}
    )
    assert [c["function"].get("arguments") for c in msg["tool_calls"]] == [{}, {}, {"k": 1}, None]
    tool = sc.history_message({"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "r"}], "junk": 1})
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": "r"}
    assert sc.chat_history([]) == []
    assert sc.chat_history([{"role": "system", "content": "s"}]) == [{"role": "system", "content": "s"}]


def test_think_splitter_edge_cases(monkeypatch):
    sc = _sidecar(monkeypatch)
    run = lambda s, pieces: [p for x in pieces for p in s.feed(x)] + s.flush()
    assert run(sc.ThinkSplitter(True), []) == []
    assert run(sc.ThinkSplitter(False), [""]) == []
    # Thinking on but the reply never closes the block: all of it is reasoning.
    assert run(sc.ThinkSplitter(True), ["still ", "thinking"]) == [("reasoning", "still "), ("reasoning", "thinking")]
    # Thinking off, model obeys: content only, released at the end.
    assert run(sc.ThinkSplitter(False), ["Par", "is"]) == [("content", "Paris")]
    # Thinking off, explicit <think>...</think> block.
    out = run(sc.ThinkSplitter(False), ["<think>r</think>A"])
    assert "".join(t for k, t in out if k == "reasoning") == "r" and "".join(t for k, t in out if k == "content") == "A"
    # Thinking off, stray reasoning longer than the window: released as content, tag dropped.
    long = "x" * (sc._STRAY_THINK_WINDOW + 1)
    out = run(sc.ThinkSplitter(False), [long, "</think>tail"])
    assert all(k == "content" for k, _ in out) and "".join(t for _, t in out) == long + "tail"
    # Thinking off, bare </think> with nothing before it.
    assert run(sc.ThinkSplitter(False), ["</think>\n\nA"]) == [("content", "A")]
