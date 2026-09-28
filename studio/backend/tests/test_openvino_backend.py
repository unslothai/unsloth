# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
        {
            "role": "assistant",
            "tool_calls": [{"function": {"name": "read", "arguments": '{"a": 1}'}}],
        }
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
        [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "q"},
            {"role": "system", "content": "b"},
        ]
    )
    assert joined == [{"role": "system", "content": "a\n\nb"}, {"role": "user", "content": "q"}]
