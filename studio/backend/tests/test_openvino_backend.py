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


def test_template_reasons_only_when_it_knows_think(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    assert ovb._template_reasons(plain) is False

    thinking = tmp_path / "thinking"
    thinking.mkdir()
    (thinking / "chat_template.jinja").write_text("{% if enable_thinking %}<think>{% endif %}")
    assert ovb._template_reasons(thinking) is True


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


FAKE_SIDECAR = textwrap.dedent(
    """
    import argparse, http.server
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--model-id"); ap.add_argument("--port", type=int)
    ap.add_argument("--max-context", type=int)
    a = ap.parse_args()
    body = ('{"context_length": %d}' % min(a.max_context or 32768, 32768)).encode()
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(body)
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
        # the context the sidecar enforces: the requested 8192 caps the model's 32768
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


def test_context_length_is_what_the_sidecar_enforces(tmp_path, monkeypatch):
    model = _ir(tmp_path / "m")
    fake = tmp_path / "sidecar.py"
    fake.write_text(FAKE_SIDECAR)
    monkeypatch.setattr(ovb, "_SIDECAR", fake)
    monkeypatch.setenv(ovb.PYTHON_ENV, sys.executable)
    backend = ovb.OpenVinoBackend()
    try:
        # asked for more than the model/KV cache holds: the sidecar's limit is reported
        resident = backend.load(str(model), 100000)
        assert resident.context_length == 32768 and resident.requested_context_length == 100000
        assert backend.upstream().context_length == 32768
    finally:
        backend.unload()


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


@pytest.fixture
def sc(monkeypatch):
    fake = types.ModuleType("openvino_genai")
    fake.GenerationFinishReason = types.SimpleNamespace(STOP = 1, LENGTH = 2)
    monkeypatch.setitem(sys.modules, "openvino_genai", fake)
    monkeypatch.delitem(sys.modules, "core.inference.openvino_sidecar", raising = False)
    from core.inference import openvino_sidecar

    return openvino_sidecar


def _think(sc, thinking, pieces):
    s = sc.ThinkSplitter(thinking)
    out = [p for x in pieces for p in s.feed(x)] + s.flush()
    return {k: "".join(t for kk, t in out if kk == k) for k in ("reasoning", "content")}, out


@pytest.mark.parametrize(
    "thinking, pieces, reasoning, content",
    [
        # Close tag split across pieces.
        (True, ["plan ", "it</th", "ink>\n\nPar", "is"], "plan it", "Paris"),
        (True, [], "", ""),
        # Never closed: all of it is reasoning.
        (True, ["still ", "thinking"], "still thinking", ""),
        # Qwen3's template ends the prompt at "assistant\n"; the model itself writes "<think>\n".
        (True, ["<think>\nplan</think>\n\nA"], "\nplan", "A"),
        (True, ["<th", "ink>", "\nplan", "</think>A"], "\nplan", "A"),
        (True, ["\n<think>plan</think>A"], "plan", "A"),
        (False, ["hi"], "", "hi"),
        (False, [""], "", ""),
        (False, ["Par", "is"], "", "Paris"),
        # Reasoning despite enable_thinking=false: an explicit block, or a bare </think>.
        (False, ["<think>r</think>A"], "r", "A"),
        (False, ["user wants X\n</th", "ink>\n\nX", "!"], "user wants X", "X!"),
        (False, ["</think>\n\nA"], "", "A"),
    ],
)
def test_think_splitter(sc, thinking, pieces, reasoning, content):
    assert _think(sc, thinking, pieces)[0] == {"reasoning": reasoning, "content": content}


def test_think_splitter_releases_stray_reasoning_past_the_window(sc):
    long = "x" * (sc._STRAY_THINK_WINDOW + 1)
    joined, out = _think(sc, False, [long, "</think>tail"])
    assert all(k == "content" for k, _ in out) and joined["content"] == long + "tail"
    # Thinking off and obeyed: content held to the end, then one piece.
    assert _think(sc, False, ["Par", "is"])[1] == [("content", "Paris")]


READ_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "parameters": {"properties": {"path": {"type": "string"}, "n": {"type": "integer"}}},
        },
    }
]


def _split(
    sc,
    text,
    tools = READ_TOOL,
    step = 1,
):
    s = sc.ToolSplitter(active = bool(tools))
    out = "".join(s.feed(text[i : i + step]) for i in range(0, len(text), step))
    rest, calls = s.finish(tools)
    return out + rest, calls


def _xml(*params):
    body = "".join(f"<parameter={k}>{v}</parameter>" for k, v in params)
    return f"<tool_call><function=read>{body}</function></tool_call>"


@pytest.mark.parametrize(
    "text, content, args",
    [
        (
            "Reading.\n<tool_call>\n<function=read>\n<parameter=path>\n42\n</parameter>\n"
            "<parameter=n>\n10\n</parameter>\n</function>\n</tool_call>",
            "Reading.",
            [{"path": "42", "n": 10}],
        ),
        (
            '<tool_call>{"name": "read", "arguments": {"path": "a"}}</tool_call>',
            "",
            [{"path": "a"}],
        ),
        # Nothing parses: the markup comes back as text.
        ("<tool_call>{not json}</tool_call>", "<tool_call>{not json}</tool_call>", []),
        ("<tool_call>junk", "<tool_call>junk", []),
        (
            '<tool_call>{"arguments": {}}</tool_call>',
            '<tool_call>{"arguments": {}}</tool_call>',
            [],
        ),
        ("<tool_call></tool_call>", "<tool_call></tool_call>", []),
        # One bad block does not sink the good one.
        ("<tool_call>{bad}</tool_call>" + _xml(("path", "\nx\n")), "", [{"path": "x"}]),
        # Cut off by max_tokens mid-call: what arrived is still parsed.
        ("<tool_call>\n<function=read>\n<parameter=path>\nab", "", [{"path": "ab"}]),
        # Non-string param that is not JSON stays a string; string params are never coerced.
        (_xml(("n", "ten"), ("path", "007")), "", [{"n": "ten", "path": "007"}]),
        # Multi-line value keeps inner newlines, trims the wrapping ones.
        (_xml(("path", "\na\nb\n")), "", [{"path": "a\nb"}]),
        # Look-alike text is not a tag, including a partial "<tool" at the very end.
        ("use <tools> or <tool_calls x", "use <tools> or <tool_calls x", []),
        ("ends with <tool_ca", "ends with <tool_ca", []),
    ],
)
def test_tool_splitter(sc, text, content, args):
    for step in (1, 3):
        out, calls = _split(sc, text, step = step)
        assert (
            out.strip() == content
            and [json.loads(c["function"]["arguments"]) for c in calls] == args
        )


def test_tool_call_ids_names_and_odd_tool_lists(sc):
    calls = _split(sc, _xml() * 2)[1]
    assert [c["index"] for c in calls] == [0, 1] and calls[0]["id"] != calls[1]["id"]
    # JSON form with arguments already a string; a tool unknown to the schema still parses.
    body = '<tool_call>{"name": "other", "arguments": "{\\"q\\": 1}"}</tool_call>'
    assert [c["function"] for c in _split(sc, body)[1]] == [
        {"name": "other", "arguments": '{"q": 1}'}
    ]
    # No tools offered: markup passes through untouched.
    assert _split(sc, "<tool_call>{}</tool_call>", tools = []) == ("<tool_call>{}</tool_call>", [])
    junk = [
        None,
        "x",
        {"type": "function"},
        {"function": {"parameters": None}},
        {"function": {"name": "read"}},
    ]
    assert _split(sc, _xml(("n", "3")), tools = junk)[1][0]["function"]["arguments"] == '{"n": 3}'


def test_history_messages(sc):
    assert sc.flatten_content(None) == ""
    parts = [{"type": "image_url", "image_url": {}}, {"type": "text", "text": "a"}, 5]
    assert sc.flatten_content(parts) == "a"
    assert sc.history_message({"content": "q"}) == {"role": "user", "content": "q"}
    # Bad or empty arguments become {} instead of breaking the template.
    calls = [
        {"function": {"name": n, "arguments": a}}
        for n, a in (("a", "{oops"), ("b", ""), ("c", {"k": 1}), ("d", '{"a": 1}'))
    ]
    msg = sc.history_message(
        {"role": "assistant", "content": None, "tool_calls": calls + [{"id": "x"}]}
    )
    assert msg["content"] == ""
    assert [c["function"].get("arguments") for c in msg["tool_calls"]] == [
        {},
        {},
        {"k": 1},
        {"a": 1},
        None,
    ]
    tool = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [{"type": "text", "text": "r"}],
        "junk": 1,
    }
    assert sc.history_message(tool) == {"role": "tool", "tool_call_id": "c1", "content": "r"}
    assert sc.chat_history([]) == []
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "q"},
        {"role": "system", "content": "b"},
    ]
    assert sc.chat_history(msgs) == [
        {"role": "system", "content": "a\n\nb"},
        {"role": "user", "content": "q"},
    ]


def test_finish_reason_and_errors(sc):
    res = lambda *r: types.SimpleNamespace(finish_reasons = list(r))
    assert sc.finish_reason(res(2), []) == "length"
    assert sc.finish_reason(res(1), []) == "stop"
    assert sc.finish_reason(None, []) == "stop"
    assert sc.finish_reason(res(2), [{"id": "c"}]) == "tool_calls"
    assert sc.finish_reason(res(1), [], out_of_room = True) == "length"
    assert (
        sc.generation_error(RuntimeError("request did not fit in the available cache budget"))[0]
        == 400
    )
    assert sc.generation_error(RuntimeError("boom")) == (500, "Generation failed: boom")


def test_model_context_reads_config(sc, tmp_path):
    assert sc.model_context(str(tmp_path)) is None  # no config.json
    cfg = tmp_path / "config.json"
    for text, want in (
        ("{broken", None),
        ('{"max_position_embeddings": 4096}', 4096),
        ('{"text_config": {"max_position_embeddings": 262144}}', 262144),
    ):
        cfg.write_text(text)
        assert sc.model_context(str(tmp_path)) == want
    # The KV cache caps it: 1 GiB / (2 layers * K+V * 2 heads * 64 dim * f16 bytes) * 0.85.
    cfg.write_text(
        json.dumps(
            {
                "max_position_embeddings": 10**7,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "hidden_size": 128,
            }
        )
    )
    assert sc.model_context(str(tmp_path), 1) == 891289


def test_split_layout_ir_is_a_model_directory(tmp_path):
    from hub.services.models.common import _is_model_directory as hub_is_model_dir
    from routes.models import _is_model_directory, _is_weight_bin

    d = tmp_path / "vlm-int4-ov"
    d.mkdir()
    (d / "config.json").write_text("{}")
    for name in ("openvino_language_model", "openvino_vision_embeddings_model"):
        (d / f"{name}.xml").write_text("")
        (d / f"{name}.bin").write_bytes(b"")
    assert _is_model_directory(d) and hub_is_model_dir(d)
    assert _is_weight_bin("openvino_language_model.bin") and not _is_weight_bin(
        "openvino_tokenizer.bin"
    )


def test_an_ir_beside_safetensors_needs_a_sidecar_python(tmp_path, monkeypatch):
    both = _ir(tmp_path / "both")
    (both / "model.safetensors").write_text("")
    monkeypatch.delenv(ovb.PYTHON_ENV, raising = False)
    monkeypatch.setattr(ovb.importlib.util, "find_spec", lambda name: None)
    # Without openvino-genai the checkpoint keeps its standard load; an explicit prefix still routes.
    assert not ovb.is_openvino_model_path(str(both))
    assert ovb.is_openvino_model_path(f"openvino:{both}")
    assert ovb.is_openvino_model_path(str(_ir(tmp_path / "ir_only")))
    monkeypatch.setattr(ovb.importlib.util, "find_spec", lambda name: object())
    assert ovb.is_openvino_model_path(str(both))
