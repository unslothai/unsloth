# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-user latency: the per-process prompt count memo and the trigger-free display strip."""

import random
from types import SimpleNamespace

import routes.inference as inference


def _counting_client(calls):
    class _Resp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(
            self,
            url,
            json = None,
            **_k,
        ):
            calls.append(url)
            if url.endswith("/input_tokens"):
                return _Resp({"input_tokens": 7 + len(str(json))})
            return _Resp({"tokens": [1, 2, 3]})

    return _Client


def test_count_memo_reuses_a_render_only_within_one_server_process(monkeypatch):
    from core.inference import llama_cpp

    class _Backend(llama_cpp.LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:8080"
        _auth_headers: dict = {}

    calls: list = []
    monkeypatch.setattr(llama_cpp.httpx, "Client", _counting_client(calls))
    monkeypatch.setattr(llama_cpp, "_COUNT_MEMO", llama_cpp.collections.OrderedDict())
    backend = _Backend.__new__(_Backend)
    backend._process = SimpleNamespace(pid = 4242)
    msgs = [{"role": "user", "content": "count me"}]

    first = backend.count_chat_tokens(msgs, prefer_native = True)
    second = backend.count_chat_tokens(msgs, prefer_native = True)
    assert first == second and len(calls) == 1
    backend.count_chat_tokens([{"role": "user", "content": "other"}], prefer_native = True)
    assert len(calls) == 2
    backend._process = SimpleNamespace(pid = 4343)
    backend.count_chat_tokens(msgs, prefer_native = True)
    assert len(calls) == 3
    backend._process = None
    backend.count_chat_tokens(msgs, prefer_native = True)
    backend.count_chat_tokens(msgs, prefer_native = True)
    assert len(calls) == 5


def test_count_memo_is_skipped_for_a_template_that_reads_the_clock(monkeypatch):
    # Granite's template renders today's date, so the same body counts differently over time.
    from core.inference import llama_cpp

    class _Backend(llama_cpp.LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:8080"
        _auth_headers: dict = {}

    calls: list = []
    monkeypatch.setattr(llama_cpp.httpx, "Client", _counting_client(calls))
    monkeypatch.setattr(llama_cpp, "_COUNT_MEMO", llama_cpp.collections.OrderedDict())
    backend = _Backend.__new__(_Backend)
    backend._process = SimpleNamespace(pid = 4242)
    backend._chat_template_override = None
    backend._chat_template = "{{ strftime_now('%B %d, %Y') }}{% for m in messages %}{% endfor %}"
    msgs = [{"role": "user", "content": "count me"}]
    backend.count_chat_tokens(msgs, prefer_native = True)
    backend.count_chat_tokens(msgs, prefer_native = True)
    assert len(calls) == 2, "a clock-reading template is counted afresh"
    backend._chat_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    backend.count_chat_tokens(msgs, prefer_native = True)
    backend.count_chat_tokens(msgs, prefer_native = True)
    assert len(calls) == 3, "a static template still hits the memo"


def test_count_memo_tells_apart_schemas_in_a_different_key_order(monkeypatch):
    """llama-server keeps a JSON object's key order and a template that walks a schema renders
    it in that order, so the same entries in another order can tokenize differently. A key with
    sorted keys served one request's count to the other."""
    from core.inference import llama_cpp

    class _Backend(llama_cpp.LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:8080"
        _auth_headers: dict = {}

    def _tool(properties):
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up.",
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        ]

    city = {"type": "string"}
    year = {"type": "integer"}
    calls: list = []
    monkeypatch.setattr(llama_cpp.httpx, "Client", _counting_client(calls))
    monkeypatch.setattr(llama_cpp, "_COUNT_MEMO", llama_cpp.collections.OrderedDict())
    backend = _Backend.__new__(_Backend)
    backend._process = SimpleNamespace(pid = 4242)
    msgs = [{"role": "user", "content": "count me"}]

    backend.count_chat_tokens(msgs, tools = _tool({"city": city, "year": year}), prefer_native = True)
    backend.count_chat_tokens(msgs, tools = _tool({"year": year, "city": city}), prefer_native = True)
    assert len(calls) == 2, "the second schema order was answered from the first one's count"
    backend.count_chat_tokens(msgs, tools = _tool({"year": year, "city": city}), prefer_native = True)
    assert len(calls) == 2, "the same render in the same order must still be a memo hit"


def test_display_strip_leaves_markup_free_text_alone_and_still_strips_markup():
    strip = inference._strip_tool_xml_for_display
    prose = "The keeper climbed the stairs, call: home soon; web_search ARGS later.\n\nTOOL_CALLS"
    assert strip(prose, auto_heal_tool_calls = True, enabled_tool_names = {"web_search"}) is prose
    leaked = 'Sure.<tool_call>{"name": "web_search", "arguments": {"q": "x"}}</tool_call>'
    assert strip(leaked, auto_heal_tool_calls = True, enabled_tool_names = {"web_search"}) == "Sure."


def test_display_strip_fast_path_matches_the_full_strip_on_random_text(monkeypatch):
    pieces = list("abcxyz ABC019\n\t.,;:!?'\"()-_=+*&^%$#@~`|/\\}]") + [
        "call:",
        "call: web_search",
        "web_search",
        "TOOL_CALLS",
        "ARGS",
        "think",
        "function=",
        "tool_call",
        "｜",
        "|>",
        "/>",
        "python_tag",
        "parameter",
    ]
    rng = random.Random(7)
    texts = ["".join(rng.choice(pieces) for _ in range(rng.randint(0, 60))) for _ in range(3000)]

    class _Never:
        @staticmethod
        def search(_text):
            return True

    monkeypatch.setattr(inference, "_DISPLAY_STRIP_TRIGGERS", _Never)
    for text in texts:
        for names in (None, set(), {"web_search", "python", "terminal"}):
            assert (
                inference._strip_tool_xml_for_display(
                    text, auto_heal_tool_calls = True, enabled_tool_names = names
                )
                == text
            )
