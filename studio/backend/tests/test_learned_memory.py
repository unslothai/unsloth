# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
import pytest
from core import chat_memory as memory
from models.inference import ChatCompletionRequest, ChatMessage


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "memory.db"

    def connect(*args, **kwargs):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_projects(id TEXT PRIMARY KEY, archived INTEGER DEFAULT 0)"
        )
        return conn

    monkeypatch.setattr(memory.studio_db, "get_connection", connect)
    settings = {"learned_memory_enabled": True}
    monkeypatch.setattr(
        memory.studio_db, "get_app_setting", lambda key, fallback = None: settings.get(key, fallback)
    )
    threads = {"chat": {"projectId": None}, "project-chat": {"projectId": "p"}}
    monkeypatch.setattr(memory.studio_db, "get_chat_thread", lambda key: threads.get(key))
    conn = connect()
    conn.execute("INSERT INTO chat_projects(id) VALUES ('p')")
    conn.commit()
    conn.close()
    return settings, connect


def _write(
    key,
    value,
    thread = "chat",
):
    return memory.execute({"scope": "personal", "key": key, "value": value}, thread)


def test_store_saves_corrects_forgets_and_refuses_unsafe_writes(db):
    assert memory.update(None, "language", "Python", 0) == 1
    assert memory.update(None, "language", "Python", 1) == 1
    with pytest.raises(ValueError, match = "changed"):
        memory.update(None, "language", "Rust", 0)
    with pytest.raises(ValueError, match = "full"):
        memory.update(None, "large", "x" * 2048, 1)
    assert memory.update(None, "language", "Rust", 1) == 2
    assert memory.read()["entries"]["language"]["value"] == "Rust"
    memory.update(None, "language", None, 2)
    assert memory.read()["entries"] == {}


def test_project_memory_is_scoped_and_dies_with_its_project(db):
    memory.update("p", "language", "Rust", 0)
    assert "Rust" not in memory.context("chat", 16384)["text"]
    assert "Rust" in memory.context("project-chat", 16384)["text"]
    conn = db[1]()
    conn.execute("DELETE FROM chat_projects WHERE id='p'")
    conn.commit()
    conn.close()
    assert memory.read("p")["entries"] == {}


def test_tool_writes_follow_the_snapshot_it_was_offered(db):
    memory.offer("chat")
    assert _write("language", "Python").startswith("Memory updated")
    assert _write("language", "Python").startswith("Memory unchanged")
    assert memory.execute(
        {"scope": "project", "key": "language", "value": "Rust"}, "chat"
    ).startswith("Error:")
    memory.update(None, "language", "Go", 1)
    assert _write("language", "Rust").startswith("Error:")
    memory.offer("project-chat")
    memory.clear(None, 2)
    assert _write("language", "Rust", "project-chat").startswith("Error:")
    assert memory.read()["entries"] == {}


def test_opted_out_chats_get_no_memory(db):
    memory.update(None, "language", "Python", 0)
    assert memory.context("incognito")["text"] == ""
    memory.offer("chat")
    db[0]["memory_disabled:chat"] = True
    assert memory.context("chat")["text"] == ""
    assert _write("language", "Rust").startswith("Error:")
    payload = ChatCompletionRequest(
        model = "test",
        thread_id = "chat",
        studio_tool_history = True,
        messages = [
            ChatMessage(role = "assistant", content = "<tool_call>update_memory</tool_call>"),
            ChatMessage(role = "user", content = "Remember another fact"),
        ],
    )
    memory.apply_to_request(payload)
    assert "Persistent memory is disabled" in payload.messages[0].content
    db[0]["learned_memory_enabled"] = False
    assert memory.offer("project-chat") == []


def test_context_is_injected_bounded_and_never_slices_facts(db):
    revision = 0
    for i in range(8):
        revision = memory.update(None, "fact_" + str(i), "語" * 20, revision)
    result = memory.context("chat", 4096)
    assert len(result["text"].encode()) <= 256
    assert result["omitted"] > 0
    assert memory.context("chat", 16)["text"] == ""
    payload = ChatCompletionRequest(
        model = "test", thread_id = "chat", messages = [ChatMessage(role = "user", content = "hi")]
    )
    memory.apply_to_request(payload)
    assert payload.messages[0].role == "system" and "語" in payload.messages[0].content


def _routed(**fields):
    from state.tool_policy import reset_tool_policy
    from routes.inference import _explicit_studio_tool_loop_requested

    reset_tool_policy()
    payload = ChatCompletionRequest(
        **{
            "model": "test",
            "thread_id": "chat",
            "enable_tools": False,
            "messages": [{"role": "user", "content": "hello"}],
            **fields,
        }
    )
    return _explicit_studio_tool_loop_requested(payload)


def _history(name):
    call = {"id": "c1", "type": "function", "function": {"name": name, "arguments": "{}"}}
    return [
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "name": name, "content": "ok"},
    ]


@pytest.mark.parametrize(
    "contract",
    [
        {"tools": [{"type": "function", "function": {"name": "client_weather"}}]},
        {"tools": [{"type": "function", "function": {"name": "w"}}], "enable_tools": None},
        {"tool_choice": "none"},
        {"response_format": {"type": "json_object"}},
        {"n": 2},
        {"messages": _history("client_weather")},
        {"messages": _history("update_memory")},
    ],
)
def test_client_contracts_keep_the_memory_tool_out(db, contract):
    assert not _routed(**contract)


@pytest.mark.parametrize(
    "fields",
    [{}, {"messages": _history("update_memory"), "studio_tool_history": True}],
)
def test_saved_chats_get_the_memory_tool(db, fields):
    assert _routed(**fields)


@pytest.mark.parametrize("studio_history", [None, False, True])
def test_disabled_memory_does_not_rewrite_client_tool_contract(db, studio_history):
    db[0]["learned_memory_enabled"] = False
    payload = ChatCompletionRequest(
        messages = _history("update_memory"),
        tools = [{"type": "function", "function": {"name": "update_memory"}}],
        studio_tool_history = studio_history,
    )
    before = payload.model_dump()
    memory.apply_to_request(payload)
    assert payload.model_dump() == before


def test_unmarked_tool_history_does_not_get_studio_warning(db):
    db[0]["learned_memory_enabled"] = False
    payload = ChatCompletionRequest(messages = _history("update_memory"))
    before = payload.model_dump()
    memory.apply_to_request(payload)
    assert payload.model_dump() == before


@pytest.mark.parametrize("extra_tool", [None, "python", "deep_research", "read_skill"])
@pytest.mark.parametrize("full_access_only", [False, True])
@pytest.mark.parametrize("memory_on", [False, True])
def test_memory_guidance_follows_the_offered_catalog(
    monkeypatch, extra_tool, full_access_only, memory_on
):
    from routes import inference

    monkeypatch.setattr(inference, "_enabled_agent_skills", lambda: [])
    tools = [{"type": "function", "function": {"name": extra_tool}}] if extra_tool else []
    if memory_on:
        tools.append(memory.TOOL)
    nudge = inference._build_tool_action_nudge(
        tools = tools, model_name = "test-4B", full_access_only = full_access_only
    )
    assert ("update_memory" in nudge) is memory_on


@pytest.mark.parametrize("stream", [False, True])
def test_responses_conversion_preserves_memory_context(db, stream):
    from models.inference import ResponsesRequest
    from routes.inference import _build_chat_request

    memory.update(None, "language", "Python", 0)
    payload = ResponsesRequest(input = "What language do I prefer?", thread_id = "chat", stream = stream)
    chat = _build_chat_request(
        payload, [ChatMessage(role = "user", content = payload.input)], stream = stream
    )
    memory.apply_to_request(chat, 8192)
    assert chat.thread_id == "chat"
    assert "personal.language" in chat.messages[0].content


@pytest.mark.parametrize("hosted", [False, True])
def test_memory_budget_uses_serving_window_after_routing(db, monkeypatch, hosted):
    import asyncio
    from types import SimpleNamespace
    from routes import inference as inf
    from auth import authentication
    from core.inference import npu_backend

    # Fits a 16K memory allowance, but not the previously resident model's 4K allowance.
    memory.update(None, "long_fact", "Amber " * 50, 0)
    resident = {"window": 4096}
    monkeypatch.setattr(inf, "_monitor_context_length", lambda: resident["window"])
    monkeypatch.setattr(inf, "_should_validate_before_switch", lambda: True)
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda _: False)
    monkeypatch.setattr(npu_backend, "peek_npu_backend", lambda: None)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = True))
    parsed = []
    extract = inf._extract_content_parts

    def record_parse(*args, **kwargs):
        result = extract(*args, **kwargs)
        parsed.append(result)
        return result

    monkeypatch.setattr(inf, "_extract_content_parts", record_parse)

    async def switch(*args, **kwargs):
        resident["window"] = 16384

    monkeypatch.setattr(inf, "_maybe_auto_switch_model", switch)

    class Captured(Exception):
        pass

    def capture(payload):
        assert "personal.long_fact" in payload.messages[0].content
        if not hosted:
            assert "personal.long_fact" in parsed[-1][0]
        raise Captured()

    async def proxy(payload, *args):
        capture(payload)

    monkeypatch.setattr(inf, "_proxy_to_external_provider", proxy)
    monkeypatch.setattr(inf, "_normalize_chat_reasoning_controls", capture)
    payload = ChatCompletionRequest(
        model = "target",
        thread_id = "chat",
        stream = True,
        messages = [{"role": "user", "content": "What do you remember?"}],
        **({"provider_type": "openai", "context_window": 16384} if hosted else {}),
    )
    request = SimpleNamespace(headers = {}, scope = {}, state = SimpleNamespace())
    with pytest.raises(Captured):
        asyncio.run(
            inf.produce_openai_chat_completions(payload, request, "t", cancel_on_disconnect = False)
        )


def test_offered_memory_scopes_follow_the_chat_without_mutating_other_offers(db):
    personal = memory.offer("chat")[0]["function"]
    project = memory.offer("project-chat")[0]["function"]
    assert personal["parameters"]["properties"]["scope"]["enum"] == ["personal"]
    assert project["parameters"]["properties"]["scope"]["enum"] == ["personal", "project"]
    assert memory.TOOL["function"]["parameters"]["properties"]["scope"]["enum"] == [
        "personal",
        "project",
    ]
    assert memory.execute(
        {"scope": "project", "key": "stack", "value": "Kotlin"}, "project-chat"
    ).startswith("Memory updated:")
    assert memory.execute(
        {"scope": "personal", "key": "editor", "value": "Helix"}, "project-chat"
    ).startswith("Memory updated:")
    assert memory.read("p")["entries"]["stack"]["value"] == "Kotlin"
    assert "editor" not in memory.read("p")["entries"]
    assert memory.read()["entries"]["editor"]["value"] == "Helix"
    assert "stack" not in memory.read()["entries"]


@pytest.mark.parametrize("enable_tools", [None, False])
def test_memory_only_chat_falls_through_without_a_managed_tool_parser(
    db, monkeypatch, enable_tools
):
    import asyncio
    from types import SimpleNamespace
    from routes import inference as inf
    from routes.managed_engine_chat import managed_tool_chat
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    monkeypatch.setattr(inf.api_monitor, "fail", lambda *args: None)
    payload = ChatCompletionRequest(
        model = "managed",
        thread_id = "chat",
        enable_tools = enable_tools,
        messages = [{"role": "user", "content": "hello"}],
    )
    backend = SimpleNamespace(
        active_model_name = "managed", models = {"managed": {"supports_tools": False}}
    )
    request = SimpleNamespace(headers = {})
    assert asyncio.run(managed_tool_chat(payload, request, backend, [], "", None)) is None


def test_memory_replay_keeps_the_tool_loop_without_a_launcher_default(db):
    from routes import inference as inf
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    payload = ChatCompletionRequest(
        model = "test",
        thread_id = "chat",
        studio_tool_history = True,
        messages = [*_history("update_memory"), {"role": "user", "content": "Forget that"}],
    )
    assert inf._effective_enable_tools(payload)
    assert not inf._tools_on_by_launcher_default_only(payload)
    assert inf._launcher_tool_default_applies(payload, False)


def test_memory_only_auto_approval_needs_no_stream(db):
    from routes import inference as inf
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    payload = ChatCompletionRequest(
        model = "test",
        thread_id = "chat",
        enable_tools = False,
        stream = False,
        permission_mode = "auto",
        messages = [{"role": "user", "content": "Remember I prefer Rust"}],
    )
    assert not inf._confirm_gate_needs_stream(payload)
    assert not inf._confirm_gate_has_no_channel(payload, False, ["update_memory"])
    assert inf._confirm_gate_has_no_channel(payload, False, ["terminal"])
    payload.enable_tools = True
    assert inf._confirm_gate_needs_stream(payload)
    payload.enable_tools = False
    payload.mcp_enabled = True
    assert inf._confirm_gate_needs_stream(payload)
    payload.mcp_enabled = False
    payload.deep_research_armed = True
    assert inf._confirm_gate_needs_stream(payload)
    payload.deep_research_armed = False
    payload.confirm_tool_calls = True
    assert inf._confirm_gate_needs_stream(payload)


@pytest.mark.parametrize("ui_events", [False, True])
@pytest.mark.parametrize("replay", [False, True])
def test_memory_does_not_restore_unsafe_launcher_tools(db, monkeypatch, ui_events, replay):
    import asyncio
    from routes import inference as inf
    from state.tool_policy import reset_tool_policy, set_tool_policy_default

    reset_tool_policy()
    set_tool_policy_default(True)
    monkeypatch.setattr(inf, "_enabled_agent_skills", lambda: [])
    monkeypatch.setattr(inf, "_thread_has_conversation_archive", lambda _: False)
    payload = ChatCompletionRequest(
        model = "test",
        thread_id = "chat",
        stream = True,
        studio_tool_history = replay,
        messages = [
            *(_history("update_memory") if replay else []),
            {"role": "user", "content": "Remember my editor"},
        ],
    )
    try:
        assert inf._launcher_tool_default_applies(payload, ui_events)
        tools = asyncio.run(
            inf._select_request_tools(
                payload,
                tools_on = True,
                mcp_allowed = False,
                ui_events = ui_events,
            )
        )
        names = inf._catalog_names(tools)
        assert "update_memory" in names
        if ui_events and not replay:
            assert "terminal" in names
        else:
            assert names == ["update_memory"]
        assert not inf._confirm_gate_has_no_channel(payload, ui_events, names)
        if not ui_events:
            assert inf._confirm_gate_has_no_channel(payload, False, ["terminal"])
    finally:
        reset_tool_policy()
