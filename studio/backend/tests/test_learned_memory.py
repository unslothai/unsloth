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
