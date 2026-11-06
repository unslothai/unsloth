import sqlite3
import pytest
from core import chat_memory as memory


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


def test_save_correct_forget_and_noop(db):
    assert memory.update(None, "language", "Python", 0) == 1
    assert memory.update(None, "language", "Python", 1) == 1
    assert "Python" in memory.context("chat", 16384)["text"]
    assert memory.update(None, "language", "Rust", 1) == 2
    assert list(memory.read()["entries"]) == ["language"]
    memory.update(None, "language", None, 2)
    assert memory.context("chat")["text"] == ""


def test_capacity_and_stale_write_preserve_data(db):
    memory.update(None, "language", "Python", 0)
    with pytest.raises(ValueError, match = "changed"):
        memory.update(None, "language", "Rust", 0)
    with pytest.raises(ValueError, match = "full"):
        memory.update(None, "large", "x" * 2048, 1)
    assert memory.read()["entries"]["language"]["value"] == "Python"


def test_project_isolation_and_cascade(db):
    memory.update("p", "language", "Rust", 0)
    assert "Rust" not in memory.context("chat", 16384)["text"]
    assert "Rust" in memory.context("project-chat", 16384)["text"]
    conn = db[1]()
    conn.execute("DELETE FROM chat_projects WHERE id='p'")
    conn.commit()
    conn.close()
    assert memory.read("p")["entries"] == {}


def test_opt_out_missing_thread_and_late_tool(db):
    memory.update(None, "language", "Python", 0)
    assert memory.context("incognito")["text"] == ""
    memory.offer("chat")
    db[0]["memory_disabled:chat"] = True
    assert memory.context("chat")["text"] == ""
    assert memory.execute(
        {"scope": "personal", "key": "language", "value": "Rust"}, "chat"
    ).startswith("Error:")
    db[0]["learned_memory_enabled"] = False
    assert memory.offer("project-chat") == []


def test_tool_revision_and_project_authority(db):
    memory.offer("chat")
    args = {"scope": "personal", "key": "language", "value": "Python"}
    assert memory.execute(args, "chat").startswith("Memory updated")
    assert memory.execute({**args, "value": "Rust"}, "chat").startswith("Memory updated")
    assert memory.execute({**args, "scope": "project"}, "chat").startswith("Error:")
    memory.update(None, "language", None, 2)
    assert memory.execute(args, "chat").startswith("Error:")
    assert memory.read()["entries"] == {}


def test_context_is_bounded_and_never_slices_facts(db):
    revision = 0
    for i in range(8):
        revision = memory.update(None, "fact_" + str(i), "語" * 20, revision)
    result = memory.context("chat", 4096)
    assert len(result["text"].encode()) <= 256
    assert result["omitted"] > 0
    assert memory.context("chat", 16)["text"] == ""


def test_clear_preserves_revision_tombstone(db):
    memory.update(None, "language", "Python", 0)
    memory.offer("chat")
    memory.clear(None, 1)
    assert memory.read()["entries"] == {}
    assert memory.execute(
        {"scope": "personal", "key": "language", "value": "Rust"}, "chat"
    ).startswith("Error:")


def test_request_injection_and_snapshot_survive_intervening_edit(db):
    from models.inference import ChatCompletionRequest, ChatMessage

    memory.update(None, "language", "Python", 0)
    payload = ChatCompletionRequest(
        model = "test", thread_id = "chat", messages = [ChatMessage(role = "user", content = "hello")]
    )
    memory.apply_to_request(payload)
    assert "Python" in payload.messages[0].content
    memory.update(None, "language", "Rust", 1)
    memory.offer("chat")
    assert memory.execute(
        {"scope": "personal", "key": "language", "value": "Go"}, "chat"
    ).startswith("Error:")
    db[0]["learned_memory_enabled"] = False
    payload = ChatCompletionRequest(
        model = "test", thread_id = "chat", messages = [ChatMessage(role = "user", content = "hello")]
    )
    memory.apply_to_request(payload)
    assert len(payload.messages) == 1


def test_accounts_use_separate_stores(db, tmp_path, monkeypatch):
    memory.update(None, "language", "Python", 0)
    other = tmp_path / "other-account.db"

    def connect_other(*args, **kwargs):
        conn = sqlite3.connect(other)
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_projects(id TEXT PRIMARY KEY, archived INTEGER DEFAULT 0)"
        )
        return conn

    monkeypatch.setattr(memory.studio_db, "get_connection", connect_other)
    assert memory.read()["entries"] == {}
    memory.update(None, "language", "Rust", 0)
    monkeypatch.setattr(memory.studio_db, "get_connection", db[1])
    assert memory.read()["entries"]["language"]["value"] == "Python"


def test_clear_empty_scope_invalidates_pending_first_write(db):
    memory.offer("chat")
    memory.clear(None, 0)
    assert memory.execute(
        {"scope": "personal", "key": "language", "value": "Python"}, "chat"
    ).startswith("Error:")
    assert memory.read()["entries"] == {}


def test_disabling_memory_explains_old_tool_history(db):
    from models.inference import ChatCompletionRequest, ChatMessage

    db[0]["memory_disabled:chat"] = True
    payload = ChatCompletionRequest(
        model = "test",
        thread_id = "chat",
        messages = [
            ChatMessage(role = "user", content = "Remember my preference"),
            ChatMessage(role = "assistant", content = "<tool_call>update_memory</tool_call>"),
            ChatMessage(role = "user", content = "Remember another fact"),
        ],
    )
    memory.apply_to_request(payload)
    assert "Persistent memory is disabled" in payload.messages[0].content
    assert memory.offer("chat") == []


def test_repeated_tool_write_reports_no_change(db):
    memory.offer("chat")
    args = {"scope": "personal", "key": "language", "value": "Python"}
    assert memory.execute(args, "chat").startswith("Memory updated:")
    assert memory.execute(args, "chat").startswith("Memory unchanged:")
    assert memory.read()["revision"] == 1


@pytest.mark.parametrize("enable_tools", [False, None])
def test_memory_does_not_take_over_client_tool_catalog(db, enable_tools):
    from types import SimpleNamespace
    from models.inference import ChatCompletionRequest
    from routes.inference import _takes_tool_passthrough, _explicit_studio_tool_loop_requested
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    payload = ChatCompletionRequest(
        model = "test",
        thread_id = "chat",
        enable_tools = enable_tools,
        messages = [{"role": "user", "content": "Check the weather"}],
        tools = [
            {
                "type": "function",
                "function": {"name": "client_weather", "parameters": {"type": "object"}},
            }
        ],
        tool_choice = {"type": "function", "function": {"name": "client_weather"}},
    )
    memory.apply_to_request(payload)
    assert _takes_tool_passthrough(payload, SimpleNamespace(supports_tools = True))
    assert not _explicit_studio_tool_loop_requested(payload)


@pytest.mark.parametrize(
    "contract",
    [
        {"tool_choice": "none"},
        {"response_format": {"type": "json_object"}},
        {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "client_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
            ]
        },
    ],
)
def test_memory_respects_other_client_contracts(db, contract):
    from models.inference import ChatCompletionRequest
    from routes.inference import _explicit_studio_tool_loop_requested
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    payload = ChatCompletionRequest(
        **{
            "model": "test",
            "thread_id": "chat",
            "enable_tools": False,
            "messages": [{"role": "user", "content": "hello"}],
            **contract,
        }
    )
    assert not _explicit_studio_tool_loop_requested(payload)


def test_plain_chat_can_still_request_only_memory(db):
    from models.inference import ChatCompletionRequest
    from routes.inference import _explicit_studio_tool_loop_requested, _effective_enable_tools
    from state.tool_policy import reset_tool_policy

    reset_tool_policy()
    payload = ChatCompletionRequest(
        model = "test",
        thread_id = "chat",
        enable_tools = False,
        messages = [{"role": "user", "content": "Remember my preference"}],
    )
    assert _explicit_studio_tool_loop_requested(payload)
    assert _effective_enable_tools(payload)
