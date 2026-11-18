# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cloud connections set to compaction_mode "rolling" or "checkpoint" never ask for a summary:
whole oldest turns go to the thread's archive. "checkpoint" resets to the latest turn and /compact
starts an epoch, as on a local model; "rolling" is the pre-summary cloud fit."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
from starlette.requests import Request

from core.inference import cloud_compaction
from storage import studio_db

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_cloud_summary_compaction import (  # noqa: E402
    _SMALL_TRIGGER,
    _SUMMARY,
    _AlwaysOverflowing,
    _OverflowAt8k,
    _SummarisingHandler,
    _bodies,
    _recall_chat,
    _saved_thread,
    _system_text,
)
from test_external_auto_compaction import _long_chat, _truncations  # noqa: E402

_EVICT = dict(compaction_mode = "checkpoint")
_ROLLING = dict(compaction_mode = "rolling")


@pytest.fixture(autouse = True)
def _no_remembered_summaries():
    cloud_compaction._memory_summaries.clear()
    cloud_compaction._cut_off_summaries.clear()
    yield
    cloud_compaction._memory_summaries.clear()
    cloud_compaction._cut_off_summaries.clear()


@pytest.fixture
def _tool_loop_runs(monkeypatch):
    async def connected(self) -> bool:
        return False

    # The tool loop polls for a disconnect and would stop before reaching the provider.
    monkeypatch.setattr(Request, "is_disconnected", connected)


def test_a_long_chat_drops_whole_oldest_turns_without_a_summary_call():
    chat = _long_chat()
    chunks, summaries, [sent] = _bodies(_SummarisingHandler, **_SMALL_TRIGGER, **_EVICT)
    assert summaries == []
    assert "[Conversation summary]" not in _system_text(sent)
    assert sent["messages"][0]["role"] == "system"
    assert "Answer in French." in sent["messages"][0]["content"]
    assert sent["messages"][-1]["content"] == "latest question"
    kept = sent["messages"][1:]
    # Whole turns: what is left is the newest part of the chat, starting on a user turn.
    assert kept == chat[-len(kept) :] and kept[0]["role"] == "user"
    assert len(kept) < len(chat) - 1
    [truncation] = _truncations(chunks)
    assert "summarized" not in truncation and "summary_failed" not in truncation
    assert truncation["dropped_messages"] == len(chat) - 1 - len(kept)


def test_the_default_mode_still_summarises():
    _, summaries, [sent] = _bodies(_SummarisingHandler, **_SMALL_TRIGGER, compaction_mode = "summary")
    assert len(summaries) == 1 and _SUMMARY in _system_text(sent)


def test_a_chat_within_the_threshold_is_forwarded_whole():
    chat = _long_chat(turns = 1)
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler, messages = chat, **_SMALL_TRIGGER, **_EVICT
    )
    assert summaries == [] and _truncations(chunks) == []
    assert len(sent["messages"]) == len(chat)


def test_auto_compact_off_forwards_the_whole_chat():
    chat = _long_chat()
    chunks, summaries, [sent] = _bodies(_SummarisingHandler, **_EVICT)
    assert summaries == [] and _truncations(chunks) == []
    assert len(sent["messages"]) == len(chat)


def test_a_summary_saved_before_switching_modes_is_not_replayed():
    thread = _saved_thread("thread-evict-after-summary")
    _, [_], _ = _bodies(_SummarisingHandler, thread_id = thread, **_SMALL_TRIGGER)
    assert studio_db.list_chat_thread_summaries(thread)
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler, thread_id = thread, **_SMALL_TRIGGER, **_EVICT
    )
    assert summaries == []
    assert _SUMMARY not in json.dumps(sent["messages"])
    [truncation] = _truncations(chunks)
    assert "summarized" not in truncation and truncation["dropped_messages"] > 0


def test_evicted_turns_are_archived_and_recalled_word_for_word(rag_home, rag_conn, stub_embeddings):
    thread = "thread-evict-recall"
    chat = _recall_chat(thread)
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler, messages = chat, thread_id = thread, **_SMALL_TRIGGER, **_EVICT
    )
    assert summaries == []
    # The turn holding the report was dropped; recall brings its exact wording back.
    assert all("055 794 3321" not in str(m["content"]) for m in sent["messages"][:-1])
    assert "055 794 3321" in str(sent["messages"][-1]["content"])
    [truncation] = _truncations(chunks)
    assert "summarized" not in truncation and truncation["recalled_chunks"] > 0


@pytest.mark.parametrize(
    "window",
    [
        dict(compaction_threshold = 200_000, context_window = 256_000),
        # A model with no catalogued window: the request carries no trigger.
        {},
    ],
)
def test_compact_starts_a_checkpoint_without_a_summary(
    rag_home, rag_conn, stub_embeddings, _tool_loop_runs, window
):
    thread = f"thread-evict-manual-{len(window)}"
    chat = _recall_chat(thread)
    chunks, summaries, chats = _bodies(
        _SummarisingHandler,
        messages = chat,
        thread_id = thread,
        compact_context = True,
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
        context_overflow = "truncate_oldest",
        **window,
        **_EVICT,
    )
    assert summaries == []
    sent = chats[0]
    turns = [m for m in sent["messages"] if m["role"] != "system"]
    assert len(turns) < len(chat) - 1
    assert "Was there a phone number on the report?" in json.dumps(turns[-1])
    assert _SUMMARY not in json.dumps(sent["messages"])
    [truncation] = _truncations(chunks)
    assert truncation["manual"] is True and truncation["checkpoint_started"] is True
    assert "summarized" not in truncation
    assert studio_db.list_chat_thread_summaries(thread) == []

    # The checkpoint stays in force on the next turn with auto-compact switched off, once the
    # client has saved the reply that carries it.
    count = len(chat) - 1
    studio_db.upsert_chat_message(
        {
            "id": f"{thread}-{count}",
            "threadId": thread,
            "role": "assistant",
            "content": [{"type": "text", "text": "It is 055 794 3321."}],
            "metadata": {"custom": {"contextTruncation": truncation}},
            "createdAt": count + 2,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": f"{thread}-{count + 1}",
            "threadId": thread,
            "role": "user",
            "content": [{"type": "text", "text": "Thanks."}],
            "createdAt": count + 3,
        }
    )
    later = [
        *chat,
        {"role": "assistant", "content": "It is 055 794 3321."},
        {"role": "user", "content": "Thanks."},
    ]
    _, summaries, chats = _bodies(
        _SummarisingHandler,
        messages = later,
        thread_id = thread,
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
        **_EVICT,
    )
    assert summaries == []
    after = [m for m in chats[0]["messages"] if m["role"] != "system"]
    assert len(after) < len(later) - 1


def test_compact_with_tools_off_still_starts_a_checkpoint(rag_home, rag_conn, stub_embeddings):
    thread = "thread-evict-manual-no-tools"
    chat = _recall_chat(thread)
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler,
        messages = chat,
        thread_id = thread,
        compact_context = True,
        context_overflow = "truncate_oldest",
        **_EVICT,
    )
    assert summaries == []
    assert len(sent["messages"]) < len(chat)
    [truncation] = _truncations(chunks)
    assert truncation["manual"] is True and truncation["checkpoint_started"] is True


def test_the_error_policy_evicts_nothing_after_an_earlier_fit():
    thread = _saved_thread("thread-evict-error-policy")
    chat = _long_chat()
    chunks, _, _ = _bodies(
        _SummarisingHandler, messages = chat, thread_id = thread, **_SMALL_TRIGGER, **_EVICT
    )
    [truncation] = _truncations(chunks)
    studio_db.upsert_chat_message(
        {
            "id": f"{thread}-reply",
            "threadId": thread,
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "metadata": {"custom": {"contextTruncation": truncation}},
            "createdAt": 2,
        }
    )
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler,
        messages = chat,
        thread_id = thread,
        context_overflow = "error",
        compaction_threshold = 6_000,
        context_window = 64_000,
        **_EVICT,
    )
    assert summaries == [] and _truncations(chunks) == []
    assert len(sent["messages"]) == len(chat)


def test_compact_without_a_saved_thread_is_ignored():
    chat = _long_chat(turns = 2)
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler,
        messages = chat,
        compact_context = True,
        context_overflow = "truncate_oldest",
        **_EVICT,
    )
    assert summaries == [] and _truncations(chunks) == []
    assert len(sent["messages"]) == len(chat)


def test_an_overflow_refusal_evicts_and_resends_without_a_summary():
    chat = _long_chat(turns = 4)
    chunks, summaries, chats = _bodies(
        _OverflowAt8k,
        messages = chat,
        context_overflow = "truncate_oldest",
        compaction_threshold = 100_000,
        **_EVICT,
    )
    assert summaries == []
    refused, resent = chats
    assert len(refused["messages"]) == len(chat)
    assert len(resent["messages"]) < len(chat)
    assert resent["messages"][-1]["content"] == "latest question"
    assert not [c for c in chunks if "error" in c]
    [truncation] = _truncations(chunks)
    assert "summarized" not in truncation and truncation["dropped_messages"] > 0


def test_a_second_overflow_is_reported_as_the_error():
    chunks, summaries, chats = _bodies(
        _AlwaysOverflowing,
        messages = _long_chat(turns = 4),
        context_overflow = "truncate_oldest",
        compaction_threshold = 100_000,
        **_EVICT,
    )
    assert summaries == [] and len(chats) == 2
    [error] = [c["error"] for c in chunks if "error" in c]
    assert "maximum context length" in error["message"]


class _RefusedPastAFit(_SummarisingHandler):
    """Refuses a prompt the 6,000-token trigger lets through, then calls a tool once."""

    overflow_chars = 14_000

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
        body = json.loads(raw)
        fits = len(json.dumps(body["messages"])) <= self.overflow_chars
        if (
            self.server.calls_tool
            and fits
            and not any(  # type: ignore[attr-defined]
                m.get("role") == "tool" for m in body["messages"]
            )
        ):
            self.server.recorded.append(body)  # type: ignore[attr-defined]
            call = {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})},
            }
            payload = (
                "".join(
                    "data: "
                    + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": end}]})
                    + "\n\n"
                    for delta, end in (
                        ({"role": "assistant", "tool_calls": [call]}, None),
                        ({}, "tool_calls"),
                    )
                )
                + "data: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.rfile = io.BytesIO(raw)
        super().do_POST()


@pytest.mark.parametrize("tool_loop", [False, True])
def test_an_overflow_after_a_fit_evicts_below_the_refused_prompt(tool_loop, monkeypatch):
    async def connected(self) -> bool:
        return False

    monkeypatch.setattr(Request, "is_disconnected", connected)
    tools = dict(enable_tools = True, enabled_tools = ["python"], permission_mode = "off")

    class Server(_RefusedPastAFit):
        def do_POST(self) -> None:  # noqa: N802
            self.server.calls_tool = tool_loop  # type: ignore[attr-defined]
            super().do_POST()

    chunks, summaries, chats = _bodies(
        Server, messages = _long_chat(), **_SMALL_TRIGGER, **_EVICT, **(tools if tool_loop else {})
    )
    sizes = [len(json.dumps(body["messages"])) for body in chats]
    assert summaries == [] and not [c for c in chunks if "error" in c]
    # The retry evicts below the refused prompt, and a tool pass after it stays there.
    assert sizes[0] > _RefusedPastAFit.overflow_chars and sizes[1] < sizes[0]
    assert all(size <= _RefusedPastAFit.overflow_chars for size in sizes[1:])
    assert len(chats) == (3 if tool_loop else 2)


def test_an_overflow_on_a_model_without_studio_tools_evicts_without_a_checkpoint(
    rag_home, rag_conn, stub_embeddings, monkeypatch
):
    import routes.inference as ri

    monkeypatch.setattr(ri, "provider_model_runs_local_tools", lambda *_: False)
    thread = "thread-evict-overflow-no-tools"
    chunks, summaries, chats = _bodies(
        _OverflowAt8k,
        messages = _recall_chat(thread),
        thread_id = thread,
        context_overflow = "truncate_oldest",
        compaction_threshold = 100_000,
        **_EVICT,
    )
    assert summaries == [] and len(chats) == 2
    [truncation] = _truncations(chunks)
    assert truncation["dropped_messages"] > 0 and not truncation.get("checkpoint_started")


def test_rolling_drops_just_enough_where_checkpoint_resets(rag_home, rag_conn, stub_embeddings):
    sent = {}
    for mode in ("rolling", "checkpoint"):
        thread = f"thread-{mode}"
        chat = _recall_chat(thread)
        chunks, summaries, [sent[mode]] = _bodies(
            _SummarisingHandler,
            messages = chat,
            thread_id = thread,
            **_SMALL_TRIGGER,
            compaction_mode = mode,
        )
        assert summaries == []
        [truncation] = _truncations(chunks)
        assert truncation["recalled_chunks"] > 0
        assert "055 794 3321" in str(sent[mode]["messages"][-1]["content"])
        assert bool(truncation.get("checkpoint")) is (mode == "checkpoint")
    assert len(sent["rolling"]["messages"]) > len(sent["checkpoint"]["messages"])


def test_rolling_compact_starts_no_checkpoint(rag_home, rag_conn, stub_embeddings):
    thread = "thread-rolling-manual"
    chat = _recall_chat(thread)
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler,
        messages = chat,
        thread_id = thread,
        compact_context = True,
        context_overflow = "truncate_oldest",
        **_ROLLING,
    )
    assert summaries == []
    assert len(sent["messages"]) == len(chat)
    assert _truncations(chunks) == []


def test_rolling_ignores_a_checkpoint_saved_before_the_switch(
    rag_home, rag_conn, stub_embeddings, _tool_loop_runs
):
    thread = "thread-switch-to-rolling"
    chat = _recall_chat(thread)
    chunks, _, _ = _bodies(
        _SummarisingHandler,
        messages = chat,
        thread_id = thread,
        compact_context = True,
        context_overflow = "truncate_oldest",
        compaction_threshold = 200_000,
        context_window = 256_000,
        **_EVICT,
    )
    [truncation] = _truncations(chunks)
    count = len(chat) - 1
    studio_db.upsert_chat_message(
        {
            "id": f"{thread}-{count}",
            "threadId": thread,
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "metadata": {"custom": {"contextTruncation": truncation}},
            "createdAt": count + 2,
        }
    )
    later = [*chat, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "Thanks."}]
    sent = {}
    for name in (thread, "thread-fresh-rolling"):
        if name != thread:
            _recall_chat(name)
        chunks, _, [sent[name]] = _bodies(
            _SummarisingHandler, messages = later, thread_id = name, **_SMALL_TRIGGER, **_ROLLING
        )
        assert not _truncations(chunks)[0].get("checkpoint")
    # Recall differs (the switched thread's archive also holds its checkpoint); the window does not.
    switched, fresh = sent[thread]["messages"], sent["thread-fresh-rolling"]["messages"]
    assert len(switched) == len(fresh) and switched[:-1] == fresh[:-1]


@pytest.mark.parametrize("mode", ["rolling", "checkpoint"])
def test_an_overflow_retry_keeps_the_reply_cap_the_window_allows(mode):
    class Refused(_SummarisingHandler):
        overflow_chars = 14_000

    _, _, chats = _bodies(
        Refused,
        messages = _long_chat(),
        context_overflow = "truncate_oldest",
        compaction_threshold = 12_000,
        context_window = 16_384,
        max_tokens = 16_000,
        compaction_mode = mode,
    )
    caps = [body.get("max_tokens") or body.get("max_completion_tokens") for body in chats]
    assert len(caps) == 2 and caps[0] < 16_000 and caps[1] <= caps[0]


def test_checkpoint_mode_resets_under_a_rolling_process_default(
    rag_home, rag_conn, stub_embeddings, monkeypatch
):
    from core.inference import checkpoint

    monkeypatch.setattr(checkpoint, "CONTEXT_POLICY", "rolling")
    thread = "thread-checkpoint-over-rolling-env"
    chat = _recall_chat(thread)
    chunks, _, _ = _bodies(
        _SummarisingHandler, messages = chat, thread_id = thread, **_SMALL_TRIGGER, **_EVICT
    )
    [truncation] = _truncations(chunks)
    assert truncation.get("checkpoint") is True


@pytest.mark.parametrize("mode", ["rolling", "checkpoint"])
def test_an_overflow_with_a_large_latest_turn_still_evicts_every_older_turn(mode):
    # The newest turn alone is more than half the refused prompt, so a half-size budget cannot fit.
    turn = "word " * 400
    chat = [{"role": "system", "content": "Answer in French."}]
    for index in range(4):
        chat += [
            {"role": "user", "content": f"q{index} {turn}"},
            {"role": "assistant", "content": f"a{index} {turn}"},
        ]
    chat.append({"role": "user", "content": "latest " + "doc " * 6000})

    class Refused(_SummarisingHandler):
        overflow_chars = len(json.dumps([chat[0], chat[-1]])) + 2_000

    chunks, _, chats = _bodies(
        Refused,
        messages = chat,
        context_overflow = "truncate_oldest",
        compaction_threshold = 200_000,
        compaction_mode = mode,
    )
    assert not [c for c in chunks if "error" in c]
    assert len(chats) == 2 and [m["role"] for m in chats[1]["messages"]] == ["system", "user"]


@pytest.mark.parametrize("mode", ["rolling", "checkpoint"])
def test_an_overflow_inside_one_tool_turn_drops_its_older_exchange(mode):
    def call(n):
        fn = {"name": "python", "arguments": "{}"}
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"c{n}", "type": "function", "function": fn}],
        }

    chat = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "Run the two checks."},
        call(1),
        {"role": "tool", "tool_call_id": "c1", "content": "log " * 6000},
        call(2),
        {"role": "tool", "tool_call_id": "c2", "content": "ok"},
    ]

    class Refused(_SummarisingHandler):
        overflow_chars = len(json.dumps(chat)) // 2

    chunks, _, chats = _bodies(
        Refused,
        messages = chat,
        context_overflow = "truncate_oldest",
        compaction_threshold = 200_000,
        compaction_mode = mode,
    )
    assert not [c for c in chunks if "error" in c]
    assert len(chats) == 2 and "log log" not in json.dumps(chats[1]["messages"])


def test_the_connection_stores_its_compaction_mode(tmp_path, monkeypatch):
    from models.providers import ProviderCreate, ProviderUpdate
    from pydantic import ValidationError
    from storage import providers_db

    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    monkeypatch.setattr(providers_db, "_schema_ready", set())
    providers_db.create_provider(
        id = "gateway",
        provider_type = "openrouter",
        display_name = "Gateway",
        base_url = "https://openrouter.ai/api/v1",
        compaction_mode = "checkpoint",
    )
    assert providers_db.get_provider("gateway")["compaction_mode"] == "checkpoint"
    assert providers_db.update_provider(id = "gateway", display_name = "Renamed")
    assert providers_db.get_provider("gateway")["compaction_mode"] == "checkpoint"
    assert providers_db.update_provider(id = "gateway", compaction_mode = None)
    assert providers_db.get_provider("gateway")["compaction_mode"] is None

    for mode in ("summary", "rolling", "checkpoint"):
        assert ProviderUpdate(compaction_mode = mode).compaction_mode == mode
    assert "compaction_mode" in ProviderUpdate(compaction_mode = None).model_fields_set
    with pytest.raises(ValidationError):
        ProviderCreate(provider_type = "openrouter", display_name = "x", compaction_mode = "evict")
