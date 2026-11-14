# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Long chats on a cloud connection are summarised, not cut: the summary fit, its fallback,
/compact, the per-connection trigger and the retry after a provider's overflow refusal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from starlette.requests import Request

import routes.inference as ri
from core.inference import cloud_compaction
from core.inference import external_provider as ep_mod
from core.inference.providers import PROVIDER_REGISTRY
from storage import studio_db

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_auto_compaction import (  # noqa: E402
    _AnthropicCompactingHandler,
    _is_summary_call,
    _long_chat,
    _proxy,
    _SummaryRefusingHandler,
    _truncations,
)
from test_external_provider_sampling_over_the_wire import _Handler  # noqa: E402

_SUMMARY = "## Goal\nThe user asked twelve questions about words."


def _sse(text: str) -> bytes:
    return (
        "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": text}}]}) + "\n\n"
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()


class _SummarisingHandler(_Handler):
    """Writes a summary when asked for one and answers every other turn."""

    # Chat turns longer than this many characters are refused the way a provider words it.
    overflow_chars: int | None = None

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.recorded.append(body)  # type: ignore[attr-defined]
        summary = _is_summary_call(body)
        if (
            not summary
            and self.overflow_chars is not None
            and len(json.dumps(body["messages"])) > self.overflow_chars
        ):
            payload = json.dumps(
                {
                    "error": {
                        "message": "This model's maximum context length is 8192 tokens. "
                        "However, your messages resulted in 9000 tokens.",
                        "code": "context_length_exceeded",
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
        else:
            payload = _sse(_SUMMARY if summary else "ok")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _OverflowAt8k(_SummarisingHandler):
    overflow_chars = 8_000


class _AlwaysOverflowing(_SummarisingHandler):
    overflow_chars = 0


def _bodies(
    handler,
    provider_type = "openrouter",
    messages = None,
    **fields,
):
    """Chunks the client saw, then the summary calls and the chat calls the provider received."""
    recorded: list[dict] = []

    class Recording(handler):
        def do_POST(self) -> None:  # noqa: N802
            self.server.recorded = recorded  # type: ignore[attr-defined]
            super().do_POST()

    chunks, _ = _proxy(provider_type, messages or _long_chat(), handler = Recording, **fields)
    summaries = [body for body in recorded if _is_summary_call(body)]
    chats = [body for body in recorded if not _is_summary_call(body)]
    return chunks, summaries, chats


def _saved_thread(thread_id: str) -> str:
    studio_db.upsert_chat_thread(
        {"id": thread_id, "title": "t", "modelType": "base", "modelId": "a-model", "createdAt": 1}
    )
    return thread_id


@pytest.fixture(autouse = True)
def _no_remembered_summaries():
    cloud_compaction._memory_summaries.clear()
    yield
    cloud_compaction._memory_summaries.clear()


# Compacts early in a window large enough for one summary call to read the whole backlog.
_SMALL_TRIGGER = dict(
    context_overflow = "truncate_oldest", compaction_threshold = 6_000, context_window = 64_000
)


def _system_text(body: dict) -> str:
    return "\n".join(m["content"] for m in body["messages"] if m["role"] == "system")


def test_a_long_chat_is_summarised_and_the_recent_turns_stay_verbatim():
    chat = _long_chat()
    chunks, summaries, chats = _bodies(_SummarisingHandler, **_SMALL_TRIGGER)
    [summary_call], [sent] = summaries, chats
    assert "question 0" in summary_call["messages"][1]["content"]
    assert "tools" not in summary_call
    system = _system_text(sent)
    assert "Answer in French." in system
    assert "[Conversation summary]" in system and _SUMMARY in system
    assert sent["messages"][-1]["content"] == "latest question"
    kept = [m for m in sent["messages"] if m["role"] != "system"]
    assert kept == chat[-len(kept) :]
    assert kept[0]["role"] == "user" and len(kept) >= 3
    [truncation] = _truncations(chunks)
    assert truncation["summarized"] is True and truncation["summary_source"] == "unsloth"
    assert truncation["checkpoint_started"] is True and truncation["fits"] is True
    assert truncation["dropped_messages"] == len(chat) - 1 - len(kept)
    assert truncation["prompt_tokens_after"] < truncation["prompt_tokens_before"]
    # The client is told a summary is being written before the reply starts.
    progress = [c for c in chunks if c.get("type") == "generation_progress"]
    assert progress == [] or progress[0]["phase"] == "compacting"


def test_a_backlog_too_long_for_one_summary_call_is_folded_in_oldest_first():
    _, summaries, [sent] = _bodies(
        _SummarisingHandler, context_overflow = "truncate_oldest", compaction_threshold = 6_000
    )
    assert 1 < len(summaries) <= cloud_compaction.SUMMARY_MAX_CALLS
    assert "question 0" in summaries[0]["messages"][1]["content"]
    assert _SUMMARY not in summaries[0]["messages"][1]["content"]
    assert all(_SUMMARY in call["messages"][1]["content"] for call in summaries[1:])
    assert _SUMMARY in _system_text(sent)


def test_a_chat_within_the_threshold_asks_for_no_summary():
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler,
        messages = _long_chat(turns = 1),
        context_overflow = "truncate_oldest",
        compaction_threshold = 6_000,
    )
    assert summaries == [] and _truncations(chunks) == []
    assert len(sent["messages"]) == len(_long_chat(turns = 1))


def test_a_failed_summary_falls_back_to_dropping_the_oldest_turns():
    chat = _long_chat()
    chunks, sent = _proxy(
        "openrouter",
        chat,
        handler = _SummaryRefusingHandler,
        context_overflow = "truncate_oldest",
        compaction_threshold = 6_000,
    )
    assert "[Conversation summary]" not in json.dumps(sent["messages"])
    assert len(sent["messages"]) < len(chat)
    assert sent["messages"][-1]["content"] == "latest question"
    [truncation] = _truncations(chunks)
    assert truncation["summary_failed"] is True
    assert "summarized" not in truncation
    assert truncation["dropped_messages"] == len(chat) - len(sent["messages"])
    assert not [c for c in chunks if "error" in c]


def test_a_saved_thread_reuses_its_summary_on_later_turns():
    thread = _saved_thread("thread-summary-reuse")
    chat = _long_chat()
    options = dict(thread_id = thread, **_SMALL_TRIGGER)
    _, first_summaries, [first] = _bodies(_SummarisingHandler, messages = chat, **options)
    assert len(first_summaries) == 1
    [saved] = studio_db.list_chat_thread_summaries(thread)
    assert saved["summary"] == _SUMMARY and saved["manual"] == 0

    # A fresh process has only the database row.
    cloud_compaction._memory_summaries.clear()
    later = [*chat, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
    chunks, summaries, [sent] = _bodies(_SummarisingHandler, messages = later, **options)
    assert summaries == []
    assert _SUMMARY in _system_text(sent)
    assert sent["messages"][-1]["content"] == "next"
    assert len(sent["messages"]) == len(first["messages"]) + 2
    [truncation] = _truncations(chunks)
    assert truncation["summarized"] is True and "checkpoint_started" not in truncation
    assert truncation["boundary_messages"] == saved["covered_messages"]


def test_an_edited_history_does_not_reuse_a_summary_of_the_old_one():
    thread = _saved_thread("thread-summary-edit")
    options = dict(thread_id = thread, **_SMALL_TRIGGER)
    _bodies(_SummarisingHandler, messages = _long_chat(), **options)
    edited = _long_chat()
    edited[1] = {"role": "user", "content": "a different first question " + "word " * 400}
    _, summaries, _ = _bodies(_SummarisingHandler, messages = edited, **options)
    assert len(summaries) == 1
    assert "a different first question" in summaries[0]["messages"][1]["content"]


def test_a_second_compaction_folds_the_first_summary_in():
    thread = _saved_thread("thread-summary-rolling")
    options = dict(thread_id = thread, **_SMALL_TRIGGER)
    chat = _long_chat()
    _bodies(_SummarisingHandler, messages = chat, **options)
    grown = list(chat)
    for index in range(12):
        grown.append({"role": "assistant", "content": f"more answer {index} " + "word " * 400})
        grown.append({"role": "user", "content": f"more question {index} " + "word " * 400})
    _, [again], [sent] = _bodies(_SummarisingHandler, messages = grown, **options)
    request = again["messages"][1]["content"]
    assert _SUMMARY in request and "more question 0" in request
    assert "<user>\nquestion 0 word" not in request
    assert len(studio_db.list_chat_thread_summaries(thread)) == 2
    assert sent["messages"][-1] == grown[-1]


def test_compact_summarises_a_cloud_chat_that_still_fits():
    thread = _saved_thread("thread-manual-compact")
    chat = _long_chat(turns = 2)
    chunks, [summary_call], [sent] = _bodies(
        _SummarisingHandler,
        messages = chat,
        thread_id = thread,
        compact_context = True,
        context_overflow = "truncate_oldest",
        compaction_threshold = 100_000,
    )
    assert "answer 1" in summary_call["messages"][1]["content"]
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert sent["messages"][-1]["content"] == "latest question"
    [truncation] = _truncations(chunks)
    assert truncation["manual"] is True and truncation["checkpoint_started"] is True
    assert truncation["dropped_messages"] == 4
    [saved] = studio_db.list_chat_thread_summaries(thread)
    assert saved["manual"] == 1

    # The summary stays in force with auto-compact switched off.
    later = [*chat, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}]
    _, summaries, [after] = _bodies(_SummarisingHandler, messages = later, thread_id = thread)
    assert summaries == []
    assert [m["role"] for m in after["messages"]] == ["system", "user", "assistant", "user"]
    assert _SUMMARY in _system_text(after)


def test_compact_without_a_saved_thread_is_ignored():
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler,
        messages = _long_chat(turns = 2),
        compact_context = True,
        context_overflow = "truncate_oldest",
    )
    assert summaries == [] and _truncations(chunks) == []
    assert len(sent["messages"]) == len(_long_chat(turns = 2))


def test_compact_on_anthropic_lowers_the_native_trigger_for_a_long_chat():
    thread = _saved_thread("thread-anthropic-native")
    chunks, sent = _proxy(
        "anthropic",
        _long_chat(turns = 60, turn = "word " * 800),
        handler = _AnthropicCompactingHandler,
        model = "claude-sonnet-4-6",
        thread_id = thread,
        compact_context = True,
        context_overflow = "truncate_oldest",
        compaction_threshold = 150_000,
    )
    assert sent["context_management"]["edits"][0]["trigger"]["value"] == 50_000
    [truncation] = _truncations(chunks)
    assert truncation["summarized"] is True and truncation["manual"] is True
    assert truncation["checkpoint_started"] is True
    assert studio_db.list_chat_thread_summaries(thread) == []


def test_compact_on_a_short_anthropic_chat_uses_the_summary_call():
    thread = _saved_thread("thread-anthropic-short")
    recorded: list[dict] = []

    class Recording(_AnthropicCompactingHandler):
        def do_POST(self) -> None:  # noqa: N802
            self.server.recorded = recorded  # type: ignore[attr-defined]
            super().do_POST()

    _proxy(
        "anthropic",
        _long_chat(turns = 2),
        handler = Recording,
        model = "claude-sonnet-4-6",
        thread_id = thread,
        compact_context = True,
        context_overflow = "truncate_oldest",
        compaction_threshold = 150_000,
    )
    summary_call, sent = recorded
    assert "handover summary" in json.dumps(summary_call["system"])
    assert "[Conversation summary]" in json.dumps(sent["system"])
    assert [m["role"] for m in sent["messages"]] == ["user"]
    assert sent["context_management"]["edits"][0]["trigger"]["value"] == 150_000
    assert len(studio_db.list_chat_thread_summaries(thread)) == 1


def test_compact_on_openai_responses_lowers_the_native_threshold(monkeypatch):
    monkeypatch.setattr(ep_mod, "_is_openai_family_cloud", lambda _base_url: True)
    captured: list[dict] = []

    class Responses(_Handler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            captured.append(json.loads(self.rfile.read(length) or b"{}"))
            self.server.recorded.append(captured[-1])  # type: ignore[attr-defined]
            sse = (
                b"event: response.completed\n"
                b'data: {"type":"response.completed",'
                b'"response":{"output":[],"usage":{"input_tokens":0,"output_tokens":0}}}\n\n'
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(sse)))
            self.end_headers()
            self.wfile.write(sse)

    _proxy(
        "openai",
        _long_chat(turns = 2),
        handler = Responses,
        model = "gpt-5.5",
        thread_id = _saved_thread("thread-openai-native"),
        compact_context = True,
        context_overflow = "truncate_oldest",
        compaction_threshold = 150_000,
    )
    [sent] = captured
    assert sent["context_management"] == [{"type": "compaction", "compact_threshold": 1_000}]


@pytest.mark.parametrize(
    "threshold, uncapped, window, trigger",
    [
        (750_000, None, 1_000_000, 200_000),
        (750_000, True, 1_000_000, 750_000),
        (750_000, True, 400_000, 400_000),
        (10_000, True, 1_000_000, 50_000),
        (10_000, None, 1_000_000, 50_000),
    ],
)
def test_the_connection_trigger_lifts_the_default_cap_on_anthropic(
    threshold, uncapped, window, trigger
):
    _, sent = _proxy(
        "anthropic",
        _long_chat(turns = 1),
        handler = _AnthropicCompactingHandler,
        model = "claude-sonnet-4-6",
        context_overflow = "truncate_oldest",
        compaction_threshold = threshold,
        compaction_uncapped = uncapped,
        context_window = window,
    )
    assert sent["context_management"]["edits"][0]["trigger"]["value"] == trigger


def test_the_native_trigger_clamps():
    assert ep_mod.anthropic_compaction_trigger(1) == 50_000
    assert ep_mod.anthropic_compaction_trigger(900_000) == 200_000
    assert ep_mod.anthropic_compaction_trigger(900_000, uncapped = True) == 900_000
    assert ep_mod.anthropic_compaction_trigger(1, uncapped = True) == 50_000
    assert ep_mod.openai_compaction_threshold(1) == 1_000
    assert ep_mod.openai_compaction_threshold(900_000) == 200_000
    assert ep_mod.openai_compaction_threshold(900_000, uncapped = True) == 900_000


def test_the_connection_trigger_sets_where_the_summary_fit_starts():
    chat = _long_chat()
    options = dict(messages = chat, context_overflow = "truncate_oldest", context_window = 64_000)
    _, summaries, _ = _bodies(_SummarisingHandler, compaction_threshold = 48_000, **options)
    assert summaries == []
    _, summaries, _ = _bodies(
        _SummarisingHandler, compaction_threshold = 6_000, compaction_uncapped = True, **options
    )
    assert len(summaries) == 1


def test_an_overflow_refusal_compacts_once_and_resends():
    chat = _long_chat(turns = 4)
    chunks, [summary_call], chats = _bodies(
        _OverflowAt8k,
        messages = chat,
        context_overflow = "truncate_oldest",
        compaction_threshold = 100_000,
    )
    refused, resent = chats
    assert len(refused["messages"]) == len(chat)
    assert [m["role"] for m in resent["messages"]] == ["system", "user"]
    assert _SUMMARY in _system_text(resent)
    assert "question 3" in summary_call["messages"][1]["content"]
    assert not [c for c in chunks if "error" in c]
    [truncation] = _truncations(chunks)
    assert truncation["summarized"] is True and truncation["checkpoint_started"] is True
    assert any(c.get("choices") and c["choices"][0]["delta"].get("content") == "ok" for c in chunks)


def test_a_second_overflow_is_reported_as_the_error():
    chunks, summaries, chats = _bodies(
        _AlwaysOverflowing,
        messages = _long_chat(turns = 4),
        context_overflow = "truncate_oldest",
        compaction_threshold = 100_000,
    )
    assert len(summaries) == 1 and len(chats) == 2
    [error] = [c["error"] for c in chunks if "error" in c]
    assert "maximum context length" in error["message"]


def test_an_overflow_with_auto_compact_off_is_not_retried():
    chunks, summaries, chats = _bodies(_AlwaysOverflowing, messages = _long_chat(turns = 4))
    assert summaries == [] and len(chats) == 1
    assert [c for c in chunks if "error" in c]


def test_a_tool_enabled_chat_recovers_from_an_overflow_refusal(monkeypatch):
    async def connected(self) -> bool:
        return False

    monkeypatch.setattr(Request, "is_disconnected", connected)
    chunks, summaries, chats = _bodies(
        _OverflowAt8k,
        messages = _long_chat(turns = 4),
        enable_tools = True,
        enabled_tools = ["python"],
        permission_mode = "off",
        context_overflow = "truncate_oldest",
        compaction_threshold = 100_000,
    )
    assert len(summaries) == 1 and len(chats) == 2
    assert chats[-1]["tools"] and _SUMMARY in _system_text(chats[-1])
    assert not [c for c in chunks if "error" in c]


@pytest.mark.parametrize(
    "message, status, overflow",
    [
        ("prompt is too long: 210000 tokens > 200000 maximum", 400, True),
        ("This model's maximum context length is 8192 tokens.", 400, True),
        ("context_length_exceeded", None, True),
        ("Input is too long for requested model.", 400, True),
        ("Rate limit reached: too many tokens per minute", 429, False),
        ("max_tokens must be less than 8192", 400, False),
        ("Invalid API key", 401, False),
        ("", None, False),
    ],
)
def test_overflow_refusals_are_told_from_other_errors(message, status, overflow):
    assert cloud_compaction.is_context_overflow_error(message, status) is overflow


def test_the_tail_starts_on_a_user_turn_and_keeps_the_newest_one():
    body = [m for m in _long_chat(turns = 3) if m["role"] != "system"]
    count = lambda message: 100  # noqa: E731
    assert cloud_compaction.tail_start(body, count, 0) == len(body) - 1
    assert cloud_compaction.tail_start(body, count, 300) == len(body) - 3
    assert cloud_compaction.tail_start(body, count, 10_000) == 0
    assert all(
        body[cloud_compaction.tail_start(body, count, n)]["role"] == "user" for n in (0, 250)
    )


def test_pinned_turns_stay_verbatim_inside_the_summarised_range():
    chat = _long_chat(turns = 2)
    pinned = chat[1]
    compacted = cloud_compaction.apply_summary(chat, "sum", 4, keep_ids = {id(pinned)})
    assert compacted[1] is pinned and compacted[-1] is chat[-1]
    assert compacted[0]["content"].endswith("sum") and len(compacted) == 3
    assert chat[0]["content"] == "Answer in French."


def test_the_connection_stores_its_auto_compact_trigger(tmp_path, monkeypatch):
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
        auto_compact_tokens = 120_000,
    )
    assert providers_db.get_provider("gateway")["auto_compact_tokens"] == 120_000
    assert providers_db.update_provider(id = "gateway", display_name = "Renamed")
    assert providers_db.get_provider("gateway")["auto_compact_tokens"] == 120_000
    assert providers_db.update_provider(id = "gateway", auto_compact_tokens = None)
    assert providers_db.get_provider("gateway")["auto_compact_tokens"] is None

    assert ProviderUpdate(auto_compact_tokens = 400_000).auto_compact_tokens == 400_000
    assert ProviderUpdate(auto_compact_tokens = None).auto_compact_tokens is None
    for rejected in (999, 2_000_001):
        with pytest.raises(ValidationError):
            ProviderCreate(
                provider_type = "openrouter", display_name = "x", auto_compact_tokens = rejected
            )


def test_a_prompt_squeezed_by_fixed_overhead_summarises_up_to_the_newest_turn():
    from models.inference import ChatCompletionRequest

    chat = _long_chat(turns = 2, turn = "word " * 40)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "latest"}],
        provider_type = "openrouter",
        model = "a-model",
        context_overflow = "truncate_oldest",
        compaction_threshold = 2_000,
        max_tokens = 256,
    )
    # A tool catalogue that takes most of the budget: the turns alone are well inside the tail.
    tools = [{"type": "function", "function": {"name": "t", "description": "d " * 3_000}}]
    plan = ri._plan_external_compaction(chat, payload, tools = tools)
    assert plan.to_summarise == chat[1:-1]
    assert ri._plan_external_compaction(chat, payload).to_summarise == []


def test_a_summary_is_still_reported_when_the_rest_cannot_be_trimmed(monkeypatch):
    refusal = {"dropped_messages": 0, "fits": False, "irreducible_tokens": 9_000}
    monkeypatch.setattr(
        ri, "_fit_external_context", lambda messages, payload, **_: (messages, refusal, 256)
    )
    chunks, summaries, [sent] = _bodies(_SummarisingHandler, **_SMALL_TRIGGER)
    assert len(summaries) == 1 and _SUMMARY in _system_text(sent)
    [truncation] = _truncations(chunks)
    assert truncation["summarized"] is True and truncation["fits"] is False
    assert truncation["dropped_messages"] > 0 and truncation["irreducible_tokens"] == 9_000


def test_turns_dropped_after_a_summary_move_the_boundary_past_it(monkeypatch):
    def fit(messages, payload, **_):
        body = [m for m in messages if m["role"] != "system"]
        kept = [m for m in messages if m["role"] == "system"] + body[2:]
        return kept, {"dropped_messages": 2, "boundary_messages": 2, "fits": True}, 256

    monkeypatch.setattr(ri, "_fit_external_context", fit)
    thread = _saved_thread("thread-summary-then-drop")
    chunks, _, _ = _bodies(_SummarisingHandler, thread_id = thread, **_SMALL_TRIGGER)
    [saved] = studio_db.list_chat_thread_summaries(thread)
    [truncation] = _truncations(chunks)
    assert "summarized" not in truncation and truncation["summary_failed"] is False
    assert truncation["dropped_messages"] == 2
    assert truncation["boundary_messages"] == saved["covered_messages"] + 2


def _recall_chat(thread: str) -> list[dict]:
    from test_external_conversation_archive import _chat, _persist

    chat = _chat()
    _persist(thread, chat)
    return chat


def test_a_summarised_chat_recalls_the_exact_wording_the_newest_turn_asks_for(
    rag_home, rag_conn, stub_embeddings
):
    thread = "thread-summary-recall"
    chat = _recall_chat(thread)
    chunks, [_], [sent] = _bodies(
        _SummarisingHandler, messages = chat, thread_id = thread, **_SMALL_TRIGGER
    )
    assert _SUMMARY in _system_text(sent)
    # The summary replaced the turn holding the report; recall brings its wording back.
    assert all("055 794 3321" not in str(m["content"]) for m in sent["messages"][:-1])
    assert "055 794 3321" in str(sent["messages"][-1]["content"])
    assert "Was there a phone number on the report?" in str(sent["messages"][-1]["content"])
    [truncation] = _truncations(chunks)
    assert truncation["summarized"] is True and truncation["recalled_chunks"] > 0


def test_a_reused_summary_still_recalls_for_the_newest_turn(rag_home, rag_conn, stub_embeddings):
    thread = "thread-summary-recall-reuse"
    chat = _recall_chat(thread)
    _bodies(_SummarisingHandler, messages = chat, thread_id = thread, **_SMALL_TRIGGER)
    later = [
        *chat,
        {"role": "assistant", "content": "I cannot see one."},
        {"role": "user", "content": "Check the cardiology report again for the telefono."},
    ]
    chunks, summaries, [sent] = _bodies(
        _SummarisingHandler, messages = later, thread_id = thread, **_SMALL_TRIGGER
    )
    assert summaries == []
    assert "055 794 3321" in str(sent["messages"][-1]["content"])
    [truncation] = _truncations(chunks)
    assert truncation["recalled_chunks"] > 0


# Every chat connection without server-side compaction: native Anthropic and OpenAI have their own
# tests, the ChatGPT subscription needs a saved login, Lemonade is a managed engine.
_SUMMARISED_PROVIDERS = sorted(
    name
    for name, info in PROVIDER_REGISTRY.items()
    if not info.get("decisions_only")
    and name not in {"anthropic", "openai", "openai_codex", "lemonade"}
)


@pytest.mark.parametrize("provider_type", _SUMMARISED_PROVIDERS)
def test_every_connection_without_native_compaction_is_summarised(provider_type):
    _, summaries, chats = _bodies(
        _SummarisingHandler, provider_type = provider_type, **_SMALL_TRIGGER
    )
    assert summaries and _SUMMARY in _system_text(chats[-1])


def test_claude_through_a_gateway_is_summarised_not_cut():
    _, [_], [sent] = _bodies(
        _SummarisingHandler,
        provider_type = "openrouter",
        model = "anthropic/claude-sonnet-4.6",
        **_SMALL_TRIGGER,
    )
    assert _SUMMARY in _system_text(sent)


def _serving(entry: dict):
    class Serving(_SummarisingHandler):
        def do_GET(self) -> None:  # noqa: N802
            payload = json.dumps({"object": "list", "data": [entry]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Serving


@pytest.mark.parametrize(
    "provider_type, entry",
    [
        # vLLM and SGLang list the engine's --max-model-len / --context-length.
        ("vllm", {"id": "a-model", "max_model_len": 8_000}),
        ("custom", {"id": "a-model", "max_model_len": 8_000, "owned_by": "sglang"}),
        ("llama_cpp", {"id": "a-model", "owned_by": "llamacpp", "meta": {"n_ctx": 8_000}}),
    ],
)
def test_a_self_hosted_server_without_a_catalogued_window_is_summarised(provider_type, entry):
    ep_mod._served_windows.clear()
    _, summaries, chats = _bodies(
        _serving(entry), provider_type = provider_type, context_overflow = "truncate_oldest"
    )
    assert summaries and _SUMMARY in _system_text(chats[-1])
