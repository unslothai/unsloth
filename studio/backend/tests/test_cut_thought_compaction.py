# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A thought the context window cut before any answer (no EOS) is not resumed: the reply ends with
the thought closed, and the thread's next request compacts, archives the thought for
search_conversation and starts with a note saying so."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_length_truncated_reasoning_continuation import (  # noqa: E402
    _LONG_THOUGHT,
    _done,
    _finish,
    _make_backend,
    _sse,
    _usage,
)

from core.inference import cut_thought  # noqa: E402

THREAD = "thread-cut"


@pytest.fixture(autouse = True)
def _fresh_markers():
    cut_thought._reset_for_tests()
    yield
    cut_thought._reset_for_tests()


@pytest.fixture
def persisted(monkeypatch):
    monkeypatch.setattr(cut_thought, "_persisted", lambda thread_id: True)


def _thought_cut(completion_tokens: int = 3_000) -> list[str]:
    return [
        _sse({"reasoning_content": _LONG_THOUGHT}),
        _finish("length"),
        _usage(completion_tokens),
        _done(),
    ]


_TEMPLATE_CLOSE = "\n</think>\n\n"


def _with_server_helpers(backend, monkeypatch):
    """Words stand in for tokens; the template closes a thought with Qwen3's marker."""

    def fake_post(path, body):
        if path == "/tokenize":
            return {"tokens": body["content"].split()}
        if path == "/detokenize":
            return {"content": " ".join(body["tokens"])}
        if path == "/apply-template":
            turn = body["messages"][-1]
            return {
                "prompt": f"<think>\n{turn['reasoning_content']}{_TEMPLATE_CLOSE}{turn['content']}"
            }
        return None

    monkeypatch.setattr(backend, "_llama_post_json", fake_post)
    return backend


def _plain(backend, **kwargs) -> list:
    return list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "Plan the migration"}],
            enable_thinking = True,
            thread_id = THREAD,
            **kwargs,
        )
    )


# ── At the cut ──


def test_the_cut_reply_ends_with_the_thought_closed_and_nothing_more(monkeypatch, persisted):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut()], payloads)

    events = _plain(backend)

    # llama-server is not asked to finish the thought or to answer.
    assert len(payloads) == 1
    texts = [e for e in events if isinstance(e, str)]
    assert texts[-1].endswith("</think>")
    assert [
        e["finish_reason"] for e in events if isinstance(e, dict) and e.get("type") == "metadata"
    ] == ["length"]


def test_the_cut_thought_is_cut_back_and_held_for_the_threads_next_request(monkeypatch, persisted):
    backend = _with_server_helpers(_make_backend(monkeypatch, [_thought_cut()], []), monkeypatch)

    _plain(backend)

    cut = cut_thought.take(THREAD)
    # The window can stop it mid-word, so the last few tokens go.
    assert cut.thought == " ".join(_LONG_THOUGHT.split()[: -cut_thought.BACKTRACK_TOKENS])
    # The model's own closing marker, read from its template.
    assert cut.close_marker == _TEMPLATE_CLOSE
    # Once.
    assert cut_thought.take(THREAD) is None


@pytest.mark.parametrize(
    "stream, kwargs",
    [
        # A clean stop.
        ([_sse({"reasoning_content": "Short."}), _finish("stop"), _done()], {}),
        # An answer the window cut: there is something to continue.
        (
            [
                _sse({"reasoning_content": "Plan."}),
                _sse({"content": "Step 1"}),
                _finish("length"),
                _done(),
            ],
            {},
        ),
        # The caller's own Max Tokens ran out, not the window: resuming the thought is right.
        (_thought_cut(completion_tokens = 512), {"max_tokens": 512}),
        # The recovery request itself: once per thread.
        (_thought_cut(), {"record_cut_thought": False}),
    ],
)
def test_only_a_thought_the_window_cut_is_held(monkeypatch, persisted, stream, kwargs):
    backend = _make_backend(monkeypatch, [stream], [])

    _plain(backend, **kwargs)

    assert cut_thought.take(THREAD) is None


def test_a_thread_studio_does_not_keep_is_left_alone(monkeypatch):
    monkeypatch.setattr(cut_thought, "_persisted", lambda thread_id: False)
    backend = _make_backend(monkeypatch, [_thought_cut()], [])

    _plain(backend)

    assert cut_thought.take(THREAD) is None


def test_a_stale_cut_is_dropped(monkeypatch, persisted):
    cut_thought.mark(THREAD, "old thought")
    real = cut_thought.time.monotonic
    monkeypatch.setattr(cut_thought.time, "monotonic", lambda: real() + cut_thought._TTL_S + 1)

    assert cut_thought.take(THREAD) is None


# ── The thread's next request ──


def _request(**kwargs):
    from models.inference import ChatCompletionRequest
    return ChatCompletionRequest(model = "local", thread_id = THREAD, stream = True, **kwargs)


def _continuation():
    """What the chat's automatic continuation sends after the cut: the thought to resume."""
    return _request(
        messages = [
            {"role": "user", "content": "Plan the migration"},
            {"role": "assistant", "content": "", "reasoning_content": _LONG_THOUGHT},
        ],
        continue_final_message = True,
        context_policy = "checkpoint",
    )


@pytest.fixture
def archived(monkeypatch, tmp_path):
    from core.inference import tools
    from state import tool_policy

    stored: list = []
    monkeypatch.setattr(
        cut_thought,
        "archive",
        lambda thread_id, thought: stored.append((thread_id, thought)) or True,
    )
    monkeypatch.setattr(tool_policy, "conversation_recall_allowed", lambda: True)
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(tmp_path))
    return stored


def _marked(thought = "Step 12 needs the ledger moved first", close = _TEMPLATE_CLOSE):
    cut_thought.mark(THREAD, thought, close)
    return thought


def test_the_next_request_finishes_the_thought_instead_of_resuming_it(persisted, archived):
    from routes import inference as inference_route

    thought = _marked()
    payload = _continuation()

    note, searchable, note_is_a_user_turn = inference_route._recover_cut_thought(payload)

    roles = [m.role for m in payload.messages]
    assert roles == ["user", "assistant", "user"]
    # Kept in the conversation as a finished thought: no answer, nothing left to resume.
    finished = payload.messages[1]
    assert finished.reasoning_content == thought and not finished.content
    assert payload.continue_final_message is False
    # The note is the first user turn after the reset.
    assert note_is_a_user_turn and payload.messages[-1].content == note
    # Forced, as /compact does.
    assert payload.compact_context is True
    assert payload.context_overflow == "truncate_oldest"
    assert archived == [(THREAD, thought)]
    assert searchable is True


def test_the_note_names_the_search_and_the_transcript(persisted, archived, tmp_path):
    from routes import inference as inference_route

    thought = _marked()
    note, _, _ = inference_route._recover_cut_thought(_continuation())

    path = tmp_path / f"transcript-{THREAD}.md"
    assert note == (
        f"The previous model stopped thinking, so re-check via search_conversation or read the "
        f"transcript at {path} and continue work."
    )
    transcript = path.read_text()
    assert "Plan the migration" in transcript
    # The thought ends with the model's own closing marker, once.
    assert (thought + _TEMPLATE_CLOSE) in transcript
    assert transcript.count(thought) == 1


def test_a_typed_next_message_keeps_it_and_gets_the_note_in_the_system_prompt(persisted, archived):
    from routes import inference as inference_route

    thought = _marked()
    payload = _request(
        messages = [
            {"role": "user", "content": "Plan the migration"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "go on"},
        ],
        context_policy = "checkpoint",
    )

    note, _, note_is_a_user_turn = inference_route._recover_cut_thought(payload)

    assert not note_is_a_user_turn
    assert [m.content for m in payload.messages if m.role == "user"] == [
        "Plan the migration",
        "go on",
    ]
    assert payload.messages[1].reasoning_content == thought
    assert payload.compact_context is True


def test_without_search_or_a_transcript_the_note_names_neither(persisted, monkeypatch):
    from routes import inference as inference_route

    monkeypatch.setattr(cut_thought, "archive", lambda thread_id, thought: False)
    from core.inference import compaction_transcript

    monkeypatch.setattr(compaction_transcript, "write", lambda *args: None)
    _marked()

    note, searchable, _ = inference_route._recover_cut_thought(_continuation())

    assert searchable is False
    assert "search_conversation" not in note and "transcript" not in note


def test_a_request_with_no_cut_is_untouched(archived):
    from routes import inference as inference_route

    payload = _continuation()

    assert inference_route._recover_cut_thought(payload) is None
    assert payload.continue_final_message is True and len(payload.messages) == 2
    assert archived == []


def test_a_caller_that_asked_for_errors_keeps_them(persisted, archived):
    from routes import inference as inference_route

    _marked()
    payload = _continuation()
    payload.context_overflow = "error"

    assert inference_route._recover_cut_thought(payload) is None
    assert payload.compact_context is None


# ── The closing marker, read from the template ──


@pytest.mark.parametrize(
    "template, close",
    [
        # Qwen3 / Qwen3.8
        (
            "{% for m in messages %}{% if m.role == 'assistant' %}<|im_start|>assistant\n<think>\n"
            "{{ m.reasoning_content }}\n</think>\n\n{{ m.content }}<|im_end|>{% endif %}{% endfor %}",
            "\n</think>\n\n",
        ),
        # Gemma 4
        (
            "{% for m in messages %}{% if m.role == 'assistant' %}<|turn>model\n<|channel>thought\n"
            "{{ m.reasoning_content }}\n<channel|>{{ m.content }}<turn|>{% endif %}{% endfor %}",
            "\n<channel|>",
        ),
        # Inkling 2
        (
            "{% for m in messages %}{% if m.role == 'assistant' %}<|message_model|><|content_thinking|>"
            "{{ m.reasoning_content }}<|end_message|><|message_model|><|content_text|>{{ m.content }}"
            "<|end_message|>{% endif %}{% endfor %}",
            "<|end_message|><|message_model|><|content_text|>",
        ),
    ],
)
def test_the_closing_marker_is_read_from_the_template(template, close):
    import jinja2
    compiled = jinja2.Environment().from_string(template)

    assert (
        cut_thought.close_marker_from_template(lambda messages: compiled.render(messages = messages))
        == close
    )


def test_a_template_that_drops_the_thought_gives_no_marker():
    assert (
        cut_thought.close_marker_from_template(lambda messages: "<|im_start|>assistant\nanswer")
        == ""
    )


# ── search_conversation finds the thought ──


@pytest.fixture
def conn(rag_home, rag_conn, stub_embeddings):
    return rag_conn


def _save_thread(thread_id, messages):
    from storage import studio_db
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "t",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    for index, message in enumerate(messages):
        studio_db.upsert_chat_message(
            {
                "id": f"{thread_id}-{index}",
                "threadId": thread_id,
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
                "createdAt": index + 2,
            }
        )


def test_search_conversation_finds_the_cut_thought(conn):
    from core.rag import conversation_archive

    branch = [{"role": "user", "content": "Plan the migration"}]
    _save_thread(THREAD, branch)

    assert cut_thought.archive(THREAD, "Step 12 needs the ledger moved first, code ZEPHYR-4411.")

    # The live branch never carries reasoning, yet the thought is this thread's own.
    hit = conversation_archive.recall(THREAD, "ZEPHYR-4411", branch_messages = branch)
    assert hit is not None and "ZEPHYR-4411" in hit[0]


def test_an_answer_off_the_branch_is_still_filtered(conn):
    from core.rag import conversation_archive

    branch = [{"role": "user", "content": "Plan the migration"}]
    _save_thread(THREAD, branch)
    conversation_archive.archive_turns(
        THREAD,
        [
            {"role": "user", "content": "Plan the migration"},
            {"role": "assistant", "content": "A retried reply, code SIBLING-9090."},
        ],
    )

    hit = conversation_archive.recall(THREAD, "SIBLING-9090", branch_messages = branch)
    assert hit is None or "SIBLING-9090" not in hit[0]


def test_the_archive_and_the_marker_agree_on_the_thought_document():
    from core.rag import conversation_archive

    # core.rag cannot import core.inference, so the label is written in both; they must match.
    assert conversation_archive._CUT_THOUGHT_DOCUMENT == cut_thought.THOUGHT_DOCUMENT


# ── First line: a thinking budget that leaves the answer room ──


@pytest.mark.parametrize(
    "context_length, prompt_tokens, max_tokens, requested, expected",
    [
        # Small window: half the room thinks, half answers.
        (1024, 0, None, None, 512),
        # Large window: a quarter of the room is kept for the answer.
        (32768, 768, None, None, 24000),
        # The caller's own budget still wins when it is smaller.
        (32768, 768, None, 2000, 2000),
        # A max_tokens below the room stops the reply first: nothing to add.
        (32768, 768, 4000, None, None),
        # No window or a prompt already over it: leave the request as it was.
        (0, 100, None, None, None),
        (1024, 2000, None, 300, 300),
    ],
)
def test_the_window_budget(context_length, prompt_tokens, max_tokens, requested, expected):
    assert (
        cut_thought.window_budget(context_length, prompt_tokens, max_tokens, requested) == expected
    )


def test_a_thinking_request_asks_llama_server_to_close_the_thought_in_time(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut()], payloads)
    backend._effective_context_length = 1024

    _plain(backend)

    budget = payloads[0]["thinking_budget_tokens"]
    assert 0 < budget <= 512


def test_a_request_with_thinking_off_gets_no_budget(monkeypatch):
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [_thought_cut()], payloads)

    list(
        backend.generate_chat_completion(
            messages = [{"role": "user", "content": "Plan the migration"}],
            enable_thinking = False,
            thread_id = THREAD,
        )
    )

    assert "thinking_budget_tokens" not in payloads[0]


def test_the_route_reparses_and_archives_off_the_event_loop():
    """The recovery rewrites messages an auto-switch may already have parsed, and its archive embeds a
    window-sized thought: the route drops that parse and runs the recovery in a thread."""
    import ast
    import inspect

    from routes import inference as inference_route

    tree = ast.parse(inspect.getsource(inference_route.produce_openai_chat_completions))
    source = ast.unparse(tree)
    assert "await asyncio.to_thread(_recover_cut_thought, payload)" in source
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "_cut_recovery is not None"
    ]
    assert any(
        isinstance(stmt, ast.Assign) and ast.unparse(stmt) == "_pre_parsed = None"
        for guard in guards
        for stmt in guard.body
    )
