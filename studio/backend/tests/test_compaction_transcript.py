# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every compacted request writes the chat's whole branch as a transcript in its sandbox folder and
names it in the <carried_forward> block, next to search_conversation."""

from __future__ import annotations

import pytest

from core.inference import checkpoint, compaction_transcript, cut_thought

THREAD = "thread-transcript"


@pytest.fixture(autouse = True)
def sandbox(monkeypatch, tmp_path):
    from core.inference import tools

    cut_thought._reset_for_tests()
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(tmp_path))
    yield tmp_path
    cut_thought._reset_for_tests()


def _branch():
    return [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Use the 2024 dataset for every chart."},
        {"role": "assistant", "content": "Noted, 2024 it is."},
        {"role": "user", "content": "Now the revenue section."},
    ]


def _with_block(messages):
    block = checkpoint.render_checkpoint(["Use the 2024 dataset for every chart."])
    return [{**messages[0], "content": messages[0]["content"] + "\n\n" + block}, *messages[1:]]


def test_the_transcript_holds_the_whole_branch(sandbox):
    path = compaction_transcript.write(THREAD, _branch())

    text = (sandbox / f"transcript-{THREAD}.md").read_text()
    assert path == str(sandbox / f"transcript-{THREAD}.md")
    assert "Use the 2024 dataset for every chart." in text and "Now the revenue section." in text
    # The system prompt is not the conversation.
    assert "Be brief." not in text


def test_the_block_names_the_transcript_beside_the_search(sandbox):
    path = compaction_transcript.write(THREAD, _branch())

    linked = compaction_transcript.link_in_block(_with_block(_branch()), path)

    system = linked[0]["content"]
    assert compaction_transcript.mention(path) in system
    assert system.index("search_conversation") < system.index(path) < system.index("- Use the 2024")


def test_the_link_is_not_carried_into_the_next_epoch_as_an_instruction(sandbox):
    path = compaction_transcript.write(THREAD, _branch())
    linked = compaction_transcript.link_in_block(_with_block(_branch()), path)

    # The next reset re-reads the block's bullets; the transcript line is not one of them.
    assert checkpoint._block_items(linked[0]["content"]) == [
        "Use the 2024 dataset for every chart."
    ]
    # And a refit strips the whole block, the line with it.
    assert path not in checkpoint._without_block(linked)[0]["content"]


def test_linking_twice_changes_nothing(sandbox):
    path = compaction_transcript.write(THREAD, _branch())
    once = compaction_transcript.link_in_block(_with_block(_branch()), path)

    assert compaction_transcript.link_in_block(once, path) == once


def test_without_a_block_nothing_is_added(sandbox):
    path = compaction_transcript.write(THREAD, _branch())

    assert compaction_transcript.link_in_block(_branch(), path) == _branch()


def test_a_cut_thought_is_labelled_with_its_closing_marker(sandbox, monkeypatch):
    monkeypatch.setattr(cut_thought, "_persisted", lambda thread_id: True)
    cut_thought.mark(THREAD, "Step 12 needs the ledger first", "\n</think>\n\n")
    cut_thought.take(THREAD)
    branch = _branch() + [
        {"role": "assistant", "content": "", "reasoning_content": "Step 12 needs the ledger first"}
    ]

    compaction_transcript.write(THREAD, branch)

    text = (sandbox / f"transcript-{THREAD}.md").read_text()
    assert "(thinking, stopped by the context window)" in text
    assert text.count("Step 12 needs the ledger first") == 1
    assert "Step 12 needs the ledger first\n</think>" in text


def test_tool_calls_and_their_results_are_in_the_transcript(sandbox):
    branch = _branch() + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "python",
                        "arguments": '{"code": "open(\'report.md\').read()"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "python",
            "content": "Revenue grew 12%.",
        },
    ]

    compaction_transcript.write(THREAD, branch)

    text = (sandbox / f"transcript-{THREAD}.md").read_text()
    assert "## assistant called python" in text and "report.md" in text
    assert "## tool result (python)" in text and "Revenue grew 12%." in text


# ── Wired into every compacted request ──


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
        if message["role"] == "system":
            continue
        studio_db.upsert_chat_message(
            {
                "id": f"{thread_id}-{index}",
                "threadId": thread_id,
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
                "createdAt": index + 2,
            }
        )


def test_a_compacted_request_writes_and_links_the_transcript(conn, sandbox):
    from core.inference.llama_cpp import _archive_and_recall

    before = _branch()
    _save_thread(THREAD, before)
    fitted = _with_block([before[0], before[-1]])

    result = _archive_and_recall(fitted, before, thread_id = THREAD, style = "inline", recall_done = True)

    path = str(sandbox / f"transcript-{THREAD}.md")
    assert compaction_transcript.mention(path) in result["conversation"][0]["content"]
    assert "Noted, 2024 it is." in (sandbox / f"transcript-{THREAD}.md").read_text()


def test_a_second_compaction_still_writes_the_whole_branch(conn, sandbox):
    from core.inference.llama_cpp import _archive_and_recall

    branch = _branch()
    _save_thread(THREAD, branch)
    # A later fit in the same run sees only the already-fitted history, not the turns evicted first.
    already_fitted = [branch[0], branch[-1]]
    fitted = _with_block(already_fitted)

    _archive_and_recall(
        fitted,
        already_fitted,
        branch_messages = branch,
        thread_id = THREAD,
        style = "inline",
        recall_done = True,
    )

    assert "Noted, 2024 it is." in (sandbox / f"transcript-{THREAD}.md").read_text()
