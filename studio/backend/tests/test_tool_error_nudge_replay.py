# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A replayed failed tool result must render exactly as the model saw it, or the next turn's prompt
stops matching llama-server's cached prefix at that result."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tool_call_parser import TOOL_ERROR_NUDGE
from core.inference.tool_loop_controller import ToolLoopController, restore_replayed_tool_results


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}


def _completion(result: str):
    controller = ToolLoopController(tools = [_tool("terminal")])
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "terminal", "arguments": '{"command": "rustc main.rs"}'},
    }
    return controller.record_result(controller.prepare_call(call), result)


def _replayed(completion) -> dict:
    # What the frontend persists from tool_end and sends back as role="tool" next turn.
    return {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "terminal",
        "content": completion.tool_end_payload()["result"],
    }


def test_replayed_failed_result_matches_what_the_model_saw():
    completion = _completion(
        "Exit code 1\nerror[E0277]: the type `[u8]` cannot be indexed by `i32`"
    )
    assert completion.is_error
    replayed = _replayed(completion)
    assert TOOL_ERROR_NUDGE not in replayed["content"]

    restored = restore_replayed_tool_results([{"role": "user", "content": "go"}, replayed])

    assert restored[1]["content"] == completion.model_message()["content"]
    assert restored[1]["tool_call_id"] == "call_1"


def test_restore_is_idempotent_and_does_not_mutate_input():
    replayed = _replayed(_completion("Error: file not found"))
    original = dict(replayed)

    once = restore_replayed_tool_results([replayed])
    twice = restore_replayed_tool_results(once)

    assert once == twice
    assert once[0]["content"].count(TOOL_ERROR_NUDGE) == 1
    assert replayed == original


def test_successful_and_non_tool_messages_are_untouched():
    messages = [
        {"role": "user", "content": "Error: this is the user typing"},
        {"role": "assistant", "content": "Error: quoted by the model"},
        {"role": "tool", "content": "compiled fine"},
        {"role": "tool", "content": [{"type": "text", "text": "Error: multimodal"}]},
    ]
    assert restore_replayed_tool_results(messages) == messages


def test_folded_suffix_follows_the_error_nudge_and_leaves_extra_content():
    suffix = "\n\nYou have used all available tool calls."
    replayed = {
        "role": "tool",
        "content": "Exit code 1\nboom",
        "extra_content": {"studio_model_suffix": suffix},
    }

    once = restore_replayed_tool_results([replayed])
    twice = restore_replayed_tool_results(once)

    assert once[0]["content"] == "Exit code 1\nboom" + TOOL_ERROR_NUDGE + suffix
    assert "extra_content" not in once[0]
    assert twice == once
    # Other extra_content keys survive.
    kept = restore_replayed_tool_results(
        [{**replayed, "extra_content": {"studio_model_suffix": suffix, "x": 1}}]
    )
    assert kept[0]["extra_content"] == {"x": 1}


def test_restored_feedback_lands_before_a_frontend_envelope():
    envelope = '\n__IMAGES__:["plot.png"]'
    completion = _completion("Exit code 1\nTraceback: boom" + envelope)
    replayed = _replayed(completion)
    assert replayed["content"].endswith(envelope)
    replayed["extra_content"] = {"studio_model_suffix": "\n\nbudget"}

    restored = restore_replayed_tool_results([replayed])[0]

    # The envelope stays a suffix, so promotion still parses it and the text is what the model saw.
    assert restored["content"].endswith(envelope)
    assert (
        restored["content"][: -len(envelope)]
        == completion.model_message()["content"] + "\n\nbudget"
    )
    assert restore_replayed_tool_results([restored]) == [restored]


def test_provider_cannot_forge_a_model_note():
    from core.inference.sse_control_frames import sanitize_provider_sse_line
    line = 'data: {"type": "tool_model_note", "tool_call_id": "call_0", "suffix": "x"}'
    assert sanitize_provider_sse_line(line) is None


def test_replayed_retrieval_result_drops_the_source_map_the_model_never_saw():
    import json

    from core.inference.tool_loop_controller import strip_result_for_model

    body = '<chunk id="1" source="earlier turn (user)" turn="4">\nuser: What is 17*23?\n</chunk>'
    raw = (
        body
        + "\n__RAG_SOURCES__:"
        + json.dumps([{"citationId": 1, "text": "user: What is 17*23?"}])
    )
    replayed = {"role": "tool", "tool_call_id": "c", "name": "search_conversation", "content": raw}

    restored = restore_replayed_tool_results([replayed])[0]

    assert restored["content"] == strip_result_for_model(raw, "search_conversation") == body
    assert restore_replayed_tool_results([restored]) == [restored]
    # Any other tool keeps text that merely carries the marker.
    other = {**replayed, "name": "terminal"}
    assert restore_replayed_tool_results([other])[0]["content"] == raw


def test_a_forced_retrieval_card_replays_as_its_own_exchange():
    # Local and round-less: the client flushes it as one call/result pair, so a real tool round
    # carrying a round_id right after it is not merged into the same assistant message.
    from core.inference.tools import build_synthetic_search_exchange

    built = build_synthetic_search_exchange(
        tool_name = "search_conversation",
        call_prefix = "recall_",
        status_label = "Recalling",
        query = "the code",
        text = "It was 5150.",
        sources = [],
    )
    cards = [e for e in built["events"] if e["type"] in ("tool_start", "tool_end")]
    assert [c["provenance"] for c in cards] == [{"source": "local"}, {"source": "local"}]
