# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The turn that spills is the model's own tool-call arguments, not the tool's result.

`tool_result_budget` sizes what a tool RETURNS. Nothing sized what the model SENT: a
whole-file `edit_file` puts the file itself in the assistant turn, the fit protects that
turn as the newest, and `ctx_shift` is off for any mmproj load, so the request is refused
with the write already on disk. These cover the two levers that answer it -- compacting
the arguments of calls that already returned, and refusing before spending a side effect
when that is not enough.
"""

import json

import pytest

from core.inference.context_refusal import describe_unservable_tool_call
from core.inference.context_window import (
    _ARG_COMPACTION_FLOOR_CHARS,
    _blamed_role,
    compact_completed_tool_arguments,
    compact_executed_call_arguments,
    compact_refused_tool_arguments,
    turn_is_servable,
)


def _call(
    call_id = "c1",
    name = "edit_file",
    **arguments,
):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _thread(
    body,
    *,
    answered = True,
    call_id = "c1",
):
    messages = [
        {"role": "user", "content": "Create a Flappy Bird game in HTML"},
        {
            "role": "assistant",
            "content": "Writing the file.",
            "tool_calls": [_call(call_id, path = "flappy-bird.html", old_string = "", new_string = body)],
        },
    ]
    if answered:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "edit_file",
                "content": f"Wrote {len(body)} chars to flappy-bird.html",
            }
        )
    return messages


def _replayed_arguments(messages):
    return messages[1]["tool_calls"][0]["function"]["arguments"]


def test_an_executed_call_gives_up_its_arguments():
    body = "<!DOCTYPE html>" + "x" * 8000
    messages = _thread(body)
    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    replayed = json.loads(_replayed_arguments(fitted))
    assert str(len(body)) in replayed["new_string"]
    # The path survives: a model that cannot see which file it just wrote answers the
    # receipt by writing it again.
    assert replayed["path"] == "flappy-bird.html"
    assert "flappy-bird.html" in replayed["new_string"]
    assert len(_replayed_arguments(fitted)) < len(_replayed_arguments(messages))


def test_what_the_tool_received_is_never_rewritten():
    """Only the replay changes, exactly as `strip_result_for_model` only changes the replay."""
    body = "x" * 8000
    messages = _thread(body)
    before = _replayed_arguments(messages)

    fitted, _ = compact_completed_tool_arguments(messages)

    assert _replayed_arguments(messages) == before
    assert body in before
    assert fitted is not messages


def test_a_call_still_awaiting_its_result_is_left_alone():
    """Rewriting the in-flight call would describe a write different from the one running."""
    messages = _thread("x" * 8000, answered = False)

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 0
    assert fitted is messages


def test_protect_last_holds_the_freshest_exchange_clear():
    messages = _thread("x" * 8000)

    _, compacted = compact_completed_tool_arguments(messages, protect_last = 2)

    assert compacted == 0


def test_small_arguments_are_not_worth_a_receipt():
    messages = _thread("x" * (_ARG_COMPACTION_FLOOR_CHARS - 1))

    _, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 0


def test_a_thread_with_no_tool_calls_is_returned_untouched():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 0
    assert fitted is messages


def test_unparseable_arguments_still_report_their_size():
    """A call that already ran cannot be re-issued from its arguments, so size is honest."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": "{not json" + "x" * 8000},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "ok"},
    ]

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    assert "8009" in fitted[0]["tool_calls"][0]["function"]["arguments"]


def test_arguments_spread_thin_are_left_rather_than_grown():
    """Receipts for many small fields cost more than the fields did."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", **{f"k{i}": "v" for i in range(200)})],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "ok"},
    ]

    _, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 0


@pytest.mark.parametrize(
    "prompt_tokens, servable",
    [(0, True), (2000, True), (4096, False), (9000, False)],
)
def test_zero_room_is_the_refusal_the_budget_could_not_express(prompt_tokens, servable):
    """`tool_result_budget` clamps at zero, which truncation reads as a number, not a stop."""
    assert (
        turn_is_servable(4096, 512, prompt_tokens) is servable
    ), f"{prompt_tokens} tokens against a 4096 window"


def test_a_tool_call_turn_is_blamed_apart_from_a_resumed_reply():
    """ "Start a new reply" re-runs the same oversized write, so the two need different advice."""
    assert _blamed_role({"role": "assistant", "content": "half a repl"}) == "assistant"
    assert _blamed_role({"role": "assistant", "tool_calls": [_call()]}) == "assistant_tool_call"
    assert _blamed_role({"role": "tool", "content": "out"}) == "tool"


def test_the_pre_execution_refusal_promises_nothing_was_written():
    """The fact the 400 could never offer, and the reason for refusing before executing."""
    message = describe_unservable_tool_call("edit_file", 4237, 4096)

    assert "edit_file" in message
    assert "4237" in message and "4096" in message
    assert "Nothing was written" in message


def test_the_refusal_says_when_history_was_already_spent():
    """Otherwise "increase the Context Length" reads as advice nobody tried."""
    assert "compacted" in describe_unservable_tool_call("edit_file", 4237, 4096, compacted_calls = 2)
    assert "compacted" not in describe_unservable_tool_call("edit_file", 4237, 4096)


def test_content_nested_in_the_edits_array_is_still_elided():
    """edit_file batches through `edits[]`, so the file content is never top level.

    Written a day apart, the batching and this compaction stopped meeting: a top-level
    pass walked `path` and `edits`, found no long string, and compacted nothing at all.
    """
    body = "<!DOCTYPE html>" + "y" * 8000
    messages = [
        {
            "role": "assistant",
            "content": "Writing.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps(
                            {
                                "path": "flappy-bird.html",
                                "edits": [{"old_string": "", "new_string": body}],
                            }
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "Created it"},
    ]

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    replayed = json.loads(fitted[0]["tool_calls"][0]["function"]["arguments"])
    assert body not in json.dumps(fitted)
    assert replayed["path"] == "flappy-bird.html"
    assert str(len(body)) in replayed["edits"][0]["new_string"]


def test_a_refused_call_is_not_described_as_written():
    """Nothing ran, so telling the model to re-read the file would send it after nothing."""
    body = "y" * 8000
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps(
                            {"path": "a.html", "edits": [{"old_string": "", "new_string": body}]}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "name": "edit_file",
            "content": "Not enough context",
        },
    ]

    fitted = compact_refused_tool_arguments(messages, "c1")

    replayed = json.dumps(fitted)
    assert body not in replayed
    assert "refused before it ran" in replayed
    assert "re-read the file" not in replayed


def test_compacting_a_refusal_leaves_other_calls_alone():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _call("keep", "edit_file", path = "a.py", note = "z" * 8000),
                _call("drop", "edit_file", path = "b.py", note = "q" * 8000),
            ],
        },
        {"role": "tool", "tool_call_id": "drop", "name": "edit_file", "content": "refused"},
    ]

    fitted = compact_refused_tool_arguments(messages, "drop")

    calls = fitted[0]["tool_calls"]
    assert "z" * 8000 in calls[0]["function"]["arguments"]
    assert "q" * 8000 not in calls[1]["function"]["arguments"]


@pytest.mark.parametrize(
    "prompt_tokens, servable",
    [
        # Observed live at a 4096 window: the gate refused these while llama-server, which
        # admits on size alone, would have served them. The model answered by retrying
        # ever smaller edits against a bar it could not see.
        (3504, True),
        (3549, True),
        (3712, True),
        # Genuinely too big: no room left to answer in.
        (3740, False),
        (4119, False),
    ],
)
def test_the_gate_refuses_only_what_the_server_would_reject(prompt_tokens, servable):
    """`prompt_budget` sets aside all of max_tokens -- right for sizing a result, far too
    strict for deciding whether a call may run at all."""
    assert turn_is_servable(4096, 512, prompt_tokens) is servable


def test_the_refusal_does_not_read_as_a_contradiction():
    """ "3740 tokens against a 4096-token window" invites the obvious objection."""
    message = describe_unservable_tool_call("edit_file", 3740, 4096)

    assert "no room to reply" in message


def test_a_receipt_cannot_be_mistaken_for_the_tools_output():
    """The first wording was quoted back by the model as "the tool result says ...".

    It then decided the sandbox had mangled its file and abandoned a working approach. The
    receipt has to name whose text it replaced, and say what it is not.
    """
    body = "y" * 8000
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps(
                            {"path": "a.html", "edits": [{"old_string": "", "new_string": body}]}
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "Created a.html"},
    ]

    replayed = json.dumps(compact_executed_call_arguments(messages, "c1"))

    assert "arguments you sent" in replayed
    assert "not the tool's output" in replayed
    # The tool's real result is the one record of what happened and is never touched.
    assert "Created a.html" in replayed
