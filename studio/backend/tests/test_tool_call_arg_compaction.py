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
    """Retargeted: the floor is on the arguments as a WHOLE, not on each string.

    This passed a single 1023-character body, one below the per-leaf floor, and asserted
    nothing was compacted. That per-leaf rule was the defect: a batched refactor of fifty
    800-character edits is forty thousand characters of window with no single string over
    the floor, so nothing was reclaimed and the call sat in the prompt permanently. What
    the floor is really for -- not spending a ~100-character receipt to save less -- is a
    statement about the total, which is what this now pins.
    """
    messages = _thread("x" * 200)

    _, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 0


def test_a_batch_of_small_edits_is_compacted_on_its_total():
    """No single string clears the per-leaf floor; together they are most of the window."""
    edits = [
        {
            "old_string": f"def old_{i}():\n" + "    pass\n" * 60,
            "new_string": f"def new_{i}():\n" + "    return 1\n" * 60,
        }
        for i in range(20)
    ]
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", "edit_file", path = "a.py", edits = edits)],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "Applied 20 edits"},
    ]

    def _args(msgs):
        return msgs[0]["tool_calls"][0]["function"]["arguments"]

    before = len(_args(messages))
    assert all(
        len(value) < _ARG_COMPACTION_FLOOR_CHARS for edit in edits for value in edit.values()
    ), "fixture must keep every leaf under the per-leaf floor"

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    assert len(_args(fitted)) < before // 2


def test_one_large_edit_beside_many_small_ones_still_compacts_all_of_them():
    """The mode was chosen from the LARGEST leaf, so a mixed batch stayed in per-leaf mode.

    One 1100-character edit beside fifty 800-character ones cleared the per-leaf floor on
    the strength of the single big leaf, compacted only that leaf, and replayed the other
    forty-odd kilobytes verbatim. A batched refactor produces exactly this shape.
    """
    edits = [{"old_string": "X" * 1100, "new_string": "Y" * 1100}]
    edits += [
        {"old_string": f"a{i:03d}" + "m" * 795, "new_string": f"b{i:03d}" + "n" * 795}
        for i in range(50)
    ]
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", "edit_file", path = "a.py", edits = edits)],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "Applied 51 edits"},
    ]

    def _args(msgs):
        return msgs[0]["tool_calls"][0]["function"]["arguments"]

    before = len(_args(messages))
    assert before > 80_000, "fixture must be the size the replay actually costs"

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    # Per-leaf mode leaves the fifty small edits whole, which is over half the payload.
    assert len(_args(fitted)) < before // 4


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
    assert "Not tool output" in replayed
    # The tool's real result is the one record of what happened and is never touched.
    assert "Created a.html" in replayed


@pytest.mark.parametrize(
    "reply",
    [
        # The approval gate's own wording, verbatim.
        "The user declined to run this tool call.",
        # The unreadable-arguments guard: the call never reached the tool.
        "Error: edit_file arguments were cut off after 82 characters and could not be "
        "read, so nothing ran. Resend as complete JSON.",
        "Error: edit_file arguments are not valid JSON, so nothing ran.",
    ],
)
def test_a_call_that_never_ran_keeps_its_arguments(reply):
    """A `role=tool` reply proves an ANSWER, not an execution.

    The completed receipt says the content is "already written" and tells the model to
    re-read the file. Handing that to a call the user DECLINED, or one whose arguments
    could not be read, states a write that never happened. The model then reports the
    file as done, or reads a path that does not exist and disbelieves the tool.
    """
    messages = _thread("x" * 8000)
    messages[-1]["content"] = reply

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 0
    assert fitted is messages
    assert "already written" not in json.dumps(fitted)


def test_an_ordinary_result_is_still_compacted_beside_one_that_did_not_run():
    """The guard must cost only the calls that never ran, not disable compaction."""
    body = "x" * 8000
    messages = _thread(body)
    messages.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c2", "edit_file", path = "b.py", new_string = "y" * 8000)],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "c2",
            "name": "edit_file",
            "content": "The user declined to run this tool call.",
        }
    )

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    assert body not in json.dumps(fitted[1])
    assert "y" * 8000 in json.dumps(fitted[3])


def _reused_id_thread():
    """Two turns, both numbering from `call_0`, which is what the parsers really emit."""
    return [
        {"role": "user", "content": "Write the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("call_0", "edit_file", path = "first.html", new_string = "a" * 4000)],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "edit_file",
            "content": "Wrote first.html",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("call_0", "edit_file", path = "second.html", new_string = "b" * 4000)],
        },
    ]


def test_compaction_does_not_reach_back_to_an_older_call_of_the_same_id():
    """Tool-call IDs are not unique across a conversation.

    The textual parsers number from `call_0` with an offset that starts at zero on every
    turn, and the structured fallback does the same when the server omits an ID, so a
    multi-round turn holds several `call_0`s. Rewriting every match hands an earlier
    call a receipt describing a LATER call's fate.
    """
    messages = _reused_id_thread()
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "edit_file",
            "content": "Wrote second.html",
        }
    )

    fitted = compact_executed_call_arguments(messages, "call_0")

    assert "b" * 4000 not in json.dumps(fitted[3]), "the current call was not compacted"
    assert "a" * 4000 in json.dumps(fitted[1]), "an earlier call of the same id was rewritten"


def test_a_refusal_does_not_relabel_an_earlier_success_as_refused():
    """The worse direction: a receipt that states a write which DID happen never did."""
    messages = _reused_id_thread()
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "edit_file",
            "content": "Not enough context",
        }
    )

    fitted = compact_refused_tool_arguments(messages, "call_0")

    assert "refused before it ran" in json.dumps(fitted[3])
    assert "refused before it ran" not in json.dumps(fitted[1])
    assert "a" * 4000 in json.dumps(fitted[1])


@pytest.mark.parametrize("tool_name", ["python", "terminal", "web_search", "mcp__server__tool"])
def test_a_tool_that_writes_no_file_is_not_told_its_arguments_are_on_disk(tool_name):
    """Selection is by size and by having been answered, which is every tool, not just
    the file ones. A 4000-character `code` argument was handed the `edit_file` receipt,
    telling the model the content was already written and that the file on disk holds
    it, so it could go looking for a file nothing had created."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", tool_name, code = "print(1)\n" * 500)],
        },
        {"role": "tool", "tool_call_id": "c1", "name": tool_name, "content": "1"},
    ]

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1, "the arguments still have to be spent; only the wording changes"
    replayed = json.dumps(fitted)
    assert "on disk" not in replayed.replace("nothing was written to disk", "")
    assert "already written" not in replayed
    assert "the call already ran" in replayed


def test_the_file_receipt_is_unchanged_for_edit_file():
    """The wording this path was proven against live must not move."""
    messages = _thread("<!DOCTYPE html>" + "x" * 8000)

    fitted, _ = compact_completed_tool_arguments(messages)

    replayed = _replayed_arguments(fitted)
    assert "already written to flappy-bird.html" in replayed
    assert "the file on disk holds it" in replayed


def test_a_refused_call_with_malformed_arguments_is_not_replayed_as_having_run():
    """The parse-error fallback had the wording hardcoded, so it contradicted the reply.

    The `tool` message beside it says nothing was written; the receipt said the call ran.
    Two accounts of one call, and the one the model can act on is the wrong one.
    """
    broken = '{"path":"a.html","new_string":"' + "z" * 4000  # never closes
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": broken},
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
    assert "z" * 4000 not in replayed
    assert "refused before it ran" in replayed
    assert "after the call ran" not in replayed


@pytest.mark.parametrize(
    "tool_name, lever",
    [
        ("edit_file", "ask for a smaller file"),
        ("python", "run a shorter program"),
        ("terminal", "run a shorter command"),
        # Not a file, not a program, and not guessable: the neutral line.
        ("mcp__server__tool", "ask for less in one call"),
    ],
)
def test_the_refusal_offers_a_lever_the_tool_actually_has(tool_name, lever):
    """The gate runs for every enabled tool, so "ask for a smaller file" was reaching
    calls with no file in them, where it is advice the user cannot act on."""
    message = describe_unservable_tool_call(tool_name, 4119, 4096)

    assert lever in message
    if tool_name != "edit_file":
        assert "smaller file" not in message


def test_an_earlier_success_does_not_vouch_for_a_later_declined_call():
    """Ids restart at call_0 every turn, so "answered" cannot be a conversation-wide set.

    The reply marker correctly skips the denial, but the older success had already put
    call_0 in the set, and the declined call was replayed as already written. The model
    is then told a file exists that it refused to create.
    """
    messages = _reused_id_thread()
    messages.append(
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "edit_file",
            "content": "The user declined to run this tool call.",
        }
    )

    fitted, compacted = compact_completed_tool_arguments(messages)

    # The first call really did run, so it is still spent.
    assert compacted == 1
    assert "a" * 4000 not in json.dumps(fitted[1])
    # The declined one keeps its arguments and is never called written.
    assert "b" * 4000 in json.dumps(fitted[3])
    assert "already written to second.html" not in json.dumps(fitted)


def test_an_edit_that_ran_and_failed_is_not_called_written():
    """The tool NAME does not settle disk state. A call that ran and changed nothing is
    not a write, and saying the content is already there invites the model to skip the
    retry the error was asking for."""
    body = "x" * 8000
    messages = _thread(body)
    messages[-1]["content"] = "Error: old_string not found in flappy-bird.html"

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1, "the arguments are still spent; only the wording changes"
    replayed = json.dumps(fitted)
    assert body not in replayed
    assert "already written" not in replayed
    assert "the file on disk holds it" not in replayed
    assert "the call already ran" in replayed


def test_a_successful_edit_still_earns_the_file_wording():
    messages = _thread("<!DOCTYPE html>" + "x" * 8000)

    fitted, _ = compact_completed_tool_arguments(messages)

    assert "already written to flappy-bird.html" in _replayed_arguments(fitted)


@pytest.mark.parametrize("tool_name", ["python", "terminal"])
def test_a_non_file_tool_is_not_told_it_wrote_nothing(tool_name):
    """The other direction of the same guess: this code may well have created files."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", tool_name, code = "open('out.txt','w')\n" * 400)],
        },
        {"role": "tool", "tool_call_id": "c1", "name": tool_name, "content": "done"},
    ]

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    replayed = json.dumps(fitted)
    assert "nothing was written" not in replayed
    assert "already written" not in replayed
    assert "the call already ran" in replayed


def test_the_destination_path_survives_aggregate_compaction():
    """The receipt promises the content is on disk, so the field naming WHICH file is
    the one thing that can never be spent. Once the aggregate floor lowers the per-leaf
    bar to 256, a long nested path was elided like content."""
    long_path = "src/" + "very_long_directory_name/" * 20 + "module.py"
    assert len(long_path) > 256
    edits = [
        {"old_string": "a" * 300, "new_string": "b" * 300},
        {"old_string": "c" * 300, "new_string": "d" * 300},
    ]
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", "edit_file", path = long_path, edits = edits)],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "edit_file", "content": "Applied 2 edits"},
    ]

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    replayed = fitted[0]["tool_calls"][0]["function"]["arguments"]
    assert long_path in replayed, "the model can no longer name the file it just changed"
    assert "a" * 300 not in replayed


@pytest.mark.parametrize(
    "tool_name, expect_file_wording",
    [("edit_file", True), ("python", False), ("terminal", False), ("mcp__s__t", False)],
)
def test_the_refusal_blames_a_file_only_when_a_file_is_involved(tool_name, expect_file_wording):
    """`_blamed_role` sent every oversized tool call to the file advice, so a program or
    an MCP payload was answered with "ask for a smaller file"."""
    from core.inference.context_window import _blamed_role  # noqa: PLC0415

    message = {"role": "assistant", "tool_calls": [_call("c1", tool_name, code = "x")]}

    role = _blamed_role(message)

    assert role == ("assistant_tool_call" if expect_file_wording else "assistant_tool_payload")


def test_a_reply_pairs_with_the_newest_pending_call_of_a_reused_id():
    """An interrupted call leaves a stale site under an id the next turn reuses.

    Pairing the reply with the OLDEST pending site marks the abandoned call's arguments
    executed while the call that actually ran keeps its arguments replayed in full: both
    halves wrong, and the expensive half is the one that stays.
    """
    body_a = "a" * 4000
    body_b = "b" * 4000
    messages = [
        {"role": "user", "content": "Write the file"},
        {
            "role": "assistant",
            "content": "",
            # Announced, then the turn was interrupted: no `tool` reply ever followed.
            "tool_calls": [_call("call_0", "edit_file", path = "first.html", new_string = body_a)],
        },
        {"role": "user", "content": "Try again"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("call_0", "edit_file", path = "second.html", new_string = body_b)],
        },
        {
            "role": "tool",
            "tool_call_id": "call_0",
            "name": "edit_file",
            "content": "Wrote second.html",
        },
    ]

    fitted, compacted = compact_completed_tool_arguments(messages)

    assert compacted == 1
    assert body_b not in json.dumps(fitted[3]), "the call that actually ran was not compacted"
    assert body_a in json.dumps(fitted[1]), "an unanswered call was called executed"


def test_a_mixed_batch_is_blamed_on_the_call_that_made_it_large():
    """Any `edit_file` in a parallel batch won the file wording, whatever its size.

    A small edit beside an oversized `python` payload was therefore diagnosed as a file
    that is too large, and the advice -- ask for a smaller file -- cannot shrink the
    payload that actually caused the refusal.
    """
    from core.inference.context_window import _blamed_role  # noqa: PLC0415

    message = {
        "role": "assistant",
        "tool_calls": [
            _call("c1", "edit_file", path = "a.py", edits = [{"old_string": "a", "new_string": "b"}]),
            _call("c2", "python", code = "x" * 40000),
        ],
    }

    assert _blamed_role(message) == "assistant_tool_payload"


def test_a_batch_whose_bulk_is_the_file_edit_still_gets_the_file_wording():
    """The other side of the same choice, so the fix cannot swallow the file case."""
    from core.inference.context_window import _blamed_role  # noqa: PLC0415

    message = {
        "role": "assistant",
        "tool_calls": [
            _call("c1", "edit_file", path = "a.py", new_string = "x" * 40000),
            _call("c2", "python", code = "print(1)"),
        ],
    }

    assert _blamed_role(message) == "assistant_tool_call"


def test_a_reply_the_window_replaced_does_not_prove_a_write():
    """`_fit_result_to_room` can swap the real answer for a stub saying there was no room.

    The stub carries none of the failure markers, so a FAILED edit was labelled "already
    written" -- under exactly the tight context that makes compaction run. Absence of
    evidence is not evidence, so the neutral wording applies.
    """
    from core.inference.context_window import _completed_phrase_for  # noqa: PLC0415
    from core.inference.tools import _zero_room_stub  # noqa: PLC0415

    stub = _zero_room_stub(2401, None, True)
    written = _completed_phrase_for("edit_file", "Wrote 2401 chars to a.html")
    omitted = _completed_phrase_for("edit_file", stub)

    assert "written" in written, "the fixture no longer describes a real write"
    assert omitted != written, "a reply the window ate was read as proof of a write"
    assert "written" not in omitted


def test_a_truncated_reply_is_inconclusive_too():
    """The other shape of the same cause: a body cut down with a notice appended."""
    from core.inference.context_window import _completed_phrase_for  # noqa: PLC0415

    cut = "Wrote 2401 ch\n\n... (truncated to 13 chars for the model; 2401 chars total)"

    assert "written" not in _completed_phrase_for("edit_file", cut)


def test_a_refused_call_is_compacted_below_the_general_floor():
    """The floor is there so a receipt never costs more than what it replaces.

    A refused call is the case where that trade is always worth making: the refusal
    message is about to be added to a prompt that already does not fit, so any reduction
    is the difference between the user reading the refusal and reading llama-server's
    context error instead.
    """
    from core.inference.context_window import (  # noqa: PLC0415
        _ARG_COMPACTION_TOTAL_FLOOR_CHARS,
        compact_refused_tool_arguments,
    )

    body = "x" * 400
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", "python", code = body)],
        },
    ]
    before = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert len(before) < _ARG_COMPACTION_TOTAL_FLOOR_CHARS, "fixture must sit under the floor"

    fitted = compact_refused_tool_arguments(messages, "c1")
    after = fitted[0]["tool_calls"][0]["function"]["arguments"]

    assert len(after) < len(before), "a refused call under the floor was left whole"
    assert body not in after


def test_a_receipt_that_would_grow_the_prompt_is_still_refused():
    """The floor's real job, kept: eliding must never cost more than it saves."""
    from core.inference.context_window import compact_refused_tool_arguments  # noqa: PLC0415

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", "python", code = "print(1)")],
        },
    ]
    before = messages[0]["tool_calls"][0]["function"]["arguments"]

    fitted = compact_refused_tool_arguments(messages, "c1")

    assert fitted[0]["tool_calls"][0]["function"]["arguments"] == before


def test_an_overflowing_tool_turn_is_blamed_on_the_call_not_the_reply():
    """Strict templates render the assistant call only once its reply is present.

    The marginal cost of that reply is therefore the reply PLUS the arguments, and
    blaming the reply alone tells the user to ask for a smaller slice of a file when what
    overflowed was the payload they cannot shrink that way.
    """
    from core.inference.context_window import _blamed_role_for_turn  # noqa: PLC0415

    messages = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_call("c1", "python", code = "x" * 40000)],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "python", "content": "ok"},
    ]

    assert _blamed_role_for_turn(messages) == "assistant_tool_payload"


def test_a_tool_reply_with_no_call_behind_it_is_still_blamed_on_itself():
    """The fallback has to survive: a reply whose call is gone is the reply's own weight."""
    from core.inference.context_window import _blamed_role_for_turn  # noqa: PLC0415

    messages = [
        {"role": "user", "content": "run it"},
        {"role": "tool", "tool_call_id": "c1", "name": "python", "content": "x" * 40000},
    ]

    assert _blamed_role_for_turn(messages) == "tool"
