# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Stop before the turn produced anything must not strand two user turns (#9484).

The abandoned turn serialises to a lone empty assistant message, every backend drops it, and
strict templates (Ministral, Gemma) then refuse the two user turns that are left touching.
``pruneOutboundHistory`` drops that turn together with the prompt that triggered it, the way
refusals are already dropped.

The prune, ``toOpenAIMessages`` and ``serializeAssistantReplayMessages`` are sliced verbatim out
of the studio sources and run under ``node`` (see ``_node_harness``), so what is asserted is the
wire shape rather than the spelling of the source. ``cancelled-turn-history-prune.test.ts`` pins
the wiring; this pins what it does, and the refusal-only prune the fix replaces is run beside it
so the case states the defect rather than only the repair.
"""

from __future__ import annotations

import textwrap

from _node_harness import (
    WORKDIR,
    read,
    require_node,
    run_harness,
    slice_between,
    source_path,
)

ADAPTER = source_path("studio/frontend/src/features/chat/api/chat-adapter.ts")
CODEX = source_path("studio/frontend/src/features/chat/codex-reasoning.ts")
CONTINUATION = source_path("studio/frontend/src/features/chat/utils/continuation.ts")

TEMP = WORKDIR / "temp" / "cancelled_turn_history_prune"

SOURCES = (ADAPTER, CODEX, CONTINUATION)

# Fixtures the sliced code reads through.
HARNESS = """
// @ts-nocheck
function readCodexReasoning(_metadata: any): any {
  return undefined;
}

function codexReasoningForToolCalls(_ledger: any, _ids: any): any {
  return undefined;
}

function getToolReplayProvenance(part: any): any {
  return part?.provenance;
}

function shouldFlushCompletedLocalToolPair(_part: any): boolean {
  return false;
}

function canReplayToolCallWithoutRoleTool(part: any): boolean {
  return part?.canReplay === true;
}

function serializeAssistantToolCallPart(part: any): any {
  return part?.toolCallId
    ? {
        id: part.toolCallId,
        type: "function",
        function: { name: part.toolName ?? "", arguments: part.args ?? "{}" },
      }
    : null;
}

function serializeToolResultPart(part: any): any {
  return part?.result === undefined
    ? null
    : { role: "tool", content: String(part.result), tool_call_id: part.toolCallId };
}

// ---- PRELUDE ENDS: verbatim studio source follows ----
"""


def _adapter_slice(start: str, end: str) -> str:
    return slice_between(read(ADAPTER), start, end)


def _send_path_slice() -> str:
    """The send path's own outbound build, wrapped so the test runs it instead of reading it.

    Sliced out of ``createOpenAIStreamAdapter``. Wiring assertions on the source text pass just
    as happily against a send path that went back to ``messages.flatMap(...)``; this one only
    passes while the payload is actually built from pruned history.
    """
    body = slice_between(
        read(ADAPTER),
        "      const survivingMessages = pruneOutboundHistory(\n"
        "        messages,\n"
        "        !isExternalRequest,\n",
        "if (selectedImageEditReference) {",
    )
    return (
        "export function buildSendPathOutbound(messages: any, isExternalRequest: boolean) {\n"
        + body
        + "  return outboundMessages;\n}\n"
    )


def _harness_source() -> str:
    return (
        HARNESS
        + _adapter_slice("function collectTextParts(", "function normalizeOpenAIReasoningItem(")
        + _adapter_slice(
            "function isAnthropicRefusalMessage(",
            "type SerializedMessage = {",
        )
        + _adapter_slice(
            "function sanitizeAssistantReplayText(",
            "function serializeAssistantReplayMessages(",
        )
        + _adapter_slice(
            "function serializeAssistantReplayMessages(",
            "function extractImageBase64(",
        )
        + slice_between(
            read(CODEX),
            "export function codexLocalToolRoundId(",
            "export function addCodexReasoning(",
        )
        + slice_between(
            read(CONTINUATION),
            "export type IncompleteReason =",
            "const INCOMPLETE_LABELS",
        )
        + """
// The refusal-only prune this fix replaces, kept so each case can show what the wire looked
// like before rather than only that it is right now.
export function pruneRefusalsOnly(messages: any[]): any[] {
  const surviving: any[] = [];
  for (const message of messages) {
    if (isAnthropicRefusalMessage(message)) {
      const last = surviving.at(-1);
      if (last && last.role === "user") surviving.pop();
      continue;
    }
    surviving.push(message);
  }
  return surviving;
}

/** Roles on the wire, after the backend drops the empty assistant turns it always drops. */
export function wireRoles(messages: any[], includeReasoningContent: boolean): string[] {
  return messages
    .flatMap((message: any) => toOpenAIMessages(message, includeReasoningContent))
    .filter((message: any) => !(message.role === "assistant" && !message.content
      && !message.tool_calls && !message.reasoning_content))
    .map((message: any) => message.role);
}

export { pruneOutboundHistory, toOpenAIMessages };
"""
        + _send_path_slice()
    )


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script, sources = SOURCES)


USER = '{ role: "user", content: [{ type: "text", text: "TEXT" }] }'
CANCELLED = (
    '{ role: "assistant", content: [], status: { type: "incomplete" },'
    ' metadata: { custom: { incomplete: { reason: "cancelled" } } } }'
)


def _script(history: str, include_reasoning: str = "true") -> str:
    return textwrap.dedent(
        f"""
        // @ts-nocheck
        import {{ pruneOutboundHistory, pruneRefusalsOnly, toOpenAIMessages, wireRoles }}
          from "./harness.ts";
        const history = {history};
        const kept = pruneOutboundHistory(history, {include_reasoning});
        console.log(JSON.stringify({{
          kept: kept.map((m) => m.role),
          keptText: kept.flatMap((m) => (m.content ?? []).filter((p) => p.type === "text")
            .map((p) => p.text)),
          wire: wireRoles(kept, {include_reasoning}),
          wireBefore: wireRoles(pruneRefusalsOnly(history), {include_reasoning}),
        }}));
        """
    )


def _user(text: str) -> str:
    return USER.replace("TEXT", text)


def test_the_defect_is_two_user_turns_touching_on_the_wire():
    """What #9484 reported: the empty turn goes, and the prompts it separated end up adjacent."""
    out = _run(_script(f"[{_user('first')}, {CANCELLED}, {_user('second')}]"))
    assert out["wireBefore"] == ["user", "user"], (
        "the refusal-only prune must still reproduce the stranded pair, or this case has "
        "stopped measuring the bug it was written for"
    )


def test_a_stop_before_any_output_takes_its_prompt_with_it():
    out = _run(_script(f"[{_user('first')}, {CANCELLED}, {_user('second')}]"))
    assert out["kept"] == ["user"]
    assert out["keptText"] == ["second"]
    assert out["wire"] == ["user"]


def test_a_reply_that_produced_text_is_kept_with_its_prompt():
    """The prune reads the wire shape, so anything the model actually said protects the pair."""
    answered = '{ role: "assistant", content: [{ type: "text", text: "an answer" }] }'
    out = _run(_script(f"[{_user('first')}, {answered}, {_user('second')}]"))
    assert out["kept"] == ["user", "assistant", "user"]
    assert out["wire"] == ["user", "assistant", "user"]


def test_a_stop_during_reasoning_is_abandoned_on_both_serialisations():
    """An incomplete turn never replays its reasoning, so the local and external builds agree.

    They pass different ``includeReasoningContent``: the recount always sends true, the request
    sends ``!isExternalRequest``. A turn cut mid-think must prune either way or the two paths
    would price and send different histories.
    """
    thinking = (
        '{ role: "assistant", content: [{ type: "reasoning", text: "let me think" }],'
        ' status: { type: "incomplete" } }'
    )
    for include_reasoning in ("true", "false"):
        out = _run(_script(f"[{_user('first')}, {thinking}, {_user('second')}]", include_reasoning))
        assert out["kept"] == ["user"], f"includeReasoningContent={include_reasoning}"
        assert out["keptText"] == ["second"]


def test_a_turn_that_called_a_tool_is_not_abandoned():
    """tool_calls are payload even with no text; dropping the pair would orphan the call."""
    called = (
        '{ role: "assistant", content: [{ type: "tool-call", toolCallId: "call_1",'
        ' toolName: "web_search", args: "{}", result: "ok" }],'
        ' status: { type: "incomplete" } }'
    )
    out = _run(_script(f"[{_user('first')}, {called}, {_user('second')}]"))
    assert out["kept"] == ["user", "assistant", "user"]
    assert out["wire"] == ["user", "assistant", "tool", "user"]


def test_refusals_are_still_pruned_with_their_prompt():
    """The behaviour the prune already had; the abandoned-turn rule shares its loop now."""
    refused = (
        '{ role: "assistant", content: [{ type: "text", text: "I cannot help with that." }],'
        " metadata: { custom: { anthropicRefusal: true } } }"
    )
    out = _run(_script(f"[{_user('first')}, {refused}, {_user('second')}]"))
    assert out["kept"] == ["user"]
    assert out["keptText"] == ["second"]


def test_back_to_back_stops_collapse_to_the_live_prompt():
    """Stop twice and both abandoned pairs go, rather than one prune uncovering the next."""
    history = f"[{_user('first')}, {CANCELLED}, {_user('second')}, {CANCELLED}, {_user('third')}]"
    out = _run(_script(history))
    assert out["kept"] == ["user"]
    assert out["keptText"] == ["third"]
    assert out["wireBefore"] == ["user", "user", "user"]


def test_a_reply_that_finished_on_reasoning_alone_keeps_its_prompt():
    """A complete reasoning-only turn is a reply, not a Stop.

    External requests serialise with ``includeReasoningContent = false``, which strips the
    reasoning and leaves an empty assistant message. Reading only the wire shape called that
    abandoned and deleted the prompt that produced it, so the question the user actually asked
    left the context on hosted providers but survived on local ones.
    """
    answered = '{ role: "assistant", content: [{ type: "reasoning", text: "thought" }] }'
    for include_reasoning in ("true", "false"):
        out = _run(_script(f"[{_user('first')}, {answered}, {_user('second')}]", include_reasoning))
        assert out["kept"] == [
            "user",
            "assistant",
            "user",
        ], f"includeReasoningContent={include_reasoning}"
        assert out["keptText"] == ["first", "second"]


def test_a_reply_that_finished_with_no_text_at_all_is_still_abandoned():
    """The turn holds nothing either serialisation could carry, so it prunes like a Stop."""
    silent = '{ role: "assistant", content: [{ type: "text", text: "" }] }'
    out = _run(_script(f"[{_user('first')}, {silent}, {_user('second')}]"))
    assert out["kept"] == ["user"]


def test_a_tool_call_the_replay_cannot_carry_prunes_with_its_prompt():
    """A resultless local call is dropped by the serialiser, so the turn carries nothing.

    OpenAI rejects an assistant ``tool_calls`` turn whose ids have no responding ``role="tool"``
    message, so the serialiser cannot rescue this by emitting the call -- it skips it, and the
    turn reaches the provider as the same lone empty assistant message a Stop with no output
    produces. Treating the call as payload only moved the defect one hop: the backend drops the
    empty assistant (``_drop_empty_assistant_sentinels``) and then merges the two user turns
    that are left touching (``_coalesce_consecutive_user_turns``), which resends the cancelled
    prompt glued to the next one and invites the tool request the user Stopped.
    """
    stopped = (
        ', status: { type: "incomplete" },'
        ' metadata: { custom: { incomplete: { reason: "cancelled" } } } }'
    )
    for marker in (stopped, " }"):
        unreplayable = (
            '{ role: "assistant", content: [{ type: "tool-call", toolCallId: "call_1",'
            ' toolName: "delete_file", args: "{}" }]' + marker
        )
        out = _run(_script(f"[{_user('first')}, {unreplayable}, {_user('second')}]"))
        assert out["kept"] == ["user"], marker
        assert out["keptText"] == ["second"]
        assert out["wire"] == ["user"]
        assert out["wireBefore"] == ["user", "user"], (
            "the refusal-only prune must still strand the pair here, or this case has stopped "
            "measuring the defect it was written for"
        )


def test_a_resultless_call_that_replays_without_role_tool_keeps_its_prompt():
    """Resultless is not the test; unreplayable is.

    Provider-native builtin cards replay through ``extra_content``/native parts and never
    produce a ``role="tool"`` message, so ``canReplayToolCallWithoutRoleTool`` lets the
    serialiser emit the call with no result. That call does reach the provider, so the prompt
    that asked for it is history and pruning the pair would delete it.
    """
    builtin = (
        '{ role: "assistant", content: [{ type: "tool-call", toolCallId: "call_1",'
        ' toolName: "web_search", args: "{}", canReplay: true }],'
        ' status: { type: "incomplete" } }'
    )
    out = _run(_script(f"[{_user('first')}, {builtin}, {_user('second')}]"))
    assert out["kept"] == ["user", "assistant", "user"]
    assert out["keptText"] == ["first", "second"]
    assert out["wire"] == ["user", "assistant", "user"]


def test_a_trailing_abandoned_turn_keeps_the_prompt_it_followed():
    """Nothing follows it, so there is no stranded pair to repair and nothing to drop.

    The token-count path rebuilds outbound history for the live thread, which after a Stop
    ends on the abandoned turn. Popping there emptied the whole history and priced the thread
    at zero.
    """
    for include_reasoning in ("true", "false"):
        out = _run(_script(f"[{_user('first')}, {CANCELLED}]", include_reasoning))
        assert out["kept"] == ["user"], f"includeReasoningContent={include_reasoning}"
        assert out["keptText"] == ["first"]


def test_a_thread_that_is_only_a_stop_keeps_its_system_prompt_and_prompt():
    history = (
        '[{ role: "system", content: [{ type: "text", text: "sys" }] }, '
        f"{_user('first')}, {CANCELLED}]"
    )
    out = _run(_script(history))
    assert out["kept"] == ["system", "user"]
    assert out["keptText"] == ["sys", "first"]


def test_the_count_and_send_paths_prune_the_same_history():
    """The recount always passes true and the request passes ``!isExternalRequest``.

    Any shape whose verdict depends on that flag prices one history and sends another, so the
    context bar and the payload disagree.
    """
    shapes = {
        "cancelled": CANCELLED,
        "text": '{ role: "assistant", content: [{ type: "text", text: "an answer" }] }',
        "reasoning_complete": '{ role: "assistant", content: [{ type: "reasoning",'
        ' text: "thought" }] }',
        "reasoning_incomplete": '{ role: "assistant", content: [{ type: "reasoning",'
        ' text: "thought" }], status: { type: "incomplete" } }',
        "tool_unreplayable": '{ role: "assistant", content: [{ type: "tool-call",'
        ' toolCallId: "call_1", toolName: "web_search", args: "{}" }] }',
        "image": '{ role: "assistant", content: [{ type: "image", image: "QUJD" }] }',
    }
    for name, shape in shapes.items():
        history = f"[{_user('first')}, {shape}, {_user('second')}]"
        counted = _run(_script(history, "true"))
        sent = _run(_script(history, "false"))
        assert counted["kept"] == sent["kept"], name
        assert counted["keptText"] == sent["keptText"], name


def _send_script(history: str, is_external: str) -> str:
    return textwrap.dedent(
        f"""
        // @ts-nocheck
        import {{ buildSendPathOutbound }} from "./harness.ts";
        const outbound = buildSendPathOutbound({history}, {is_external});
        console.log(JSON.stringify({{
          roles: outbound.map((m) => m.role),
          contents: outbound.map((m) => m.content),
        }}));
        """
    )


def test_the_send_path_builds_its_payload_out_of_pruned_history():
    """Run the send path's own outbound build, not a restatement of it.

    A send path that stopped pruning would still satisfy every wiring assertion on the source
    text, so the abandoned pair is put through the real slice here.
    """
    for is_external in ("false", "true"):
        out = _run(_send_script(f"[{_user('first')}, {CANCELLED}, {_user('second')}]", is_external))
        assert out["roles"] == ["user"], f"isExternalRequest={is_external}"
        assert out["contents"] == ["second"]


def test_the_send_path_still_carries_an_answered_exchange():
    """The counterpart: pruning must not be the send path quietly dropping history."""
    answered = '{ role: "assistant", content: [{ type: "text", text: "an answer" }] }'
    out = _run(_send_script(f"[{_user('first')}, {answered}, {_user('second')}]", "false"))
    assert out["roles"] == ["user", "assistant", "user"]
    assert out["contents"] == ["first", "an answer", "second"]


def test_a_stop_that_produced_only_whitespace_takes_its_prompt_with_it():
    """Whitespace is not an answer, and the backend already agrees.

    ``_build_external_messages`` drops any assistant turn whose string content trims away
    (studio/backend/routes/inference.py). Keeping the pair here because "   " is truthy in JS
    only moved the defect one hop: the backend removed the assistant alone and put the two user
    turns back on the wire touching.
    """
    for blank in ("   ", "\\n\\t"):
        whitespace = (
            '{ role: "assistant", content: [{ type: "text", text: "%s" }],'
            ' status: { type: "incomplete" },'
            ' metadata: { custom: { incomplete: { reason: "cancelled" } } } }' % blank
        )
        out = _run(_script(f"[{_user('first')}, {whitespace}, {_user('second')}]"))
        assert out["kept"] == ["user"], repr(blank)
        assert out["keptText"] == ["second"]
        assert out["wireBefore"] == ["user", "assistant", "user"], (
            "the refusal-only prune kept the whitespace turn, which is the hop the backend "
            "trim then undid"
        )


def test_whitespace_only_replies_leave_no_pair_for_the_backend_to_split():
    """The same shape without a Stop marker: the backend trims it either way."""
    whitespace = '{ role: "assistant", content: [{ type: "text", text: "  " }] }'
    out = _run(_script(f"[{_user('first')}, {whitespace}, {_user('second')}]"))
    assert out["kept"] == ["user"]
