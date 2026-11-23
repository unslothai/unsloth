# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Stop before any output fills the turn with incompleteLabel, so user turns still alternate."""

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
    """Slices the real outbound build from the adapter so the test runs it, not its source text."""
    body = slice_between(
        read(ADAPTER),
        "      const survivingMessages = pruneOutboundHistory(messages, replayReasoning);",
        "if (selectedImageEditReference) {",
    )
    return (
        "export function buildSendPathOutbound(messages: any, isExternalRequest: boolean) {\n"
        "  const supportsStudioToolsForThisTurn = false, studioLocalCodeTools: string[] = [];\n"
        # No test here carries provider compaction, so its replay target stays unset.
        "  const externalProvider = undefined, externalSelection = undefined;\n"
        "  const providerCompactionTargetConnectionKey = null;\n"
        "  const toExternalBackendProviderType = (providerType: any) => providerType;\n"
        # The provider-dependent flag is covered by external-preserve-thinking.test.ts.
        + "  const replayReasoning = !isExternalRequest;\n"
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
            "export function stripContinuationOverlap(",
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

/** Same as wireRoles, but skips fillStoppedAssistantReplay so the #9484 pair still shows. */
export function wireRolesBeforeFill(messages: any[], includeReasoningContent: boolean): string[] {
  return messages
    .flatMap((message: any) => message.role === "assistant"
      ? serializeAssistantReplayMessages(message, includeReasoningContent)
      : toOpenAIMessages(message, includeReasoningContent))
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
# Nothing yielded, so status is the only record; a failure looks the same under "error".
STOPPED_UNMARKED = (
    '{ role: "assistant", content: [], status: { type: "incomplete", reason: "cancelled" } }'
)
FAILED_UNMARKED = (
    '{ role: "assistant", content: [], status: { type: "incomplete", reason: "error" } }'
)
STOPPED = "Response stopped"
INTERRUPTED = "Response interrupted"


def _script(history: str, include_reasoning: str = "true") -> str:
    return textwrap.dedent(
        f"""
        // @ts-nocheck
        import {{ pruneOutboundHistory, pruneRefusalsOnly, toOpenAIMessages, wireRoles, wireRolesBeforeFill }}
          from "./harness.ts";
        const history = {history};
        const kept = pruneOutboundHistory(history, {include_reasoning});
        console.log(JSON.stringify({{
          kept: kept.map((m) => m.role),
          keptText: kept.flatMap((m) => (m.content ?? []).filter((p) => p.type === "text")
            .map((p) => p.text)),
          wire: wireRoles(kept, {include_reasoning}),
          wireBefore: wireRolesBeforeFill(pruneRefusalsOnly(history), {include_reasoning}),
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


def test_a_stop_before_any_output_keeps_its_prompt_on_the_wire():
    """#10428: the prompt must still transmit; #9484 wanted alternating roles, not deletion."""
    out = _run(_script(f"[{_user('first')}, {CANCELLED}, {_user('second')}]"))
    assert out["kept"] == ["user", "assistant", "user"]
    assert out["keptText"] == ["first", "second"]
    assert out["wire"] == ["user", "assistant", "user"]
    assert out["wireBefore"] == ["user", "user"]


def test_a_reply_that_produced_text_is_kept_with_its_prompt():
    """The prune reads the wire shape, so anything the model actually said protects the pair."""
    answered = '{ role: "assistant", content: [{ type: "text", text: "an answer" }] }'
    out = _run(_script(f"[{_user('first')}, {answered}, {_user('second')}]"))
    assert out["kept"] == ["user", "assistant", "user"]
    assert out["wire"] == ["user", "assistant", "user"]


def test_a_stop_during_reasoning_keeps_its_prompt_on_both_serialisations():
    """A turn stopped mid-reasoning keeps its prompt on both paths, which pass different reasoning flags."""
    thinking = (
        '{ role: "assistant", content: [{ type: "reasoning", text: "let me think" }],'
        ' status: { type: "incomplete" } }'
    )
    for include_reasoning in ("true", "false"):
        out = _run(_script(f"[{_user('first')}, {thinking}, {_user('second')}]", include_reasoning))
        assert out["kept"] == [
            "user",
            "assistant",
            "user",
        ], f"includeReasoningContent={include_reasoning}"
        assert out["keptText"] == ["first", "second"]
        assert out["wire"] == ["user", "assistant", "user"]


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


def test_back_to_back_stops_keep_every_interrupted_prompt():
    history = f"[{_user('first')}, {CANCELLED}, {_user('second')}, {CANCELLED}, {_user('third')}]"
    out = _run(_script(history))
    assert out["kept"] == ["user", "assistant", "user", "assistant", "user"]
    assert out["keptText"] == ["first", "second", "third"]
    assert out["wire"] == ["user", "assistant", "user", "assistant", "user"]
    assert out["wireBefore"] == ["user", "user", "user"]


def test_a_reply_that_finished_on_reasoning_alone_keeps_its_prompt():
    """A reasoning-only reply is a real answer, not a Stop, so its prompt stays on hosted providers too."""
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
    """A tool call the replay drops leaves an empty turn, so the prompt is pruned with it, not merged."""
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
        assert out["wireBefore"] == ["user", "user"], (
            "the refusal-only prune must still strand the pair here, or this case has stopped "
            "measuring the defect it was written for"
        )
        if marker == stopped:
            assert out["kept"] == ["user", "assistant", "user"], marker
            assert out["keptText"] == ["first", "second"]
            assert out["wire"] == ["user", "assistant", "user"]
        else:
            assert out["kept"] == ["user"], marker
            assert out["keptText"] == ["second"]
            assert out["wire"] == ["user"]


def test_a_resultless_call_that_replays_without_role_tool_keeps_its_prompt():
    """A resultless call that replays without role=tool still reaches the provider, so its prompt stays."""
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
    """A trailing abandoned turn must keep its prompt, or the token count of the live thread reads zero."""
    for include_reasoning in ("true", "false"):
        out = _run(_script(f"[{_user('first')}, {CANCELLED}]", include_reasoning))
        assert out["kept"] == [
            "user",
            "assistant",
        ], f"includeReasoningContent={include_reasoning}"
        assert out["keptText"] == ["first"]
        assert out["wire"] == ["user", "assistant"]


def test_a_thread_that_is_only_a_stop_keeps_its_system_prompt_and_prompt():
    history = (
        '[{ role: "system", content: [{ type: "text", text: "sys" }] }, '
        f"{_user('first')}, {CANCELLED}]"
    )
    out = _run(_script(history))
    assert out["kept"] == ["system", "user", "assistant"]
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
    """Runs the send path's real outbound build, so a regression to unpruned messages fails the test."""
    for is_external in ("false", "true"):
        out = _run(_send_script(f"[{_user('first')}, {CANCELLED}, {_user('second')}]", is_external))
        assert out["roles"] == ["user", "assistant", "user"], f"isExternalRequest={is_external}"
        assert out["contents"] == ["first", STOPPED, "second"]


def test_a_stop_with_no_persisted_marker_still_reads_as_a_stop():
    """The real shape: nothing is yielded before the first token, so only the status is left."""
    for is_external in ("false", "true"):
        out = _run(
            _send_script(f"[{_user('first')}, {STOPPED_UNMARKED}, {_user('second')}]", is_external)
        )
        assert out["roles"] == ["user", "assistant", "user"], f"isExternalRequest={is_external}"
        assert out["contents"] == ["first", STOPPED, "second"]


def test_a_generation_that_failed_is_not_replayed_as_a_stop():
    """Same empty shape under ``reason: "error"``: the prompt stays, but "stopped" misreports why."""
    for is_external in ("false", "true"):
        out = _run(
            _send_script(f"[{_user('first')}, {FAILED_UNMARKED}, {_user('second')}]", is_external)
        )
        assert out["roles"] == ["user", "assistant", "user"], f"isExternalRequest={is_external}"
        assert out["contents"] == ["first", INTERRUPTED, "second"]


def test_a_reloaded_stop_keeps_its_prompt_once_the_marker_was_persisted():
    """Reloaded, not in-session: ``status`` rebuilds as ``complete``, so the persisted marker is
    all that still says this was stopped. The durable path stamps it; the next test is the gap."""
    reloaded = (
        '{ role: "assistant", content: [], status: { type: "complete", reason: "unknown" },'
        ' metadata: { custom: { incomplete: { reason: "cancelled" } } } }'
    )
    for is_external in ("false", "true"):
        out = _run(_send_script(f"[{_user('first')}, {reloaded}, {_user('second')}]", is_external))
        assert out["roles"] == ["user", "assistant", "user"], f"isExternalRequest={is_external}"
        assert out["contents"] == ["first", STOPPED, "second"]


def test_a_reloaded_stop_with_no_persisted_marker_is_still_dropped():
    """The known limit, pinned as a decision: nothing persists a marker, so the reloaded row is
    indistinguishable from a silent reply. Closing it means persisting at Stop time, not here."""
    reloaded_unmarked = (
        '{ role: "assistant", content: [], status: { type: "complete", reason: "unknown" },'
        " metadata: { custom: {} } }"
    )
    for is_external in ("false", "true"):
        out = _run(
            _send_script(f"[{_user('first')}, {reloaded_unmarked}, {_user('second')}]", is_external)
        )
        assert out["roles"] == ["user"], f"isExternalRequest={is_external}"
        assert out["contents"] == ["second"]


def test_the_send_path_still_carries_an_answered_exchange():
    """The counterpart: pruning must not be the send path quietly dropping history."""
    answered = '{ role: "assistant", content: [{ type: "text", text: "an answer" }] }'
    out = _run(_send_script(f"[{_user('first')}, {answered}, {_user('second')}]", "false"))
    assert out["roles"] == ["user", "assistant", "user"]
    assert out["contents"] == ["first", "an answer", "second"]


def test_a_stop_that_produced_only_whitespace_keeps_its_prompt():
    """Whitespace-only output counts as no answer, matching the backend, so the prompt is kept."""
    for blank in ("   ", "\\n\\t"):
        whitespace = (
            '{ role: "assistant", content: [{ type: "text", text: "%s" }],'
            ' status: { type: "incomplete" },'
            ' metadata: { custom: { incomplete: { reason: "cancelled" } } } }' % blank
        )
        out = _run(_script(f"[{_user('first')}, {whitespace}, {_user('second')}]"))
        assert out["kept"] == ["user", "assistant", "user"], repr(blank)
        assert [text for text in out["keptText"] if text.strip()] == ["first", "second"]
        assert out["wire"] == ["user", "assistant", "user"]
        assert out["wireBefore"] == ["user", "assistant", "user"], (
            "the refusal-only prune kept the whitespace turn, which is the hop the backend "
            "trim then undid"
        )


def test_whitespace_only_replies_leave_no_pair_for_the_backend_to_split():
    """The same shape without a Stop marker: the backend trims it either way."""
    whitespace = '{ role: "assistant", content: [{ type: "text", text: "  " }] }'
    out = _run(_script(f"[{_user('first')}, {whitespace}, {_user('second')}]"))
    assert out["kept"] == ["user"]


def test_a_refusal_is_never_filled_back_onto_the_wire():
    """Refilling a refusal would put a stop label on the wire for a message the serialiser suppresses."""
    refusals = (
        " metadata: { custom: { anthropicRefusal: true } } }",
        ' status: { type: "incomplete", reason: "cancelled" },'
        " metadata: { custom: { anthropicRefusal: true,"
        ' incomplete: { reason: "cancelled" } } } }',
    )
    for tail in refusals:
        refused = '{ role: "assistant", content: [],' + tail
        out = _run(
            textwrap.dedent(
                f"""
                // @ts-nocheck
                import {{ toOpenAIMessages }} from "./harness.ts";
                console.log(JSON.stringify({{
                  serialized: toOpenAIMessages({refused}, true),
                }}));
                """
            )
        )
        assert out["serialized"] == [], tail
