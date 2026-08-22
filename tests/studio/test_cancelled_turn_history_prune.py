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

# Fixtures the sliced code reads through. Only the tool-call helpers are stand-ins: the turns
# that decide this behaviour carry text, reasoning or nothing at all, and a tool call has to be
# real enough to prove a turn carrying one is NOT abandoned.
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

function canReplayToolCallWithoutRoleTool(_part: any): boolean {
  return false;
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


def _harness_source() -> str:
    return (
        HARNESS
        + _adapter_slice("function collectTextParts(", "function normalizeOpenAIReasoningItem(")
        + _adapter_slice(
            "// Refusal flag stamped on assistant metadata",
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
            "/** Why a turn ended before the model was done. */",
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
    )


def _run(script: str) -> dict:
    require_node(SOURCES)
    return run_harness(TEMP, _harness_source(), script)


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
        out = _run(
            _script(f"[{_user('first')}, {thinking}, {_user('second')}]", include_reasoning)
        )
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
        ' metadata: { custom: { anthropicRefusal: true } } }'
    )
    out = _run(_script(f"[{_user('first')}, {refused}, {_user('second')}]"))
    assert out["kept"] == ["user"]
    assert out["keptText"] == ["second"]


def test_back_to_back_stops_collapse_to_the_live_prompt():
    """Stop twice and both abandoned pairs go, rather than one prune uncovering the next."""
    history = (
        f"[{_user('first')}, {CANCELLED}, {_user('second')}, {CANCELLED}, {_user('third')}]"
    )
    out = _run(_script(history))
    assert out["kept"] == ["user"]
    assert out["keptText"] == ["third"]
    assert out["wireBefore"] == ["user", "user", "user"]
