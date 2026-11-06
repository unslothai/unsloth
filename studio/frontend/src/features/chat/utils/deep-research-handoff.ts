// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const DEEP_RESEARCH_STARTED_MARKER = "Deep Research has started";

/** `CreateResearchRun.question`'s max_length; a longer question would 422. */
export const DEEP_RESEARCH_QUESTION_MAX_CHARS = 2000;

export type DeepResearchToolEvent = {
  type?: unknown;
  tool_call_id?: unknown;
  arguments?: unknown;
  result?: unknown;
  awaiting_confirmation?: unknown;
};

export type DeepResearchHandoff = {
  /** The question to research; "" means the user's own message, null means no handoff yet. */
  question: string | null;
  pendingCallId: string;
  pendingQuestion: string;
  hiddenCallIds: Set<string>;
};

export function newDeepResearchHandoff(): DeepResearchHandoff {
  return {
    question: null,
    pendingCallId: "",
    pendingQuestion: "",
    hiddenCallIds: new Set(),
  };
}

export function readDeepResearchToolEvent(
  handoff: DeepResearchHandoff,
  event: DeepResearchToolEvent,
): boolean {
  const callId =
    typeof event.tool_call_id === "string" ? event.tool_call_id : "";
  if (event.type === "tool_start") {
    // The local loop emits a provisional empty tool_start first; remember only the first real question.
    const args = event.arguments;
    const question =
      args && typeof args === "object"
        ? String((args as { question?: unknown }).question ?? "").trim()
        : "";
    if (handoff.question === null && question) {
      handoff.pendingCallId = callId;
      handoff.pendingQuestion = Array.from(question)
        .slice(0, DEEP_RESEARCH_QUESTION_MAX_CHARS)
        .join("");
    }
    // A gated call keeps its Allow/Deny card, or the loop blocks on a verdict forever.
    if (event.awaiting_confirmation === true) {
      return false;
    }
    handoff.hiddenCallIds.add(callId);
    return true;
  }
  if (event.type !== "tool_end") {
    return false;
  }
  // Only the started marker proves the tool ran; denied or skipped calls also send tool_end.
  const result = typeof event.result === "string" ? event.result : "";
  if (
    result.startsWith(DEEP_RESEARCH_STARTED_MARKER) &&
    handoff.question === null
  ) {
    handoff.question =
      callId === handoff.pendingCallId ? handoff.pendingQuestion : "";
  }
  return handoff.hiddenCallIds.has(callId);
}
