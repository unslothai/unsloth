// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Reading the model's Deep Research handoff off the tool events every loop publishes.
 *  `deep_research` is an ordinary tool to the three loops that run it, so there is no bespoke
 *  frame to key on: the question rides on `tool_start` and the result on `tool_end`. */

/** The opening of the backend's `DEEP_RESEARCH_STARTED`, which is what says the call ran. */
export const DEEP_RESEARCH_STARTED_MARKER = "Deep Research has started";

/** `CreateResearchRun.question`'s max_length; a longer question would 422 the whole handoff. */
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

/** Fold one `deep_research` tool event into the handoff. @returns whether the adapter should
 *  swallow the event instead of drawing a tool card. An ungated call is hidden on both halves:
 *  the research card is the reply, and a tool pill would just say the same thing. */
export function readDeepResearchToolEvent(
  handoff: DeepResearchHandoff,
  event: DeepResearchToolEvent,
): boolean {
  const callId =
    typeof event.tool_call_id === "string" ? event.tool_call_id : "";
  if (event.type === "tool_start") {
    // The local loop paints a provisional tool_start with empty arguments before the real one, so
    // only a start carrying a question is remembered, and only the first: a second call in the
    // same turn is the model repeating itself.
    const args = event.arguments;
    const question =
      args && typeof args === "object"
        ? String((args as { question?: unknown }).question ?? "").trim()
        : "";
    if (handoff.question === null && question) {
      handoff.pendingCallId = callId;
      // Clamped, not refused: the endpoint rejects a longer one and the whole handoff would fail,
      // where a small model padding the question still asked something researchable.
      handoff.pendingQuestion = Array.from(question)
        .slice(0, DEEP_RESEARCH_QUESTION_MAX_CHARS)
        .join("");
    }
    // A gated call has to keep its Allow / Deny card: the loop blocks on a verdict, so hiding the
    // card asks the user nothing and the turn hangs there.
    if (event.awaiting_confirmation === true) {
      return false;
    }
    handoff.hiddenCallIds.add(callId);
    return true;
  }
  if (event.type !== "tool_end") {
    return false;
  }
  // The result is what says the tool ran. tool_end on its own does not: a denied, skipped,
  // truncated or budget-exhausted call is closed by the same event, and handing off on one
  // researches a question the loop refused to pass on, spending the chat's one run.
  const result = typeof event.result === "string" ? event.result : "";
  if (
    result.startsWith(DEEP_RESEARCH_STARTED_MARKER) &&
    handoff.question === null
  ) {
    // The run starts on the question the model passed; if that could not be read, on the user's own
    // message, which is what research did before the model had a say.
    handoff.question =
      callId === handoff.pendingCallId ? handoff.pendingQuestion : "";
  }
  // Close whatever this pair opened: a hidden start gets a hidden end, and a card the user was
  // asked to approve gets closed by the event that ends it.
  return handoff.hiddenCallIds.has(callId);
}
