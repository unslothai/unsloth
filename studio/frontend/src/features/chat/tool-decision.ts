// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolveToolConfirmation, ToolApprovalGoneError } from "./api/chat-api";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

/**
 * "retry": the post failed on the way out and is worth pressing again. "gone": the backend
 * is no longer holding the approval, so pressing again can only fail the same way.
 */
export type ToolDecisionResult = "ok" | "retry" | "gone" | "missing";

/**
 * Allow or deny a tool call parked for approval: what the chat card's buttons and the
 * floating orb's popup both do. The pending entry clears only once the backend has taken
 * the decision, so a failed post leaves the call answerable.
 */
export async function decideToolCall(
  toolCallId: string,
  decision: "allow" | "deny",
): Promise<ToolDecisionResult> {
  const store = useChatRuntimeStore.getState();
  if (!Object.prototype.hasOwnProperty.call(store.toolConfirmations, toolCallId)) return "missing";
  const { sessionId, approvalId } = store.toolConfirmations[toolCallId];
  try {
    if (!(await resolveToolConfirmation(sessionId, approvalId, decision))) return "retry";
  } catch (err) {
    return err instanceof ToolApprovalGoneError ? "gone" : "retry";
  }
  useChatRuntimeStore.getState().clearToolConfirmation(toolCallId);
  return "ok";
}
