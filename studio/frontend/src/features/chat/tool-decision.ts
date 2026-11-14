// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolveToolConfirmation, ToolApprovalGoneError } from "./api/chat-api";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

export type ToolDecisionResult = "ok" | "retry" | "gone" | "missing" | "needs_card";

// The pending entry clears only once the backend took the decision, so a failed post stays answerable.
export async function decideToolCall(
  toolCallId: string,
  decision: "allow" | "deny",
  alwaysToolName?: string,
): Promise<ToolDecisionResult> {
  const store = useChatRuntimeStore.getState();
  if (!Object.prototype.hasOwnProperty.call(store.toolConfirmations, toolCallId)) return "missing";
  const { sessionId, approvalId, autoAllowKey, imageDisclosure } = store.toolConfirmations[toolCallId];
  // Sharing an image is allowed only on the chat card, which names where it goes and that it may be kept.
  if (decision === "allow" && imageDisclosure) return "needs_card";
  try {
    if (!(await resolveToolConfirmation(sessionId, approvalId, decision))) return "retry";
  } catch (err) {
    return err instanceof ToolApprovalGoneError ? "gone" : "retry";
  }
  const after =
  useChatRuntimeStore.getState();
  // Granted only once the backend has taken the decision, never on the click.
  if (decision === "allow" && alwaysToolName && autoAllowKey) after.allowToolAlways(autoAllowKey, alwaysToolName);
  after.clearToolConfirmation(toolCallId);
  return "ok";
}
