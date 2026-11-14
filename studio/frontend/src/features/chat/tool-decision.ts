// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolveToolConfirmation, ToolApprovalGoneError } from "./api/chat-api";
import { useChatRuntimeStore } from "./stores/chat-runtime-store";

export type ToolDecisionResult = "ok" | "retry" | "gone" | "missing";

export async function decideToolCall(
  toolCallId: string,
  decision: "allow" | "deny",
  alwaysToolName?: string,
): Promise<ToolDecisionResult> {
  const store = useChatRuntimeStore.getState();
  if (!Object.prototype.hasOwnProperty.call(store.toolConfirmations, toolCallId)) return "missing";
  const { sessionId, approvalId, autoAllowKey } = store.toolConfirmations[toolCallId];
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
