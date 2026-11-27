// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Re-pointed runs: check the run's assistantMessageId; a run naming none belongs to anyone. */
export function researchReplyOwnsRun(
  boundAssistantMessageId: unknown,
  messageId: unknown,
): boolean {
  if (typeof boundAssistantMessageId !== "string" || !boundAssistantMessageId) {
    return true;
  }
  if (typeof messageId !== "string" || !messageId) {
    return true;
  }
  return boundAssistantMessageId === messageId;
}
