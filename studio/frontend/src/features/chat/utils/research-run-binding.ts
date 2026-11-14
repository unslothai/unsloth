// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A stopped run is re-pointed at the next reply, so stale in-memory copies must check
 *  the run's own assistantMessageId. A run that names none belongs to whoever asks. */
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
