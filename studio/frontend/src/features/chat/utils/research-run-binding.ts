// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Whether a reply still owns the research run its metadata names. A thread holds one run for its
 *  lifetime, so a stopped run is re-pointed at the next question and keeps its id. The server
 *  moves the binding to the new reply, but a message already in memory keeps the copy it was
 *  loaded with, and the run store is keyed by run id, so the stopped reply would render whatever
 *  that run is doing now, in two places at once. The run's own `assistantMessageId` is the answer,
 *  since that is exactly what the server moves. A run that never named one belongs to whoever asks. */
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
