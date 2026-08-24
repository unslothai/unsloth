// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a thread already holds a research answer, asked once per message array rather than once
 * per keystroke.
 *
 * The composer asks this to decide whether deep research is still offerable, inside a
 * `useAuiState` selector -- and `useAuiState` is `useSyncExternalStore` with the selector AS
 * getSnapshot, so every store write runs every selector. The composer is controlled off store
 * state, so every character typed is a store write, and this scan walked every message through
 * the state proxy per keystroke (React may even run getSnapshot twice before commit) for an
 * answer that never changed while typing.
 *
 * assistant-ui rebuilds the message array on every repository change, so the array identity is
 * exactly the revision the answer depends on. Keying on it makes typing O(1) and keeps the answer
 * identical to the scan it replaces, including when a research reply's metadata arrives later:
 * that arrival replaces the array.
 *
 * Two cache misses, both perf-only and neither worse than the old unconditional scan: a streaming
 * reply rebuilds the array per token, and `MessageRepository.getMessages(headId)` for an explicit
 * non-head branch builds a fresh array per call, so a caller reading a branch head instead of
 * `thread.messages` would miss permanently.
 *
 * Weak, so a dead thread takes its entry with it. Sibling of research-reply-owners.ts, which
 * answers the per-message ownership question the same way; separate because one cache holding
 * both answers off the same revision would hand a caller the other's.
 */

export type ResearchPresenceMessage = { metadata?: unknown };

const presenceByMessages = new WeakMap<object, boolean>();

/** Whether any message carries `metadata.custom.researchRunId` as a string. */
export function messageHasResearchRunId(
  message: ResearchPresenceMessage,
): boolean {
  const custom = (
    message.metadata as { custom?: { researchRunId?: unknown } } | undefined
  )?.custom;
  return typeof custom?.researchRunId === "string";
}

/** @param messages the thread's message array, used as the revision key. */
export function threadHasResearchMessage(
  messages: readonly ResearchPresenceMessage[],
): boolean {
  const known = presenceByMessages.get(messages);
  if (known !== undefined) {
    return known;
  }
  const answer = messages.some(messageHasResearchRunId);
  presenceByMessages.set(messages, answer);
  return answer;
}
