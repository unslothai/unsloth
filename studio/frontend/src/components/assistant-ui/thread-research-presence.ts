// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a thread already holds a research answer, answered once per message array rather than
 * once per keystroke.
 *
 * The composer asks this to decide whether deep research is still offerable. It asks it inside a
 * `useAuiState` selector, and `useAuiState` is `useSyncExternalStore` with the selector AS the
 * getSnapshot, so the selector runs on every store notification -- and the composer is controlled
 * off store state, so every character typed is a store notification. The scan therefore walked
 * every message in the thread, reading metadata through the state proxy, once per keystroke, and
 * React may run getSnapshot a second time before commit. The answer never changed while typing:
 * the boolean is the same, so nothing re-rendered, and the walk was pure cost.
 *
 * assistant-ui rebuilds its message array whenever the repository changes, so the array's
 * identity is exactly the revision the answer depends on. Keying on it makes typing O(1) and
 * keeps the answer byte-identical to the scan it replaces, including for a thread whose research
 * reply sits in metadata that arrives later: that arrival replaces the array.
 *
 * Two cases where the cache simply misses, both perf-only and neither worse than the scan it
 * replaces, which ran on every notification regardless. While a reply streams, the array is
 * rebuilt per token, so every arrival is a miss. And `MessageRepository.getMessages(headId)` for
 * an explicit non-head branch builds a fresh array on every call, so anything reading a branch
 * head rather than `thread.messages` would miss permanently.
 *
 * Weak, so a thread that goes away takes its entry with it. Sibling of research-reply-owners.ts,
 * which answers the per-message ownership question the same way; kept separate because they key
 * different answers off the same revision and one cache holding both would hand a caller the
 * other's.
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

/**
 * @param messages the thread's message array, used as the revision key. Called with the same
 *   array repeatedly while nothing changes, which is the case this exists for.
 */
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
