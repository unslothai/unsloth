// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Keyed on the message array identity (rebuilt on every repository change) so typing, which runs
 * every useAuiState selector, does not rescan the thread. Separate cache from
 * research-reply-owners.ts so the two answers never mix.
 */

export type ResearchPresenceMessage = { metadata?: unknown };

const presenceByMessages = new WeakMap<object, boolean>();
const presenceByLiveRun = new WeakMap<object, Map<string, boolean>>();

/** Only a finished run counts; running or stopped runs keep research offered. No status counts. */
export function messageHasResearchRunId(
  message: ResearchPresenceMessage,
  liveRunId?: string,
): boolean {
  const custom = (
    message.metadata as
      | {
          custom?: {
            researchRunId?: unknown;
            researchStatus?: unknown;
            researchRun?: { status?: unknown };
          };
        }
      | undefined
  )?.custom;
  if (typeof custom?.researchRunId !== "string") {
    return false;
  }
  if (custom.researchRunId === liveRunId) {
    return false;
  }
  const status = custom.researchRun?.status ?? custom.researchStatus;
  if (typeof status !== "string") {
    return true;
  }
  return status === "completed" || status === "failed";
}

export function threadHasResearchMessage(
  messages: readonly ResearchPresenceMessage[],
  liveRunId?: string,
): boolean {
  if (liveRunId) {
    const knownByRun = presenceByLiveRun.get(messages);
    const known = knownByRun?.get(liveRunId);
    if (known !== undefined) {
      return known;
    }
    const answer = messages.some((message) =>
      messageHasResearchRunId(message, liveRunId),
    );
    const next = knownByRun ?? new Map<string, boolean>();
    next.set(liveRunId, answer);
    if (!knownByRun) {
      presenceByLiveRun.set(messages, next);
    }
    return answer;
  }
  const known = presenceByMessages.get(messages);
  if (known !== undefined) {
    return known;
  }
  const answer = messages.some((message) => messageHasResearchRunId(message));
  presenceByMessages.set(messages, answer);
  return answer;
}
