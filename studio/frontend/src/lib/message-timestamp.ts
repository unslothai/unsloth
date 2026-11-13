// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function messageTimestamp(message: {
  createdAt?: Date;
  metadata?: { custom?: Record<string, unknown> };
}): number | undefined {
  if (message.metadata?.custom?.createdAtEstimated === true) return undefined;
  const time = message.createdAt?.getTime();
  return time !== undefined && Number.isFinite(time) ? time : undefined;
}

/** When a reply finished, else when it was created; none while it is running or awaiting tool approval. */
export function replyFinishedTimestamp(message: {
  createdAt?: Date;
  status?: { type?: string };
  metadata?: { custom?: Record<string, unknown> };
}): number | undefined {
  const status = message.status?.type;
  if (status === "running" || status === "requires-action") return undefined;
  const custom = message.metadata?.custom;
  // Deep Research replies record only their run; completedAt stays null until it ends.
  const research = custom?.researchRun as { completedAt?: unknown } | undefined;
  if (research && typeof research === "object") {
    return typeof research.completedAt === "number" && Number.isFinite(research.completedAt)
      ? research.completedAt
      : undefined;
  }
  const details = custom?.responseDetails as { finishedAt?: unknown } | undefined;
  const finishedAt =
    typeof details?.finishedAt === "number" && Number.isFinite(details.finishedAt)
      ? details.finishedAt
      : undefined;
  const timing = custom?.timing as
    | { streamStartTime?: unknown; totalStreamTime?: unknown }
    | undefined;
  const timingEnd =
    typeof timing?.streamStartTime === "number" && typeof timing.totalStreamTime === "number"
      ? timing.streamStartTime + timing.totalStreamTime
      : undefined;
  const streamEnd = timingEnd !== undefined && Number.isFinite(timingEnd) ? timingEnd : undefined;
  // A stopped or failed reply's timing is its last round; responseDetails may be an earlier tool round's.
  const end = status === "incomplete" ? (streamEnd ?? finishedAt) : (finishedAt ?? streamEnd);
  return end ?? messageTimestamp(message);
}
