// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Importers may need a synthetic ordering key when no send time is known. */
export function messageTimestamp(message: {
  createdAt?: Date;
  metadata?: { custom?: Record<string, unknown> };
}): number | undefined {
  if (message.metadata?.custom?.createdAtEstimated === true) return undefined;
  const time = message.createdAt?.getTime();
  return time !== undefined && Number.isFinite(time) ? time : undefined;
}

/** When a reply finished, else when it was created; none while it is still running. */
export function replyFinishedTimestamp(message: {
  createdAt?: Date;
  status?: { type?: string };
  metadata?: { custom?: Record<string, unknown> };
}): number | undefined {
  if (message.status?.type === "running") return undefined;
  const custom = message.metadata?.custom;
  // Deep Research replies record only their run; completedAt stays null until it ends.
  const research = custom?.researchRun as { completedAt?: unknown } | undefined;
  if (research && typeof research === "object") {
    return typeof research.completedAt === "number" && Number.isFinite(research.completedAt)
      ? research.completedAt
      : undefined;
  }
  const details = custom?.responseDetails as { finishedAt?: unknown } | undefined;
  if (typeof details?.finishedAt === "number" && Number.isFinite(details.finishedAt)) {
    return details.finishedAt;
  }
  // Stopped and failed replies persist only their stream timing, which ends where they stopped.
  const timing = custom?.timing as
    | { streamStartTime?: unknown; totalStreamTime?: unknown }
    | undefined;
  if (
    typeof timing?.streamStartTime === "number" &&
    typeof timing.totalStreamTime === "number"
  ) {
    const end = timing.streamStartTime + timing.totalStreamTime;
    if (Number.isFinite(end)) return end;
  }
  return messageTimestamp(message);
}
