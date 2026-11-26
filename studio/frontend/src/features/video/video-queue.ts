// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A busy engine queues a submit server-side (one FIFO, round-robin across accounts); the page
// shows the wait, then hands over to the generate-progress poll once the job renders.

import type { VideoGenerateProgress, VideoQueueJob } from "./api";

export const VIDEO_QUEUE_POLL_MS = 2000;

export interface QueuedVideo {
  videoId: string;
  position: number | null;
  etaSeconds: number | null;
}

/** "Queued #2, about 3 min"; the ETA part is dropped when unknown. */
export function queuedVideoLabel(position: number | null, etaSeconds: number | null): string {
  const head = position != null && position > 0 ? `Queued #${position}` : "Queued";
  if (etaSeconds == null || !Number.isFinite(etaSeconds) || etaSeconds < 0) return head;
  if (etaSeconds < 60) return `${head}, less than a minute`;
  return `${head}, about ${Math.ceil(etaSeconds / 60)} min`;
}

export type QueuedVideoNext =
  | { kind: "queued"; job: QueuedVideo }
  | { kind: "started" }
  | { kind: "failed"; error: string | null }
  | { kind: "cancelled" };

/** What a queue poll means for the job the page waits on. Absent = it renders now. "starting" is
 *  still waiting: that attempt can go back to "queued" or end "failed" without ever rendering. */
export function queuedVideoNext(jobs: VideoQueueJob[], videoId: string): QueuedVideoNext {
  const job = jobs.find((j) => j.video_id === videoId);
  if (!job) return { kind: "started" };
  if (job.status === "failed") return { kind: "failed", error: job.error };
  if (job.status === "cancelled") return { kind: "cancelled" };
  return {
    kind: "queued",
    job: { videoId, position: job.queue_position, etaSeconds: job.eta_seconds },
  };
}

/** After a reload, the account's oldest waiting job is the one the page resumes. */
export function oldestQueuedVideo(jobs: VideoQueueJob[]): QueuedVideo | null {
  let oldest: VideoQueueJob | null = null;
  for (const job of jobs) {
    // "starting" too: it has no progress record yet, and its start may still take a model load.
    if (job.status !== "queued" && job.status !== "starting") continue;
    if (oldest === null || job.created_at < oldest.created_at) oldest = job;
  }
  return oldest
    ? { videoId: oldest.video_id, position: oldest.queue_position, etaSeconds: oldest.eta_seconds }
    : null;
}

/** A terminal phase tagged with another job's id is a stale record, not our outcome. */
export function progressIsForVideo(p: VideoGenerateProgress, videoId: string | null): boolean {
  return !videoId || !p.video_id || p.video_id === videoId;
}
