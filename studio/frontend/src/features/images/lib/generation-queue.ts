// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure helpers for the Images page's view of the server-side generation queue. */

import type { ImageQueueJob } from "../api.ts";

const FINISHED = new Set(["done", "failed", "cancelled"]);

export function queuedJobs(jobs: readonly ImageQueueJob[]): ImageQueueJob[] {
  return jobs.filter((job) => job.status === "queued");
}

export function runningJob(jobs: readonly ImageQueueJob[]): ImageQueueJob | null {
  return jobs.find((job) => job.status === "running") ?? null;
}

/** Whether the queue still has work, so the page keeps polling and Generate keeps lining up behind it. */
export function queueBusy(jobs: readonly ImageQueueJob[]): boolean {
  return jobs.some((job) => job.status === "queued" || job.status === "running");
}

/** Jobs that finished between two polls. One already finished on the first poll (a reload) is not news. */
export function newlyFinished(
  prev: readonly ImageQueueJob[],
  next: readonly ImageQueueJob[],
): ImageQueueJob[] {
  const before = new Map(prev.map((job) => [job.id, job.status]));
  return next.filter((job) => {
    const was = before.get(job.id);
    return FINISHED.has(job.status) && was !== undefined && !FINISHED.has(was);
  });
}

export function promptSnippet(prompt: string, max = 60): string {
  const flat = prompt.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat;
}

export function queueJobSummary(job: ImageQueueJob): string {
  return [
    `${job.width}×${job.height}`,
    job.steps === null ? "model steps" : `${job.steps} steps`,
    job.seed === null ? "random seed" : `seed ${job.seed}`,
    ...(job.batch_size > 1 ? [`×${job.batch_size}`] : []),
  ].join(" · ");
}
