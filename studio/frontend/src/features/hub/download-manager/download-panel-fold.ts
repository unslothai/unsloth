// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ManagedDownload } from "./download-manager-types.ts";
import { presentedProgress } from "./download-presentation.ts";

/** How long a steadily downloading panel stays open before it folds into its icon. */
export const AUTO_FOLD_MS = 10_000;

/** A row the user should see: anything but a plain running transfer (done, failed, cancelling, warning). */
export function needsAttention(job: ManagedDownload | undefined): boolean {
  return !job || job.state !== "running" || Boolean(job.error);
}

export type FoldSnapshot = {
  jobKeys: readonly string[];
  queued: number;
  /** Job keys whose row needs attention. */
  attention: readonly string[];
};

export function foldSnapshot(
  jobKeys: readonly string[],
  jobs: Readonly<Record<string, ManagedDownload>>,
  queued: number,
): FoldSnapshot {
  return {
    jobKeys,
    queued,
    attention: jobKeys.filter((key) => needsAttention(jobs[key])),
  };
}

/** Open the panel when a download is added or a row newly needs attention, so new downloads,
 *  finishes and errors are never hidden behind the icon. A queued entry turning into a job is
 *  the same download moving on, not a new one. */
export function shouldPopOpen(prev: FoldSnapshot, next: FoldSnapshot): boolean {
  if (next.queued > prev.queued) return true;
  const seen = new Set(prev.jobKeys);
  const added = next.jobKeys.filter((key) => !seen.has(key)).length;
  if (added > prev.queued - next.queued) return true;
  const flagged = new Set(prev.attention);
  return next.attention.some((key) => !flagged.has(key));
}

/** Combined progress of the running downloads, 0..1, or null when nothing is measurable. */
export function aggregateFraction(
  jobs: Iterable<ManagedDownload>,
): number | null {
  let expected = 0;
  let downloaded = 0;
  for (const job of jobs) {
    if (job.state !== "running") continue;
    const progress = presentedProgress(job);
    if (!(progress.expectedBytes > 0)) return null;
    expected += progress.expectedBytes;
    downloaded += Math.min(progress.downloadedBytes, progress.expectedBytes);
  }
  if (expected <= 0) return null;
  return Math.min(1, Math.max(0, downloaded / expected));
}
