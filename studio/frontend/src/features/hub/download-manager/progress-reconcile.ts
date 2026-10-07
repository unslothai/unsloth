// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DOWNLOAD_KIND } from "./constants";
import { MAX_PROGRESS_FRACTION } from "./download-manager-config";
import type {
  FloorHold,
  ManagedDownload,
  ProgressLike,
} from "./download-manager-types";

export function hasObservedExpectedBytes(job: ManagedDownload): boolean {
  // Finalized bytes only: an `.incomplete` blob at full size is not yet verified.
  return (
    job.expectedBytes > 0 &&
    job.completedBytes >= job.expectedBytes &&
    job.completeOnDisk
  );
}

// Unvalidated wire value; NaN/Infinity would render "NaN%". 0 means "not measured".
function finiteReading(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

// Xet commits bytes in batches, so a small transfer can sit at 0 B; show activity instead.
export function isIndeterminateProgress(
  progress: {
    downloadedBytes: number;
    fraction: number;
  },
  cancelling = false,
): boolean {
  return !cancelling && progress.downloadedBytes <= 0 && progress.fraction <= 0;
}

export function resolveProgressUpdate(
  job: ManagedDownload,
  progressResp: ProgressLike,
  opts: { resetMonotonic?: boolean; skipFloor?: boolean } = {},
): {
  expected: number;
  downloadedBytes: number;
  measuredTransfer: boolean;
  completedBytes: number;
  completeOnDisk: boolean;
  fraction: number;
  madeProgress: boolean;
} {
  const reported = finiteReading(progressResp.expected_bytes);
  const isGgufVariantJob =
    job.kind === DOWNLOAD_KIND.MODEL && job.variant !== null;
  const backendOwnsGgufProgress = isGgufVariantJob && reported > 0;
  // Snapshots stay monotonic to absorb jitter; a new generation or retry drops the high-water mark.
  const resetMonotonic = opts.resetMonotonic === true;
  const trustBackend = backendOwnsGgufProgress || resetMonotonic;
  const expected = trustBackend
    ? reported > 0
      ? reported
      : job.expectedBytes
    : Math.max(reported > 0 ? reported : job.expectedBytes, job.expectedBytes);
  const previousDownloadedBytes = job.downloadedBytes;
  const reportedCompleted = finiteReading(progressResp.completed_bytes);
  // Hold the last reading through an all-zero poll (an unmeasurable, negatively cached answer).
  // Not a high-water mark: bytes may legitimately drop within a generation (Xet->HTTP fallback).
  const reportedDownloaded = finiteReading(progressResp.downloaded_bytes);
  // Whether this poll measured the counter or held it; a held figure belongs to the previous total.
  const measuredTransfer = resetMonotonic || reportedDownloaded > 0;
  const downloadedBytes = measuredTransfer
    ? Math.max(0, reportedDownloaded)
    : Math.max(previousDownloadedBytes, 0);
  const measuredCompleted = resetMonotonic || reportedCompleted > 0;
  const completedBytes = resetMonotonic
    ? Math.max(0, reportedCompleted)
    : measuredCompleted
      ? reportedCompleted
      : Math.max(job.completedBytes, 0);
  // Honour completion only when this poll measured completed bytes, so both come from one reading.
  const completeOnDisk = progressResp.complete_on_disk === true && measuredCompleted;
  const madeProgress =
    resetMonotonic ||
    // An unmeasured scan returns zeroes; do not let the idle grace finalize it as gone.
    progressResp.cache_measured === false ||
    downloadedBytes > previousDownloadedBytes ||
    expected !== job.expectedBytes;
  const reportedFraction = finiteReading(progressResp.progress);
  const rawFraction =
    reportedFraction > 0
      ? reportedFraction
      : expected > 0
        ? downloadedBytes / expected
        : 0;
  const cappedFraction = Math.min(rawFraction, MAX_PROGRESS_FRACTION);
  // Keep the variant bar monotonic (shared blobs/ dir can dip a reading), except across a
  // generation/attempt change; `skipFloor` covers polls still reading the killed partial.
  const fraction =
    isGgufVariantJob && !resetMonotonic && opts.skipFloor !== true
      ? Math.max(cappedFraction, job.fraction)
      : cappedFraction;
  return {
    expected,
    downloadedBytes,
    measuredTransfer,
    completedBytes,
    completeOnDisk,
    fraction,
    madeProgress,
  };
}

function runCounterChanged(
  previous: number | undefined,
  current: number | undefined,
): boolean {
  return (
    Number.isSafeInteger(current) &&
    Number.isSafeInteger(previous) &&
    current !== previous
  );
}

export function serverRunChange(
  known: { generation?: number; attempt?: number },
  status: { generation?: number; attempt?: number },
): "generation" | "attempt" | null {
  if (runCounterChanged(known.generation, status.generation)) {
    return "generation";
  }
  return runCounterChanged(known.attempt, status.attempt) ? "attempt" : null;
}

// Bytes left rise only once the retry worker purges the killed partial.
export function floorHoldEnded(
  hold: FloorHold,
  expectedBytes: number,
  downloadedBytes: number,
  now: number,
): boolean {
  return (
    expectedBytes - downloadedBytes > hold.remainingBytes || now >= hold.until
  );
}
