// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One backend progress reading reconciled against the job's current figures.
// Split out of poll-loop so it can be exercised on its own: poll-loop reaches
// the store, the toaster and the API client, and this is the part that decides
// what a card shows and when a download counts as finished.

import { DOWNLOAD_KIND } from "./constants";
import { MAX_PROGRESS_FRACTION } from "./download-manager-config";
import type { ManagedDownload, ProgressLike } from "./download-manager-types";

export function hasObservedExpectedBytes(job: ManagedDownload): boolean {
  // Finalized bytes only: an `.incomplete` blob hitting expected size isn't
  // finished until the backend verifies it's usable on disk.
  return (
    job.expectedBytes > 0 &&
    job.completedBytes >= job.expectedBytes &&
    job.completeOnDisk
  );
}

export function resolveProgressUpdate(
  job: ManagedDownload,
  progressResp: ProgressLike,
  opts: { resetMonotonic?: boolean } = {},
): {
  expected: number;
  downloadedBytes: number;
  completedBytes: number;
  completeOnDisk: boolean;
  fraction: number;
  madeProgress: boolean;
} {
  const reported = progressResp.expected_bytes;
  const isGgufVariantJob =
    job.kind === DOWNLOAD_KIND.MODEL && job.variant !== null;
  const backendOwnsGgufProgress = isGgufVariantJob && reported > 0;
  // GGUF totals are backend-owned (non-monotonic); snapshots stay monotonic to
  // absorb jitter, but a generation bump (XET redownload, restart, re-adoption)
  // must drop the stale high-water mark and snap to the new run's bytes.
  const resetMonotonic = opts.resetMonotonic === true;
  const trustBackend = backendOwnsGgufProgress || resetMonotonic;
  const expected = trustBackend
    ? reported > 0
      ? reported
      : job.expectedBytes
    : Math.max(reported > 0 ? reported : job.expectedBytes, job.expectedBytes);
  const previousDownloadedBytes = job.downloadedBytes;
  const reportedCompleted = progressResp.completed_bytes ?? 0;
  // Hold the last reading through a poll that found nothing, rather than
  // publishing the "I could not measure this" answer as a measurement. The
  // backend recomputes both counters from the shared per-repo blobs/ dir every
  // poll, and when it cannot resolve a variant's expected files it returns an
  // all-zero reading against the caller's catalog-hinted total; the failure
  // behind that is negatively cached, so every poll for the whole TTL says the
  // same thing and a finished card reads "0 B of 33 GB" until the app restarts.
  //
  // Deliberately NOT a high-water mark, which is what the fraction below uses.
  // Bytes have legitimate reasons to fall inside one generation and the
  // fraction does not: an XET run that falls back to HTTP re-claims with the
  // same generation and a freshly computed completed_baseline_bytes, and
  // resuming in XET mode purges the partial outright. A floor would pin the
  // card near full for the whole retry and starve the rolling-window rate of
  // the movement it needs to publish a speed or an ETA. Only a zero is ignored,
  // so every real change -- up or down -- still lands on the next poll.
  const downloadedBytes = resetMonotonic
    ? Math.max(0, progressResp.downloaded_bytes)
    : progressResp.downloaded_bytes > 0
      ? progressResp.downloaded_bytes
      : Math.max(previousDownloadedBytes, 0);
  // Same rule, and it cannot manufacture a completion: hasObservedExpectedBytes
  // also needs completeOnDisk, which is never held over, and the backend only
  // ever sets it on a reading whose own completed_bytes already cleared the bar.
  const completedBytes = resetMonotonic
    ? Math.max(0, reportedCompleted)
    : reportedCompleted > 0
      ? reportedCompleted
      : Math.max(job.completedBytes, 0);
  const completeOnDisk = progressResp.complete_on_disk === true;
  const madeProgress =
    resetMonotonic ||
    downloadedBytes > previousDownloadedBytes ||
    expected !== job.expectedBytes;
  const rawFraction =
    progressResp.progress > 0
      ? progressResp.progress
      : expected > 0
        ? downloadedBytes / expected
        : 0;
  const cappedFraction = Math.min(rawFraction, MAX_PROGRESS_FRACTION);
  // Keep the GGUF variant bar monotonic: backend progress is recomputed from the
  // shared per-repo blobs/ dir, so a sibling quant, generation bump, or
  // no-metadata poll can dip one reading. Resets via startJob's seed fraction.
  const fraction = isGgufVariantJob
    ? Math.max(cappedFraction, job.fraction)
    : cappedFraction;
  return {
    expected,
    downloadedBytes,
    completedBytes,
    completeOnDisk,
    fraction,
    madeProgress,
  };
}
