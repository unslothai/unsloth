// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One backend progress reading reconciled against the job's current figures.
// Split out of poll-loop so it can be exercised on its own: poll-loop reaches
// the store, the toaster and the API client, and this is the part that decides
// what a card shows and when a download counts as finished.

import { MAX_PROGRESS_FRACTION } from "./download-manager-config";
import { DOWNLOAD_KIND } from "./constants";
import type {
  ManagedDownload,
  ProgressLike,
} from "./download-manager-types";

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
  // The byte counters are high-water marks for every job kind, the way the
  // fraction below already is, and for the same reason: the backend recomputes
  // them from the shared per-repo blobs/ dir, so one poll that cannot resolve
  // the variant's expected files reports zero. Only the total is backend-owned.
  // Letting that single reading through rewrote a finished card to "0 B of
  // 33 GB" permanently, and since completion needs completedBytes to reach
  // expectedBytes the job never finalized and kept its Retry/Resume controls.
  // resetMonotonic still drops the mark: a new generation's bytes are its own.
  const downloadedBytes = resetMonotonic
    ? Math.max(0, progressResp.downloaded_bytes)
    : Math.max(previousDownloadedBytes, progressResp.downloaded_bytes, 0);
  const completedBytes = resetMonotonic
    ? Math.max(0, progressResp.completed_bytes ?? 0)
    : Math.max(job.completedBytes, progressResp.completed_bytes ?? 0, 0);
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
