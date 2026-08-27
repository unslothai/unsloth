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

// A number off the wire (a byte count or a fraction), or 0 when it is not a usable
// one. The payload is
// cast from raw JSON with no runtime validation (apiGetProgress -> ProgressLike),
// and a NaN or an Infinity survives every comparison below to reach the store as
// a width of "NaN%" on the progress bar. 0 reads as "not measured", which is the
// conservative answer here: it holds the card's last figure rather than
// publishing a garbage one, and never manufactures a completion.
function finiteReading(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function resolveProgressUpdate(
  job: ManagedDownload,
  progressResp: ProgressLike,
  opts: { resetMonotonic?: boolean } = {},
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
  const reportedCompleted = finiteReading(progressResp.completed_bytes);
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
  const reportedDownloaded = finiteReading(progressResp.downloaded_bytes);
  // Whether this poll MEASURED the transfer counter or merely held the last
  // one. A held figure is fine to draw a bar with, but it is priced against the
  // PREVIOUS total, so a caller subtracting it from the current expectedBytes
  // gets a remainder for a plan that no longer exists. The XET fallback reclaim
  // is exactly that pairing: the retry's first reading is a legitimate 0 against
  // a shrunken total, and the 3 GB held behind it wipe out a 0.5 GB remainder.
  const measuredTransfer = resetMonotonic || reportedDownloaded > 0;
  const downloadedBytes = measuredTransfer
    ? Math.max(0, reportedDownloaded)
    : Math.max(previousDownloadedBytes, 0);
  // Same rule for the finalized counter.
  const measuredCompleted = resetMonotonic || reportedCompleted > 0;
  const completedBytes = resetMonotonic
    ? Math.max(0, reportedCompleted)
    : measuredCompleted
      ? reportedCompleted
      : Math.max(job.completedBytes, 0);
  // ...and the completion flag is only honoured on a poll that measured the
  // counter it is a statement about. Holding the bytes over made a new pairing
  // reachable that the plain reading never was: LAST poll's completed_bytes
  // beside THIS poll's complete_on_disk, a completion neither reading showed on
  // its own. The backend does hold the invariant that would rule that out
  // (complete_on_disk is only set once completed_bytes clears the total), but a
  // false "complete" retires the card and drops the download, and that is not an
  // outcome to leave resting on a cross-tier promise no code here checks. Both
  // halves of hasObservedExpectedBytes now provably come from the same reading.
  const completeOnDisk = progressResp.complete_on_disk === true && measuredCompleted;
  const madeProgress =
    resetMonotonic ||
    // An UNMEASURED scan is not an idle one. The backend says cache_measured false when it
    // could not read the cache at all, and the response is then all zeroes -- which the idle
    // grace reads as "nothing is happening" and finalizes the job as gone. The initial adopt
    // probe already refuses to retire on that shape; without this the protection lasted only
    // until the first adopted poll loop ran its grace out.
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
  // Keep the GGUF variant bar monotonic: backend progress is recomputed from the
  // shared per-repo blobs/ dir, so a sibling quant, generation bump, or
  // no-metadata poll can dip one reading. Resets via startJob's seed fraction.
  //
  // NOT across a generation change, though. `resetMonotonic` is the caller saying another
  // client restarted this job, and it already clears the byte counters -- carrying the old
  // generation's high-water mark over pinned a retry that starts at 0 B to the previous run's
  // 99% for its entire life, which is the stale card this whole path exists to remove.
  const fraction =
    isGgufVariantJob && !resetMonotonic
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
