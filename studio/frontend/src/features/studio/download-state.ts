// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept apart from the overlay so the rules are testable without mounting it.

// Restated, not imported, to stay out of the chat feature's type graph.
export type DownloadProgressReading = {
  downloaded_bytes: number;
  completed_bytes: number;
  expected_bytes: number;
  progress: number;
  // Optional: treating a missing value as true would settle an unverified row.
  complete_on_disk?: boolean;
  // omitted, not nulled, when the cache could not be scanned at all.
  cache_path?: string | null;
};

export type DownloadState = {
  downloadedBytes: number;
  // `.incomplete` blobs are not counted.
  completedBytes: number;
  totalBytes: number;
  percent: number;
  cachePath: string | null;
  completeOnDisk: boolean;
  settled: boolean;
  // Not `!settled`: an orphaned `.incomplete` blob never settles but is not transferring.
  moving: boolean;
};

export const EMPTY_DOWNLOAD_STATE: DownloadState = {
  downloadedBytes: 0,
  completedBytes: 0,
  totalBytes: 0,
  percent: 0,
  cachePath: null,
  completeOnDisk: false,
  settled: false,
  moving: false,
};

// A run fetches a repo subset, so expected bytes are never reached; standing still means settled.
export function downloadStateFromProgress(
  reading: DownloadProgressReading,
  previous: DownloadState = EMPTY_DOWNLOAD_STATE,
): DownloadState {
  const completeOnDisk = reading.complete_on_disk ?? false;
  const totalBytes = reading.expected_bytes;
  const downloadedBytes = reading.downloaded_bytes;
  const completedBytes = reading.completed_bytes;
  const nothingInFlight = downloadedBytes > 0 && downloadedBytes === completedBytes;
  // Two polls, because huggingface_hub finalizes blobs one at a time.
  const unchanged =
    downloadedBytes === previous.downloadedBytes &&
    completedBytes === previous.completedBytes;
  return {
    downloadedBytes,
    completedBytes,
    totalBytes,
    percent: totalBytes > 0 ? Math.min(100, Math.round(reading.progress * 100)) : 0,
    cachePath: reading.cache_path ?? null,
    completeOnDisk,
    settled: completeOnDisk || (nothingInFlight && unchanged),
    // A verified snapshot stops the poll, so `true` here would freeze the row's preparation step.
    moving: !completeOnDisk && !unchanged,
  };
}

// The backend caps cached progress at 99%, so present a settled resource as ready.
export function coerceCachedStateReady(state: DownloadState): DownloadState {
  if (!state.cachePath) return state;
  if (!state.settled && state.downloadedBytes > 0 && state.percent < 100) {
    return state;
  }
  if (state.downloadedBytes <= 0) {
    // An unreadable size settles so it cannot hang.
    return state.totalBytes > 0
      ? state
      : { ...state, percent: 100, settled: true };
  }
  // expected_bytes counts files this run never wanted.
  return {
    ...state,
    completedBytes: state.downloadedBytes,
    totalBytes: state.downloadedBytes,
    percent: 100,
    settled: true,
  };
}
