// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The download state the training-start overlay renders, plus the coercion that
// decides whether a resource is still arriving or merely unverified. Kept apart
// from the overlay so the rule can be exercised without mounting the component.

/**
 * The download-progress payload, as `getDownloadProgress` and `getDatasetDownloadProgress`
 * return it. Restated rather than imported so this module stays out of the chat feature's
 * type graph, and required field-by-field so a caller cannot drop one silently.
 */
export type DownloadProgressReading = {
  downloaded_bytes: number;
  completed_bytes: number;
  expected_bytes: number;
  progress: number;
  complete_on_disk: boolean;
  /** Omitted, not nulled, when the cache could not be scanned at all. */
  cache_path?: string | null;
};

export type DownloadState = {
  downloadedBytes: number;
  // Finalized-blob bytes only, mirroring the backend field of the same name:
  // bytes still landing in a `.incomplete` blob count toward `downloadedBytes`
  // but not here.
  completedBytes: number;
  totalBytes: number;
  percent: number;
  cachePath: string | null;
  // The backend verified a usable snapshot on disk. Only a reading sets this.
  completeOnDisk: boolean;
  // Nothing is left to transfer, by verification or by standing still. See
  // `downloadStateFromProgress` for why the weaker signal is needed at all.
  settled: boolean;
};

export const EMPTY_DOWNLOAD_STATE: DownloadState = {
  downloadedBytes: 0,
  completedBytes: 0,
  totalBytes: 0,
  percent: 0,
  cachePath: null,
  completeOnDisk: false,
  settled: false,
};

/**
 * Fold one poll into the running state.
 *
 * `expected_bytes` counts every file in the repo, but a training run fetches a subset --
 * `Qwen3.5-0.8B-Base` skips `README.md`, `LICENSE` and `.gitattributes`, and a dataset skips
 * the configs `load_dataset()` did not ask for. `completed_bytes` therefore never reaches
 * `expected_bytes`, the backend's verification never fires, and `progress` stays pinned at its
 * 0.99 cap for a resource that is entirely present.
 *
 * Standing still with no blob in flight is the signal instead, and it needs two polls:
 * huggingface_hub finalizes each blob as it lands, so a single reading cannot tell "finished"
 * from "between files". Once settled a resource stays settled -- the run only consumes it.
 */
export function downloadStateFromProgress(
  reading: DownloadProgressReading,
  previous: DownloadState = EMPTY_DOWNLOAD_STATE,
): DownloadState {
  const totalBytes = reading.expected_bytes;
  const downloadedBytes = reading.downloaded_bytes;
  const completedBytes = reading.completed_bytes;
  const nothingInFlight = downloadedBytes > 0 && downloadedBytes === completedBytes;
  const unchanged =
    downloadedBytes === previous.downloadedBytes &&
    completedBytes === previous.completedBytes;
  return {
    downloadedBytes,
    completedBytes,
    totalBytes,
    percent: totalBytes > 0 ? Math.min(100, Math.round(reading.progress * 100)) : 0,
    cachePath: reading.cache_path ?? null,
    completeOnDisk: reading.complete_on_disk,
    settled:
      reading.complete_on_disk ||
      previous.settled ||
      (nothingInFlight && unchanged),
  };
}

/**
 * Present a settled resource as ready.
 *
 * The backend holds progress at 99% until a Studio download manifest verifies the snapshot on
 * disk, and cache entries made outside Studio never get one, so they stayed rendered as an
 * active transfer indefinitely (#7858).
 *
 * The total is rewritten to the bytes actually fetched rather than left at `expected_bytes`,
 * which counts files this run never wanted: reporting "28.1 MB / 28.1 MB" for a 14.0 MB fetch
 * would trade a stuck bar for a wrong one.
 */
export function coerceCachedStateReady(state: DownloadState): DownloadState {
  if (!state.cachePath) return state;
  if (!state.settled && state.downloadedBytes > 0 && state.percent < 100) {
    return state;
  }
  if (state.downloadedBytes <= 0) {
    return { ...state, percent: 100, settled: true };
  }
  return {
    ...state,
    completedBytes: state.downloadedBytes,
    totalBytes: state.downloadedBytes,
    percent: 100,
    settled: true,
  };
}
