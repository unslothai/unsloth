


// the download state the training-start overlay renders, kept apart from it so the rules can
// be exercised without mounting the component.

// restated rather than imported from the chat feature so this module stays out of its type graph.
export type DownloadProgressReading = {
  downloaded_bytes: number;
  completed_bytes: number;
  expected_bytes: number;
  progress: number;
  // optional on purpose: a backend older than this field is the one case where treating a
  // missing value as `true` would settle a row that nothing verified.
  complete_on_disk?: boolean;
  // omitted, not nulled, when the cache could not be scanned at all.
  cache_path?: string | null;
};

export type DownloadState = {
  downloadedBytes: number;
  // finalized blobs only; bytes still landing in a `.incomplete` blob are not counted here.
  completedBytes: number;
  totalBytes: number;
  percent: number;
  cachePath: string | null;
  // the backend verified a usable snapshot on disk.
  completeOnDisk: boolean;
  // nothing left to transfer, by verification or by standing still.
  settled: boolean;
  // bytes moved between the last two readings, and the snapshot is not verified. Not
  // `!settled`: an orphaned `.incomplete` blob keeps `downloaded !== completed` forever, so a
  // row that will never settle is still not transferring anything.
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

// a training run fetches a subset of the repo, so completed_bytes never reaches expected_bytes
// and the backend's verification never fires; standing still is the settled signal instead.
export function downloadStateFromProgress(
  reading: DownloadProgressReading,
  previous: DownloadState = EMPTY_DOWNLOAD_STATE,
): DownloadState {
  const completeOnDisk = reading.complete_on_disk ?? false;
  const totalBytes = reading.expected_bytes;
  const downloadedBytes = reading.downloaded_bytes;
  const completedBytes = reading.completed_bytes;
  const nothingInFlight = downloadedBytes > 0 && downloadedBytes === completedBytes;
  // two polls, because huggingface_hub finalizes each blob as it lands and a single quiet reading happens between files.
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
    // A verified snapshot is not transferring whatever the byte counts did since the last
    // reading -- and it is the condition the poll stops on, so a `true` here would freeze
    // and suppress this row's preparation step for the rest of the run.
    moving: !completeOnDisk && !unchanged,
  };
}

// presents a settled resource as ready, which the backend's 99% cap never does on its own (#7858).
export function coerceCachedStateReady(state: DownloadState): DownloadState {
  if (!state.cachePath) return state;
  if (!state.settled && state.downloadedBytes > 0 && state.percent < 100) {
    return state;
  }
  if (state.downloadedBytes <= 0) {
    // an unreadable-size entry settles so it cannot hang; one with bytes still expected has not started.
    return state.totalBytes > 0
      ? state
      : { ...state, percent: 100, settled: true };
  }
  // the total becomes what was fetched, not expected_bytes, which counts files this run never wanted.
  return {
    ...state,
    completedBytes: state.downloadedBytes,
    totalBytes: state.downloadedBytes,
    percent: 100,
    settled: true,
  };
}
