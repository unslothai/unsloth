


// The download state the training-start overlay renders, plus the coercion that
// decides whether a resource is still arriving or merely unverified. Kept apart
// from the overlay so the rule can be exercised without mounting the component.

export type DownloadState = {
  downloadedBytes: number;
  // Finalized-blob bytes only, mirroring the backend field of the same name:
  // bytes still landing in a `.incomplete` blob count toward `downloadedBytes`
  // but not here.
  completedBytes: number;
  totalBytes: number;
  percent: number;
  cachePath: string | null;
};

export const EMPTY_DOWNLOAD_STATE: DownloadState = {
  downloadedBytes: 0,
  completedBytes: 0,
  totalBytes: 0,
  percent: 0,
  cachePath: null,
};

/**
 * Present a cached resource as ready once nothing is left to transfer.
 *
 * The backend holds progress at 99% until a Studio download manifest verifies the
 * snapshot on disk. Hugging Face cache entries created outside Studio -- or before
 * the manifest system existed -- never get one, so they stay unverified with every
 * byte already present, and the overlay rendered them as an active transfer
 * indefinitely (#7858).
 *
 * `completedBytes` is what separates the two cases: it counts finalized blobs
 * only, so all of them being present means nothing is in flight and the remaining
 * gap is verification rather than transfer. A resumed download -- where finalized
 * bytes can sit near the total while `.incomplete` blobs are still growing -- keeps
 * reporting progress, which is what the backend's 99% cap exists to protect.
 */
export function coerceCachedStateReady(state: DownloadState): DownloadState {
  if (!state.cachePath) return state;
  const everyByteFinalized =
    state.totalBytes > 0 && state.completedBytes >= state.totalBytes;
  if (state.downloadedBytes > 0 && state.percent < 100 && !everyByteFinalized) {
    return state;
  }
  const totalBytes =
    state.totalBytes > 0 ? state.totalBytes : state.downloadedBytes;
  if (totalBytes <= 0) {
    return { ...state, percent: 100 };
  }
  return {
    ...state,
    downloadedBytes: totalBytes,
    completedBytes: totalBytes,
    totalBytes,
    percent: 100,
  };
}
