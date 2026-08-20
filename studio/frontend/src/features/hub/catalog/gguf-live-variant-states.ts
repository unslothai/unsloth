


import type { ManagedDownload } from "../download-manager";
import type { GgufVariantDetail } from "../inventory";
import { normalizeGgufVariantIdentity } from "../lib/model-identity";

export type LiveGgufVariantState = {
  state: ManagedDownload["state"];
  expectedBytes: number;
  transferredBytes: number;
  /** Whether `transferredBytes` came from a poll that measured it. */
  measuredTransfer: boolean;
  startedAt: number;
};

export function activeDownloadState(state: ManagedDownload["state"] | undefined): boolean {
  return state === "running" || state === "cancelling";
}

function terminalPartialState(state: ManagedDownload["state"] | undefined): boolean {
  return state === "cancelled" || state === "error";
}

function completedDownloadState(
  state: ManagedDownload["state"] | undefined,
): boolean {
  return state === "complete";
}

export function createLiveGgufVariantStatesSelector(repoId: string): (state: {
  jobs: Record<string, ManagedDownload>;
}) => Map<string, LiveGgufVariantState> {
  const repoKey = repoId.trim().toLowerCase();
  let cache: { signature: string; states: Map<string, LiveGgufVariantState> } = {
    signature: "",
    states: new Map(),
  };
  return (state) => {
    const entries: Array<[string, LiveGgufVariantState]> = [];
    for (const job of Object.values(state.jobs)) {
      if (job.kind !== "model" || !job.variant) continue;
      if (job.repoId.trim().toLowerCase() !== repoKey) continue;
      const live =
        activeDownloadState(job.state) ||
        completedDownloadState(job.state) ||
        (terminalPartialState(job.state) &&
          Math.max(job.downloadedBytes, job.completedBytes) > 0);
      if (!live) continue;
      entries.push([
        normalizeGgufVariantIdentity(job.variant),
        {
          state: job.state,
          expectedBytes: job.expectedBytes,
          // The in-flight counter alone, never max(downloaded, completed): a max
          // could only fire on a HELD completed figure, which is priced against
          // the previous baseline. After an XET-to-HTTP reclaim that subtracted
          // 3 GB from a 0.5 GB remainder and the row read "0 B left".
          transferredBytes: job.downloadedBytes,
          // Undefined is a job that has not polled yet, which has no held
          // figure to forward; only an explicit false is a held reading.
          measuredTransfer: job.measuredTransfer !== false,
          startedAt: job.startedAt,
        },
      ]);
    }
    entries.sort(([left], [right]) => left.localeCompare(right));
    const signature = JSON.stringify(entries);
    if (signature === cache.signature) return cache.states;
    cache = { signature, states: new Map(entries) };
    return cache.states;
  };
}

export function applyLiveGgufVariantStates(
  variants: readonly GgufVariantDetail[],
  liveStates: ReadonlyMap<string, LiveGgufVariantState>,
): GgufVariantDetail[] {
  return variants.map((variant) => {
    const live = liveStates.get(normalizeGgufVariantIdentity(variant.quant));
    if (!live) return variant;
    const liveComplete = completedDownloadState(live.state);
    const livePartial =
      activeDownloadState(live.state) || terminalPartialState(live.state);
    const expectedBytes = Math.max(
      live.expectedBytes,
      variant.download_size_bytes ?? 0,
      variant.size_bytes,
    );
    // The row says "N left", so N has to follow the transfer of a running job,
    // and the backend's measured remainder covers every other case.
    //
    // Both terms come from the job and only from the job. snapshot_progress nets
    // completed_baseline_bytes -- files a previous quant already fetched, an
    // mmproj most often -- out of expected_bytes AND downloaded_bytes alike, so
    // the pair is self-consistent while the catalog totals above are not.
    // Subtracting the transfer from `expectedBytes`, the larger of the two
    // scopes, adds that baseline back: 1 GB reused and 1 GB fetched of a 5 GB
    // plan would read 4 GB left rather than 3 GB.
    //
    // Only while running: `transferredBytes` is progress, not bytes a resume can
    // reuse. From huggingface_hub 1.18 the partial is process-unique and unlinked
    // in a finally, so an interrupted transfer is refetched whole, which is how
    // `existing_blob_bytes` already prices it. Subtracting a dead job's progress
    // read "1.0 GB left" for a cancelled 18 GB download with all 18 GB to fetch.
    //
    // And only off a MEASURED reading: a held one is priced against the previous,
    // larger total, so mixing it with the current one read "0 B left".
    const liveRemaining =
      activeDownloadState(live.state) &&
      live.measuredTransfer &&
      live.expectedBytes > 0 &&
      live.transferredBytes > 0
        ? Math.max(live.expectedBytes - live.transferredBytes, 0)
        : null;
    return {
      ...variant,
      downloaded: liveComplete ? true : livePartial ? false : variant.downloaded,
      partial: liveComplete ? false : livePartial || variant.partial,
      download_size_bytes:
        expectedBytes > 0 ? expectedBytes : variant.download_size_bytes,
      download_remaining_bytes:
        liveComplete
          ? variant.download_remaining_bytes
          : liveRemaining ?? variant.download_remaining_bytes,
    };
  });
}
