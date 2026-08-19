// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
          // The in-flight counter alone. snapshot_progress builds the pair as
          // downloaded = completed + in-progress and nets the same baseline out
          // of both, so a single reading never has completed above downloaded
          // and the max could only ever fire on a HELD completed figure -- and a
          // held one is priced against the PREVIOUS baseline. An XET run that
          // falls back to HTTP re-claims in the same generation with a baseline
          // recomputed from disk, which now covers everything the XET attempt
          // finalized, so the new run reports completed 0 against a shrunken
          // total while the card still holds the old run's finalized bytes.
          // Taking the max there subtracted 3 GB from a 0.5 GB remainder and the
          // row read "0 B left" with the transfer barely started.
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
    // The row says "N left", so N has to follow the transfer. Only
    // download_size_bytes moved here, leaving the label on the remainder the
    // one-time variant fetch measured -- or, for a download that started after
    // it, on the full total. A running job carries its own progress, so derive
    // the remainder from that and keep the fetched figure for everything else.
    //
    // Both terms come from the job and only from the job. snapshot_progress
    // nets completed_baseline_bytes -- files a previous quant already fetched,
    // an mmproj most often -- out of expected_bytes AND downloaded_bytes alike,
    // so the pair is self-consistent while the catalog totals above are not.
    // Subtracting the job's transfer from expectedBytes, which takes the larger
    // of the two scopes, would add that baseline straight back: 1 GB reused and
    // 1 GB fetched of a 5 GB plan would read 4 GB left rather than 3 GB.
    //
    // Only while the job is running. `transferredBytes` is the last progress
    // reading, not bytes a resume can reuse, and the two part company the
    // moment the worker exits: from huggingface_hub 1.18 the partial is opened
    // "wb" under a process-unique name and unlinked in a finally, so an
    // interrupted in-file transfer is refetched whole. `existing_blob_bytes`
    // already prices it that way, whole shards and resumable partials only, so
    // subtracting the dead job's progress here overwrote a correct backend
    // reading with a far smaller one: cancelling a one-file 18 GB download at
    // 17 GB read "1.0 GB left" for a transfer with all 18 GB still to fetch.
    //
    // And only off a MEASURED reading. resolveProgressUpdate holds the previous
    // transfer through a poll that reported zero, which the retry after an XET
    // fallback or an XET resume that purged its partial legitimately does: the
    // reclaim recomputes the baseline from disk, so the first reading is a real
    // 0 against a total shrunk by everything already finalized. Both terms have
    // to come from the same reading, and a held one is priced against the
    // previous, larger total -- subtracting 3 GB of it from the 0.5 GB that
    // remains read "0 B left" until the retry moved its first byte. The backend
    // remainder below already covers that window, as it does for a terminal row.
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
