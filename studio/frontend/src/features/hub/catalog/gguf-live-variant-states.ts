// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ManagedDownload } from "../download-manager";
import type { GgufVariantDetail } from "../inventory";
import { normalizeGgufVariantIdentity } from "../lib/model-identity";

export type LiveGgufVariantState = {
  state: ManagedDownload["state"];
  expectedBytes: number;
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
    return {
      ...variant,
      downloaded: liveComplete ? true : livePartial ? false : variant.downloaded,
      partial: liveComplete ? false : livePartial || variant.partial,
      download_size_bytes:
        expectedBytes > 0 ? expectedBytes : variant.download_size_bytes,
    };
  });
}
