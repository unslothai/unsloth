// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { disposableTimeoutSignal } from "../lib/abort-signals";
import {
  getActiveDatasetDownloads,
  getAllActiveModelDownloads,
  type DownloadJobState,
} from "./api";
import { DOWNLOAD_KIND } from "./constants";
import { ACTIVE_STATES, POLL_REQUEST_TIMEOUT_MS } from "./download-manager-config";
import {
  apiGetProgress,
  apiGetStatus,
  isRequestTimeout,
} from "./download-api-adapter";
import type {
  DownloadRequest,
  ManagedDownload,
} from "./download-manager-types";
import {
  getState,
  jobKeyOf,
  removeJob,
  repoKeyOf,
} from "./download-manager-state";
import {
  adoptJob,
  applyProgressUpdate,
  finalize,
  hasObservedExpectedBytes,
} from "./poll-loop";
import { runtimeRegistry } from "./runtime-registry";

const HYDRATE_STATUS_TIMEOUT_RETRIES = 4;
const HYDRATE_STATUS_RETRY_MS = 2_500;
const HYDRATE_ADOPTION_RETRY_MS = 2_500;
const HYDRATE_ADOPTION_MAX_RETRIES = 12;

let hydrated = false;

type HydratedIdleProbeResult = "active" | "gone" | "settled";

function safeGeneration(value: number | undefined): number | undefined {
  return Number.isSafeInteger(value) ? value : undefined;
}

async function withHydrationTimeout<T>(
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const timeout = disposableTimeoutSignal(POLL_REQUEST_TIMEOUT_MS);
  try {
    return await request(timeout.signal);
  } finally {
    timeout.dispose();
  }
}

function removeLocalActivePeers(
  kind: DownloadRequest["kind"],
  repoId: string,
  variant: string | null,
): void {
  const activeRepoKey = repoKeyOf(kind, repoId);
  const activeJobKey = jobKeyOf(kind, repoId, variant);
  const snapshotJobKey = jobKeyOf(kind, repoId, null);
  for (const job of Object.values(getState().jobs)) {
    if (!ACTIVE_STATES.has(job.state)) continue;
    if (repoKeyOf(job.kind, job.repoId) !== activeRepoKey) continue;
    if (variant !== null && kind === DOWNLOAD_KIND.MODEL) {
      if (job.key === activeJobKey) continue;
      if (job.key === snapshotJobKey) removeJob(job.key);
      continue;
    }
    if (job.key !== activeJobKey) removeJob(job.key);
  }
}

async function adoptActiveModelDownloads(): Promise<void> {
  const downloads = await withHydrationTimeout((signal) =>
    getAllActiveModelDownloads(signal),
  );
  for (const download of downloads) {
    const repoId = download.repo_id?.trim();
    if (!repoId || !ACTIVE_STATES.has(download.state)) continue;
    removeLocalActivePeers(
      DOWNLOAD_KIND.MODEL,
      repoId,
      download.variant ?? null,
    );
    adoptJob(
      {
        kind: DOWNLOAD_KIND.MODEL,
        repoId,
        variant: download.variant ?? null,
        expectedBytes: 0,
      },
      safeGeneration(download.generation),
      download.state,
    );
  }
}

async function adoptActiveDatasetDownloads(): Promise<void> {
  const downloads = await withHydrationTimeout((signal) =>
    getActiveDatasetDownloads(signal),
  );
  for (const download of downloads) {
    const repoId = download.repo_id?.trim();
    if (!repoId || !ACTIVE_STATES.has(download.state)) continue;
    removeLocalActivePeers(DOWNLOAD_KIND.DATASET, repoId, null);
    adoptJob(
      {
        kind: DOWNLOAD_KIND.DATASET,
        repoId,
        variant: null,
        expectedBytes: 0,
      },
      safeGeneration(download.generation),
      download.state,
    );
  }
}

type BackendAdoptionSide = "model" | "dataset";

const BACKEND_ADOPTERS: Record<BackendAdoptionSide, () => Promise<void>> = {
  model: adoptActiveModelDownloads,
  dataset: adoptActiveDatasetDownloads,
};

// Adopt backend-running downloads this client doesn't know about. Retry only
// the failed side(s) so backend readiness settles without duplicating jobs.
async function hydrateBackendActiveDownloads(
  attempt = 0,
  sides: readonly BackendAdoptionSide[] = ["model", "dataset"],
): Promise<void> {
  const results = await Promise.allSettled(
    sides.map((side) => BACKEND_ADOPTERS[side]()),
  );
  const pending = sides.filter(
    (_, index) => results[index].status === "rejected",
  );
  if (pending.length === 0 || attempt >= HYDRATE_ADOPTION_MAX_RETRIES) return;
  scheduleBackendAdoptionRetry(attempt + 1, pending);
}

function scheduleBackendAdoptionRetry(
  attempt: number,
  sides: readonly BackendAdoptionSide[],
): void {
  const timer = window.setTimeout(() => {
    runtimeRegistry.hydrationRetryTimers.delete(timer);
    void hydrateBackendActiveDownloads(attempt, sides);
  }, HYDRATE_ADOPTION_RETRY_MS);
  runtimeRegistry.hydrationRetryTimers.add(timer);
}

async function probeHydratedIdleProgress(
  key: string,
  job: ManagedDownload,
): Promise<HydratedIdleProbeResult> {
  try {
    const progressResp = await withHydrationTimeout((signal) =>
      apiGetProgress(job, signal),
    );
    const current = getState().jobs[key];
    if (!current || !ACTIVE_STATES.has(current.state)) return "settled";
    const { downloadedBytes } = applyProgressUpdate(key, current, progressResp);
    const updated = getState().jobs[key];
    if (!updated || !ACTIVE_STATES.has(updated.state)) return "settled";
    if (hasObservedExpectedBytes(updated)) {
      finalize(key, "complete", { bytes: updated.downloadedBytes });
      return "settled";
    }
    return downloadedBytes > 0 ? "active" : "gone";
  } catch {
    return "active";
  }
}

async function settleHydratedJob(
  key: string,
  req: DownloadRequest,
  status: {
    state: DownloadJobState;
    error?: string | null;
    generation?: number;
  },
): Promise<void> {
  if (status.state === "running" || status.state === "cancelling") {
    adoptJob(req, safeGeneration(status.generation), status.state);
  } else if (status.state === "complete") {
    finalize(key, "complete");
  } else if (status.state === "cancelled") {
    finalize(key, "cancelled");
  } else if (status.state === "error") {
    finalize(key, "error", { error: status.error ?? null });
  } else {
    const job = getState().jobs[key];
    if (job && hasObservedExpectedBytes(job)) {
      finalize(key, "complete", { bytes: job.downloadedBytes });
    } else if (job?.state === "running" || job?.state === "cancelling") {
      const probeResult = await probeHydratedIdleProgress(key, job);
      if (probeResult === "settled") return;
      if (probeResult === "active" && job.state === "running") {
        adoptJob(req);
        return;
      }
      const latest = getState().jobs[key];
      if (latest && hasObservedExpectedBytes(latest)) {
        finalize(key, "complete", { bytes: latest.downloadedBytes });
      } else {
        finalize(key, job.state === "cancelling" ? "cancelled" : "gone");
      }
    } else {
      finalize(key, "gone");
    }
  }
}

function scheduleHydrationProbeRetry(
  key: string,
  req: DownloadRequest,
  attempt: number,
): void {
  const timer = window.setTimeout(() => {
    runtimeRegistry.hydrationRetryTimers.delete(timer);
    const job = getState().jobs[key];
    if (!job || !ACTIVE_STATES.has(job.state)) return;
    void probeHydratedJob(key, req, attempt);
  }, HYDRATE_STATUS_RETRY_MS);
  runtimeRegistry.hydrationRetryTimers.add(timer);
}

async function probeHydratedJob(
  key: string,
  req: DownloadRequest,
  attempt: number,
): Promise<void> {
  const job = getState().jobs[key];
  if (!job || !ACTIVE_STATES.has(job.state)) return;
  try {
    const status = await withHydrationTimeout((signal) =>
      apiGetStatus(job, signal),
    );
    const current = getState().jobs[key];
    if (!current || !ACTIVE_STATES.has(current.state)) return;
    await settleHydratedJob(key, req, status);
  } catch (error) {
    if (isRequestTimeout(error) && attempt < HYDRATE_STATUS_TIMEOUT_RETRIES) {
      scheduleHydrationProbeRetry(key, req, attempt + 1);
      return;
    }
    adoptJob(req);
  }
}

export function hydrateDownloadManager(): void {
  if (hydrated) return;
  hydrated = true;
  void hydrateBackendActiveDownloads();
  const jobs = Object.values(getState().jobs);
  for (const job of jobs) {
    if (!ACTIVE_STATES.has(job.state)) {
      removeJob(job.key);
      continue;
    }
    const req: DownloadRequest = {
      kind: job.kind,
      repoId: job.repoId,
      variant: job.variant,
      expectedBytes: job.expectedBytes,
    };
    void probeHydratedJob(job.key, req, 0);
  }
}

export function resetHydrationState(): void {
  hydrated = false;
}
