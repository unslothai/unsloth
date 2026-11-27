// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type TransferSample,
  type TransferStats,
  appendSample,
  computeTransferStats,
} from "@/lib/transfer-stats";
import { DOWNLOAD_KIND } from "./constants";
import { patchJob, putJob, scheduleRemoval } from "./download-manager-state";

/**
 * Downloads owned by another subsystem (STT sidecars) that report progress here, since the
 * poll loop cannot drive them.
 */

const TERMINAL_LINGER_MS = 4_000;

interface ExternalJob {
  cancel: () => Promise<void> | void;
  samples: TransferSample[];
}

const externalJobs = new Map<string, ExternalJob>();

export function isExternalJob(key: string): boolean {
  return externalJobs.has(key);
}

export async function cancelExternalJob(key: string): Promise<void> {
  const job = externalJobs.get(key);
  if (!job) return;
  patchJob(key, { state: "cancelling" });
  try {
    await job.cancel();
  } catch {
    // Progress updates do not reset state, so restore the row instead of leaving it "cancelling".
    if (externalJobs.has(key)) patchJob(key, { state: "running" });
  }
}

export function startExternalJob(init: {
  key: string;
  repoId: string;
  variant: string | null;
  expectedBytes: number;
  cancel: () => Promise<void> | void;
}): void {
  externalJobs.set(init.key, { cancel: init.cancel, samples: [] });
  putJob({
    key: init.key,
    kind: DOWNLOAD_KIND.MODEL,
    repoId: init.repoId,
    variant: init.variant,
    state: "running",
    downloadedBytes: 0,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: Math.max(0, init.expectedBytes),
    fraction: 0,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: Date.now(),
    external: true,
  });
}

export function updateExternalJob(
  key: string,
  progress: { downloadedBytes: number; expectedBytes: number },
): void {
  const job = externalJobs.get(key);
  if (!job) return;
  const downloadedBytes = Math.max(0, progress.downloadedBytes);
  const expectedBytes = Math.max(0, progress.expectedBytes);
  // Hidden tabs clamp tracker timers to ~1/min, which the estimator would read as burst cadence.
  let stats: TransferStats | null = null;
  if (typeof document !== "undefined" && document.hidden) {
    job.samples.length = 0;
  } else {
    appendSample(job.samples, Date.now() / 1000, downloadedBytes);
    stats = computeTransferStats(job.samples, expectedBytes);
  }
  patchJob(key, {
    downloadedBytes,
    expectedBytes,
    fraction:
      expectedBytes > 0 ? Math.min(1, downloadedBytes / expectedBytes) : 0,
    // Unstable early samples read as absurd rates; show nothing until settled.
    bytesPerSec: stats?.stable ? stats.rateBytesPerSecond : 0,
    etaSeconds: stats?.stable ? stats.etaSeconds : 0,
  });
}

export function finishExternalJob(
  key: string,
  outcome: "complete" | "cancelled" | "error",
  error?: string | null,
): void {
  if (!externalJobs.delete(key)) return;
  patchJob(key, {
    state: outcome,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: outcome === "error" ? (error ?? "Download failed.") : null,
    ...(outcome === "complete" ? { fraction: 1, completeOnDisk: true } : {}),
  });
  scheduleRemoval(key, TERMINAL_LINGER_MS);
}

export function updateExternalActivity(
  key: string,
  activity: string,
  details: string[],
): void {
  if (externalJobs.has(key)) patchJob(key, { activity, details });
}
