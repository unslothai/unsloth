// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FolderSyncJob } from "../types/rag";

/** Percent shown on an in-progress chip. Caps below 100 while work remains so
 * Math.round cannot report 100% (e.g. 1340/1342) with the spinner still running. */
export function chipProgressPercent(
  progress: number | null | undefined,
  processing: boolean,
): number | null {
  if (progress == null || !Number.isFinite(progress)) return null;
  const raw = progress <= 1 ? progress * 100 : progress;
  const rounded = Math.round(Math.max(0, Math.min(100, raw)));
  if (processing && rounded >= 100) return 99;
  return rounded;
}

export type FolderSyncCounts = {
  discovered: number;
  /** Files already indexed from an earlier sync, plus this sync's processed work. */
  ready: number;
  /** Files this sync still has to handle (ingest/update/delete). */
  workTotal: number | null;
  processed: number;
};

/**
 * Align the Source folders row with the progress bar by counting unchanged
 * (already-indexed) files toward "ready", using the job's progress to recover
 * how many files this sync still needs to touch.
 */
export function folderSyncCounts(job: FolderSyncJob): FolderSyncCounts | null {
  const discovered = job.discoveredFiles;
  if (discovered == null || !Number.isFinite(discovered) || discovered < 0) {
    return null;
  }
  const processed = Math.max(0, job.processedFiles ?? 0);
  const progress = job.progress;
  let workTotal: number | null = null;
  if (
    typeof progress === "number" &&
    Number.isFinite(progress) &&
    progress > 0 &&
    processed > 0
  ) {
    workTotal = Math.max(
      processed,
      Math.round(processed / Math.min(progress, 1)),
    );
  } else if (
    typeof progress === "number" &&
    Number.isFinite(progress) &&
    progress >= 1
  ) {
    workTotal = processed;
  }

  if (workTotal != null && workTotal > discovered) {
    // Deletions can make work exceed files still on disk; fall back to the bar's basis.
    return {
      discovered,
      ready: Math.min(processed, workTotal),
      workTotal,
      processed,
    };
  }

  if (workTotal != null && workTotal >= 0) {
    const unchanged = Math.max(0, discovered - workTotal);
    const ready = Math.min(discovered, unchanged + Math.min(processed, workTotal));
    return { discovered, ready, workTotal, processed };
  }

  // No progress tick yet: only this sync's deltas are known.
  return {
    discovered,
    ready: Math.min(discovered, processed),
    workTotal: null,
    processed,
  };
}

export function folderSyncJobSummary(job: FolderSyncJob): {
  text: string;
  title: string;
  progressPercent: number | null;
} {
  if (job.status === "failed") {
    return {
      text: job.error ?? "Sync failed",
      title: "Folder sync failed.",
      progressPercent: null,
    };
  }
  if (job.status === "completed") {
    const indexed = job.indexedFiles ?? job.processedFiles;
    return {
      text:
        indexed == null
          ? "Sync complete"
          : `${indexed} file${indexed === 1 ? "" : "s"} indexed this sync`,
      title:
        "Files added or updated in this sync. Unchanged files from earlier syncs stay indexed.",
      progressPercent: 100,
    };
  }

  const counts = folderSyncCounts(job);
  if (counts == null) {
    return {
      text: job.stage || "Scanning folder",
      title: "Scanning the folder for supported files.",
      progressPercent: null,
    };
  }

  const { discovered, ready, workTotal, processed } = counts;
  if (workTotal != null && workTotal > discovered) {
    // Progress bar basis (files needing work), when it cannot map onto discovered.
    return {
      text: `${Math.min(processed, workTotal)} of ${workTotal} files this sync`,
      title: `${discovered} supported files in the folder. The bar tracks files this sync still needs to add, update, or remove.`,
      progressPercent: (Math.min(processed, workTotal) / Math.max(workTotal, 1)) * 100,
    };
  }

  const progressPercent = discovered > 0 ? (ready / discovered) * 100 : 0;
  const title =
    workTotal == null
      ? `Supported files found in the folder. Unchanged files from earlier syncs count as ready once sync progress is known.`
      : `${ready} of ${discovered} files ready for search. ${workTotal} needed work this sync; the bar includes already-indexed files.`;

  return {
    text: `${ready} of ${discovered} files ready`,
    title,
    progressPercent,
  };
}

export function linkedFolderChipLabel(
  name: string,
  total: number,
  indexing: number,
): string {
  if (indexing <= 0) {
    return `${name} · ${total} file${total === 1 ? "" : "s"}`;
  }
  const done = Math.max(0, total - indexing);
  return `${name} · ${done} of ${total} indexed`;
}

export function linkedFolderChipTitle(
  name: string,
  total: number,
  indexing: number,
): string {
  if (indexing <= 0) {
    return `${name} — ${total} file${total === 1 ? "" : "s"} already added to this project`;
  }
  const done = Math.max(0, total - indexing);
  return `${name} — ${done} of ${total} added files finished indexing (${indexing} still running). Shared with every chat in this project.`;
}
