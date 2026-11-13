// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const SOURCE_FOLDER_DESCRIPTION =
  "Keep supported documents in sync for chats to search. Linking does not grant access to edit the original files.";

export function sourceFolderUnavailableMessage(isDesktop: boolean): string {
  return isDesktop
    ? "Folder selection is unavailable with this connection. You can still upload documents."
    : "Open Unsloth Desktop to add a source folder. You can still upload documents.";
}

/** What a source folder's sync line counts, shown on hover. */
export const FOLDER_SYNC_HELP =
  "Synced files have been read and indexed (or set aside because they can't be), out of every supported file found in the folder. Chats can search a file as soon as it is synced.";

/** Folder-wide sync progress, or null before the scan has counted the folder.
 * `syncedFiles` includes files an earlier sync already indexed; `processedFiles` counts
 * only this sync's changes, so it can stop short of the folder's total. */
export function folderSyncProgress(job: {
  discoveredFiles?: number | null;
  syncedFiles?: number | null;
}): { synced: number; discovered: number } | null {
  const discovered = job.discoveredFiles ?? 0;
  if (job.syncedFiles == null || discovered <= 0) return null;
  return { synced: Math.min(job.syncedFiles, discovered), discovered };
}

export function folderSyncLabel(synced: number, discovered: number): string {
  return `${synced} of ${discovered} file${discovered === 1 ? "" : "s"} synced`;
}

/** A linked folder's documents in the project: ready to search vs still indexing. */
export function linkedFolderDocumentsSummary(
  name: string,
  total: number,
  indexing: number,
): { label: string; title: string } {
  const files = (n: number) => `${n} file${n === 1 ? "" : "s"}`;
  if (indexing <= 0) {
    return {
      label: files(total),
      title: `${name}: ${files(total)}, kept in sync and shared with every chat in this project. Manage it in the project's Sources tab.`,
    };
  }
  const ready = total - indexing;
  return {
    label: `${ready} ready · ${indexing} indexing`,
    title: `${name}: ${ready} of the ${files(total)} added from this folder so far ${ready === 1 ? "is" : "are"} ready to search and ${indexing} ${indexing === 1 ? "is" : "are"} still being indexed. The folder sync adds more files as it reaches them; the project's Sources tab shows how far it has got.`,
  };
}

/** A running chip's percentage: rounded down, so a few files still indexing never read 100%. */
export function runningPercent(progress: number): number {
  const value = progress <= 1 ? progress * 100 : progress;
  return Math.max(0, Math.min(99, Math.floor(value)));
}
