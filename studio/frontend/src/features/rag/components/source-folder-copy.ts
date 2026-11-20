// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const SOURCE_FOLDER_DESCRIPTION =
  "Keep supported documents in sync for chats to search. Choose a folder on this server or link one from Desktop. Linking does not grant access to edit the original files.";

export function sourceFolderUnavailableMessage(isDesktop: boolean): string {
  return isDesktop
    ? "Local folder selection is unavailable with this connection. Use Browse server to choose a folder on the Studio host."
    : "Use Browse server to choose a folder on the Studio host, or open Unsloth Desktop to link a local folder.";
}

export const FOLDER_SYNC_HELP =
  "Synced files have been read and indexed (or set aside because they can't be), out of every supported file found in the folder. Chats can search a file as soon as it is synced.";

/** Null before the scan counts the folder; processedFiles would stop short of the total. */
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

/** Floored so a chip still indexing never reads 100%. */
export function runningPercent(progress: number): number {
  const value = progress <= 1 ? progress * 100 : progress;
  return Math.max(0, Math.min(99, Math.floor(value)));
}
