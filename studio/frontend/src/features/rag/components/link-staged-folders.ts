// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NativeDocumentFolderSelection } from "@/features/native-intents";
import { toast } from "@/lib/toast";
import {
  announceProjectSourcesUpdated,
  createLinkedFolder,
  noteProjectWork,
  watchProjectFolderJob,
} from "../api/rag-api";
import { markProjectSourcesPending } from "./project-source-dropzone";
import { EXPIRY_GRACE_MS } from "./staged-source";

/** LEASE_TTL in native_backend_lease.rs, for shells that do not send expiresAtMs. */
const FOLDER_LEASE_TTL_MS = 2 * 60_000;

export type StagedFolder = NativeDocumentFolderSelection & {
  expiresAtMs: number;
};

export function stageFolder(
  selected: NativeDocumentFolderSelection,
): StagedFolder {
  return {
    ...selected,
    expiresAtMs: selected.expiresAtMs ?? Date.now() + FOLDER_LEASE_TTL_MS,
  };
}

/** Expired, or too close to it to survive the request. */
export function isFolderExpired(folder: StagedFolder, now: number): boolean {
  return folder.expiresAtMs - EXPIRY_GRACE_MS <= now;
}

/** Link folders picked before the project existed. A failed link toasts and never blocks creation. */
export async function linkStagedFolders(
  projectId: string,
  folders: StagedFolder[],
): Promise<void> {
  if (folders.length === 0) return;
  markProjectSourcesPending(projectId);
  noteProjectWork(projectId, 1);
  try {
    for (const folder of folders) {
      try {
        if (isFolderExpired(folder, Date.now())) {
          throw new Error(
            "The selection expired. Link it again from the project.",
          );
        }
        const { job } = await createLinkedFolder(
          { type: "project", id: projectId },
          folder.token,
          folder.displayName,
        );
        watchProjectFolderJob(projectId, job.id);
      } catch (error) {
        toast.error(`Could not link ${folder.displayName}`, {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    }
  } finally {
    announceProjectSourcesUpdated(projectId);
    noteProjectWork(projectId, -1);
  }
}
