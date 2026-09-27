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

export type StagedFolder = NativeDocumentFolderSelection;

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
