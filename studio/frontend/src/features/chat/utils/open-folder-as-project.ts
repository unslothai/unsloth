// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { pickNativeDocumentFolder } from "@/features/native-intents";
import {
  announceProjectSourcesUpdated,
  createLinkedFolder,
  noteProjectWork,
  watchProjectFolderJob,
} from "@/features/rag/api/rag-api";
import { markProjectSourcesPending } from "@/features/rag/components/project-source-dropzone";
import { toast } from "@/lib/toast";
import { createChatProject } from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";

let opening = false;

/** File > Open Folder: pick a folder, make a project named after it with the folder linked, and
 *  land on its Sources. Resolves to the project, even when only the link failed, or null when
 *  cancelled or no project was made. */
export async function openFolderAsProject(): Promise<ProjectRecord | null> {
  if (opening) return null;
  opening = true;
  try {
    const selected = await pickNativeDocumentFolder();
    if (!selected) return null;
    const project = await createChatProject(selected.displayName);
    noteProjectWork(project.id, 1);
    try {
      const { job } = await createLinkedFolder(
        { type: "project", id: project.id },
        selected.token,
        selected.displayName,
      );
      watchProjectFolderJob(project.id, job.id);
      announceProjectSourcesUpdated(project.id);
    } catch (error) {
      // Kept, not deleted: chats may have joined it meanwhile, and deleting a project deletes
      // them. Its Sources has Link folder to retry.
      toast.error("Could not link folder", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      noteProjectWork(project.id, -1);
    }
    markProjectSourcesPending(project.id);
    return project;
  } catch (error) {
    toast.error("Could not open folder", {
      description: error instanceof Error ? error.message : String(error),
    });
    return null;
  } finally {
    opening = false;
  }
}
