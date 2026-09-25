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
import { createChatProject, deleteChatProject } from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";

let opening = false;

/** File > Open Folder: pick a folder, make a project named after it with the folder linked, and
 *  land on its Sources. Resolves to the new project, or null when cancelled or failed. */
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
    } catch (error) {
      // An empty project named after a folder it does not hold would only mislead.
      await deleteChatProject(project.id).catch(() => undefined);
      throw error;
    } finally {
      noteProjectWork(project.id, -1);
    }
    announceProjectSourcesUpdated(project.id);
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
