// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { pickNativeDocumentFolder } from "@/features/native-intents";
import {
  announceProjectSourcesUpdated,
  createLinkedFolder,
  noteProjectWork,
  watchProjectFolderJob,
} from "@/features/rag/api/rag-api";
import {
  isProjectLandingMounted,
  markProjectSourcesPending,
} from "@/features/rag/components/project-source-dropzone";
import { toast } from "@/lib/toast";
import { useSyncExternalStore } from "react";
import { createChatProject } from "../hooks/use-chat-projects";
import type { ProjectRecord } from "../types";

let opening = false;
const openingListeners = new Set<() => void>();

function setOpening(value: boolean): void {
  opening = value;
  openingListeners.forEach((listener) => listener());
}

export function useOpeningFolder(): boolean {
  return useSyncExternalStore(
    (listener) => {
      openingListeners.add(listener);
      return () => openingListeners.delete(listener);
    },
    () => opening,
  );
}

export async function openFolderAsProject(): Promise<ProjectRecord | null> {
  if (opening) return null;
  setOpening(true);
  try {
    const selected = await pickNativeDocumentFolder();
    if (!selected) return null;
    const project = await createChatProject(selected.displayName);
    markProjectSourcesPending(project.id);
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
      // Kept, not deleted: deleting a project deletes chats that may have joined it.
      toast.error("Could not link folder", {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      noteProjectWork(project.id, -1);
    }
    if (!isProjectLandingMounted(project.id)) markProjectSourcesPending(project.id);
    return project;
  } catch (error) {
    toast.error("Could not open folder", {
      description: error instanceof Error ? error.message : String(error),
    });
    return null;
  } finally {
    setOpening(false);
  }
}
