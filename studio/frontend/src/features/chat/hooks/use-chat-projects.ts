// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { CHAT_HISTORY_UPDATED_EVENT } from "../api/chat-api";
import type { ProjectRecord } from "../types";
import {
  createStoredChatProject,
  deleteStoredChatProject,
  isExpectedBackgroundChatStorageError,
  listStoredChatProjects,
  moveStoredChatItemToProject,
  updateStoredChatProject,
} from "../utils/chat-history-storage";
import type { SidebarItem } from "./use-chat-sidebar-items";

let cachedProjects: ProjectRecord[] = [];

export function useChatProjects(): {
  projects: ProjectRecord[];
  isLoading: boolean;
  hasLoaded: boolean;
} {
  const [projects, setProjects] = useState<ProjectRecord[]>(cachedProjects);
  const [isLoading, setIsLoading] = useState(cachedProjects.length === 0);
  const [hasLoaded, setHasLoaded] = useState(cachedProjects.length > 0);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      if (!cancelled) setIsLoading(true);
      try {
        const next = await listStoredChatProjects({ includeArchived: false });
        cachedProjects = next;
        if (!cancelled) setProjects(next);
      } catch (error) {
        if (isExpectedBackgroundChatStorageError(error)) {
          return;
        }
        if (!cancelled) throw error;
      } finally {
        if (!cancelled) {
          setHasLoaded(true);
          setIsLoading(false);
        }
      }
    }

    const onHistoryUpdated = () => {
      void load();
    };

    void load();
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    return () => {
      cancelled = true;
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    };
  }, []);

  return { projects, isLoading, hasLoaded };
}

export async function createChatProject(name: string): Promise<ProjectRecord> {
  return createStoredChatProject(name);
}

export async function renameChatProject(
  projectId: string,
  name: string,
): Promise<void> {
  const trimmed = name.trim();
  if (!trimmed) throw new Error("Project name is required.");
  await updateStoredChatProject(projectId, { name: trimmed });
}

export async function updateChatProjectInstructions(
  projectId: string,
  instructions: string,
): Promise<void> {
  await updateStoredChatProject(projectId, { instructions: instructions.trim() });
}

export async function deleteChatProject(
  projectId: string,
  args: { deleteFiles?: boolean } = {},
): Promise<void> {
  await deleteStoredChatProject(projectId, args);
}

export async function moveChatItemToProject(
  item: SidebarItem,
  projectId: string | null,
): Promise<void> {
  await moveStoredChatItemToProject(item, projectId);
}
