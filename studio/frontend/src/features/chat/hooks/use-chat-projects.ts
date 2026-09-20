// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, useSyncExternalStore } from "react";
import { CHAT_PROJECTS_UPDATED_EVENT } from "../api/chat-api";
import type { ProjectRecord } from "../types";
import {
  createStoredChatProject,
  deleteStoredChatProject,
  getStoredChatProject,
  isExpectedBackgroundChatStorageError,
  listStoredChatProjects,
  moveStoredChatItemToProject,
  updateStoredChatProject,
} from "../utils/chat-history-storage";
import { offerToDeleteKeptSandboxes } from "../utils/offer-kept-sandbox-files";
import type { SidebarItem } from "./use-chat-sidebar-items";

let cachedProjects: ProjectRecord[] = [];
let projectsLoaded = false;
let projectsRequest: Promise<ProjectRecord[]> | null = null;
let projectsRefreshPending = false;
let lastProjectsUpdateEvent: Event | null = null;
const projectSubscribers = new Set<() => void>();

function subscribeToProjects(onStoreChange: () => void): () => void {
  projectSubscribers.add(onStoreChange);
  return () => projectSubscribers.delete(onStoreChange);
}

function getProjectsSnapshot(): ProjectRecord[] {
  return cachedProjects;
}

function publishProjects(projects: ProjectRecord[]): void {
  cachedProjects = projects;
  projectsLoaded = true;
  for (const onStoreChange of projectSubscribers) onStoreChange();
}

function loadProjects(
  force = false,
  followUpIfPending = false,
): Promise<ProjectRecord[]> {
  if (projectsRequest) {
    if (followUpIfPending) projectsRefreshPending = true;
    return projectsRequest;
  }
  if (!force && projectsLoaded) {
    return Promise.resolve(cachedProjects);
  }

  async function run(): Promise<ProjectRecord[]> {
    let nextProjects: ProjectRecord[] | null = null;
    do {
      projectsRefreshPending = false;
      try {
        const next = await listStoredChatProjects({ includeArchived: false });
        nextProjects = Array.isArray(next) ? next : [];
      } catch (error) {
        if (!isExpectedBackgroundChatStorageError(error)) throw error;
        nextProjects = null;
      }
    } while (projectsRefreshPending);
    if (nextProjects !== null) publishProjects(nextProjects);
    return cachedProjects;
  }

  const request = run().finally(() => {
    projectsRequest = null;
  });
  projectsRequest = request;
  return request;
}

export function useChatProjects(): {
  projects: ProjectRecord[];
  isLoading: boolean;
  hasLoaded: boolean;
} {
  const projects = useSyncExternalStore(
    subscribeToProjects,
    getProjectsSnapshot,
    getProjectsSnapshot,
  );
  const [isLoading, setIsLoading] = useState(!projectsLoaded);
  const [hasLoaded, setHasLoaded] = useState(projectsLoaded);

  useEffect(() => {
    let cancelled = false;

    async function refresh(force = false, followUpIfPending = false) {
      if (!force && projectsLoaded) return;
      if (!cancelled && !projectsLoaded) setIsLoading(true);
      try {
        await loadProjects(force, followUpIfPending);
      } catch (error) {
        // Every caller below is `void refresh(...)`, so nobody is listening: an unexpected
        // failure here becomes an unhandled rejection rather than a handled error. That stayed
        // invisible while only the sidebar and the projects page mounted this hook, because both
        // run where the projects route answers. A chat image mounts it too, once per image, so a
        // route that 404s turned one failure into one rejection per rendered image. The cached
        // rows are left as they are and the caller reads `hasLoaded`; a project that cannot be
        // read is the same to a reader as a project that is not there yet.
        console.debug("Could not refresh the project list", error);
      } finally {
        if (!cancelled) {
          setHasLoaded(true);
          setIsLoading(false);
        }
      }
    }

    const onProjectsUpdated = (event: Event) => {
      const followUpIfPending = event !== lastProjectsUpdateEvent;
      lastProjectsUpdateEvent = event;
      void refresh(true, followUpIfPending);
    };
    // Cached rows render immediately, then one shared request reconciles changes made by another
    // browser tab or API client.
    void refresh(projectsLoaded);
    window.addEventListener(CHAT_PROJECTS_UPDATED_EVENT, onProjectsUpdated);
    return () => {
      cancelled = true;
      window.removeEventListener(CHAT_PROJECTS_UPDATED_EVENT, onProjectsUpdated);
    };
  }, []);

  return { projects, isLoading, hasLoaded };
}

export async function createChatProject(
  name: string,
  workspace?: { nativePathLease: string },
): Promise<ProjectRecord> {
  return createStoredChatProject(name, workspace);
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

export async function setChatProjectWorkspace(
  projectId: string,
  workspace:
    | { kind: "managed" }
    | { kind: "external"; nativePathLease: string },
): Promise<void> {
  await updateStoredChatProject(projectId, {
    workspaceKind: workspace.kind,
    ...(workspace.kind === "external"
      ? { nativePathLease: workspace.nativePathLease }
      : {}),
  });
}

export async function deleteChatProject(
  projectId: string,
  args: { deleteFiles?: boolean } = {},
): Promise<void> {
  const kept = await deleteStoredChatProject(projectId, args);
  // The member chats went with the project, so their own sandboxes are reachable from nothing: the
  // same offer an ordinary chat delete makes, and a sandbox the backend could not remove is kept
  // even when asked to go.
  offerToDeleteKeptSandboxes(kept);
}

export async function moveChatItemToProject(
  item: SidebarItem,
  projectId: string | null,
): Promise<void> {
  await moveStoredChatItemToProject(item, projectId);
}

/**
 * The project a chat is scoped to, including one the project list does not carry.
 *
 * `loadProjects` asks for non-archived projects only, but a chat opened from the
 * archived view keeps its project scope, so its row is never in that list. A caller
 * that reads "missing" as "still loading" then waits for a row that cannot arrive.
 * Archived projects are fetched one at a time instead of widening the shared list,
 * which is what the sidebar and the projects page render.
 */
export function useScopedChatProject(projectId: string | null | undefined): {
  project: ProjectRecord | undefined;
  isResolving: boolean;
} {
  const { projects, hasLoaded } = useChatProjects();
  const listed = projectId
    ? projects.find((candidate) => candidate.id === projectId)
    : undefined;
  // Keyed by the id it was read for, so a scope change is spotted by comparison
  // rather than by clearing state from inside the effect.
  const [fetched, setFetched] = useState<{
    id: string;
    project: ProjectRecord | null;
  } | null>(null);
  const resolvedFor = fetched?.id === projectId ? fetched : null;
  const alreadyRead = resolvedFor !== null;

  useEffect(() => {
    if (!projectId || listed || !hasLoaded || alreadyRead) return;
    let cancelled = false;
    void (async () => {
      try {
        const project = await getStoredChatProject(projectId);
        if (!cancelled) setFetched({ id: projectId, project: project ?? null });
      } catch (error) {
        // Same contract as the list: a project that cannot be read is reported as
        // absent rather than left resolving for ever.
        console.debug("Could not read the scoped project", error);
        if (!cancelled) setFetched({ id: projectId, project: null });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectId, listed, hasLoaded, alreadyRead]);

  if (!projectId) return { project: undefined, isResolving: false };
  if (listed) return { project: listed, isResolving: false };
  return {
    project: resolvedFor?.project ?? undefined,
    isResolving: !hasLoaded || resolvedFor === null,
  };
}
