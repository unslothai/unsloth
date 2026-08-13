// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  getAuthSessionEpoch,
} from "@/features/auth/session";
import { useEffect, useState, useSyncExternalStore } from "react";
import { CHAT_PROJECTS_UPDATED_EVENT } from "../api/chat-api";
import type { ProjectRecord } from "../types";
import {
  createStoredChatProject,
  deleteStoredChatProject,
  isExpectedBackgroundChatStorageError,
  listStoredChatProjects,
  moveStoredChatItemToProject,
  updateStoredChatProject,
} from "../utils/chat-history-storage";
import { offerToDeleteKeptSandboxes } from "../utils/offer-kept-sandbox-files";
import type { SidebarItem } from "./use-chat-sidebar-items";

// Shared identity so useSyncExternalStore never sees a fresh array for the same empty state.
const NO_PROJECTS: ProjectRecord[] = [];

let cachedProjects: ProjectRecord[] = NO_PROJECTS;
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

// The cache lives at module scope and a web logout only navigates, so without this the next
// account renders the previous user's project names until its own request lands.
export function resetChatProjectsState(): void {
  cachedProjects = NO_PROJECTS;
  projectsLoaded = false;
  projectsRequest = null;
  projectsRefreshPending = false;
  lastProjectsUpdateEvent = null;
  for (const onStoreChange of projectSubscribers) onStoreChange();
}

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, resetChatProjectsState);
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
    const epoch = getAuthSessionEpoch();
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
    // A request that straddles a logout describes the account it started under.
    if (epoch !== getAuthSessionEpoch()) return NO_PROJECTS;
    if (nextProjects !== null) publishProjects(nextProjects);
    return cachedProjects;
  }

  const request = run().finally(() => {
    // A session reset clears this slot, so only the request that owns it may release it.
    if (projectsRequest === request) projectsRequest = null;
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
    // Cached rows render immediately, then one shared request reconciles
    // changes made by another browser tab or API client.
    void refresh(projectsLoaded);
    window.addEventListener(CHAT_PROJECTS_UPDATED_EVENT, onProjectsUpdated);
    return () => {
      cancelled = true;
      window.removeEventListener(CHAT_PROJECTS_UPDATED_EVENT, onProjectsUpdated);
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
  const kept = await deleteStoredChatProject(projectId, args);
  // The member chats went with the project, so their own sandboxes are
  // reachable from nothing: the same offer an ordinary chat delete makes, and
  // a sandbox the backend could not remove is kept even when asked to go.
  offerToDeleteKeptSandboxes(kept);
}

export async function moveChatItemToProject(
  item: SidebarItem,
  projectId: string | null,
): Promise<void> {
  await moveStoredChatItemToProject(item, projectId);
}
