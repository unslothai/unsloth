// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Nested, per-project folders for organizing a project's chats. Persisted client-side, the
// same way sidebar custom sections are (see sidebar-organization-store.ts); no backend schema.
// A folder belongs to exactly one project and has a parent folder in the same project, or null
// for a top-level folder. A chat with no mapping, or one pointing at a folder that no longer
// exists in its project, sits at the project root.

import { create } from "zustand";
import { persist } from "zustand/middleware";

import { syncAcrossWindows } from "../../../lib/sync-across-windows.ts";

export const PROJECT_FOLDERS_STORAGE_KEY = "unsloth_project_folders";

export const FOLDER_NAME_MAX = 60;

export interface ProjectFolder {
  id: string;
  projectId: string;
  /** Parent folder in the same project, or null for a top-level folder. */
  parentId: string | null;
  name: string;
  createdAt?: number;
  modifiedAt?: number;
}

export interface ProjectFoldersState {
  folders: ProjectFolder[];
  /** chatId -> folderId. A missing or dangling entry means the chat is at the project root. */
  folderByChatId: Record<string, string>;
  createFolder: (
    projectId: string,
    parentId: string | null,
    name: string,
  ) => string | null;
  renameFolder: (folderId: string, name: string) => void;
  /** Deletes a folder and promotes its child folders and chats to the folder's parent. */
  deleteFolder: (folderId: string) => void;
  /** Reparents a folder; a no-op if the move would form a cycle or cross projects. */
  moveFolder: (folderId: string, parentId: string | null) => void;
  setChatFolder: (chatId: string, folderId: string | null) => void;
  setChatsFolder: (chatIds: string[], folderId: string | null) => void;
}

export function normalizeFolderName(name: string): string {
  return name.replace(/\s+/g, " ").trim().slice(0, FOLDER_NAME_MAX);
}

function newFolderId(): string {
  // randomUUID needs a secure context, which a LAN address over plain http is not.
  return typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Prototype-free map so chat ids like "constructor" do not read as filed. */
export function chatFolderMap(
  from?: Record<string, string>,
): Record<string, string> {
  const map = Object.create(null) as Record<string, string>;
  if (from) {
    for (const [chatId, folderId] of Object.entries(from)) {
      map[chatId] = folderId;
    }
  }
  return map;
}

function byId(folders: readonly ProjectFolder[]): Map<string, ProjectFolder> {
  return new Map(folders.map((folder) => [folder.id, folder]));
}

/** Top-level-first children of `parentId` within `projectId`, in stored order. */
export function childFolders(
  folders: readonly ProjectFolder[],
  projectId: string,
  parentId: string | null,
): ProjectFolder[] {
  return folders.filter(
    (folder) => folder.projectId === projectId && folder.parentId === parentId,
  );
}

/** Whether `nodeId` is `ancestorId` or sits anywhere inside its subtree. */
export function isDescendant(
  folders: readonly ProjectFolder[],
  ancestorId: string,
  nodeId: string | null,
): boolean {
  const lookup = byId(folders);
  let current = nodeId;
  const seen = new Set<string>(); // guards against an already-corrupt cycle in storage
  while (current) {
    if (current === ancestorId) {
      return true;
    }
    if (seen.has(current)) {
      return false;
    }
    seen.add(current);
    current = lookup.get(current)?.parentId ?? null;
  }
  return false;
}

/** A reparent is rejected when the target is the folder itself or one of its descendants. */
export function wouldCreateCycle(
  folders: readonly ProjectFolder[],
  folderId: string,
  targetParentId: string | null,
): boolean {
  if (targetParentId === null) {
    return false;
  }
  if (targetParentId === folderId) {
    return true;
  }
  return isDescendant(folders, folderId, targetParentId);
}

/** `folderId` plus every folder nested under it, within the same project. */
export function collectSubtreeIds(
  folders: readonly ProjectFolder[],
  folderId: string,
): Set<string> {
  const out = new Set<string>([folderId]);
  let added = true;
  while (added) {
    added = false;
    for (const folder of folders) {
      if (folder.parentId && out.has(folder.parentId) && !out.has(folder.id)) {
        out.add(folder.id);
        added = true;
      }
    }
  }
  return out;
}

/** Nesting depth of a folder (a top-level folder is 0). */
export function folderDepth(
  folders: readonly ProjectFolder[],
  folderId: string,
): number {
  const lookup = byId(folders);
  let depth = 0;
  let current = lookup.get(folderId)?.parentId ?? null;
  const seen = new Set<string>();
  while (current && !seen.has(current)) {
    seen.add(current);
    depth += 1;
    current = lookup.get(current)?.parentId ?? null;
  }
  return depth;
}

/**
 * The folder a chat is filed under within `projectId`, or null for the project root.
 * A mapping to a folder that is gone, or that belongs to another project, reads as root.
 */
export function resolveChatFolderId(
  folders: readonly ProjectFolder[],
  folderByChatId: Record<string, string>,
  chatId: string,
  projectId: string,
): string | null {
  const folderId = folderByChatId[chatId];
  if (!folderId) {
    return null;
  }
  const folder = byId(folders).get(folderId);
  return folder && folder.projectId === projectId ? folderId : null;
}

/** Chats filed directly under `folderId` (null = project root), preserving `chatIds` order. */
export function chatsInFolder(
  folders: readonly ProjectFolder[],
  folderByChatId: Record<string, string>,
  chatIds: readonly string[],
  projectId: string,
  folderId: string | null,
): string[] {
  return chatIds.filter(
    (chatId) =>
      resolveChatFolderId(folders, folderByChatId, chatId, projectId) ===
      folderId,
  );
}

function readTime(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** Validates one persisted folder, or null if it is malformed or a duplicate id. */
function parsePersistedFolder(
  raw: unknown,
  seen: ReadonlySet<string>,
): ProjectFolder | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const entry = raw as Partial<ProjectFolder>;
  if (typeof entry.id !== "string" || !entry.id || seen.has(entry.id)) {
    return null;
  }
  if (typeof entry.projectId !== "string" || !entry.projectId) {
    return null;
  }
  const name =
    typeof entry.name === "string" ? normalizeFolderName(entry.name) : "";
  if (!name) {
    return null;
  }
  return {
    id: entry.id,
    projectId: entry.projectId,
    parentId: typeof entry.parentId === "string" ? entry.parentId : null,
    name,
    ...(readTime(entry.createdAt) ? { createdAt: entry.createdAt } : {}),
    ...(readTime(entry.modifiedAt) ? { modifiedAt: entry.modifiedAt } : {}),
  };
}

export function mergePersistedProjectFolders(
  persisted: unknown,
  current: ProjectFoldersState,
): ProjectFoldersState {
  const saved = persisted as Partial<ProjectFoldersState> | undefined;
  const folders: ProjectFolder[] = [];
  const seen = new Set<string>();
  if (Array.isArray(saved?.folders)) {
    for (const raw of saved.folders as unknown[]) {
      const folder = parsePersistedFolder(raw, seen);
      if (!folder) {
        continue;
      }
      seen.add(folder.id);
      folders.push(folder);
    }
  }
  // Drop parent links whose target did not survive validation, so no folder dangles.
  for (const folder of folders) {
    if (folder.parentId && !seen.has(folder.parentId)) {
      folder.parentId = null;
    }
  }
  const folderByChatId = chatFolderMap();
  if (saved?.folderByChatId && typeof saved.folderByChatId === "object") {
    for (const [chatId, folderId] of Object.entries(saved.folderByChatId)) {
      if (typeof folderId === "string" && seen.has(folderId)) {
        folderByChatId[chatId] = folderId;
      }
    }
  }
  return { ...current, folders, folderByChatId };
}

export const useProjectFoldersStore = create<ProjectFoldersState>()(
  persist(
    (set) => ({
      folders: [],
      folderByChatId: chatFolderMap(),
      createFolder: (projectId, parentId, name) => {
        const clean = normalizeFolderName(name);
        if (!(clean && projectId)) {
          return null;
        }
        const id = newFolderId();
        let created = false;
        set((state) => {
          // A parent must exist in the same project, else the folder would dangle.
          if (parentId !== null) {
            const parent = state.folders.find(
              (folder) => folder.id === parentId,
            );
            if (!parent || parent.projectId !== projectId) {
              return state;
            }
          }
          created = true;
          const now = Date.now();
          return {
            folders: [
              ...state.folders,
              {
                id,
                projectId,
                parentId,
                name: clean,
                createdAt: now,
                modifiedAt: now,
              },
            ],
          };
        });
        return created ? id : null;
      },
      renameFolder: (folderId, name) => {
        const clean = normalizeFolderName(name);
        if (!clean) {
          return;
        }
        set((state) => ({
          folders: state.folders.map((folder) =>
            folder.id === folderId && folder.name !== clean
              ? { ...folder, name: clean, modifiedAt: Date.now() }
              : folder,
          ),
        }));
      },
      deleteFolder: (folderId) =>
        set((state) => {
          const target = state.folders.find((folder) => folder.id === folderId);
          if (!target) {
            return state;
          }
          const promoteTo = target.parentId; // children and chats move up one level
          const folders = state.folders
            .filter((folder) => folder.id !== folderId)
            .map((folder) =>
              folder.parentId === folderId
                ? { ...folder, parentId: promoteTo }
                : folder,
            );
          const folderByChatId = chatFolderMap();
          for (const [chatId, mapped] of Object.entries(state.folderByChatId)) {
            if (mapped !== folderId) {
              folderByChatId[chatId] = mapped;
            } else if (promoteTo !== null) {
              folderByChatId[chatId] = promoteTo;
            }
            // promoteTo === null: the chat returns to the project root (no mapping).
          }
          return { folders, folderByChatId };
        }),
      moveFolder: (folderId, parentId) =>
        set((state) => {
          const target = state.folders.find((folder) => folder.id === folderId);
          if (!target || target.parentId === parentId) {
            return state;
          }
          if (parentId !== null) {
            const parent = state.folders.find(
              (folder) => folder.id === parentId,
            );
            if (!parent || parent.projectId !== target.projectId) {
              return state;
            }
          }
          if (wouldCreateCycle(state.folders, folderId, parentId)) {
            return state;
          }
          return {
            folders: state.folders.map((folder) =>
              folder.id === folderId
                ? { ...folder, parentId, modifiedAt: Date.now() }
                : folder,
            ),
          };
        }),
      setChatFolder: (chatId, folderId) =>
        set((state) => {
          if (
            folderId &&
            !state.folders.some((folder) => folder.id === folderId)
          ) {
            return state;
          }
          const next = chatFolderMap(state.folderByChatId);
          if (folderId) {
            next[chatId] = folderId;
          } else {
            delete next[chatId];
          }
          return { folderByChatId: next };
        }),
      setChatsFolder: (chatIds, folderId) =>
        set((state) => {
          if (
            folderId &&
            !state.folders.some((folder) => folder.id === folderId)
          ) {
            return state;
          }
          const next = chatFolderMap(state.folderByChatId);
          for (const chatId of chatIds) {
            if (folderId) {
              next[chatId] = folderId;
            } else {
              delete next[chatId];
            }
          }
          return { folderByChatId: next };
        }),
    }),
    {
      name: PROJECT_FOLDERS_STORAGE_KEY,
      merge: mergePersistedProjectFolders,
    },
  ),
);

syncAcrossWindows(useProjectFoldersStore);
