// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT, getAuthSessionEpoch } from "@/features/auth";
import { deleteFineTunedModel, emitChatAttachmentDeleted } from "@/features/chat";
import { translate } from "@/i18n";
import { type GalleryKind, notifyGalleryChanged } from "@/lib/gallery-flags";
import {
  type LibraryFolder,
  type LibraryItem,
  type LibraryUploadBatch,
  createLibraryFolder,
  deleteLibraryFolder,
  deleteLibraryItem,
  errorMessage,
  getLibrary,
  markLibraryItemOpened,
  updateLibraryFolder,
  updateLibraryItem,
  uploadLibraryFiles,
} from "./api";
import { useLibraryFavoritesStore } from "./favorites-store";
import { clearCachedObjectUrls } from "./object-url-cache";

type ItemPatch = { name?: string; favorite?: boolean; folderId?: string | null };
type FolderPatch = { name?: string; parentId?: string | null };

interface LibraryState {
  items: LibraryItem[];
  folders: LibraryFolder[];
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  refresh: () => Promise<void>;
  patchItem: (id: string, patch: ItemPatch) => Promise<void>;
  removeItem: (id: string) => Promise<void>;
  markOpened: (id: string) => void;
  upload: (batch: LibraryUploadBatch, folderId: string | null) => Promise<string[]>;
  addFolder: (name: string, parentId: string | null) => Promise<LibraryFolder>;
  patchFolder: (id: string, patch: FolderPatch) => Promise<void>;
  removeFolder: (id: string) => Promise<void>;
}

const GALLERIES: Record<string, GalleryKind> = { image: "images", video: "videos", audio: "audio" };

let refreshGeneration = 0;
let snapshots = 0;

function undoEdit<T extends { id: string }>(now: T[], before: T[], after: T[] | undefined): T[] {
  if (!after) return now;
  const edited = new Map(after.map((entry) => [entry.id, entry]));
  const prior = new Map(
    before.filter((entry) => edited.get(entry.id) !== entry).map((entry) => [entry.id, entry]),
  );
  const restored = now.map((entry) => {
    const was = prior.get(entry.id);
    const made = edited.get(entry.id);
    if (!was || !made) return entry;
    let undone = entry;
    for (const key of new Set([...Object.keys(was), ...Object.keys(made)]) as Set<keyof T>) {
      if (made[key] !== was[key] && entry[key] === made[key]) {
        undone = { ...undone, [key]: was[key] };
      }
    }
    return undone;
  });
  const present = new Set(now.map((entry) => entry.id));
  for (const entry of prior.values()) {
    if (!edited.has(entry.id) && !present.has(entry.id)) {
      restored.splice(Math.min(before.indexOf(entry), restored.length), 0, entry);
    }
  }
  return restored;
}

// Edits apply here before the server has them, so a snapshot fetched while one is in flight, or
// started before one, can predate it and would undo it on screen. Such a snapshot is dropped and
// fetched again once every edit has settled; a failed edit rolls back the same way. A sign-out
// starts the count again: an edit of the account that left must not hold back the next one's.
let session = 0;
let inFlight = 0;
let edits = 0;
let stale = false;
let undos: (() => void)[] = [];

export const useLibraryStore = create<LibraryState>((set, get) => {
  async function optimistic(
    apply: (state: LibraryState) => Partial<LibraryState>,
    request: () => Promise<void>,
    undoElsewhere?: () => void,
  ): Promise<void> {
    const before = get();
    const after = apply(before);
    const epoch = getAuthSessionEpoch();
    const started = session;
    set(after);
    edits += 1;
    inFlight += 1;
    try {
      await request();
    } catch (error) {
      if (started === session) {
        stale = true;
        undos.push(() => {
          if (getAuthSessionEpoch() !== epoch) return;
          set((state) => ({
            items: undoEdit(state.items, before.items, after.items),
            folders: undoEdit(state.folders, before.folders, after.folders),
          }));
          undoElsewhere?.();
        });
      }
      throw error;
    } finally {
      if (started === session) inFlight -= 1;
      if (started === session && inFlight === 0 && stale) {
        stale = false;
        const pending = undos;
        undos = [];
        const landed = snapshots;
        await get().refresh();
        if (snapshots === landed) for (const undo of pending.reverse()) undo();
      }
    }
  }

  return {
    items: [],
    folders: [],
    status: "idle",
    error: null,
    refresh: async () => {
      const generation = ++refreshGeneration;
      const editsBefore = edits;
      const favoritesStart = useLibraryFavoritesStore.getState().begin();
      if (get().status === "idle") set({ status: "loading" });
      try {
        const { items, folders } = await getLibrary();
        if (generation !== refreshGeneration) return;
        if (inFlight > 0) {
          stale = true;
          return;
        }
        if (edits !== editsBefore) return get().refresh();
        const favorites = useLibraryFavoritesStore.getState().adopt(
          favoritesStart,
          new Set(items.filter((item) => item.favorite).map((item) => item.id)),
        );
        snapshots += 1;
        set({
          items: items.map((item) =>
            favorites.has(item.id) === item.favorite
              ? item
              : { ...item, favorite: !item.favorite },
          ),
          folders,
          status: "ready",
          error: null,
        });
      } catch (error) {
        if (generation !== refreshGeneration) return;
        set({
          status: get().status === "ready" ? "ready" : "error",
          error: errorMessage(error),
        });
      }
    },
    patchItem: (id, patch) => {
      const star =
        patch.favorite !== undefined
          ? useLibraryFavoritesStore.getState().mark(id, patch.favorite)
          : undefined;
      return optimistic(
        (state) => ({
          items: state.items.map((item) =>
            item.id === id ? { ...item, ...patch } : item,
          ),
        }),
        () => updateLibraryItem(id, patch).finally(star?.settle),
        star?.undo,
      );
    },
    removeItem: (id) => {
      const listed = get().items.find((item) => item.id === id);
      const model = listed?.model;
      const fingerprint = listed?.fingerprint;
      const star = useLibraryFavoritesStore.getState().mark(id, false);
      return optimistic(
        (state) => ({ items: state.items.filter((item) => item.id !== id) }),
        async () => {
          const epoch = getAuthSessionEpoch();
          try {
            if (model) {
              await deleteFineTunedModel({
                modelPath: model.path,
                source: model.origin,
                exportType: model.exportType,
              });
              return;
            }
            await deleteLibraryItem(id, fingerprint);
          } finally {
            star.settle();
          }
          // Signed in as another account meanwhile: its chats can hold an attachment of this id.
          if (getAuthSessionEpoch() !== epoch) return;
          const [kind, messageId, ...rest] = id.split(":");
          const gallery = GALLERIES[kind];
          if (gallery) notifyGalleryChanged(gallery);
          if (kind === "attachment" && messageId && rest.length > 0) {
            emitChatAttachmentDeleted({
              messageId: decodeURIComponent(messageId),
              attachmentId: rest.join(":"),
            });
          }
        },
        star.undo,
      );
    },
    markOpened: (id) => {
      const openedAt = Date.now();
      set((state) => ({
        items: state.items.map((item) => (item.id === id ? { ...item, openedAt } : item)),
      }));
      void markLibraryItemOpened(id).catch(() => undefined);
    },
    upload: async (batch, folderId) => {
      try {
        return await uploadLibraryFiles(batch, folderId);
      } finally {
        await get().refresh();
      }
    },
    addFolder: async (name, parentId) => {
      const epoch = getAuthSessionEpoch();
      const folder = await createLibraryFolder(name, parentId);
      if (getAuthSessionEpoch() !== epoch) {
        throw new Error(translate("library.toast.signedOutBeforeFolder"));
      }
      edits += 1;
      set((state) => ({
        folders: [folder, ...state.folders.filter((f) => f.id !== folder.id)],
      }));
      return folder;
    },
    patchFolder: (id, patch) =>
      optimistic(
        (state) => ({
          folders: state.folders.map((folder) =>
            folder.id === id
              ? { ...folder, ...patch, updatedAt: Date.now() }
              : folder,
          ),
        }),
        () => updateLibraryFolder(id, patch),
      ),
    removeFolder: (id) =>
      optimistic(
        (state) => {
          const parentId =
            state.folders.find((folder) => folder.id === id)?.parentId ?? null;
          return {
            folders: state.folders
              .filter((folder) => folder.id !== id)
              .map((folder) =>
                folder.parentId === id ? { ...folder, parentId } : folder,
              ),
            items: state.items.map((item) =>
              item.folderId === id ? { ...item, folderId: parentId } : item,
            ),
          };
        },
        () => deleteLibraryFolder(id),
      ),
  };
});

// The store is module state, so a sign-out must drop it or the next account sees these files.
if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    refreshGeneration += 1;
    session += 1;
    inFlight = 0;
    stale = false;
    undos = [];
    useLibraryStore.setState({ items: [], folders: [], status: "idle", error: null });
    clearCachedObjectUrls();
  });
}

