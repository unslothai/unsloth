// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT, getAuthSessionEpoch } from "@/features/auth";
import { deleteFineTunedModel } from "@/features/chat";
import { translate } from "@/i18n";
import { type GalleryKind, notifyGalleryChanged } from "@/lib/gallery-flags";
import {
  type LibraryFolder,
  type LibraryItem,
  type LibraryUploadBatch,
  createLibraryFolder,
  deleteLibraryFolder,
  deleteLibraryItem,
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

/** The gallery page behind each generated item's id prefix. */
const GALLERIES: Record<string, GalleryKind> = { image: "images", video: "videos", audio: "audio" };

// Bumped by every refresh and by sign-out, so only the newest request may commit its snapshot.
let refreshGeneration = 0;

// Edits apply locally first so menus feel instant, and roll back to the server's view on failure.
export const useLibraryStore = create<LibraryState>((set, get) => {
  // Edits apply here before the server has them, so a snapshot fetched while one is in flight, or
  // started before one, can predate it and would undo it on screen. Such a snapshot is dropped and
  // fetched again once every edit has settled; a failed edit rolls back the same way.
  let inFlight = 0;
  let edits = 0;
  let stale = false;
  async function optimistic(
    apply: (state: LibraryState) => Partial<LibraryState>,
    request: () => Promise<void>,
  ): Promise<void> {
    set(apply(get()));
    edits += 1;
    inFlight += 1;
    try {
      await request();
    } catch (error) {
      stale = true;
      throw error;
    } finally {
      inFlight -= 1;
      if (inFlight === 0 && stale) {
        stale = false;
        await get().refresh();
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
        // A star toggled on Images or Video meanwhile is newer than this snapshot.
        const favorites = useLibraryFavoritesStore.getState().adopt(
          favoritesStart,
          new Set(items.filter((item) => item.favorite).map((item) => item.id)),
        );
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
          error: error instanceof Error ? error.message : String(error),
        });
      }
    },
    patchItem: (id, patch) => {
      // Settled with the request, before a failure's refresh reads the server's value back.
      const settle =
        patch.favorite !== undefined
          ? useLibraryFavoritesStore.getState().mark(id, patch.favorite)
          : undefined;
      return optimistic(
        (state) => ({
          items: state.items.map((item) =>
            item.id === id ? { ...item, ...patch } : item,
          ),
        }),
        () => updateLibraryItem(id, patch).finally(settle),
      );
    },
    removeItem: (id) => {
      const model = get().items.find((item) => item.id === id)?.model;
      const settle = useLibraryFavoritesStore.getState().mark(id, false);
      return optimistic(
        (state) => ({ items: state.items.filter((item) => item.id !== id) }),
        // Fine-tunes go through the models route, which refuses while one is training or loaded,
        // and drops the Library's name, folder and star with the files.
        async () => {
          try {
            if (model) {
              await deleteFineTunedModel({
                modelPath: model.path,
                source: model.origin,
                exportType: model.exportType,
              });
              return;
            }
            await deleteLibraryItem(id);
          } finally {
            settle();
          }
          // Those pages stay mounted off-screen and would keep showing it.
          const gallery = GALLERIES[id.slice(0, id.indexOf(":"))];
          if (gallery) notifyGalleryChanged(gallery);
        },
      );
    },
    // Best effort: a lost open only leaves Last activity a little stale.
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
        // A large batch goes as several requests, so a failure can follow some that landed.
        await get().refresh();
      }
    },
    addFolder: async (name, parentId) => {
      const epoch = getAuthSessionEpoch();
      const folder = await createLibraryFolder(name, parentId);
      // Signed out meanwhile: it belongs to the account that left, not the one here now.
      if (getAuthSessionEpoch() !== epoch) {
        throw new Error(translate("library.toast.signedOutBeforeFolder"));
      }
      // A refresh started before it landed would drop it; the count sends that one back for more.
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
    // What the folder held moves up to its parent, mirroring the backend.
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
    useLibraryStore.setState({ items: [], folders: [], status: "idle", error: null });
    clearCachedObjectUrls();
  });
}

