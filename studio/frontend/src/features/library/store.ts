// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { announceGalleryRemoval } from "@/lib/gallery-removal";
import {
  type LibraryFolder,
  type LibraryItem,
  type LibraryUploadBatch,
  createLibraryFolder,
  deleteLibraryFolder,
  deleteLibraryItem,
  getLibrary,
  updateLibraryFolder,
  updateLibraryItem,
  uploadLibraryFiles,
} from "./api";
import { useLibraryFavoritesStore } from "./favorites-store";
import { clearCachedObjectUrls } from "./hooks";

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
  upload: (batch: LibraryUploadBatch, folderId: string | null) => Promise<string[]>;
  addFolder: (name: string, parentId: string | null) => Promise<LibraryFolder>;
  patchFolder: (id: string, patch: FolderPatch) => Promise<void>;
  removeFolder: (id: string) => Promise<void>;
}

/** "image:abc" to ["image", "abc"]. */
function splitItemId(id: string): [string, string] {
  const at = id.indexOf(":");
  return [id.slice(0, at), id.slice(at + 1)];
}

// Bumped by every refresh and by sign-out, so only the newest request may commit its snapshot.
let refreshGeneration = 0;

// Edits apply locally first so menus feel instant, and roll back to the server's view on failure.
export const useLibraryStore = create<LibraryState>((set, get) => {
  async function optimistic(
    apply: (state: LibraryState) => Partial<LibraryState>,
    request: () => Promise<void>,
  ): Promise<void> {
    set(apply(get()));
    try {
      await request();
    } catch (error) {
      await get().refresh();
      throw error;
    }
  }

  return {
    items: [],
    folders: [],
    status: "idle",
    error: null,
    refresh: async () => {
      const generation = ++refreshGeneration;
      if (get().status === "idle") set({ status: "loading" });
      try {
        const { items, folders } = await getLibrary();
        if (generation !== refreshGeneration) return;
        set({ items, folders, status: "ready", error: null });
        useLibraryFavoritesStore.setState({
          ids: new Set(items.filter((item) => item.favorite).map((item) => item.id)),
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
      if (patch.favorite !== undefined) {
        useLibraryFavoritesStore.getState().mark(id, patch.favorite);
      }
      return optimistic(
        (state) => ({
          items: state.items.map((item) =>
            item.id === id ? { ...item, ...patch } : item,
          ),
        }),
        () => updateLibraryItem(id, patch),
      );
    },
    removeItem: (id) =>
      optimistic(
        (state) => ({ items: state.items.filter((item) => item.id !== id) }),
        async () => {
          await deleteLibraryItem(id);
          const [kind, ref] = splitItemId(id);
          if (kind === "image" || kind === "video" || kind === "audio") {
            announceGalleryRemoval(kind, ref);
          }
        },
      ),
    upload: async (batch, folderId) => {
      const ids = await uploadLibraryFiles(batch, folderId);
      await get().refresh();
      return ids;
    },
    addFolder: async (name, parentId) => {
      const folder = await createLibraryFolder(name, parentId);
      set((state) => ({ folders: [folder, ...state.folders] }));
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
    useLibraryFavoritesStore.setState({ ids: new Set() });
    clearCachedObjectUrls();
  });
}

export type LibraryView = "grid" | "list";

export const useLibraryViewStore = create<{
  view: LibraryView;
  setView: (view: LibraryView) => void;
}>()(
  persist(
    (set) => ({
      view: "grid",
      setView: (view) => set({ view }),
    }),
    { name: "unsloth_library_view" },
  ),
);
