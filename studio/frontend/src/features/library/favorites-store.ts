// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { create } from "zustand";
import { toast } from "@/lib/toast";
import { getLibraryFavorites, updateLibraryItem } from "./api";

interface FavoritesState {
  ids: ReadonlySet<string>;
  load: () => Promise<void>;
  /** Local only: the Library page mirrors its own edits here. */
  mark: (id: string, favorite: boolean) => void;
  setFavorite: (id: string, favorite: boolean) => Promise<void>;
}

/** Library favorites by item id, for pages (Images, Video) that mark them without loading the
 *  whole Library. */
export const useLibraryFavoritesStore = create<FavoritesState>((set, get) => ({
  ids: new Set(),
  load: async () => {
    try {
      set({ ids: new Set(await getLibraryFavorites()) });
    } catch {
      // Favorites are a convenience here; the page works without them.
    }
  },
  mark: (id, favorite) => {
    const ids = new Set(get().ids);
    if (favorite) ids.add(id);
    else ids.delete(id);
    set({ ids });
  },
  setFavorite: async (id, favorite) => {
    get().mark(id, favorite);
    try {
      await updateLibraryItem(id, { favorite });
      toast.success(favorite ? "Added to Favorites" : "Removed from Favorites");
    } catch (error) {
      get().mark(id, !favorite);
      toast.error("Could not update favorites", {
        description: error instanceof Error ? error.message : String(error),
      });
    }
  },
}));

/** Loads favorites once the calling page mounts; ids are `<source>:<id>`, e.g. `image:abc`. */
export function useLibraryFavorites() {
  const ids = useLibraryFavoritesStore((s) => s.ids);
  const load = useLibraryFavoritesStore((s) => s.load);
  const setFavorite = useLibraryFavoritesStore((s) => s.setFavorite);
  useEffect(() => {
    void load();
  }, [load]);
  return {
    isFavorite: (id: string) => ids.has(id),
    toggleFavorite: (id: string) => void setFavorite(id, !ids.has(id)),
  };
}
