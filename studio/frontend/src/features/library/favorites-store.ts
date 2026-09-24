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

const latestAttempt = new Map<string, number>();

/** Library favorites by item id, for pages (Images, Video) that mark them without loading the
 *  whole Library. */
export const useLibraryFavoritesStore = create<FavoritesState>((set, get) => {
  function apply(id: string, favorite: boolean): void {
    const ids = new Set(get().ids);
    if (favorite) ids.add(id);
    else ids.delete(id);
    set({ ids });
  }

  return {
    ids: new Set(),
    load: async () => {
      const touchedBefore = new Map(latestAttempt);
      try {
        const loaded = new Set(await getLibraryFavorites());
        // A star toggled while this was loading is newer than the snapshot; keep it.
        const ids = get().ids;
        for (const [id, attempt] of latestAttempt) {
          if (touchedBefore.get(id) === attempt) continue;
          if (ids.has(id)) loaded.add(id);
          else loaded.delete(id);
        }
        set({ ids: loaded });
      } catch {
        // Favorites are a convenience here; the page works without them.
      }
    },
    mark: (id, favorite) => {
      // An attempt too, so a load already in flight keeps this mark rather than its older snapshot.
      latestAttempt.set(id, (latestAttempt.get(id) ?? 0) + 1);
      apply(id, favorite);
    },
    setFavorite: async (id, favorite) => {
      const attempt = (latestAttempt.get(id) ?? 0) + 1;
      latestAttempt.set(id, attempt);
      apply(id, favorite);
      try {
        await updateLibraryItem(id, { favorite });
        if (latestAttempt.get(id) !== attempt) return;
        toast.success(
          favorite ? "Added to Favorites" : "Removed from Favorites",
        );
      } catch (error) {
        // A newer toggle owns the star now.
        if (latestAttempt.get(id) !== attempt) return;
        apply(id, !favorite);
        toast.error("Could not update favorites", {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
  };
});

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
