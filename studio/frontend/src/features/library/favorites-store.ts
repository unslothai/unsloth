// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { getLibraryFavorites, updateLibraryItem } from "./api";

/** What `adopt` needs to tell stars toggled since from a snapshot being fetched. */
export interface FavoritesSnapshotStart {
  attempts: ReadonlyMap<string, number>;
  /** Saving as the fetch went out, so its answer may predate them even if they finish first. */
  pending: ReadonlySet<string>;
  session: number;
}

interface FavoritesState {
  ids: ReadonlySet<string>;
  load: () => Promise<void>;
  /** Call before fetching a snapshot of favorites. */
  begin: () => FavoritesSnapshotStart;
  /** Takes a snapshot fetched since `begin`, keeping stars toggled meanwhile. Returns the result. */
  adopt: (start: FavoritesSnapshotStart, loaded: ReadonlySet<string>) => ReadonlySet<string>;
  /** Local only: the Library page mirrors its own edits here, and calls the returned `settle` once
   *  its request has. */
  mark: (id: string, favorite: boolean) => () => void;
  setFavorite: (id: string, favorite: boolean) => Promise<void>;
}

const latestAttempt = new Map<string, number>();
// Toggles whose request has not settled: a snapshot can predate them even if fetched after.
const pending = new Map<string, number>();
// Bumped on sign-out, so a load started for the last account never lands for the next one.
let session = 0;

/** Library favorites by item id, for pages (Images, Video) that mark them without loading the
 *  whole Library. */
export const useLibraryFavoritesStore = create<FavoritesState>((set, get) => {
  /** Marks a request for `id` in flight; the returned function settles it, once. */
  function track(id: string): () => void {
    const trackedSession = session;
    pending.set(id, (pending.get(id) ?? 0) + 1);
    let settled = false;
    return () => {
      if (settled || trackedSession !== session) return;
      settled = true;
      const left = (pending.get(id) ?? 1) - 1;
      if (left > 0) pending.set(id, left);
      else pending.delete(id);
    };
  }

  function apply(id: string, favorite: boolean): void {
    const ids = new Set(get().ids);
    if (favorite) ids.add(id);
    else ids.delete(id);
    set({ ids });
  }

  return {
    ids: new Set(),
    load: async () => {
      const start = get().begin();
      try {
        get().adopt(start, new Set(await getLibraryFavorites()));
      } catch {
        // Favorites are a convenience here; the page works without them.
      }
    },
    begin: () => ({ attempts: new Map(latestAttempt), pending: new Set(pending.keys()), session }),
    adopt: (start, loaded) => {
      if (start.session !== session) return get().ids;
      // A star toggled while this was loading, or still being saved, is newer than the snapshot.
      const ids = get().ids;
      const next = new Set(loaded);
      for (const [id, attempt] of latestAttempt) {
        const settledBefore = !start.pending.has(id) && !pending.has(id);
        if (start.attempts.get(id) === attempt && settledBefore) continue;
        if (ids.has(id)) next.add(id);
        else next.delete(id);
      }
      set({ ids: next });
      return next;
    },
    mark: (id, favorite) => {
      // An attempt too, so a load already in flight keeps this mark rather than its older snapshot.
      latestAttempt.set(id, (latestAttempt.get(id) ?? 0) + 1);
      apply(id, favorite);
      return track(id);
    },
    setFavorite: async (id, favorite) => {
      const attempt = (latestAttempt.get(id) ?? 0) + 1;
      latestAttempt.set(id, attempt);
      apply(id, favorite);
      const settle = track(id);
      try {
        await updateLibraryItem(id, { favorite });
        settle();
        if (latestAttempt.get(id) !== attempt) return;
        toast.success(
          translate(
            favorite ? "library.toast.addedToFavorites" : "library.toast.removedFromFavorites",
          ),
        );
      } catch (error) {
        settle();
        // A newer toggle owns the star now.
        if (latestAttempt.get(id) !== attempt) return;
        apply(id, !favorite);
        toast.error(translate("library.toast.favoritesFailed"), {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    },
  };
});

// Module state, so a sign-out must drop it here: Images and Video load this without the Library.
if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    session += 1;
    latestAttempt.clear();
    pending.clear();
    useLibraryFavoritesStore.setState({ ids: new Set() });
  });
}

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
