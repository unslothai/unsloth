// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { accountDatabaseName } from "@/lib/account-transition";
import { MAX_TITLE_CHARS, MAX_URL_CHARS } from "./history-store";

export type BookmarkFolder = "toolbar" | "other";
export type Bookmark = {
  id: string;
  url: string;
  title: string;
  folder: BookmarkFolder;
  addedAt: number;
  /** The page's icon when last seen, as a small PNG data: URL. */
  icon?: string;
};

export const MAX_BOOKMARKS = 2000;
// A 32px PNG is a few KB; anything far bigger isn't one, and would eat into Studio's storage.
const MAX_ICON_CHARS = 24 * 1024;

let nextId = 0;
const newId = () => `${Date.now().toString(36)}-${(nextId++).toString(36)}`;

interface BrowserBookmarksState {
  bookmarks: Bookmark[];
  lastFolder: BookmarkFolder;
  /** Saves `url`, or returns the bookmark it already has; null for an address too long to keep. */
  addBookmark: (url: string, title: string, folder?: BookmarkFolder) => Bookmark | null;
  updateBookmark: (id: string, patch: { title?: string; folder?: BookmarkFolder }) => void;
  removeBookmark: (id: string) => void;
  setBookmarkIcon: (id: string, icon: string) => void;
  restoreBookmark: (bookmark: Bookmark, index: number) => void;
  /** Saves many at once (an import), skipping saved addresses; how many it added, and how many
   *  new ones it left out at the bookmark limit. */
  importBookmarks: (
    items: { url: string; title: string; folder: BookmarkFolder; addedAt?: number }[],
  ) => { added: number; leftOut: number };
}

export const useBrowserBookmarksStore = create<BrowserBookmarksState>()(
  persist(
    (set, get) => ({
      bookmarks: [],
      lastFolder: "toolbar",
      addBookmark: (url, title, folder) => {
        const existing = get().bookmarks.find((bookmark) => bookmark.url === url);
        if (existing) return existing;
        if (url.length > MAX_URL_CHARS || get().bookmarks.length >= MAX_BOOKMARKS) return null;
        const bookmark: Bookmark = {
          id: newId(),
          url,
          title: title.trim().slice(0, MAX_TITLE_CHARS),
          folder: folder ?? get().lastFolder,
          addedAt: Date.now(),
        };
        set((state) => ({ bookmarks: [...state.bookmarks, bookmark] }));
        return bookmark;
      },
      updateBookmark: (id, patch) =>
        set((state) => ({
          bookmarks: state.bookmarks.map((bookmark) =>
            bookmark.id === id
              ? {
                  ...bookmark,
                  ...(patch.title !== undefined ? { title: patch.title.trim().slice(0, MAX_TITLE_CHARS) } : {}),
                  ...(patch.folder ? { folder: patch.folder } : {}),
                }
              : bookmark,
          ),
          lastFolder: patch.folder ?? state.lastFolder,
        })),
      setBookmarkIcon: (id, icon) =>
        set((state) => {
          if (!icon.startsWith("data:image/") || icon.length > MAX_ICON_CHARS) return state;
          const current = state.bookmarks.find((bookmark) => bookmark.id === id);
          if (!current || current.icon === icon) return state;
          return { bookmarks: state.bookmarks.map((bookmark) => (bookmark.id === id ? { ...bookmark, icon } : bookmark)) };
        }),
      removeBookmark: (id) => set((state) => ({ bookmarks: state.bookmarks.filter((bookmark) => bookmark.id !== id) })),
      importBookmarks: (items) => {
        const saved = new Set(get().bookmarks.map((bookmark) => bookmark.url));
        const added: Bookmark[] = [];
        let leftOut = 0;
        for (const item of items) {
          if (item.url.length > MAX_URL_CHARS || saved.has(item.url)) continue;
          saved.add(item.url);
          if (get().bookmarks.length + added.length >= MAX_BOOKMARKS) {
            leftOut++;
            continue;
          }
          added.push({
            id: newId(),
            url: item.url,
            title: item.title.trim().slice(0, MAX_TITLE_CHARS),
            folder: item.folder,
            addedAt: item.addedAt ?? Date.now(),
          });
        }
        // One write instead of one per bookmark.
        if (added.length > 0) set((state) => ({ bookmarks: [...state.bookmarks, ...added] }));
        return { added: added.length, leftOut };
      },
      restoreBookmark: (bookmark, index) =>
        set((state) => {
          if (state.bookmarks.some((other) => other.id === bookmark.id || other.url === bookmark.url)) return state;
          const bookmarks = [...state.bookmarks];
          bookmarks.splice(Math.min(index, bookmarks.length), 0, bookmark);
          return { bookmarks };
        }),
    }),
    {
      name: accountDatabaseName("unsloth_browser_bookmarks"),
      version: 1,
      storage: createJSONStorage(() => localStorage),
    },
  ),
);

export function useBookmarkFor(url: string | null): Bookmark | undefined {
  return useBrowserBookmarksStore((state) =>
    url ? state.bookmarks.find((bookmark) => bookmark.url === url) : undefined,
  );
}
