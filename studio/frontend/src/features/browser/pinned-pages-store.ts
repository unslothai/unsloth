// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The tab showing a pinned page carries its id (BrowserTab.pinnedId).

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { accountDatabaseName } from "@/lib/account-transition";
import { MAX_TITLE_CHARS, MAX_URL_CHARS } from "./history-store";

export type PinnedPage = {
  id: string;
  url: string;
  title: string;
  pinnedAt: number;
};

const MAX_PINNED_PAGES = 100;

let nextId = 0;
const newId = () => `pin-${Date.now().toString(36)}-${(nextId++).toString(36)}`;

interface PinnedPagesState {
  pages: PinnedPage[];
  /** Pins `url`, or returns the pin it already has; null for an address too long to keep. */
  pin: (url: string, title: string) => PinnedPage | null;
  unpin: (id: string) => void;
  rename: (id: string, title: string) => void;
}

export const usePinnedPagesStore = create<PinnedPagesState>()(
  persist(
    (set, get) => ({
      pages: [],
      pin: (url, title) => {
        const existing = get().pages.find((page) => page.url === url);
        if (existing) return existing;
        if (!/^https?:\/\//i.test(url) || url.length > MAX_URL_CHARS || get().pages.length >= MAX_PINNED_PAGES) {
          return null;
        }
        const page: PinnedPage = {
          id: newId(),
          url,
          title: title.trim().slice(0, MAX_TITLE_CHARS),
          pinnedAt: Date.now(),
        };
        set((state) => ({ pages: [...state.pages, page] }));
        return page;
      },
      unpin: (id) => set((state) => ({ pages: state.pages.filter((page) => page.id !== id) })),
      rename: (id, title) => {
        const trimmed = title.trim().slice(0, MAX_TITLE_CHARS);
        if (!trimmed) return;
        set((state) => ({ pages: state.pages.map((page) => (page.id === id ? { ...page, title: trimmed } : page)) }));
      },
    }),
    {
      name: accountDatabaseName("unsloth_browser_pinned_pages"),
      version: 1,
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
