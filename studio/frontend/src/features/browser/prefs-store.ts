// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { SearchEngineId } from "./address";

interface BrowserPrefsState {
  /** Open chat links in the browser panel instead of the default browser. */
  openLinksInBrowser: boolean;
  /** Open attached documents in the browser panel instead of the preview dialog. */
  openFilesInBrowser: boolean;
  searchEngine: SearchEngineId;
  /** Show the query and fragment in the address bar, not just the site and path. */
  showFullUrl: boolean;
  setOpenLinksInBrowser: (value: boolean) => void;
  setOpenFilesInBrowser: (value: boolean) => void;
  setSearchEngine: (value: SearchEngineId) => void;
  setShowFullUrl: (value: boolean) => void;
}

export const useBrowserPrefsStore = create<BrowserPrefsState>()(
  persist(
    (set) => ({
      openLinksInBrowser: false,
      openFilesInBrowser: true,
      searchEngine: "duckduckgo",
      showFullUrl: false,
      setOpenLinksInBrowser: (openLinksInBrowser) => set({ openLinksInBrowser }),
      setOpenFilesInBrowser: (openFilesInBrowser) => set({ openFilesInBrowser }),
      setSearchEngine: (searchEngine) => set({ searchEngine }),
      setShowFullUrl: (showFullUrl) => set({ showFullUrl }),
    }),
    {
      name: "unsloth_browser_prefs",
      // v0 defaulted links to the panel; reset to the default browser.
      version: 1,
      migrate: (persisted, version) => {
        const state = (persisted ?? {}) as Partial<BrowserPrefsState>;
        return version < 1 ? { ...state, openLinksInBrowser: false } : state;
      },
    },
  ),
);
