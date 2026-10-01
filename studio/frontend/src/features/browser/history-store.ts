// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { type StateStorage, createJSONStorage, persist } from "zustand/middleware";

export type HistoryItem = { id: string; url: string; title: string; visitedAt: number };
export type DownloadItem = {
  id: string;
  name: string;
  /** Where it came from; null for a chat attachment. */
  url: string | null;
  size: number;
  contentType: string;
  downloadedAt: number;
};

const MAX_HISTORY = 1000;
const MAX_DOWNLOADS = 200;
// Pages pick their URLs and titles; cap them so history can't fill Studio's storage.
const MAX_URL_CHARS = 2048;
const MAX_TITLE_CHARS = 200;
const PERSIST_DELAY_MS = 1000;

/** localStorage with batched writes, since history is one big JSON value; a full storage is ignored. */
function deferredLocalStorage(): StateStorage {
  const pending = new Map<string, string>();
  let timer: ReturnType<typeof setTimeout> | null = null;
  const flush = () => {
    timer = null;
    for (const [name, value] of pending) {
      try {
        localStorage.setItem(name, value);
      } catch {
        // Quota or private mode: keep the in-memory history.
      }
    }
    pending.clear();
  };
  if (typeof window !== "undefined") window.addEventListener("pagehide", flush);
  return {
    getItem: (name) => pending.get(name) ?? localStorage.getItem(name),
    setItem: (name, value) => {
      pending.set(name, value);
      timer ??= setTimeout(flush, PERSIST_DELAY_MS);
    },
    removeItem: (name) => {
      pending.delete(name);
      localStorage.removeItem(name);
    },
  };
}

let nextId = 0;
const newId = () => `${Date.now().toString(36)}-${(nextId++).toString(36)}`;

interface BrowserHistoryState {
  history: HistoryItem[];
  downloads: DownloadItem[];
  recordVisit: (url: string, title: string) => void;
  recordDownload: (item: Omit<DownloadItem, "id" | "downloadedAt">) => void;
  removeVisit: (id: string) => void;
  removeVisits: (ids: ReadonlySet<string>) => void;
  removeDownload: (id: string) => void;
  clearHistory: () => void;
  clearDownloads: () => void;
}

/** Pages visited and files downloaded in the browser panel, newest first. Kept in this browser only. */
export const useBrowserHistoryStore = create<BrowserHistoryState>()(
  persist(
    (set) => ({
      history: [],
      downloads: [],
      recordVisit: (url, fullTitle) =>
        set((state) => {
          if (url.length > MAX_URL_CHARS) return state;
          const title = fullTitle.slice(0, MAX_TITLE_CHARS);
          const [latest, ...rest] = state.history;
          // A reload or title update of the same page is one visit.
          if (latest?.url === url) {
            return { history: [{ ...latest, title: title || latest.title, visitedAt: Date.now() }, ...rest] };
          }
          const item = { id: newId(), url, title, visitedAt: Date.now() };
          return { history: [item, ...state.history].slice(0, MAX_HISTORY) };
        }),
      recordDownload: (item) =>
        set((state) => ({
          downloads: [{ ...item, id: newId(), downloadedAt: Date.now() }, ...state.downloads].slice(0, MAX_DOWNLOADS),
        })),
      removeVisit: (id) => set((state) => ({ history: state.history.filter((item) => item.id !== id) })),
      removeVisits: (ids) => set((state) => ({ history: state.history.filter((item) => !ids.has(item.id)) })),
      removeDownload: (id) => set((state) => ({ downloads: state.downloads.filter((item) => item.id !== id) })),
      clearHistory: () => set({ history: [] }),
      clearDownloads: () => set({ downloads: [] }),
    }),
    {
      name: "unsloth_browser_history",
      version: 1,
      storage: createJSONStorage(deferredLocalStorage),
    },
  ),
);
