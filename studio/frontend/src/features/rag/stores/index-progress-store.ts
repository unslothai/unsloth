// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** Tracks every document in the current upload batch so the toast can show
 *  ONE aggregate "Indexing documents" entry (with overall %) instead of a
 *  separate toast per file. Unlike the per-job rag-store map, entries are
 *  registered at addDoc time, so queued-but-not-yet-started files (held by
 *  the concurrency semaphore) are counted in the denominator too. */

export type IndexEntryStatus = "queued" | "indexing" | "ready" | "error";

export interface IndexEntry {
  filename: string;
  status: IndexEntryStatus;
  /** 0..1; only meaningful while `indexing`. */
  progress: number;
}

interface IndexProgressState {
  entries: Record<string, IndexEntry>;
  add: (id: string, filename: string) => void;
  setIndexing: (id: string) => void;
  setProgress: (id: string, progress: number) => void;
  setReady: (id: string) => void;
  setError: (id: string) => void;
  clear: () => void;
}

function patch(
  set: (fn: (s: IndexProgressState) => Partial<IndexProgressState>) => void,
  id: string,
  changes: Partial<IndexEntry>,
): void {
  set((s) => {
    const existing = s.entries[id];
    if (!existing) return s;
    return { entries: { ...s.entries, [id]: { ...existing, ...changes } } };
  });
}

export const useIndexProgressStore = create<IndexProgressState>((set) => ({
  entries: {},
  add: (id, filename) =>
    set((s) => ({
      entries: {
        ...s.entries,
        [id]: { filename, status: "queued", progress: 0 },
      },
    })),
  setIndexing: (id) => patch(set, id, { status: "indexing" }),
  setProgress: (id, progress) =>
    patch(set, id, { status: "indexing", progress }),
  setReady: (id) => patch(set, id, { status: "ready", progress: 1 }),
  setError: (id) => patch(set, id, { status: "error" }),
  clear: () => set({ entries: {} }),
}));
