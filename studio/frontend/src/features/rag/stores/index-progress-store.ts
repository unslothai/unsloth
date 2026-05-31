// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** Tracks every document in the current upload batch so the toast shows ONE
 *  aggregate "Indexing documents" entry (overall %) instead of one per file.
 *  Unlike the per-job rag-store map, entries register at addDoc time, so
 *  queued files held by the concurrency semaphore count in the denominator. */

export type IndexEntryStatus = "queued" | "indexing" | "ready" | "error";

export interface IndexEntry {
  filename: string;
  status: IndexEntryStatus;
  /** 0..1; only meaningful while `indexing`. */
  progress: number;
  /** Chunks this file produced (from the job's complete event); 0 until done. */
  chunks: number;
  /** Tear down this upload and remove its document from the index. Registered
   *  by the upload surface so the toast can cancel the whole batch without
   *  owning the per-file job/SSE/semaphore handles. */
  cancel?: () => Promise<void> | void;
}

interface IndexProgressState {
  entries: Record<string, IndexEntry>;
  add: (id: string, filename: string) => void;
  setIndexing: (id: string) => void;
  setProgress: (id: string, progress: number) => void;
  setReady: (id: string, chunks?: number) => void;
  setError: (id: string) => void;
  setCancel: (id: string, cancel: () => Promise<void> | void) => void;
  cancelAll: () => Promise<void>;
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

export const useIndexProgressStore = create<IndexProgressState>((set, get) => ({
  entries: {},
  add: (id, filename) =>
    set((s) => ({
      entries: {
        ...s.entries,
        [id]: { filename, status: "queued", progress: 0, chunks: 0 },
      },
    })),
  setIndexing: (id) => patch(set, id, { status: "indexing" }),
  setProgress: (id, progress) =>
    patch(set, id, { status: "indexing", progress }),
  setReady: (id, chunks = 0) =>
    patch(set, id, { status: "ready", progress: 1, chunks }),
  setError: (id) => patch(set, id, { status: "error" }),
  setCancel: (id, cancel) => patch(set, id, { cancel }),
  // Cancel every file in the batch (running, queued, finished) to restore the
  // pre-batch index state, then drop all toast entries.
  cancelAll: async () => {
    const handles = Object.values(get().entries)
      .map((e) => e.cancel)
      .filter((c): c is NonNullable<typeof c> => Boolean(c));
    await Promise.allSettled(handles.map((c) => c()));
    set({ entries: {} });
  },
  clear: () => set({ entries: {} }),
}));
