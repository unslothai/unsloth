// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export interface BenchmarkStore {
  /** Whether benchmark mode is currently active */
  benchmarkMode: boolean;
  /** Name for the current benchmark run */
  benchmarkName: string;
  /** Ordered list of model IDs the user has selected for this benchmark */
  benchmarkSelectedModelIds: string[];
  /** The benchmark ID that is currently running or was last active */
  activeBenchmarkId: string | null;
  /** Maps modelId → threadId for the current benchmark run */
  activeBenchmarkThreadIds: Record<string, string>;
  /** Callback set by chat-page so the thread composer can trigger a benchmark send */
  benchmarkSendFn: ((text: string) => void) | null;

  toggleBenchmarkMode: () => void;
  disableBenchmarkMode: () => void;
  setBenchmarkName: (name: string) => void;
  toggleModelSelection: (modelId: string) => void;
  clearModelSelection: () => void;
  setActiveBenchmark: (id: string, threadIds: Record<string, string>) => void;
  clearActiveBenchmark: () => void;
  setBenchmarkSendFn: (fn: ((text: string) => void) | null) => void;
}

export const useBenchmarkStore = create<BenchmarkStore>((set, get) => ({
  benchmarkMode: false,
  benchmarkName: `Benchmark ${new Date().toLocaleDateString()}`,
  benchmarkSelectedModelIds: [],
  activeBenchmarkId: null,
  activeBenchmarkThreadIds: {},
  benchmarkSendFn: null,

  toggleBenchmarkMode: () => {
    const next = !get().benchmarkMode;
    set({
      benchmarkMode: next,
      // Reset name and selection when entering a fresh benchmark session
      benchmarkName: next
        ? `Benchmark ${new Date().toLocaleDateString()}`
        : get().benchmarkName,
      benchmarkSelectedModelIds: next ? get().benchmarkSelectedModelIds : [],
      activeBenchmarkId: next ? null : get().activeBenchmarkId,
      activeBenchmarkThreadIds: next ? {} : get().activeBenchmarkThreadIds,
    });
  },

  disableBenchmarkMode: () =>
    set({
      benchmarkMode: false,
      benchmarkSelectedModelIds: [],
      activeBenchmarkId: null,
      activeBenchmarkThreadIds: {},
    }),

  setBenchmarkName: (name) => set({ benchmarkName: name }),

  toggleModelSelection: (modelId) => {
    const ids = get().benchmarkSelectedModelIds;
    set({
      benchmarkSelectedModelIds: ids.includes(modelId)
        ? ids.filter((id) => id !== modelId)
        : [...ids, modelId],
    });
  },

  clearModelSelection: () => set({ benchmarkSelectedModelIds: [] }),

  setActiveBenchmark: (id, threadIds) =>
    set({ activeBenchmarkId: id, activeBenchmarkThreadIds: threadIds }),

  clearActiveBenchmark: () =>
    set({ activeBenchmarkId: null, activeBenchmarkThreadIds: {} }),

  setBenchmarkSendFn: (fn) => set({ benchmarkSendFn: fn }),
}));
