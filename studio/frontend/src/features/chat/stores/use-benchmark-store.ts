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
  /** Callback set by chat-page so other components can trigger a multi-prompt benchmark run */
  benchmarkSendFn: ((texts: string[]) => void) | null;
  /** Text to pre-fill in the composer (set by Prompt Storage when loading a single prompt) */
  pendingComposerText: string | null;

  toggleBenchmarkMode: () => void;
  disableBenchmarkMode: () => void;
  setBenchmarkName: (name: string) => void;
  toggleModelSelection: (modelId: string) => void;
  clearModelSelection: () => void;
  setActiveBenchmark: (id: string, threadIds: Record<string, string>) => void;
  clearActiveBenchmark: () => void;
  setBenchmarkSendFn: (fn: ((texts: string[]) => void) | null) => void;
  setPendingComposerText: (text: string | null) => void;
}

export const useBenchmarkStore = create<BenchmarkStore>((set, get) => ({
  benchmarkMode: false,
  benchmarkName: `Benchmark ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
  benchmarkSelectedModelIds: [],
  activeBenchmarkId: null,
  activeBenchmarkThreadIds: {},
  benchmarkSendFn: null,
  pendingComposerText: null,

  toggleBenchmarkMode: () => {
    const next = !get().benchmarkMode;
    set({
      benchmarkMode: next,
      // Reset name and selection when entering a fresh benchmark session
      benchmarkName: next
        ? `Benchmark ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
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
  setPendingComposerText: (text) => set({ pendingComposerText: text }),
}));
