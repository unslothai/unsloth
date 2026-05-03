// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export interface PromptEvalStore {
  /** Whether Prompt Eval mode is currently active */
  promptEvalMode: boolean;
  /** Name for the current prompt eval run */
  promptEvalName: string;
  /** Ordered list of model IDs the user has selected for this benchmark */
  promptEvalSelectedModelIds: string[];
  /** The benchmark ID that is currently running or was last active */
  activePromptEvalId: string | null;
  /** Maps modelId → threadId for the current prompt eval run */
  activePromptEvalThreadIds: Record<string, string>;
  /** Callback set by chat-page so other components can trigger a multi-prompt prompt eval run */
  promptEvalSendFn: ((texts: string[]) => void) | null;
  /** Text to pre-fill in the composer (set by Prompt Storage when loading a single prompt) */
  pendingComposerText: string | null;

  togglePromptEvalMode: () => void;
  disablePromptEvalMode: () => void;
  setPromptEvalName: (name: string) => void;
  toggleModelSelection: (modelId: string) => void;
  clearModelSelection: () => void;
  setActiveBenchmark: (id: string, threadIds: Record<string, string>) => void;
  clearActiveBenchmark: () => void;
  setPromptEvalSendFn: (fn: ((texts: string[]) => void) | null) => void;
  setPendingComposerText: (text: string | null) => void;
}

export const usePromptEvalStore = create<PromptEvalStore>((set, get) => ({
  promptEvalMode: false,
  promptEvalName: `Benchmark ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
  promptEvalSelectedModelIds: [],
  activePromptEvalId: null,
  activePromptEvalThreadIds: {},
  promptEvalSendFn: null,
  pendingComposerText: null,

  togglePromptEvalMode: () => {
    const next = !get().promptEvalMode;
    set({
      promptEvalMode: next,
      // Reset name and selection when entering a fresh prompt eval session
      promptEvalName: next
        ? `Benchmark ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
        : get().promptEvalName,
      promptEvalSelectedModelIds: next ? get().promptEvalSelectedModelIds : [],
      activePromptEvalId: next ? null : get().activePromptEvalId,
      activePromptEvalThreadIds: next ? {} : get().activePromptEvalThreadIds,
    });
  },

  disablePromptEvalMode: () =>
    set({
      promptEvalMode: false,
      promptEvalSelectedModelIds: [],
      activePromptEvalId: null,
      activePromptEvalThreadIds: {},
    }),

  setPromptEvalName: (name) => set({ promptEvalName: name }),

  toggleModelSelection: (modelId) => {
    const ids = get().promptEvalSelectedModelIds;
    set({
      promptEvalSelectedModelIds: ids.includes(modelId)
        ? ids.filter((id) => id !== modelId)
        : [...ids, modelId],
    });
  },

  clearModelSelection: () => set({ promptEvalSelectedModelIds: [] }),

  setActiveBenchmark: (id, threadIds) =>
    set({ activePromptEvalId: id, activePromptEvalThreadIds: threadIds }),

  clearActiveBenchmark: () =>
    set({ activePromptEvalId: null, activePromptEvalThreadIds: {} }),

  setPromptEvalSendFn: (fn) => set({ promptEvalSendFn: fn }),
  setPendingComposerText: (text) => set({ pendingComposerText: text }),
}));
