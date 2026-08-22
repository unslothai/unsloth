// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type EmbeddingModelSettings,
  loadEmbeddingModelSettings,
} from "../api/embedding-model";

/**
 * One copy of the embedding model setting for both surfaces that edit it
 * (General and Data). Component state would let a save made on one tab finish
 * after the other has mounted and read the old value, leaving the tab on
 * screen showing a model the toast just said was replaced.
 */
interface EmbeddingModelState {
  settings: EmbeddingModelSettings | null;
  loadError: string | null;
  /** Bumped by every committed mutation, so a slower read cannot undo it. */
  revision: number;
  applySettings: (settings: EmbeddingModelSettings) => void;
  load: () => Promise<void>;
}

export const useEmbeddingModelStore = create<EmbeddingModelState>(
  (set, get) => ({
    settings: null,
    loadError: null,
    revision: 0,
    applySettings: (settings) =>
      set((state) => ({
        settings,
        loadError: null,
        revision: state.revision + 1,
      })),
    load: async () => {
      const revision = get().revision;
      try {
        const settings = await loadEmbeddingModelSettings();
        // A save committed while this read was in flight, so the read is stale.
        if (get().revision !== revision) return;
        set({ settings, loadError: null });
      } catch (error) {
        if (get().revision !== revision) return;
        set({
          loadError: error instanceof Error ? error.message : "",
        });
      }
    },
  }),
);
