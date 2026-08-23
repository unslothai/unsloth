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

// Both surfaces read on mount, so two reads overlap with no save between them
// to bump the revision. Only the newest may commit, or a slow one lands last
// and shows an error, or an older model, over what the newer one just read.
let latestLoad = 0;

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
      const load = ++latestLoad;
      const stale = () => get().revision !== revision || load !== latestLoad;
      try {
        const settings = await loadEmbeddingModelSettings();
        // A save committed, or a later read started, while this was in flight.
        if (stale()) return;
        set({ settings, loadError: null });
      } catch (error) {
        if (stale()) return;
        set({
          loadError: error instanceof Error ? error.message : "",
        });
      }
    },
  }),
);
