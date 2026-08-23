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
  /**
   * Run a write and commit its answer. Returns false when a later write
   * started first, so the caller leaves the field to that one.
   */
  save: (request: () => Promise<EmbeddingModelSettings>) => Promise<boolean>;
  load: () => Promise<void>;
}

// Both surfaces read on mount, so two reads overlap with no save between them
// to bump the revision. Only the newest may commit, or a slow one lands last
// and shows an error, or an older model, over what the newer one just read.
let latestLoad = 0;

// Each surface keeps its own pending flag, so the one the user switches to can
// write while the first write is still out. Request order is the best guess at
// which one the user meant, but not proof of what the backend ended on: the
// later one can fail verification, or persist first. So the newest answer wins
// the moment it lands, and once every overlapping write has settled the store
// re-reads instead of trusting the guess.
let latestSave = 0;
let savesInFlight = 0;
let saveWasSuperseded = false;

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
    save: async (request) => {
      const save = ++latestSave;
      savesInFlight += 1;
      try {
        const settings = await request();
        if (save !== latestSave) {
          saveWasSuperseded = true;
          return false;
        }
        get().applySettings(settings);
        return true;
      } catch (error) {
        // A failed write leaves the backend on whatever the others wrote, so
        // an overlap has to be settled by a read, not by request order.
        if (save !== latestSave || savesInFlight > 1) saveWasSuperseded = true;
        throw error;
      } finally {
        savesInFlight -= 1;
        if (savesInFlight === 0 && saveWasSuperseded) {
          saveWasSuperseded = false;
          void get().load();
        }
      }
    },
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
