// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type EmbeddingModelSettings,
  loadEmbeddingModelSettings,
} from "../api/embedding-model";

/** Shared by General and Data so a save on one tab cannot leave the other showing the old model. */
interface EmbeddingModelState {
  settings: EmbeddingModelSettings | null;
  loadError: string | null;
  /** Bumped by every committed mutation, so a slower read cannot undo it. */
  revision: number;
  applySettings: (settings: EmbeddingModelSettings) => void;
  beginSave: () => number;
  isSaveCurrent: (reservation: number) => boolean;
  /** False when a later write started first. */
  save: (
    request: () => Promise<EmbeddingModelSettings>,
    reservation?: number,
  ) => Promise<boolean>;
  /** Takes no save-order slot, or it would retire another surface's in-flight selection. */
  applyResidency: (
    request: () => Promise<EmbeddingModelSettings>,
  ) => Promise<void>;
  load: () => Promise<void>;
}

// Both surfaces read on mount; only the newest read may commit.
let latestLoad = 0;

// Request order is only a guess at the backend's final state: newest answer wins, then the store
// re-reads once every overlapping write has settled.
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
    beginSave: () => ++latestSave,
    isSaveCurrent: (reservation) => reservation === latestSave,
    save: async (request, reservation) => {
      const save = reservation ?? ++latestSave;
      // An older selection must not become the newest write because its preflight finished last.
      if (save !== latestSave) return false;
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
        // After a failed overlapped write, settle by a read, not by request order.
        if (save !== latestSave) {
          saveWasSuperseded = true;
          return false;
        }
        if (savesInFlight > 1) saveWasSuperseded = true;
        throw error;
      } finally {
        savesInFlight -= 1;
        if (savesInFlight === 0 && saveWasSuperseded) {
          saveWasSuperseded = false;
          void get().load();
        }
      }
    },
    applyResidency: async (request) => {
      savesInFlight += 1;
      try {
        const settings = await request();
        // A selection is still out; let the settling re-read carry the residency change.
        if (savesInFlight > 1) {
          saveWasSuperseded = true;
          return;
        }
        get().applySettings(settings);
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
