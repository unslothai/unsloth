// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export interface Artifact {
  id: string;
  title: string;
  language: string | null;
  content: string;
  /** Version history, most recent last */
  history: string[];
  /** Index into history array for current view */
  activeVersion: number;
  createdAt: number;
}

interface ArtifactStore {
  artifacts: Artifact[];
  activeArtifactId: string | null;
  panelOpen: boolean;

  addArtifact: (artifact: Omit<Artifact, "history" | "activeVersion">) => void;
  updateArtifactContent: (id: string, content: string) => void;
  setActiveArtifact: (id: string | null) => void;
  setActiveVersion: (id: string, version: number) => void;
  removeArtifact: (id: string) => void;
  setPanelOpen: (open: boolean) => void;
  clearArtifacts: () => void;
}

export const useArtifactStore = create<ArtifactStore>((set, get) => ({
  artifacts: [],
  activeArtifactId: null,
  panelOpen: false,

  addArtifact: (artifact) => {
    const existing = get().artifacts.find((a) => a.id === artifact.id);
    if (existing) return;
    set((state) => ({
      artifacts: [
        ...state.artifacts,
        {
          ...artifact,
          history: [artifact.content],
          activeVersion: 0,
        },
      ],
      activeArtifactId: artifact.id,
      panelOpen: true,
    }));
  },

  updateArtifactContent: (id, content) => {
    set((state) => ({
      artifacts: state.artifacts.map((a) =>
        a.id === id
          ? {
              ...a,
              content,
              history: [...a.history, content],
              activeVersion: a.history.length,
            }
          : a,
      ),
    }));
  },

  setActiveArtifact: (id) => {
    set({ activeArtifactId: id, panelOpen: id !== null });
  },

  setActiveVersion: (id, version) => {
    set((state) => ({
      artifacts: state.artifacts.map((a) =>
        a.id === id
          ? { ...a, activeVersion: version, content: a.history[version] ?? a.content }
          : a,
      ),
    }));
  },

  removeArtifact: (id) => {
    set((state) => {
      const filtered = state.artifacts.filter((a) => a.id !== id);
      return {
        artifacts: filtered,
        activeArtifactId:
          state.activeArtifactId === id
            ? filtered[0]?.id ?? null
            : state.activeArtifactId,
        panelOpen: filtered.length > 0,
      };
    });
  },

  setPanelOpen: (open) => set({ panelOpen: open }),

  clearArtifacts: () =>
    set({ artifacts: [], activeArtifactId: null, panelOpen: false }),
}));
