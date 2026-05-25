// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { ChatArtifact, ChatArtifactSurface } from "./types";

type ChatArtifactsState = {
  artifactsById: Record<string, ChatArtifact>;
  selectedArtifactId: string | null;
  surface: ChatArtifactSurface;
  openArtifact: (
    artifact: ChatArtifact,
    options?: { surface?: ChatArtifactSurface },
  ) => void;
  updateArtifact: (artifact: ChatArtifact) => void;
  closeArtifactSurface: () => void;
  clearArtifactsForThread: (threadId: string | null | undefined) => void;
  resetArtifacts: () => void;
};

export const useChatArtifactsStore = create<ChatArtifactsState>((set) => ({
  artifactsById: {},
  selectedArtifactId: null,
  surface: "panel",
  openArtifact: (artifact, options) =>
    set((state) => ({
      artifactsById: {
        [artifact.id]: artifact,
      },
      selectedArtifactId: artifact.id,
      surface: options?.surface ?? state.surface,
    })),
  updateArtifact: (artifact) =>
    set((state) =>
      state.artifactsById[artifact.id]
        ? { artifactsById: { [artifact.id]: artifact } }
        : state,
    ),
  closeArtifactSurface: () =>
    set({ artifactsById: {}, selectedArtifactId: null, surface: "panel" }),
  clearArtifactsForThread: (threadId) =>
    set((state) => {
      if (!threadId) return state;
      const artifactsById = Object.fromEntries(
        Object.entries(state.artifactsById).filter(
          ([, artifact]) => artifact.threadId !== threadId,
        ),
      );
      const selected = state.selectedArtifactId
        ? artifactsById[state.selectedArtifactId]
        : null;
      return {
        artifactsById,
        selectedArtifactId: selected ? selected.id : null,
      };
    }),
  resetArtifacts: () =>
    set({
      artifactsById: {},
      selectedArtifactId: null,
      surface: "panel",
    }),
}));

export function useSelectedChatArtifact(): ChatArtifact | null {
  return useChatArtifactsStore((state) =>
    state.selectedArtifactId
      ? (state.artifactsById[state.selectedArtifactId] ?? null)
      : null,
  );
}
