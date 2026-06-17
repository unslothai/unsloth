// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { ArtifactViewMode } from "./html-frame";
import type { ChatArtifact, ChatArtifactSurface } from "./types";

const autoOpenedArtifactIds = new Set<string>();

export function hasAutoOpenedArtifact(artifactId: string): boolean {
  return autoOpenedArtifactIds.has(artifactId);
}

export function rememberAutoOpenedArtifact(artifactId: string): void {
  autoOpenedArtifactIds.add(artifactId);
}

export function clearAutoOpenedArtifacts(): void {
  autoOpenedArtifactIds.clear();
}

type ChatArtifactsState = {
  artifactsById: Record<string, ChatArtifact>;
  selectedArtifactId: string | null;
  surface: ChatArtifactSurface;
  // View the surface should show on the next open (Preview vs Code button).
  requestedView: ArtifactViewMode;
  openArtifact: (
    artifact: ChatArtifact,
    options?: { surface?: ChatArtifactSurface; view?: ArtifactViewMode },
  ) => void;
  updateArtifact: (artifact: ChatArtifact) => void;
  closeArtifactSurface: () => void;
  clearArtifactsForThread: (threadId: string | null | undefined) => void;
  clearOrphanedArtifacts: () => void;
  resetArtifacts: () => void;
};

export const useChatArtifactsStore = create<ChatArtifactsState>((set) => ({
  artifactsById: {},
  selectedArtifactId: null,
  surface: "panel",
  requestedView: "preview",
  openArtifact: (artifact, options) =>
    set((state) => ({
      artifactsById: {
        ...state.artifactsById,
        [artifact.id]: artifact,
      },
      selectedArtifactId: artifact.id,
      surface: options?.surface ?? state.surface,
      requestedView: options?.view ?? "preview",
    })),
  updateArtifact: (artifact) =>
    set((state) =>
      state.artifactsById[artifact.id]
        ? {
            artifactsById: {
              ...state.artifactsById,
              [artifact.id]: artifact,
            },
          }
        : state,
    ),
  closeArtifactSurface: () =>
    set({ selectedArtifactId: null, surface: "panel" }),
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
  clearOrphanedArtifacts: () =>
    set((state) => {
      const artifactsById = Object.fromEntries(
        Object.entries(state.artifactsById).filter(
          ([, artifact]) => artifact.threadId != null,
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
