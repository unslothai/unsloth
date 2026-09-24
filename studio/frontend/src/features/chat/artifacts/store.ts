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
  // Bumped on every open, including reopening the one already selected, which is
  // otherwise invisible to anything watching the selected ID.
  openSequence: number;
  surface: ChatArtifactSurface;
  // View the surface should show on the next open (Preview vs Code button), and the one
  // it is showing now once it is open.
  requestedView: ArtifactViewMode;
  openArtifact: (
    artifact: ChatArtifact,
    options?: { surface?: ChatArtifactSurface; view?: ArtifactViewMode },
  ) => void;
  // The open surface switched views from its own header. Written back so the card that
  // opened it can still tell "already on screen" from "switch to the other one".
  setArtifactView: (view: ArtifactViewMode) => void;
  // Text the canvas's Fix button wants in the composer, waiting for a component that has a
  // composer to reach. The fullscreen overlay renders outside the chat runtime, so it
  // cannot stage the text itself; it leaves it here and the thread picks it up. Never sent,
  // only typed in: the user reads it and presses send.
  pendingFixPrompt: string | null;
  stageFixPrompt: (prompt: string) => void;
  clearFixPrompt: () => void;
  updateArtifact: (artifact: ChatArtifact) => void;
  closeArtifactSurface: () => void;
  clearArtifactsForThread: (threadId: string | null | undefined) => void;
  clearOrphanedArtifacts: () => void;
  resetArtifacts: () => void;
};

export const useChatArtifactsStore = create<ChatArtifactsState>((set) => ({
  artifactsById: {},
  selectedArtifactId: null,
  openSequence: 0,
  surface: "panel",
  requestedView: "preview",
  openArtifact: (artifact, options) =>
    set((state) => ({
      artifactsById: {
        ...state.artifactsById,
        [artifact.id]: artifact,
      },
      selectedArtifactId: artifact.id,
      openSequence: state.openSequence + 1,
      surface: options?.surface ?? state.surface,
      requestedView: options?.view ?? "preview",
    })),
  setArtifactView: (view) => set({ requestedView: view }),
  pendingFixPrompt: null,
  stageFixPrompt: (prompt) => set({ pendingFixPrompt: prompt }),
  clearFixPrompt: () => set({ pendingFixPrompt: null }),
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
      pendingFixPrompt: null,
    }),
}));

export function useSelectedChatArtifact(): ChatArtifact | null {
  return useChatArtifactsStore((state) =>
    state.selectedArtifactId
      ? (state.artifactsById[state.selectedArtifactId] ?? null)
      : null,
  );
}
