// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// HTML artifacts open in the browser; this keeps the chat-side state.
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
  // Fix text waiting for a composer. Never sent.
  pendingFixPrompt: string | null;
  stageFixPrompt: (prompt: string) => void;
  clearFixPrompt: () => void;
  // Set once a React preview finds no Node, so every React card can say why.
  reactPreviewUnavailable: boolean;
  markReactPreviewUnavailable: () => void;
};

export const useChatArtifactsStore = create<ChatArtifactsState>((set) => ({
  pendingFixPrompt: null,
  stageFixPrompt: (prompt) => set({ pendingFixPrompt: prompt }),
  clearFixPrompt: () => set({ pendingFixPrompt: null }),
  reactPreviewUnavailable: false,
  markReactPreviewUnavailable: () => set({ reactPreviewUnavailable: true }),
}));
