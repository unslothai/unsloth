// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// Where each forked chat's inherited history ends. Published on load: message rows render from
// a propless slot, so they cannot be handed the thread record. Missing here means no divider.
export interface ForkBoundaryState {
  /** Thread id -> that thread's own copy of the last inherited message. */
  boundaryByThreadId: Record<string, string>;
  setForkBoundary: (threadId: string, messageId: string | null) => void;
}

export const useForkBoundaryStore = create<ForkBoundaryState>()((set) => ({
  boundaryByThreadId: {},
  setForkBoundary: (threadId, messageId) =>
    set((state) => {
      const current = state.boundaryByThreadId[threadId];
      if (messageId === null) {
        if (current === undefined) return state;
        const next = { ...state.boundaryByThreadId };
        delete next[threadId];
        return { boundaryByThreadId: next };
      }
      if (current === messageId) return state;
      return {
        boundaryByThreadId: { ...state.boundaryByThreadId, [threadId]: messageId },
      };
    }),
}));

export function setForkBoundary(
  threadId: string,
  messageId: string | null | undefined,
): void {
  useForkBoundaryStore.getState().setForkBoundary(threadId, messageId ?? null);
}
