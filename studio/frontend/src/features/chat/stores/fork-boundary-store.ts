// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** Where a fork's inherited history ends, and the chat it came from. */
export interface ForkBoundary {
  /** This thread's own copy of the last inherited message. */
  messageId: string;
  /** The chat the divider links back to, null once it is gone. */
  sourceThreadId: string | null;
}

// Published on load: message rows render from a propless slot, so they cannot be handed the
// thread record. A thread missing here shows no divider.
export interface ForkBoundaryState {
  boundaryByThreadId: Record<string, ForkBoundary>;
  setForkBoundary: (threadId: string, boundary: ForkBoundary | null) => void;
}

function same(a: ForkBoundary | undefined, b: ForkBoundary): boolean {
  return a?.messageId === b.messageId && a?.sourceThreadId === b.sourceThreadId;
}

export const useForkBoundaryStore = create<ForkBoundaryState>()((set) => ({
  boundaryByThreadId: {},
  setForkBoundary: (threadId, boundary) =>
    set((state) => {
      const current = state.boundaryByThreadId[threadId];
      if (boundary === null) {
        if (current === undefined) return state;
        const next = { ...state.boundaryByThreadId };
        delete next[threadId];
        return { boundaryByThreadId: next };
      }
      // Same values keep the identity, so the rows subscribed to this do not re-render.
      if (same(current, boundary)) return state;
      return {
        boundaryByThreadId: { ...state.boundaryByThreadId, [threadId]: boundary },
      };
    }),
}));

export function setForkBoundary(
  threadId: string,
  messageId: string | null | undefined,
  sourceThreadId?: string | null,
): void {
  useForkBoundaryStore
    .getState()
    .setForkBoundary(
      threadId,
      messageId ? { messageId, sourceThreadId: sourceThreadId ?? null } : null,
    );
}
