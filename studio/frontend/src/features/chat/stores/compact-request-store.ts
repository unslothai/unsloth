// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// Threads whose next request compacts (/compact). Session only: the compaction itself happens on
// the server when that request arrives, and its record is what persists.
export interface CompactRequestState {
  requested: Record<string, true>;
  request: (threadId: string) => void;
  clear: (threadId: string) => void;
}

export const useCompactRequestStore = create<CompactRequestState>()((set) => ({
  requested: {},
  request: (threadId) =>
    set((state) =>
      state.requested[threadId]
        ? state
        : { requested: { ...state.requested, [threadId]: true } },
    ),
  clear: (threadId) =>
    set((state) => {
      if (!state.requested[threadId]) return state;
      const requested = { ...state.requested };
      delete requested[threadId];
      return { requested };
    }),
}));

export function requestThreadCompaction(threadId: string): void {
  useCompactRequestStore.getState().request(threadId);
}

/** Whether the thread's next request should compact, clearing the request so it applies once. */
export function takeThreadCompactionRequest(threadId: string | undefined): boolean {
  if (!threadId) return false;
  const state = useCompactRequestStore.getState();
  if (!state.requested[threadId]) return false;
  state.clear(threadId);
  return true;
}

/** `/compact` typed alone. Anything after it is an ordinary message. */
export function isCompactCommand(text: string): boolean {
  return text.trim().toLowerCase() === "/compact";
}
