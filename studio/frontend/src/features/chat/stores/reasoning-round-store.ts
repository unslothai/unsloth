// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// Session only. A round missing here counts as closed, so tools never flash before their block.
export interface ReasoningRoundState {
  open: Record<string, boolean>;
  setRoundOpen: (key: string, open: boolean) => void;
  clearRound: (key: string) => void;
}

export const useReasoningRoundStore = create<ReasoningRoundState>()((set) => ({
  open: {},
  setRoundOpen: (key, open) =>
    set((state) =>
      state.open[key] === open
        ? state
        : { open: { ...state.open, [key]: open } },
    ),
  clearRound: (key) =>
    set((state) => {
      if (!(key in state.open)) return state;
      const open: Record<string, boolean> = {};
      for (const [id, value] of Object.entries(state.open)) {
        if (id !== key) open[id] = value;
      }
      return { open };
    }),
}));

export function setReasoningRoundOpen(key: string, open: boolean): void {
  useReasoningRoundStore.getState().setRoundOpen(key, open);
}

export function clearReasoningRound(key: string): void {
  useReasoningRoundStore.getState().clearRound(key);
}
