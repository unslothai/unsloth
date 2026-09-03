// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * Whether the bar is up and what is typed in it. Nothing else: the index, matches and ranges are
 * refs inside the bar, which only exists while it is open.
 */
interface FindInPageStore {
  open: boolean;
  query: string;
  /**
   * Bumped on every press of the chord, so pressing it again re-focuses the field and selects what
   * is in it. A counter rather than a boolean: two presses in a row have to be two events.
   */
  focusToken: number;
  requestFocus: () => void;
  close: () => void;
  setQuery: (query: string) => void;
  /** Forget the search entirely, query included. For leaving the shell, not for closing the bar. */
  reset: () => void;
}

export const useFindInPageStore = create<FindInPageStore>((set) => ({
  open: false,
  query: "",
  focusToken: 0,
  requestFocus: () =>
    set((state) => ({ open: true, focusToken: state.focusToken + 1 })),
  // The query survives a close, so re-opening offers the last search again, selected.
  close: () => set({ open: false }),
  setQuery: (query) => set({ query }),
  reset: () => set({ open: false, query: "" }),
}));
