// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Client-side pin state for projects, keyed by project id. Kept in
// localStorage. Pinned projects drive the sidebar "Projects" section; new pins
// are prepended so the most recently pinned project sorts first.
export interface PinnedProjectsState {
  pinnedIds: string[];
  togglePin: (id: string) => void;
  unpin: (id: string) => void;
}

export const usePinnedProjectsStore = create<PinnedProjectsState>()(
  persist(
    (set) => ({
      pinnedIds: [],
      togglePin: (id) =>
        set((state) => ({
          pinnedIds: state.pinnedIds.includes(id)
            ? state.pinnedIds.filter((x) => x !== id)
            : [id, ...state.pinnedIds],
        })),
      unpin: (id) =>
        set((state) => ({
          pinnedIds: state.pinnedIds.filter((x) => x !== id),
        })),
    }),
    {
      name: "unsloth_pinned_projects",
      merge: (persisted, current) => {
        const saved = persisted as Partial<PinnedProjectsState> | undefined;
        return {
          ...current,
          pinnedIds: Array.isArray(saved?.pinnedIds) ? saved.pinnedIds : [],
        };
      },
    },
  ),
);
