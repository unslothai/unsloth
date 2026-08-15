// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

/** How the sidebar arranges chats that belong to a project. */
export type SidebarOrganizeBy = "project" | "list";
/** How chat rows are ordered inside whichever list they land in. */
export type SidebarChatSort = "priority" | "updated" | "manual";

// Manual order is per list: dragging a chat in one project must not move it in
// another list showing the same chat. Recents and Pinned get their own keys.
export const RECENTS_ORDER_SCOPE = "recents";
export const PINNED_ORDER_SCOPE = "pinned";

export function projectOrderScope(projectId: string): string {
  return `project:${projectId}`;
}

export interface SidebarOrganizationState {
  organizeBy: SidebarOrganizeBy;
  chatSort: SidebarChatSort;
  // Pinned sorts on its own. Pin order already is a manual order, so it
  // defaults to "manual" and stays put while the lists below re-sort.
  pinnedSort: SidebarChatSort;
  /** Scope key -> chat row ids, in the order the user dragged them into. */
  manualOrder: Record<string, string[]>;
  setOrganizeBy: (value: SidebarOrganizeBy) => void;
  setChatSort: (value: SidebarChatSort) => void;
  setPinnedSort: (value: SidebarChatSort) => void;
  setManualOrder: (scope: string, ids: string[]) => void;
}

/**
 * Moves `draggedId` into `targetId`'s slot, keeping the rest in order. Returns
 * `ids` itself when either row is missing, so a stale drop is a no-op.
 */
export function reorderIds(
  ids: string[],
  draggedId: string,
  targetId: string,
): string[] {
  if (draggedId === targetId) return ids;
  const from = ids.indexOf(draggedId);
  const to = ids.indexOf(targetId);
  if (from === -1 || to === -1) return ids;
  const next = [...ids];
  next.splice(from, 1);
  next.splice(to, 0, draggedId);
  return next;
}

export const useSidebarOrganizationStore = create<SidebarOrganizationState>()(
  persist(
    (set) => ({
      organizeBy: "project",
      chatSort: "priority",
      pinnedSort: "manual",
      manualOrder: {},
      setOrganizeBy: (value) => set({ organizeBy: value }),
      setChatSort: (value) => set({ chatSort: value }),
      setPinnedSort: (value) => set({ pinnedSort: value }),
      setManualOrder: (scope, ids) =>
        set((state) => ({
          manualOrder: { ...state.manualOrder, [scope]: ids },
        })),
    }),
    {
      name: "unsloth_sidebar_organization",
      merge: (persisted, current) => {
        const saved = persisted as
          | Partial<SidebarOrganizationState>
          | undefined;
        // Validated per field: an old or half-written payload keeps defaults.
        const organizeBy: SidebarOrganizeBy =
          saved?.organizeBy === "list" ? "list" : "project";
        const readSort = (
          value: unknown,
          fallback: SidebarChatSort,
        ): SidebarChatSort =>
          value === "priority" || value === "updated" || value === "manual"
            ? value
            : fallback;
        const chatSort = readSort(saved?.chatSort, "priority");
        const pinnedSort = readSort(saved?.pinnedSort, "manual");
        const manualOrder: Record<string, string[]> = {};
        if (saved?.manualOrder && typeof saved.manualOrder === "object") {
          for (const [scope, ids] of Object.entries(saved.manualOrder)) {
            if (Array.isArray(ids)) {
              manualOrder[scope] = ids.filter(
                (id): id is string => typeof id === "string",
              );
            }
          }
        }
        return { ...current, organizeBy, chatSort, pinnedSort, manualOrder };
      },
    },
  ),
);
