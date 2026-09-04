// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

/** How the sidebar arranges chats that belong to a project. */
export type SidebarOrganizeBy = "project" | "list";
/** How chat rows are ordered inside whichever list they land in. */
export type SidebarChatSort = "priority" | "updated" | "manual";

// Defined in a leaf module and re-exported here so existing importers are unchanged: this
// store is in an import cycle, so a binding defined here would be readable too late. See
// sidebar-organization-keys.ts.
export { SIDEBAR_ORGANIZATION_STORAGE_KEY } from "./sidebar-organization-keys.ts";
import { SIDEBAR_ORGANIZATION_STORAGE_KEY } from "./sidebar-organization-keys.ts";

// Manual order is per list: dragging a chat in one project must not move it in another list
// showing the same chat. Each list gets its own key.
export const RECENTS_ORDER_SCOPE = "recents";
export const PINNED_ORDER_SCOPE = "pinned";
// The project folders themselves, which drag regardless of the chat sort.
export const PROJECT_ORDER_SCOPE = "projects";

export function projectOrderScope(projectId: string): string {
  return `project:${projectId}`;
}

export interface SidebarOrganizationState {
  organizeBy: SidebarOrganizeBy;
  chatSort: SidebarChatSort;
  // Pinned sorts on its own. Pin order already is a manual order, so it defaults to "manual" and
  // stays put while the lists below re-sort.
  pinnedSort: SidebarChatSort;
  /** Scope key -> row ids, in the order the user dragged them into. */
  manualOrder: Record<string, string[]>;
  setOrganizeBy: (value: SidebarOrganizeBy) => void;
  setChatSort: (value: SidebarChatSort) => void;
  setPinnedSort: (value: SidebarChatSort) => void;
  setManualOrder: (scope: string, ids: string[]) => void;
}

/** Moves `draggedId` into `targetId`'s slot, keeping the rest in order. Returns `ids` itself
 *  when either row is missing, so a stale drop is a no-op. */
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

/** Whether a chat belongs in Recents. With the Projects section on, a project chat lives in its
 *  folder and listing it twice is noise; with it off there are no folders. */
export function showsInRecents(
  projectId: string | null | undefined,
  organizeBy: SidebarOrganizeBy,
): boolean {
  return organizeBy === "list" || !projectId;
}

/** Which edge of the target row the drop indicator belongs on, matching where `reorderIds`
 *  actually lands the row: after the target when dragging down, before it when dragging up. */
export function dropEdgeFor(
  ids: string[],
  draggedId: string,
  targetId: string,
): "top" | "bottom" {
  const from = ids.indexOf(draggedId);
  const to = ids.indexOf(targetId);
  return from !== -1 && to !== -1 && from < to ? "bottom" : "top";
}

/** Moves a row one slot up or down. The menu path to the same reorder that dragging does, for
 *  touch and keyboard, which never see a `dragstart`. */
export function moveIdBy(
  ids: string[],
  id: string,
  delta: number,
): string[] {
  const from = ids.indexOf(id);
  if (from === -1) return ids;
  const to = from + delta;
  if (to < 0 || to >= ids.length) return ids;
  const next = [...ids];
  next.splice(from, 1);
  next.splice(to, 0, id);
  return next;
}

/** Applies a saved order to `items`, leaving rows it does not mention in their incoming order
 *  and on top. A row the user never dragged is new to the list, so it stays where the list's
 *  own rule put it rather than sinking. */
export function applyManualOrder<T>(
  items: T[],
  order: string[] | undefined,
  getId: (item: T) => string,
): T[] {
  if (!order?.length) return items;
  const rank = new Map(order.map((id, index) => [id, index]));
  // Sort is stable, so two unranked rows keep their relative order.
  return [...items].sort(
    (a, b) => (rank.get(getId(a)) ?? -1) - (rank.get(getId(b)) ?? -1),
  );
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
      name: SIDEBAR_ORGANIZATION_STORAGE_KEY,
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
