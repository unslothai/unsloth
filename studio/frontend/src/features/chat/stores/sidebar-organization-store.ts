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
// Pinned folders reorder among themselves: one list's drag must not renumber the other's.
export const PINNED_PROJECT_ORDER_SCOPE = "pinned-projects";

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

/** Drops `draggedId` against the `edge` side of `targetId`, keeping the rest in order. The edge
 *  comes from which half of the target row the pointer is over, so the row lands exactly where
 *  the insertion line was drawn. Returns `ids` itself when a row is missing or the drop changes
 *  nothing, so a stale or pointless drop is a no-op the caller can skip persisting. */
export function insertIdAt(
  ids: string[],
  draggedId: string,
  targetId: string,
  edge: "top" | "bottom",
): string[] {
  if (draggedId === targetId) return ids;
  if (!ids.includes(draggedId) || !ids.includes(targetId)) return ids;
  const next = placeIdAt(ids, draggedId, targetId, edge);
  return next.every((id, index) => id === ids[index]) ? ids : next;
}

/** The same landing for a row the list does not hold yet, which is what a chat dropped into
 *  Pinned is. An unknown target puts it last, since there is no slot to aim at. */
export function placeIdAt(
  ids: string[],
  id: string,
  targetId: string | null,
  edge: "top" | "bottom",
): string[] {
  const next = ids.filter((existing) => existing !== id);
  const at = targetId === null ? -1 : next.indexOf(targetId);
  if (at === -1) return [...next, id];
  next.splice(at + (edge === "bottom" ? 1 : 0), 0, id);
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

/** Where a folder dragged over another folder's chats lands. A folder's rows are the folder row
 *  and the chats under it, and in Pinned those chats separate one folder from the next: without
 *  this the only target is the folder row itself, with a whole block of rows between two of them
 *  that answer nothing. The block is read as one strip: how far down it the pointer is decides
 *  which end of the folder the drop lands on, so the line flips once, in the middle, rather than
 *  at every row. Returns null when the row belongs to the dragged folder or the drop would move
 *  nothing. */
export function folderDropTarget(params: {
  /** The folder being dragged. */
  draggedId: string;
  /** The folder order it drags within: Projects' or Pinned's. */
  folderIds: string[];
  /** The folder whose block the pointer is over. */
  folderId: string;
  /** The hovered chat's place among that folder's chats, and which half of it the pointer is
   *  over: a folder holding one chat has to answer both ends from that row alone. */
  rowIndex: number;
  rowCount: number;
  pointerEdge: "top" | "bottom";
}): { edge: "top" | "bottom"; next: string[] } | null {
  const { draggedId, folderIds, folderId, rowIndex, rowCount } = params;
  if (draggedId === folderId || rowIndex < 0) return null;
  // How many row-halves down the block the pointer is, against its length.
  const at = rowIndex + (params.pointerEdge === "bottom" ? 1 : 0);
  const edge = at * 2 >= rowCount ? "bottom" : ("top" as const);
  const next = insertIdAt(folderIds, draggedId, folderId, edge);
  return next === folderIds ? null : { edge, next };
}

/** Which half of a row the pointer is over, which is the edge the row being dragged will land
 *  on. Read off the row's own box, so the insertion line follows the cursor rather than the
 *  two rows' index order. */
export function dropEdgeAt(
  rect: { top: number; height: number },
  pointerY: number,
): "top" | "bottom" {
  return pointerY >= rect.top + rect.height / 2 ? "bottom" : "top";
}

/** Moves a row one slot up or down. The keyboard path to the same reorder that dragging does:
 *  a keyboard never sees a `dragstart`, so alt + arrow drives this instead. */
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
        return {
          ...current,
          organizeBy,
          chatSort,
          pinnedSort,
          manualOrder,
        };
      },
    },
  ),
);
