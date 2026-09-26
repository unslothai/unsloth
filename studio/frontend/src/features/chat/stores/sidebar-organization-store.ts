// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

/** How the sidebar arranges chats that belong to a project. */
export type SidebarOrganizeBy = "project" | "list";
/** How chat rows are ordered inside whichever list they land in. */
export type SidebarChatSort = "updated" | "manual";
/** How project folders are ordered in the Projects section. */
export type SidebarProjectSort = "updated" | "name" | "created" | "manual";

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

/** A section the user made: a named list holding chats and project folders. */
export interface SidebarCustomSection {
  id: string;
  name: string;
  /** How its chats are ordered. Manual by default, like Pinned: it is a list the user built. */
  sort: SidebarChatSort;
}

/** The fixed section the "Show" toggles can hide, beside the user's own. Pinned always shows, as
 *  in ChatGPT, and Recents never hides: its header carries the menu that brings the others back. */
export const PROJECTS_SECTION_KEY = "projects";
/** Pinned's place in the section order. Only Recents has no place there: it is always last. */
export const PINNED_SECTION_KEY = "pinned";

// Section ids are generated here, but a hand-edited or restored payload can carry anything, so
// every scope built from one is prefixed and cannot collide with the fixed scopes above.
const CUSTOM_SECTION_PREFIX = "section:";

/** The manual-order scope of a custom section, which is also its drop-target section key. */
export function customSectionScope(sectionId: string): `section:${string}` {
  return `${CUSTOM_SECTION_PREFIX}${sectionId}`;
}

/** The custom section a scope names, or null for every built-in list. */
export function customSectionIdOf(scope: string): string | null {
  return scope.startsWith(CUSTOM_SECTION_PREFIX)
    ? scope.slice(CUSTOM_SECTION_PREFIX.length)
    : null;
}

export const CUSTOM_SECTION_NAME_MAX = 60;

/**
 * The order the sections above Recents are drawn in: Pinned, Projects and the user's own, by key
 * (a custom section by its id). A saved order is read over what exists now, so a key that went
 * away is dropped and one it never had gets its default place: Pinned first, a custom section
 * just above Projects in the order the list keeps them, Projects last.
 */
export function resolveSectionOrder(
  saved: readonly string[],
  customSections: readonly SidebarCustomSection[],
): string[] {
  const customIds = customSections.map((section) => section.id);
  const valid = new Set([PINNED_SECTION_KEY, PROJECTS_SECTION_KEY, ...customIds]);
  const out: string[] = [];
  for (const key of saved) {
    if (valid.has(key) && !out.includes(key)) out.push(key);
  }
  if (!out.includes(PINNED_SECTION_KEY)) out.unshift(PINNED_SECTION_KEY);
  for (const id of customIds) {
    if (out.includes(id)) continue;
    const projects = out.indexOf(PROJECTS_SECTION_KEY);
    out.splice(projects === -1 ? out.length : projects, 0, id);
  }
  if (!out.includes(PROJECTS_SECTION_KEY)) out.push(PROJECTS_SECTION_KEY);
  return out;
}

/** The custom sections in the order `order` draws them, so every list of them (the Show
 *  toggles, the Section submenus) reads top to bottom like the sidebar. */
function inSectionOrder(
  customSections: SidebarCustomSection[],
  order: readonly string[],
): SidebarCustomSection[] {
  const rank = new Map(order.map((key, index) => [key, index]));
  return [...customSections].sort(
    (a, b) => (rank.get(a.id) ?? 0) - (rank.get(b.id) ?? 0),
  );
}

function newSectionId(): string {
  // randomUUID needs a secure context, which a LAN address over plain http is not.
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** The assignments left once every one `drop` answers true for is gone. */
function withoutSection(
  map: Record<string, string>,
  drop: (sectionId: string) => boolean,
): Record<string, string> {
  const next: Record<string, string> = {};
  for (const [rowId, sectionId] of Object.entries(map)) {
    if (!drop(sectionId)) next[rowId] = sectionId;
  }
  return next;
}

export interface SidebarOrganizationState {
  organizeBy: SidebarOrganizeBy;
  chatSort: SidebarChatSort;
  // Pinned sorts on its own. Pin order already is a manual order, so it defaults to "manual" and
  // stays put while the lists below re-sort.
  pinnedSort: SidebarChatSort;
  // Manual by default: the drag order, falling back to last activity.
  projectSort: SidebarProjectSort;
  /** Scope key -> row ids, in the order the user dragged them into. */
  manualOrder: Record<string, string[]>;
  /** User-made sections, in the order they are drawn. */
  customSections: SidebarCustomSection[];
  /** Chat row id -> the custom section it is filed in. Pinned still wins: a pinned chat shows
   *  in Pinned and keeps its section for when it is unpinned. */
  sectionByChatId: Record<string, string>;
  /** Project id -> the custom section its folder is drawn in, on the same terms. */
  sectionByProjectId: Record<string, string>;
  /** Section keys the "Show" toggles turned off: Projects or a custom section's id. */
  hiddenSections: string[];
  /** The order of the sections above Recents, as dragged; read through resolveSectionOrder. */
  sectionOrder: string[];
  /** A new chat started from a custom section's header, filed there once it has an id. Not
   *  saved: it is only good for the new chat on screen now. */
  pendingNewChatSection: { sectionId: string; nonce: string } | null;
  setOrganizeBy: (value: SidebarOrganizeBy) => void;
  setChatSort: (value: SidebarChatSort) => void;
  setPinnedSort: (value: SidebarChatSort) => void;
  setProjectSort: (value: SidebarProjectSort) => void;
  setManualOrder: (scope: string, ids: string[]) => void;
  /** Adds a section at the top of the custom ones and returns its id, or null for a blank name. */
  createCustomSection: (name: string) => string | null;
  renameCustomSection: (sectionId: string, name: string) => void;
  /** Removes the section. Its rows go back to wherever they would be without it. */
  deleteCustomSection: (sectionId: string) => void;
  setCustomSectionSort: (sectionId: string, sort: SidebarChatSort) => void;
  /** Files chats into a section, or out of every section with null. */
  setChatsSection: (chatIds: string[], sectionId: string | null) => void;
  setProjectsSection: (projectIds: string[], sectionId: string | null) => void;
  setSectionHidden: (key: string, hidden: boolean) => void;
  /** Drops the section `key` against the `edge` side of `targetKey`, as a row drag lands. */
  moveSection: (key: string, targetKey: string, edge: "top" | "bottom") => void;
  setPendingNewChatSection: (pending: { sectionId: string; nonce: string } | null) => void;
}

/** Trims and bounds a section name. Empty when there is nothing to name it with. */
export function normalizeSectionName(name: string): string {
  return name.replace(/\s+/g, " ").trim().slice(0, CUSTOM_SECTION_NAME_MAX);
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

/** Reads a saved payload over the defaults, validated per field: an old, hand-edited or
 *  half-written payload keeps defaults wherever it is wrong. */
export function mergePersistedOrganization(
  persisted: unknown,
  current: SidebarOrganizationState,
): SidebarOrganizationState {
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
    value === "updated" || value === "manual" ? value : fallback;
  // A saved "priority", no longer offered, falls back to each list's default.
  const chatSort = readSort(saved?.chatSort, "updated");
  const pinnedSort = readSort(saved?.pinnedSort, "manual");
  const projectSort: SidebarProjectSort =
    saved?.projectSort === "updated" ||
    saved?.projectSort === "name" ||
    saved?.projectSort === "created"
      ? saved.projectSort
      : "manual";
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
  const customSections: SidebarCustomSection[] = [];
  const seen = new Set<string>();
  if (Array.isArray(saved?.customSections)) {
    for (const raw of saved.customSections as unknown[]) {
      if (!raw || typeof raw !== "object") continue;
      const entry = raw as Partial<SidebarCustomSection>;
      if (typeof entry.id !== "string" || !entry.id || seen.has(entry.id)) continue;
      const name =
        typeof entry.name === "string" ? normalizeSectionName(entry.name) : "";
      if (!name) continue;
      seen.add(entry.id);
      customSections.push({
        id: entry.id,
        name,
        sort: readSort(entry.sort, "manual"),
      });
    }
  }
  // An assignment to a section that no longer exists is dropped, so its row comes back.
  const readAssignments = (value: unknown): Record<string, string> => {
    const out: Record<string, string> = {};
    if (!value || typeof value !== "object") return out;
    for (const [rowId, sectionId] of Object.entries(value)) {
      if (typeof sectionId === "string" && seen.has(sectionId)) {
        out[rowId] = sectionId;
      }
    }
    return out;
  };
  const hiddenSections = Array.isArray(saved?.hiddenSections)
    ? [
        ...new Set(
          (saved.hiddenSections as unknown[]).filter(
            (key): key is string =>
              key === PROJECTS_SECTION_KEY ||
              (typeof key === "string" && seen.has(key)),
          ),
        ),
      ]
    : [];
  const sectionOrder = Array.isArray(saved?.sectionOrder)
    ? resolveSectionOrder(
        (saved.sectionOrder as unknown[]).filter(
          (key): key is string => typeof key === "string",
        ),
        customSections,
      )
    : [];
  return {
    ...current,
    organizeBy,
    chatSort,
    pinnedSort,
    projectSort,
    manualOrder,
    customSections,
    sectionByChatId: readAssignments(saved?.sectionByChatId),
    sectionByProjectId: readAssignments(saved?.sectionByProjectId),
    hiddenSections,
    sectionOrder,
  };
}

export const useSidebarOrganizationStore = create<SidebarOrganizationState>()(
  persist(
    (set) => ({
      organizeBy: "project",
      chatSort: "updated",
      pinnedSort: "manual",
      projectSort: "manual",
      manualOrder: {},
      customSections: [],
      sectionByChatId: {},
      sectionByProjectId: {},
      hiddenSections: [],
      sectionOrder: [],
      pendingNewChatSection: null,
      setOrganizeBy: (value) => set({ organizeBy: value }),
      setChatSort: (value) => set({ chatSort: value }),
      setPinnedSort: (value) => set({ pinnedSort: value }),
      setProjectSort: (value) => set({ projectSort: value }),
      setManualOrder: (scope, ids) =>
        set((state) => ({
          manualOrder: { ...state.manualOrder, [scope]: ids },
        })),
      createCustomSection: (name) => {
        const clean = normalizeSectionName(name);
        if (!clean) return null;
        const id = newSectionId();
        set((state) => {
          // Newest first among the user's own, which open under Pinned unless dragged elsewhere:
          // it is the one about to be filled, so it opens where the eye already is.
          const order = resolveSectionOrder(state.sectionOrder, state.customSections);
          const firstCustom = order.findIndex((key) =>
            state.customSections.some((section) => section.id === key),
          );
          const at = firstCustom !== -1 ? firstCustom : order.indexOf(PINNED_SECTION_KEY) + 1;
          return {
            customSections: [{ id, name: clean, sort: "manual" }, ...state.customSections],
            sectionOrder: [...order.slice(0, at), id, ...order.slice(at)],
          };
        });
        return id;
      },
      renameCustomSection: (sectionId, name) => {
        const clean = normalizeSectionName(name);
        if (!clean) return;
        set((state) => ({
          customSections: state.customSections.map((section) =>
            section.id === sectionId ? { ...section, name: clean } : section,
          ),
        }));
      },
      deleteCustomSection: (sectionId) =>
        set((state) => {
          const manualOrder = { ...state.manualOrder };
          delete manualOrder[customSectionScope(sectionId)];
          const gone = (id: string) => id === sectionId;
          return {
            customSections: state.customSections.filter((s) => s.id !== sectionId),
            sectionByChatId: withoutSection(state.sectionByChatId, gone),
            sectionByProjectId: withoutSection(state.sectionByProjectId, gone),
            hiddenSections: state.hiddenSections.filter((key) => key !== sectionId),
            sectionOrder: state.sectionOrder.filter((key) => key !== sectionId),
            manualOrder,
          };
        }),
      moveSection: (key, targetKey, edge) =>
        set((state) => {
          const order = resolveSectionOrder(state.sectionOrder, state.customSections);
          const next = insertIdAt(order, key, targetKey, edge);
          if (next === order) return state;
          return {
            sectionOrder: next,
            customSections: inSectionOrder(state.customSections, next),
          };
        }),
      setPendingNewChatSection: (pending) => set({ pendingNewChatSection: pending }),
      setCustomSectionSort: (sectionId, sort) =>
        set((state) => ({
          customSections: state.customSections.map((section) =>
            section.id === sectionId ? { ...section, sort } : section,
          ),
        })),
      setChatsSection: (chatIds, sectionId) =>
        set((state) => {
          if (sectionId && !state.customSections.some((s) => s.id === sectionId)) {
            return state;
          }
          const next = { ...state.sectionByChatId };
          for (const id of chatIds) {
            if (sectionId) next[id] = sectionId;
            else delete next[id];
          }
          return { sectionByChatId: next };
        }),
      setProjectsSection: (projectIds, sectionId) =>
        set((state) => {
          if (sectionId && !state.customSections.some((s) => s.id === sectionId)) {
            return state;
          }
          const next = { ...state.sectionByProjectId };
          for (const id of projectIds) {
            if (sectionId) next[id] = sectionId;
            else delete next[id];
          }
          return { sectionByProjectId: next };
        }),
      setSectionHidden: (key, hidden) =>
        set((state) => {
          const has = state.hiddenSections.includes(key);
          if (has === hidden) return state;
          return {
            hiddenSections: hidden
              ? [...state.hiddenSections, key]
              : state.hiddenSections.filter((existing) => existing !== key),
          };
        }),
    }),
    {
      name: SIDEBAR_ORGANIZATION_STORAGE_KEY,
      merge: mergePersistedOrganization,
      partialize: (state) => {
        const saved: Partial<SidebarOrganizationState> = { ...state };
        delete saved.pendingNewChatSection;
        return saved;
      },
    },
  ),
);
