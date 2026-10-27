// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type SidebarOrganizeBy = "project" | "list";
export type SidebarChatSort = "updated" | "manual";
export type SidebarProjectSort = "updated" | "name" | "created" | "manual";

// Re-exported from a leaf module because this store is in an import cycle.
export { SIDEBAR_ORGANIZATION_STORAGE_KEY } from "./sidebar-organization-keys.ts";
import { SIDEBAR_ORGANIZATION_STORAGE_KEY } from "./sidebar-organization-keys.ts";

// Manual order is per list, so each list gets its own key.
export const RECENTS_ORDER_SCOPE = "recents";
export const PINNED_ORDER_SCOPE = "pinned";
export const PROJECT_ORDER_SCOPE = "projects";
export const PINNED_PROJECT_ORDER_SCOPE = "pinned-projects";

export function projectOrderScope(projectId: string): string {
  return `project:${projectId}`;
}

export interface SidebarCustomSection {
  id: string;
  name: string;
  sort: SidebarChatSort;
  createdAt?: number;
  modifiedAt?: number;
}

export const PROJECTS_SECTION_KEY = "projects";
export const PINNED_SECTION_KEY = "pinned";

// Prefixed so a hand-edited section id cannot collide with the fixed scopes above.
const CUSTOM_SECTION_PREFIX = "section:";

export function customSectionScope(sectionId: string): `section:${string}` {
  return `${CUSTOM_SECTION_PREFIX}${sectionId}`;
}

export function customSectionIdOf(scope: string): string | null {
  return scope.startsWith(CUSTOM_SECTION_PREFIX)
    ? scope.slice(CUSTOM_SECTION_PREFIX.length)
    : null;
}

export const CUSTOM_SECTION_NAME_MAX = 60;

/** Section order above Recents; missing keys are dropped and new ones get a default place. */
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

export function inSectionOrder(
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

/** Prototype-free map so row ids like "constructor" do not read as filed. */
export function assignmentMap(from?: Record<string, string>): Record<string, string> {
  const map = Object.create(null) as Record<string, string>;
  if (from) for (const [rowId, sectionId] of Object.entries(from)) map[rowId] = sectionId;
  return map;
}

function withoutSection(
  map: Record<string, string>,
  drop: (sectionId: string) => boolean,
): Record<string, string> {
  const next = assignmentMap();
  for (const [rowId, sectionId] of Object.entries(map)) {
    if (!drop(sectionId)) next[rowId] = sectionId;
  }
  return next;
}

export interface SidebarOrganizationState {
  organizeBy: SidebarOrganizeBy;
  chatSort: SidebarChatSort;
  pinnedSort: SidebarChatSort;
  projectSort: SidebarProjectSort;
  manualOrder: Record<string, string[]>;
  customSections: SidebarCustomSection[];
  sectionByChatId: Record<string, string>;
  sectionByProjectId: Record<string, string>;
  /** Pinned page id -> the custom section it shows in instead of Pinned. */
  sectionByPageId: Record<string, string>;
  /** Section keys the "Show" toggles turned off: Projects or a custom section's id. */
  hiddenSections: string[];
  sectionOrder: string[];
  /** Not persisted; only applies to the new chat on screen now. */
  pendingNewChatSection: { sectionId: string; nonce: string; compare?: string } | null;
  setOrganizeBy: (value: SidebarOrganizeBy) => void;
  setChatSort: (value: SidebarChatSort) => void;
  setPinnedSort: (value: SidebarChatSort) => void;
  setProjectSort: (value: SidebarProjectSort) => void;
  setManualOrder: (scope: string, ids: string[]) => void;
  createCustomSection: (name: string) => string | null;
  renameCustomSection: (sectionId: string, name: string) => void;
  deleteCustomSection: (sectionId: string) => void;
  setCustomSectionSort: (sectionId: string, sort: SidebarChatSort) => void;
  setChatsSection: (chatIds: string[], sectionId: string | null) => void;
  setProjectsSection: (projectIds: string[], sectionId: string | null) => void;
  setPagesSection: (pageIds: string[], sectionId: string | null) => void;
  setSectionHidden: (key: string, hidden: boolean) => void;
  moveSection: (key: string, targetKey: string, edge: "top" | "bottom") => void;
  setPendingNewChatSection: (pending: SidebarOrganizationState["pendingNewChatSection"]) => void;
}

function readTime(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function touchSections(
  sections: SidebarCustomSection[],
  ids: ReadonlySet<string>,
): SidebarCustomSection[] {
  if (ids.size === 0) return sections;
  const now = Date.now();
  return sections.map((section) => (ids.has(section.id) ? { ...section, modifiedAt: now } : section));
}

export function normalizeSectionName(name: string): string {
  return name.replace(/\s+/g, " ").trim().slice(0, CUSTOM_SECTION_NAME_MAX);
}

/** Returns `ids` itself when nothing changes, so callers can skip persisting. */
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

export function showsInRecents(
  projectId: string | null | undefined,
  organizeBy: SidebarOrganizeBy,
): boolean {
  return organizeBy === "list" || !projectId;
}

/** Treats a folder's block of rows as one strip so the drop line flips once, mid-block. */
export function folderDropTarget(params: {
  draggedId: string;
  folderIds: string[];
  folderId: string;
  rowIndex: number;
  rowCount: number;
  pointerEdge: "top" | "bottom";
}): { edge: "top" | "bottom"; next: string[] } | null {
  const { draggedId, folderIds, folderId, rowIndex, rowCount } = params;
  if (draggedId === folderId || rowIndex < 0) return null;
  const at = rowIndex + (params.pointerEdge === "bottom" ? 1 : 0);
  const edge = at * 2 >= rowCount ? "bottom" : ("top" as const);
  const next = insertIdAt(folderIds, draggedId, folderId, edge);
  return next === folderIds ? null : { edge, next };
}

export function dropEdgeAt(
  rect: { top: number; height: number },
  pointerY: number,
): "top" | "bottom" {
  return pointerY >= rect.top + rect.height / 2 ? "bottom" : "top";
}

/** Keyboard path for reorder: keyboards never fire dragstart, so alt+arrow drives this. */
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

/** Rows the saved order does not mention stay on top in their incoming order. */
export function applyManualOrder<T>(
  items: T[],
  order: string[] | undefined,
  getId: (item: T) => string,
): T[] {
  if (!order?.length) return items;
  const rank = new Map(order.map((id, index) => [id, index]));
  return [...items].sort(
    (a, b) => (rank.get(getId(a)) ?? -1) - (rank.get(getId(b)) ?? -1),
  );
}

export function mergePersistedOrganization(
  persisted: unknown,
  current: SidebarOrganizationState,
): SidebarOrganizationState {
  const saved = persisted as
    | Partial<SidebarOrganizationState>
    | undefined;
  const organizeBy: SidebarOrganizeBy =
    saved?.organizeBy === "list" ? "list" : "project";
  const readSort = (
    value: unknown,
    fallback: SidebarChatSort,
  ): SidebarChatSort =>
    value === "updated" || value === "manual" ? value : fallback;
  const chatSort = readSort(saved?.chatSort, "updated");
  const pinnedSort = readSort(saved?.pinnedSort, "manual");
  // A saved automatic folder sort is no longer offered, so it falls back to Manual.
  const projectSort: SidebarProjectSort = "manual";
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
      // A section keyed like Pinned or Projects would be drawn as them and hide its rows.
      if (entry.id === PINNED_SECTION_KEY || entry.id === PROJECTS_SECTION_KEY) continue;
      const name =
        typeof entry.name === "string" ? normalizeSectionName(entry.name) : "";
      if (!name) continue;
      seen.add(entry.id);
      customSections.push({
        id: entry.id,
        name,
        sort: readSort(entry.sort, "manual"),
        ...(readTime(entry.createdAt) ? { createdAt: entry.createdAt } : {}),
        ...(readTime(entry.modifiedAt) ? { modifiedAt: entry.modifiedAt } : {}),
      });
    }
  }
  const readAssignments = (value: unknown): Record<string, string> => {
    const out = assignmentMap();
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
    sectionByPageId: readAssignments(saved?.sectionByPageId),
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
      sectionByChatId: assignmentMap(),
      sectionByProjectId: assignmentMap(),
      sectionByPageId: assignmentMap(),
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
          const order = resolveSectionOrder(state.sectionOrder, state.customSections);
          const firstCustom = order.findIndex((key) =>
            state.customSections.some((section) => section.id === key),
          );
          const at = firstCustom !== -1 ? firstCustom : order.indexOf(PINNED_SECTION_KEY) + 1;
          return {
            customSections: [
              { id, name: clean, sort: "manual", createdAt: Date.now(), modifiedAt: Date.now() },
              ...state.customSections,
            ],
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
            section.id === sectionId && section.name !== clean
              ? { ...section, name: clean, modifiedAt: Date.now() }
              : section,
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
            sectionByPageId: withoutSection(state.sectionByPageId, gone),
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
          const next = assignmentMap(state.sectionByChatId);
          const touched = new Set<string>();
          for (const id of chatIds) {
            const from = next[id];
            if (from === (sectionId ?? undefined)) continue;
            if (from) touched.add(from);
            if (sectionId) {
              next[id] = sectionId;
              touched.add(sectionId);
            } else delete next[id];
          }
          return {
            sectionByChatId: next,
            customSections: touchSections(state.customSections, touched),
          };
        }),
      setProjectsSection: (projectIds, sectionId) =>
        set((state) => {
          if (sectionId && !state.customSections.some((s) => s.id === sectionId)) {
            return state;
          }
          const next = assignmentMap(state.sectionByProjectId);
          const touched = new Set<string>();
          for (const id of projectIds) {
            const from = next[id];
            if (from === (sectionId ?? undefined)) continue;
            if (from) touched.add(from);
            if (sectionId) {
              next[id] = sectionId;
              touched.add(sectionId);
            } else delete next[id];
          }
          return {
            sectionByProjectId: next,
            customSections: touchSections(state.customSections, touched),
          };
        }),
      setPagesSection: (pageIds, sectionId) =>
        set((state) => {
          if (sectionId && !state.customSections.some((s) => s.id === sectionId)) {
            return state;
          }
          const next = assignmentMap(state.sectionByPageId);
          const touched = new Set<string>();
          for (const id of pageIds) {
            const from = next[id];
            if (from === (sectionId ?? undefined)) continue;
            if (from) touched.add(from);
            if (sectionId) {
              next[id] = sectionId;
              touched.add(sectionId);
            } else delete next[id];
          }
          return {
            sectionByPageId: next,
            customSections: touchSections(state.customSections, touched),
          };
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
