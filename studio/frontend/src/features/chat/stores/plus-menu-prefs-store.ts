// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { syncAcrossWindows } from "../../../lib/sync-across-windows.ts";

// Core items and the "More" overflow are always shown and intentionally not listed here.
export type PlusMenuItemId =
  | "deepResearch"
  | "chatWithFiles"
  | "browserUse"
  | "computerUse"
  | "mcp"
  | "skills"
  | "savedPrompts"
  | "compareChat"
  | "exportChat"
  | "projects";

// Default order; Settings > Chat lets the user reorder it.
export const PLUS_MENU_ORDER: PlusMenuItemId[] = [
  "deepResearch",
  "chatWithFiles",
  "browserUse",
  "computerUse",
  "mcp",
  "skills",
  "savedPrompts",
  "compareChat",
  "exportChat",
  "projects",
];

// Chat with files and Skills sit at the top level; everything else, MCP included, lives under
// "More". Projects is a workspace the user opts into, not a per-message attachment.
const DEFAULT_PINS: Record<PlusMenuItemId, boolean> = {
  deepResearch: true,
  chatWithFiles: true,
  browserUse: true,
  computerUse: true,
  mcp: false,
  skills: true,
  projects: false,
  savedPrompts: false,
  compareChat: false,
  exportChat: false,
};

export function plusMenuAtDefaults(
  pins: Record<PlusMenuItemId, boolean>,
  order: readonly PlusMenuItemId[],
): boolean {
  return (
    order.length === PLUS_MENU_ORDER.length &&
    order.every((id, index) => id === PLUS_MENU_ORDER[index]) &&
    PLUS_MENU_ORDER.every((id) => pins[id] === DEFAULT_PINS[id])
  );
}

export const PLUS_MENU_PINS_STORAGE_KEY = "unsloth_plus_menu_pins";

/** Drops unknown ids; ids missing from an older save go after their default predecessor. */
export function normalizePlusMenuOrder(saved: unknown): PlusMenuItemId[] {
  const known = new Set<string>(PLUS_MENU_ORDER);
  const order: PlusMenuItemId[] = [];
  if (Array.isArray(saved)) {
    for (const id of saved) {
      if (typeof id === "string" && known.has(id) && !order.includes(id as PlusMenuItemId)) {
        order.push(id as PlusMenuItemId);
      }
    }
  }
  PLUS_MENU_ORDER.forEach((id, index) => {
    if (order.includes(id)) return;
    const before = PLUS_MENU_ORDER.slice(0, index).reverse().find((prev) => order.includes(prev));
    order.splice(before ? order.indexOf(before) + 1 : 0, 0, id);
  });
  return order;
}

/** `ids` with `fixed` put back at its index in `saved`: an item the list doesn't move (Deep research)
 *  still orders the More submenu when switched off, so a move of another item must not shift it. */
export function keepSavedSlot(
  ids: readonly PlusMenuItemId[],
  saved: readonly PlusMenuItemId[],
  fixed: PlusMenuItemId,
): PlusMenuItemId[] {
  const rest = ids.filter((id) => id !== fixed);
  const at = saved.indexOf(fixed);
  const index = at < 0 ? 0 : Math.min(at, rest.length);
  return [...rest.slice(0, index), fixed, ...rest.slice(index)];
}

export function splitPlusMenuItems(
  order: readonly PlusMenuItemId[],
  pins: Record<PlusMenuItemId, boolean>,
): { pinned: PlusMenuItemId[]; overflow: PlusMenuItemId[] } {
  return {
    pinned: order.filter((id) => pins[id]),
    overflow: order.filter((id) => !pins[id]),
  };
}

export interface PlusMenuPrefsState {
  pins: Record<PlusMenuItemId, boolean>;
  setPin: (id: PlusMenuItemId, value: boolean) => void;
  togglePin: (id: PlusMenuItemId) => void;
  order: PlusMenuItemId[];
  setOrder: (order: PlusMenuItemId[]) => void;
  /** Default order and pins; saved prompt and list pins are kept. */
  resetMenu: () => void;
  // Prompt ids pinned to the submenu; kept client-side since prompts use stable server ids.
  pinnedPromptIds: string[];
  togglePinnedPrompt: (id: string) => void;
  pinnedListIds: string[];
  togglePinnedList: (id: string) => void;
}

export const usePlusMenuPrefsStore = create<PlusMenuPrefsState>()(
  persist(
    (set) => ({
      pins: { ...DEFAULT_PINS },
      setPin: (id, value) =>
        set((state) => ({ pins: { ...state.pins, [id]: value } })),
      togglePin: (id) =>
        set((state) => ({ pins: { ...state.pins, [id]: !state.pins[id] } })),
      order: [...PLUS_MENU_ORDER],
      setOrder: (order) => set({ order: normalizePlusMenuOrder(order) }),
      resetMenu: () => set({ pins: { ...DEFAULT_PINS }, order: [...PLUS_MENU_ORDER] }),
      pinnedPromptIds: [],
      togglePinnedPrompt: (id) =>
        set((state) => ({
          pinnedPromptIds: state.pinnedPromptIds.includes(id)
            ? state.pinnedPromptIds.filter((x) => x !== id)
            : [...state.pinnedPromptIds, id],
        })),
      pinnedListIds: [],
      togglePinnedList: (id) =>
        set((state) => ({
          pinnedListIds: state.pinnedListIds.includes(id)
            ? state.pinnedListIds.filter((x) => x !== id)
            : [...state.pinnedListIds, id],
        })),
    }),
    {
      name: PLUS_MENU_PINS_STORAGE_KEY,
      version: 1,
      // Saved pins hold every default, so older installs would keep MCP on top without this.
      migrate: (persisted, version) => {
        const saved = (persisted ?? {}) as Partial<PlusMenuPrefsState>;
        if (version < 1 && saved.pins) return { ...saved, pins: { ...saved.pins, mcp: false } };
        return saved;
      },
      // Backfills ids added in a later release so older persisted state resolves every item; a
      // retired id survives the spread but is inert since PLUS_MENU_ORDER no longer names it.
      merge: (persisted, current) => {
        const saved = persisted as Partial<PlusMenuPrefsState> | undefined;
        return {
          ...current,
          pins: { ...DEFAULT_PINS, ...(saved?.pins ?? {}) },
          order: normalizePlusMenuOrder(saved?.order),
          pinnedPromptIds: saved?.pinnedPromptIds ?? [],
          pinnedListIds: saved?.pinnedListIds ?? [],
        };
      },
    },
  ),
);

syncAcrossWindows(usePlusMenuPrefsStore);
