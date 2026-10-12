// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Core items and the "More" overflow are always shown and intentionally not listed here.
export type PlusMenuItemId =
  | "chatWithFiles"
  | "mcp"
  | "skills"
  | "savedPrompts"
  | "compareChat"
  | "exportChat"
  | "projects";

export const PLUS_MENU_ORDER: PlusMenuItemId[] = [
  "chatWithFiles",
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
  chatWithFiles: true,
  mcp: false,
  skills: true,
  projects: false,
  savedPrompts: false,
  compareChat: false,
  exportChat: false,
};

export const PLUS_MENU_PINS_STORAGE_KEY = "unsloth_plus_menu_pins";

export interface PlusMenuPrefsState {
  pins: Record<PlusMenuItemId, boolean>;
  setPin: (id: PlusMenuItemId, value: boolean) => void;
  togglePin: (id: PlusMenuItemId) => void;
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
      // Version 1 moved MCP under "More". Saved pins hold every default, so without this an
      // existing install would keep it at the top; the user can pin it back from "More".
      migrate: (persisted, version) => {
        const saved = (persisted ?? {}) as Partial<PlusMenuPrefsState>;
        if (version < 1 && saved.pins) return { ...saved, pins: { ...saved.pins, mcp: false } };
        return saved;
      },
      // Backfill any ids added in a later release so persisted state from an older version still
      // resolves every menu item. A retired id survives the spread but PLUS_MENU_ORDER no
      // longer names it, so it is inert.
      merge: (persisted, current) => {
        const saved = persisted as Partial<PlusMenuPrefsState> | undefined;
        return {
          ...current,
          pins: { ...DEFAULT_PINS, ...(saved?.pins ?? {}) },
          pinnedPromptIds: saved?.pinnedPromptIds ?? [],
          pinnedListIds: saved?.pinnedListIds ?? [],
        };
      },
    },
  ),
);
