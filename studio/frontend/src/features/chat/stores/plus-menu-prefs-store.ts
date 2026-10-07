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

const DEFAULT_PINS: Record<PlusMenuItemId, boolean> = {
  chatWithFiles: true,
  mcp: true,
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
      // Backfill ids added in later releases; retired ids survive the spread but are inert.
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
