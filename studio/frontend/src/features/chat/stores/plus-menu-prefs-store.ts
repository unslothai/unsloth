// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Adjustable items in the composer "+" menu. The core items (Add photos &
// files, Web search, Code) and the "More" overflow itself are always shown and
// are intentionally NOT represented here.
export type PlusMenuItemId =
  | "chatWithFiles"
  | "mcp"
  | "savedPrompts"
  | "compareChat"
  | "exportChat"
  | "canvas"
  | "projects"
  | "bypassPermissions";

// Canonical order used both for the pinned items at the top level and for the
// items that fall into the "More" overflow submenu.
export const PLUS_MENU_ORDER: PlusMenuItemId[] = [
  "chatWithFiles",
  "mcp",
  "savedPrompts",
  "compareChat",
  "exportChat",
  "canvas",
  "projects",
  "bypassPermissions",
];

// Defaults reproduce the historical layout: Chat with Files, MCP and Projects
// pinned to the top level; everything else living under "More".
const DEFAULT_PINS: Record<PlusMenuItemId, boolean> = {
  chatWithFiles: true,
  mcp: true,
  projects: true,
  savedPrompts: false,
  compareChat: false,
  exportChat: false,
  canvas: false,
  // Lives under "More" by default; it is a rarely toggled, dangerous mode.
  bypassPermissions: false,
};

export const PLUS_MENU_PINS_STORAGE_KEY = "unsloth_plus_menu_pins";

export interface PlusMenuPrefsState {
  pins: Record<PlusMenuItemId, boolean>;
  setPin: (id: PlusMenuItemId, value: boolean) => void;
  togglePin: (id: PlusMenuItemId) => void;
  // Ids of saved prompts the user pinned into the "Saved prompts" submenu.
  // Kept client-side (like the menu pins above) since prompts are addressed by
  // their stable server id.
  pinnedPromptIds: string[];
  togglePinnedPrompt: (id: string) => void;
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
    }),
    {
      name: PLUS_MENU_PINS_STORAGE_KEY,
      // Backfill any ids added in a later release so persisted state from an
      // older version still resolves every menu item.
      merge: (persisted, current) => {
        const saved = persisted as Partial<PlusMenuPrefsState> | undefined;
        return {
          ...current,
          pins: { ...DEFAULT_PINS, ...(saved?.pins ?? {}) },
          pinnedPromptIds: saved?.pinnedPromptIds ?? [],
        };
      },
    },
  ),
);
