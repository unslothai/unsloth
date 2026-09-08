// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  PASTED_TEXT_DEFAULT_MIN_CHARS,
  PASTED_TEXT_THRESHOLD_CHOICES,
} from "../utils/pasted-text.ts";

// Client-side chat UI prefs kept in localStorage, not the chat DB. confirmDeleteChats: off skips
// the delete confirm dialog. alwaysDeleteChatFiles: on also removes the sandbox folder.
// showModelDisclaimer: off hides the "LLMs can make mistakes" footer. showResponseModel: on shows
// the producing model on responses. collapseThinkingByDefault: on keeps thinking collapsed.
// pastedTextMinChars: paste length that becomes a .txt attachment; 0 is off.
export interface ChatPreferencesState {
  confirmDeleteChats: boolean;
  setConfirmDeleteChats: (value: boolean) => void;
  alwaysDeleteChatFiles: boolean;
  setAlwaysDeleteChatFiles: (value: boolean) => void;
  showModelDisclaimer: boolean;
  setShowModelDisclaimer: (value: boolean) => void;
  showResponseModel: boolean;
  setShowResponseModel: (value: boolean) => void;
  collapseThinkingByDefault: boolean;
  setCollapseThinkingByDefault: (value: boolean) => void;
  collapseToolActivityByDefault: boolean;
  setCollapseToolActivityByDefault: (value: boolean) => void;
  pastedTextMinChars: number;
  setPastedTextMinChars: (value: number) => void;
}

// A stale stored value would leave the dropdown blank and unfixable.
function normalisePastedTextMinChars(value: unknown): number {
  return PASTED_TEXT_THRESHOLD_CHOICES.includes(
    value as (typeof PASTED_TEXT_THRESHOLD_CHOICES)[number],
  )
    ? (value as number)
    : PASTED_TEXT_DEFAULT_MIN_CHARS;
}

export const useChatPreferencesStore = create<ChatPreferencesState>()(
  persist(
    (set) => ({
      confirmDeleteChats: true,
      setConfirmDeleteChats: (confirmDeleteChats) =>
        set({ confirmDeleteChats }),
      // Off by default: deleting files is the destructive half, so it stays opt in.
      alwaysDeleteChatFiles: false,
      setAlwaysDeleteChatFiles: (alwaysDeleteChatFiles) =>
        set({ alwaysDeleteChatFiles }),
      showModelDisclaimer: false,
      setShowModelDisclaimer: (showModelDisclaimer) =>
        set({ showModelDisclaimer }),
      showResponseModel: false,
      setShowResponseModel: (showResponseModel) => set({ showResponseModel }),
      collapseThinkingByDefault: false,
      setCollapseThinkingByDefault: (collapseThinkingByDefault) =>
        set({ collapseThinkingByDefault }),
      collapseToolActivityByDefault: true,
      setCollapseToolActivityByDefault: (collapseToolActivityByDefault) =>
        set({ collapseToolActivityByDefault }),
      pastedTextMinChars: PASTED_TEXT_DEFAULT_MIN_CHARS,
      setPastedTextMinChars: (pastedTextMinChars) =>
        set({ pastedTextMinChars }),
    }),
    {
      name: "unsloth_chat_preferences",
      merge: (persisted, current) => {
        const saved = persisted as Partial<ChatPreferencesState> | undefined;
        return {
          ...current,
          confirmDeleteChats: saved?.confirmDeleteChats ?? true,
          alwaysDeleteChatFiles: saved?.alwaysDeleteChatFiles ?? false,
          showModelDisclaimer: saved?.showModelDisclaimer ?? false,
          showResponseModel: saved?.showResponseModel ?? false,
          collapseThinkingByDefault: saved?.collapseThinkingByDefault ?? false,
          collapseToolActivityByDefault:
            saved?.collapseToolActivityByDefault ?? true,
          pastedTextMinChars: normalisePastedTextMinChars(
            saved?.pastedTextMinChars,
          ),
        };
      },
    },
  ),
);
