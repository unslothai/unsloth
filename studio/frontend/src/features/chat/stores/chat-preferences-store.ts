// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Client-side chat UI prefs kept in localStorage, not the chat DB.
// confirmDeleteChats: when off, deleting a chat skips the confirm dialog.
// alwaysDeleteChatFiles: when on, deleting a chat also removes its sandbox
// folder, without having to ask for it each time.
// showModelDisclaimer: when off, hide the "LLMs can make mistakes" footer note.
// showResponseModel: when on, assistant responses show the producing model.
// collapseThinkingByDefault: when on, thinking stays collapsed instead of
// streaming open.
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
      showModelDisclaimer: true,
      setShowModelDisclaimer: (showModelDisclaimer) =>
        set({ showModelDisclaimer }),
      showResponseModel: false,
      setShowResponseModel: (showResponseModel) =>
        set({ showResponseModel }),
      collapseThinkingByDefault: false,
      setCollapseThinkingByDefault: (collapseThinkingByDefault) =>
        set({ collapseThinkingByDefault }),
    }),
    {
      name: "unsloth_chat_preferences",
      merge: (persisted, current) => {
        const saved = persisted as Partial<ChatPreferencesState> | undefined;
        return {
          ...current,
          confirmDeleteChats: saved?.confirmDeleteChats ?? true,
          alwaysDeleteChatFiles: saved?.alwaysDeleteChatFiles ?? false,
          showModelDisclaimer: saved?.showModelDisclaimer ?? true,
          showResponseModel: saved?.showResponseModel ?? false,
          collapseThinkingByDefault: saved?.collapseThinkingByDefault ?? false,
        };
      },
    },
  ),
);
