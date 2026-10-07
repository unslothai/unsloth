// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  type ComposerSendShortcut,
  type ComposerFollowUpBehavior,
  normalizeComposerPreferences,
} from "../utils/composer-preferences.ts";
import {
  PASTED_TEXT_DEFAULT_MIN_CHARS,
  PASTED_TEXT_THRESHOLD_CHOICES,
} from "../utils/pasted-text.ts";
import {
  DEFAULT_THINKING_VISIBILITY,
  DEFAULT_TOOL_VISIBILITY,
  type DisplayVisibility,
  migrateVisibility,
} from "../utils/display-visibility.ts";

export interface ChatPreferencesState {
  plainTextComposer: boolean;
  setPlainTextComposer: (value: boolean) => void;
  showContextWindowUsage: boolean;
  setShowContextWindowUsage: (value: boolean) => void;
  sendShortcut: ComposerSendShortcut;
  setSendShortcut: (value: ComposerSendShortcut) => void;
  followUpBehavior: ComposerFollowUpBehavior;
  setFollowUpBehavior: (value: ComposerFollowUpBehavior) => void;
  confirmDeleteChats: boolean;
  setConfirmDeleteChats: (value: boolean) => void;
  alwaysDeleteChatFiles: boolean;
  setAlwaysDeleteChatFiles: (value: boolean) => void;
  showModelDisclaimer: boolean;
  setShowModelDisclaimer: (value: boolean) => void;
  showResponseModel: boolean;
  setShowResponseModel: (value: boolean) => void;
  showInlineEditResponse: boolean;
  setShowInlineEditResponse: (value: boolean) => void;
  thinkingVisibility: DisplayVisibility;
  setThinkingVisibility: (value: DisplayVisibility) => void;
  toolVisibility: DisplayVisibility;
  setToolVisibility: (value: DisplayVisibility) => void;
  foldToolActivityIntoThinking: boolean;
  setFoldToolActivityIntoThinking: (value: boolean) => void;
  pastedTextMinChars: number;
  setPastedTextMinChars: (value: number) => void;
  autoScrollWhileGenerating: boolean;
  setAutoScrollWhileGenerating: (value: boolean) => void;
  showScrollToBottomButton: boolean;
  setShowScrollToBottomButton: (value: boolean) => void;
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
      ...normalizeComposerPreferences(null),
      setPlainTextComposer: (plainTextComposer) => set({ plainTextComposer }),
      setShowContextWindowUsage: (showContextWindowUsage) =>
        set({ showContextWindowUsage }),
      setSendShortcut: (sendShortcut) => set({ sendShortcut }),
      setFollowUpBehavior: (followUpBehavior) => set({ followUpBehavior }),
      confirmDeleteChats: true,
      setConfirmDeleteChats: (confirmDeleteChats) =>
        set({ confirmDeleteChats }),
      alwaysDeleteChatFiles: false,
      setAlwaysDeleteChatFiles: (alwaysDeleteChatFiles) =>
        set({ alwaysDeleteChatFiles }),
      showModelDisclaimer: false,
      setShowModelDisclaimer: (showModelDisclaimer) =>
        set({ showModelDisclaimer }),
      showResponseModel: false,
      setShowResponseModel: (showResponseModel) => set({ showResponseModel }),
      showInlineEditResponse: false,
      setShowInlineEditResponse: (showInlineEditResponse) =>
        set({ showInlineEditResponse }),
      thinkingVisibility: DEFAULT_THINKING_VISIBILITY,
      setThinkingVisibility: (thinkingVisibility) =>
        set({ thinkingVisibility }),
      toolVisibility: DEFAULT_TOOL_VISIBILITY,
      setToolVisibility: (toolVisibility) => set({ toolVisibility }),
      foldToolActivityIntoThinking: false,
      setFoldToolActivityIntoThinking: (foldToolActivityIntoThinking) =>
        set({ foldToolActivityIntoThinking }),
      pastedTextMinChars: PASTED_TEXT_DEFAULT_MIN_CHARS,
      setPastedTextMinChars: (pastedTextMinChars) =>
        set({ pastedTextMinChars }),
      autoScrollWhileGenerating: true,
      setAutoScrollWhileGenerating: (autoScrollWhileGenerating) =>
        set({ autoScrollWhileGenerating }),
      showScrollToBottomButton: true,
      setShowScrollToBottomButton: (showScrollToBottomButton) =>
        set({ showScrollToBottomButton }),
    }),
    {
      name: "unsloth_chat_preferences",
      merge: (persisted, current) => {
        const saved = persisted as Partial<ChatPreferencesState> | undefined;
        // Records written before the three-state settings carry two booleans instead.
        const legacy = persisted as
          | {
              collapseThinkingByDefault?: unknown;
              collapseToolActivityByDefault?: unknown;
            }
          | undefined;
        return {
          ...current,
          ...normalizeComposerPreferences(saved),
          confirmDeleteChats: saved?.confirmDeleteChats ?? true,
          alwaysDeleteChatFiles: saved?.alwaysDeleteChatFiles ?? false,
          showModelDisclaimer: saved?.showModelDisclaimer ?? false,
          showResponseModel: saved?.showResponseModel ?? false,
          showInlineEditResponse: saved?.showInlineEditResponse ?? false,
          thinkingVisibility: migrateVisibility(
            saved?.thinkingVisibility,
            legacy?.collapseThinkingByDefault,
            DEFAULT_THINKING_VISIBILITY,
          ),
          toolVisibility: migrateVisibility(
            saved?.toolVisibility,
            legacy?.collapseToolActivityByDefault,
            DEFAULT_TOOL_VISIBILITY,
          ),
          foldToolActivityIntoThinking:
            saved?.foldToolActivityIntoThinking ?? false,
          pastedTextMinChars: normalisePastedTextMinChars(
            saved?.pastedTextMinChars,
          ),
          autoScrollWhileGenerating: saved?.autoScrollWhileGenerating ?? true,
          showScrollToBottomButton: saved?.showScrollToBottomButton ?? true,
        };
      },
    },
  ),
);
