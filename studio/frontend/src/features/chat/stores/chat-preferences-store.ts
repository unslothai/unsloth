// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { syncAcrossWindows } from "../../../lib/sync-across-windows.ts";
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
  showResponseModel: boolean;
  setShowResponseModel: (value: boolean) => void;
  showLinkPreviews: boolean;
  setShowLinkPreviews: (value: boolean) => void;
  showReplyTimes: boolean;
  setShowReplyTimes: (value: boolean) => void;
  showGenerationDetails: boolean;
  setShowGenerationDetails: (value: boolean) => void;
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
  notifyWhenReplyFinishes: boolean;
  setNotifyWhenReplyFinishes: (value: boolean) => void;
  filesPanelCollapsed: boolean;
  setFilesPanelCollapsed: (value: boolean) => void;
  /** New chat composer height, 0 (top) to 100 (docked at the bottom); 50 is the middle. */
  newChatComposerPosition: number;
  setNewChatComposerPosition: (value: number) => void;
  showTurnNavigation: boolean;
  setShowTurnNavigation: (value: boolean) => void;
  richTextPaste: boolean;
  setRichTextPaste: (value: boolean) => void;
}

export const COMPOSER_POSITION_MIDDLE = 50;
export const COMPOSER_POSITION_BOTTOM = 100;
// Records from the old on/off setting carry only newChatComposerAtBottom.
export function normaliseComposerPosition(value: unknown, legacyBottom = false): number {
  if (typeof value === "number" && Number.isFinite(value))
    return Math.max(0, Math.min(COMPOSER_POSITION_BOTTOM, Math.round(value)));
  return legacyBottom ? COMPOSER_POSITION_BOTTOM : COMPOSER_POSITION_MIDDLE;
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
      showResponseModel: false,
      setShowResponseModel: (showResponseModel) => set({ showResponseModel }),
      showLinkPreviews: true,
      setShowLinkPreviews: (showLinkPreviews) => set({ showLinkPreviews }),
      showReplyTimes: false,
      setShowReplyTimes: (showReplyTimes) => set({ showReplyTimes }),
      showGenerationDetails: false,
      setShowGenerationDetails: (showGenerationDetails) =>
        set({ showGenerationDetails }),
      showInlineEditResponse: true,
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
      notifyWhenReplyFinishes: false,
      setNotifyWhenReplyFinishes: (notifyWhenReplyFinishes) =>
        set({ notifyWhenReplyFinishes }),
      filesPanelCollapsed: false,
      setFilesPanelCollapsed: (filesPanelCollapsed) =>
        set({ filesPanelCollapsed }),
      newChatComposerPosition: COMPOSER_POSITION_MIDDLE,
      setNewChatComposerPosition: (value) =>
        set({ newChatComposerPosition: normaliseComposerPosition(value) }),
      showTurnNavigation: true,
      setShowTurnNavigation: (showTurnNavigation) => set({ showTurnNavigation }),
      richTextPaste: true,
      setRichTextPaste: (richTextPaste) => set({ richTextPaste }),
    }),
    {
      name: "unsloth_chat_preferences",
      // v0 stored the old off default for everyone, so it cannot tell a choice from the default.
      version: 1,
      migrate: (persisted, version) =>
        version < 1 && persisted && typeof persisted === "object"
          ? { ...persisted, showInlineEditResponse: true }
          : persisted,
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
          showResponseModel: saved?.showResponseModel ?? false,
          showLinkPreviews: saved?.showLinkPreviews !== false,
          showReplyTimes: saved?.showReplyTimes === true,
          showGenerationDetails: saved?.showGenerationDetails === true,
          showInlineEditResponse: saved?.showInlineEditResponse ?? true,
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
          notifyWhenReplyFinishes: saved?.notifyWhenReplyFinishes === true,
          filesPanelCollapsed: saved?.filesPanelCollapsed === true,
          showTurnNavigation: saved?.showTurnNavigation !== false,
          richTextPaste: saved?.richTextPaste !== false,
          newChatComposerPosition: normaliseComposerPosition(
            saved?.newChatComposerPosition,
            (persisted as { newChatComposerAtBottom?: unknown } | undefined)
              ?.newChatComposerAtBottom === true,
          ),
        };
      },
    },
  ),
);

syncAcrossWindows(useChatPreferencesStore);
