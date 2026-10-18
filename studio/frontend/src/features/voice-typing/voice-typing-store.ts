// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Persisted preferences for system-wide voice typing. The controller reads them at
// call time, so changes apply without a reload.

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { DEFAULT_VOICE_TYPING_SHORTCUT } from "./shortcut";

export type VoiceTypingMode = "hold" | "toggle" | "hybrid";

interface VoiceTypingSettings {
  /** Master switch; when off the shortcut is inert and the overlay stays hidden. */
  enabled: boolean;
  shortcut: string;
  mode: VoiceTypingMode;
  /** Show the floating recording indicator on screen. */
  hudEnabled: boolean;
  /** Short tones when recording starts and stops. */
  soundEnabled: boolean;
  /** Append a space so consecutive dictations do not run together. */
  appendTrailingSpace: boolean;
  /** Off: the previous clipboard content is restored right after pasting. */
  keepOnClipboard: boolean;
  setEnabled: (enabled: boolean) => void;
  setShortcut: (shortcut: string) => void;
  setMode: (mode: VoiceTypingMode) => void;
  setHudEnabled: (enabled: boolean) => void;
  setSoundEnabled: (enabled: boolean) => void;
  setAppendTrailingSpace: (enabled: boolean) => void;
  setKeepOnClipboard: (enabled: boolean) => void;
}

export const useVoiceTypingStore = create<VoiceTypingSettings>()(
  persist(
    (set) => ({
      enabled: false,
      shortcut: DEFAULT_VOICE_TYPING_SHORTCUT,
      mode: "hybrid",
      hudEnabled: true,
      soundEnabled: true,
      appendTrailingSpace: true,
      keepOnClipboard: false,
      setEnabled: (enabled) => set({ enabled }),
      setShortcut: (shortcut) => set({ shortcut }),
      setMode: (mode) => set({ mode }),
      setHudEnabled: (hudEnabled) => set({ hudEnabled }),
      setSoundEnabled: (soundEnabled) => set({ soundEnabled }),
      setAppendTrailingSpace: (appendTrailingSpace) =>
        set({ appendTrailingSpace }),
      setKeepOnClipboard: (keepOnClipboard) => set({ keepOnClipboard }),
    }),
    {
      name: "unsloth_voice_typing",
      version: 1,
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
