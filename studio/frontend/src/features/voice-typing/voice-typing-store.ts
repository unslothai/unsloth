// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Persisted preferences for system-wide voice typing. The controller reads them at
// call time, so changes apply without a reload.

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { DEFAULT_VOICE_TYPING_SHORTCUT } from "./shortcut";

export type VoiceTypingMode = "hold" | "toggle" | "hybrid";

/** What the local dictation stack can do right now; refreshed by the controller. */
export type VoiceTypingReadiness =
  | "unknown"
  | "ready"
  | "loading"
  | "will-load"
  | "needs-download"
  | "engine-missing"
  | "custom";

interface VoiceTypingSettings {
  /** Master switch; when off, pressing the shortcut explains how to turn it on. */
  enabled: boolean;
  shortcut: string;
  mode: VoiceTypingMode;
  /** Show the floating recording indicator on screen. */
  hudEnabled: boolean;
  /** Show the words as they are recognised inside the indicator. */
  hudShowTranscript: boolean;
  /** Short tones when recording starts and stops. */
  soundEnabled: boolean;
  /** Append a space so consecutive dictations do not run together. */
  appendTrailingSpace: boolean;
  /** Off: the previous clipboard content is restored right after pasting. */
  keepOnClipboard: boolean;
  /** Load the dictation model as soon as voice typing is switched on. */
  preloadModel: boolean;
  /** End a dictation by itself when the speaker pauses for long enough. */
  autoStopSilence: boolean;
  /** Silence before the automatic stop, in milliseconds. */
  autoStopDelayMs: number;
  /** How long the microphone stays open after a dictation for an instant restart. */
  micWarmSeconds: number;
  /** Drop "um", "uh" and similar sounds from each segment. */
  removeFillerWords: boolean;
  /** Turn spoken punctuation ("comma", "new line") into marks. */
  spokenPunctuation: boolean;
  /** Press Enter after inserting, for chat boxes that send on Enter. */
  submitAfterInsert: boolean;
  /** Off: the transcript waits on the clipboard for a manual paste. */
  pasteAutomatically: boolean;
  /** Runtime view of the dictation model, not a preference. */
  readiness: VoiceTypingReadiness;
  setEnabled: (enabled: boolean) => void;
  setShortcut: (shortcut: string) => void;
  setMode: (mode: VoiceTypingMode) => void;
  setHudEnabled: (enabled: boolean) => void;
  setHudShowTranscript: (enabled: boolean) => void;
  setSoundEnabled: (enabled: boolean) => void;
  setAppendTrailingSpace: (enabled: boolean) => void;
  setKeepOnClipboard: (enabled: boolean) => void;
  setPreloadModel: (enabled: boolean) => void;
  setAutoStopSilence: (enabled: boolean) => void;
  setAutoStopDelayMs: (delayMs: number) => void;
  setMicWarmSeconds: (seconds: number) => void;
  setRemoveFillerWords: (enabled: boolean) => void;
  setSpokenPunctuation: (enabled: boolean) => void;
  setSubmitAfterInsert: (enabled: boolean) => void;
  setPasteAutomatically: (enabled: boolean) => void;
  setReadiness: (readiness: VoiceTypingReadiness) => void;
}

export const useVoiceTypingStore = create<VoiceTypingSettings>()(
  persist(
    (set) => ({
      // Ships armed: pressing the shortcut is the only thing that ever happens,
      // and every off or failed path now explains itself. Off reads as "broken".
      enabled: true,
      shortcut: DEFAULT_VOICE_TYPING_SHORTCUT,
      mode: "hybrid",
      hudEnabled: true,
      hudShowTranscript: true,
      soundEnabled: true,
      appendTrailingSpace: true,
      keepOnClipboard: false,
      preloadModel: true,
      autoStopSilence: true,
      autoStopDelayMs: 2500,
      micWarmSeconds: 15,
      removeFillerWords: true,
      spokenPunctuation: false,
      submitAfterInsert: false,
      pasteAutomatically: true,
      readiness: "unknown",
      setEnabled: (enabled) => set({ enabled }),
      setShortcut: (shortcut) => set({ shortcut }),
      setMode: (mode) => set({ mode }),
      setHudEnabled: (hudEnabled) => set({ hudEnabled }),
      setHudShowTranscript: (hudShowTranscript) => set({ hudShowTranscript }),
      setSoundEnabled: (soundEnabled) => set({ soundEnabled }),
      setAppendTrailingSpace: (appendTrailingSpace) =>
        set({ appendTrailingSpace }),
      setKeepOnClipboard: (keepOnClipboard) => set({ keepOnClipboard }),
      setPreloadModel: (preloadModel) => set({ preloadModel }),
      setAutoStopSilence: (autoStopSilence) => set({ autoStopSilence }),
      setAutoStopDelayMs: (autoStopDelayMs) => set({ autoStopDelayMs }),
      setMicWarmSeconds: (micWarmSeconds) => set({ micWarmSeconds }),
      setRemoveFillerWords: (removeFillerWords) => set({ removeFillerWords }),
      setSpokenPunctuation: (spokenPunctuation) => set({ spokenPunctuation }),
      setSubmitAfterInsert: (submitAfterInsert) => set({ submitAfterInsert }),
      setPasteAutomatically: (pasteAutomatically) =>
        set({ pasteAutomatically }),
      setReadiness: (readiness) => set({ readiness }),
    }),
    {
      name: "unsloth_voice_typing",
      version: 2,
      storage: createJSONStorage(() => localStorage),
      // Readiness is live state; restoring a stale value would mislabel the model.
      partialize: (state) => ({
        enabled: state.enabled,
        shortcut: state.shortcut,
        mode: state.mode,
        hudEnabled: state.hudEnabled,
        hudShowTranscript: state.hudShowTranscript,
        soundEnabled: state.soundEnabled,
        appendTrailingSpace: state.appendTrailingSpace,
        keepOnClipboard: state.keepOnClipboard,
        preloadModel: state.preloadModel,
        autoStopSilence: state.autoStopSilence,
        autoStopDelayMs: state.autoStopDelayMs,
        micWarmSeconds: state.micWarmSeconds,
        removeFillerWords: state.removeFillerWords,
        spokenPunctuation: state.spokenPunctuation,
        submitAfterInsert: state.submitAfterInsert,
        pasteAutomatically: state.pasteAutomatically,
      }),
      // Version 2 turns the shortcut on for stores written before the feature made
      // itself visible; an off shortcut that does nothing reads as broken.
      migrate: (persisted, version) => {
        const saved = (persisted ?? {}) as Partial<VoiceTypingSettings>;
        if (version < 2) return { ...saved, enabled: true };
        return saved;
      },
    },
  ),
);
