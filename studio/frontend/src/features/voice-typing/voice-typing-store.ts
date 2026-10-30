// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Persisted preferences for system-wide voice typing. The controller reads them at
// call time, so changes apply without a reload.

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import { DEFAULT_VOICE_TYPING_SHORTCUT } from "./shortcut";
import {
  VOICE_TYPING_SOUND_THEMES,
  type VoiceTypingSoundTheme,
} from "./voice-typing-sounds";

export type VoiceTypingMode = "hold" | "toggle" | "hybrid";

export const VOICE_TYPING_PASTE_METHODS = [
  "ctrl_v",
  "ctrl_shift_v",
  "shift_insert",
  "direct",
] as const;
export type VoiceTypingPasteMethod = (typeof VOICE_TYPING_PASTE_METHODS)[number];

export const VOICE_TYPING_HUD_POSITIONS = ["bottom", "top"] as const;
export type VoiceTypingHudPosition = (typeof VOICE_TYPING_HUD_POSITIONS)[number];

/** What the local dictation stack can do right now; refreshed by the controller. */
export type VoiceTypingReadiness =
  | "unknown"
  | "ready"
  | "loading"
  | "will-load"
  | "needs-download"
  | "engine-missing"
  | "custom";

/** `-1` keeps the microphone open until voice typing is switched off. */
export const MIC_WARM_ALWAYS = -1;

/** A stored payload can be old, hand-edited or truncated; repair the fields whose
 *  ranges matter before the UI or the controller ever sees them. */
function repair(
  saved: Partial<VoiceTypingSettings> & Record<string, unknown>,
): Partial<VoiceTypingSettings> {
  if (
    saved.micWarmSeconds !== MIC_WARM_ALWAYS &&
    (typeof saved.micWarmSeconds !== "number" ||
      !Number.isFinite(saved.micWarmSeconds) ||
      saved.micWarmSeconds < 0)
  ) {
    saved.micWarmSeconds = 15;
  }
  if (
    typeof saved.soundVolume !== "number" ||
    !Number.isFinite(saved.soundVolume)
  ) {
    saved.soundVolume = 0.55;
  } else {
    saved.soundVolume = Math.min(1, Math.max(0, saved.soundVolume));
  }
  if (
    typeof saved.soundTheme !== "string" ||
    !VOICE_TYPING_SOUND_THEMES.includes(
      saved.soundTheme as VoiceTypingSoundTheme,
    )
  ) {
    saved.soundTheme = "chime";
  }
  if (typeof saved.copyLastTranscript !== "boolean") {
    saved.copyLastTranscript = false;
  }
  if (
    typeof saved.pasteMethod !== "string" ||
    !VOICE_TYPING_PASTE_METHODS.includes(
      saved.pasteMethod as VoiceTypingPasteMethod,
    )
  ) {
    saved.pasteMethod = "ctrl_v";
  }
  if (
    typeof saved.hudPosition !== "string" ||
    !VOICE_TYPING_HUD_POSITIONS.includes(
      saved.hudPosition as VoiceTypingHudPosition,
    )
  ) {
    saved.hudPosition = "bottom";
  }
  if (typeof saved.muteWhileRecording !== "boolean") {
    saved.muteWhileRecording = false;
  }
  return saved;
}

interface VoiceTypingSettings {
  /** Master switch. Off by default: a global shortcut is a strong default. */
  enabled: boolean;
  shortcut: string;
  mode: VoiceTypingMode;
  /** Short tones when recording starts and stops. */
  soundEnabled: boolean;
  /** Which tone family the feedback sounds use. */
  soundTheme: VoiceTypingSoundTheme;
  /** Feedback volume, 0 to 1. */
  soundVolume: number;
  /** Append a space so consecutive dictations do not run together. */
  appendTrailingSpace: boolean;
  /** Copy the finished transcript once, in addition to pasting it. */
  copyLastTranscript: boolean;
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
  /** Keystroke strategy used when `pasteAutomatically` is on. */
  pasteMethod: VoiceTypingPasteMethod;
  /** Where the floating indicator sits on the active monitor's work area. */
  hudPosition: VoiceTypingHudPosition;
  /** Temporarily mute system audio playback while the microphone is recording. */
  muteWhileRecording: boolean;
  /** Runtime view of the dictation model, not a preference. */
  readiness: VoiceTypingReadiness;
  /** Runtime memory of the most recent transcript in this session for quick copy. */
  lastTranscript: string;
  setEnabled: (enabled: boolean) => void;
  setShortcut: (shortcut: string) => void;
  setMode: (mode: VoiceTypingMode) => void;
  setSoundEnabled: (enabled: boolean) => void;
  setSoundTheme: (theme: VoiceTypingSoundTheme) => void;
  setSoundVolume: (volume: number) => void;
  setAppendTrailingSpace: (enabled: boolean) => void;
  setCopyLastTranscript: (enabled: boolean) => void;
  setPreloadModel: (enabled: boolean) => void;
  setAutoStopSilence: (enabled: boolean) => void;
  setAutoStopDelayMs: (delayMs: number) => void;
  setMicWarmSeconds: (seconds: number) => void;
  setRemoveFillerWords: (enabled: boolean) => void;
  setSpokenPunctuation: (enabled: boolean) => void;
  setSubmitAfterInsert: (enabled: boolean) => void;
  setPasteAutomatically: (enabled: boolean) => void;
  setPasteMethod: (method: VoiceTypingPasteMethod) => void;
  setHudPosition: (position: VoiceTypingHudPosition) => void;
  setMuteWhileRecording: (enabled: boolean) => void;
  setReadiness: (readiness: VoiceTypingReadiness) => void;
  setLastTranscript: (text: string) => void;
}

export const useVoiceTypingStore = create<VoiceTypingSettings>()(
  persist(
    (set) => ({
      enabled: false,
      shortcut: DEFAULT_VOICE_TYPING_SHORTCUT,
      mode: "hybrid",
      soundEnabled: true,
      soundTheme: "chime",
      soundVolume: 0.55,
      appendTrailingSpace: true,
      copyLastTranscript: false,
      preloadModel: true,
      autoStopSilence: true,
      autoStopDelayMs: 2500,
      micWarmSeconds: 15,
      removeFillerWords: true,
      spokenPunctuation: false,
      submitAfterInsert: false,
      pasteAutomatically: true,
      pasteMethod: "ctrl_v",
      hudPosition: "bottom",
      muteWhileRecording: false,
      readiness: "unknown",
      lastTranscript: "",
      setEnabled: (enabled) => set({ enabled }),
      setShortcut: (shortcut) => set({ shortcut }),
      setMode: (mode) => set({ mode }),
      setSoundEnabled: (soundEnabled) => set({ soundEnabled }),
      setSoundTheme: (soundTheme) => set({ soundTheme }),
      setSoundVolume: (soundVolume) =>
        set({ soundVolume: Math.min(1, Math.max(0, soundVolume)) }),
      setAppendTrailingSpace: (appendTrailingSpace) =>
        set({ appendTrailingSpace }),
      setCopyLastTranscript: (copyLastTranscript) =>
        set({ copyLastTranscript }),
      setPreloadModel: (preloadModel) => set({ preloadModel }),
      setAutoStopSilence: (autoStopSilence) => set({ autoStopSilence }),
      setAutoStopDelayMs: (autoStopDelayMs) => set({ autoStopDelayMs }),
      setMicWarmSeconds: (micWarmSeconds) => set({ micWarmSeconds }),
      setRemoveFillerWords: (removeFillerWords) => set({ removeFillerWords }),
      setSpokenPunctuation: (spokenPunctuation) => set({ spokenPunctuation }),
      setSubmitAfterInsert: (submitAfterInsert) => set({ submitAfterInsert }),
      setPasteAutomatically: (pasteAutomatically) =>
        set({ pasteAutomatically }),
      setPasteMethod: (pasteMethod) => set({ pasteMethod }),
      setHudPosition: (hudPosition) => set({ hudPosition }),
      setMuteWhileRecording: (muteWhileRecording) =>
        set({ muteWhileRecording }),
      setReadiness: (readiness) => set({ readiness }),
      setLastTranscript: (lastTranscript) => set({ lastTranscript }),
    }),
    {
      name: "unsloth_voice_typing",
      version: 3,
      storage: createJSONStorage(() => localStorage),
      // Readiness and lastTranscript are live state; restoring a stale value would mislabel the model.
      partialize: (state) => ({
        enabled: state.enabled,
        shortcut: state.shortcut,
        mode: state.mode,
        soundEnabled: state.soundEnabled,
        soundTheme: state.soundTheme,
        soundVolume: state.soundVolume,
        appendTrailingSpace: state.appendTrailingSpace,
        copyLastTranscript: state.copyLastTranscript,
        preloadModel: state.preloadModel,
        autoStopSilence: state.autoStopSilence,
        autoStopDelayMs: state.autoStopDelayMs,
        micWarmSeconds: state.micWarmSeconds,
        removeFillerWords: state.removeFillerWords,
        spokenPunctuation: state.spokenPunctuation,
        submitAfterInsert: state.submitAfterInsert,
        pasteAutomatically: state.pasteAutomatically,
        pasteMethod: state.pasteMethod,
        hudPosition: state.hudPosition,
        muteWhileRecording: state.muteWhileRecording,
      }),
      // Version 3 makes the master switch opt-in. Earlier builds armed the global
      // shortcut by default, which is too strong a default for an app-wide hook.
      // The old "keep transcript on clipboard" flag becomes the explicit copy-once
      // preference so nobody loses a workflow they had switched on.
      migrate: (persisted, version) => {
        const source = (persisted ?? {}) as Record<string, unknown>;
        const saved = Object.fromEntries(
          Object.entries(source).filter(
            ([key]) =>
              key !== "hudEnabled" &&
              key !== "hudShowTranscript" &&
              key !== "keepOnClipboard",
          ),
        ) as Partial<VoiceTypingSettings>;
        if (version < 3) {
          saved.copyLastTranscript = source.keepOnClipboard === true;
          saved.enabled = false;
        }
        return repair(saved);
      },
      // The same repairs apply to a current-version payload: a corrupt or truncated
      // write must not put an out-of-range value in front of the user.
      merge: (persisted, current) =>
        ({
          ...current,
          ...repair(
            (persisted ?? {}) as Partial<VoiceTypingSettings> &
              Record<string, unknown>,
          ),
        }) as VoiceTypingSettings,
    },
  ),
);
