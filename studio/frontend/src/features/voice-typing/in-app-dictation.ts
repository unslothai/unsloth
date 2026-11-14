// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { invoke } from "@tauri-apps/api/core";

import { playVoiceTypingCue } from "./voice-typing-sounds";
import { useVoiceTypingStore } from "./voice-typing-store";

/** Apply the voice-typing audio settings to in-app composer dictation. */
export function setInAppDictationMuted(muted: boolean): void {
  if (!isTauri) {
    return;
  }
  invoke("voice_typing_set_audio_mute", { mute: muted }).catch(() => {
    // Muting is best effort; dictation must never fail because of it.
  });
}

export function playInAppDictationCue(cue: "start" | "stop" | "cancel"): void {
  const { soundEnabled, soundTheme, soundVolume } =
    useVoiceTypingStore.getState();
  if (soundEnabled) {
    playVoiceTypingCue(cue, soundTheme, soundVolume);
  }
}
