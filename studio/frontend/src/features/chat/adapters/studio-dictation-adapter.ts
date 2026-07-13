// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { StudioModelDictationAdapter } from "./studio-model-dictation-adapter";
import {
  type StudioDictationSession,
  StudioWebSpeechDictationAdapter,
} from "./studio-web-speech-dictation-adapter";

/**
 * Routes dictation to the engine chosen in Voice settings, resolved at listen()
 * time so switching engines applies without reloading the chat runtime.
 * Falls back to whichever engine the browser supports.
 */
export class StudioDictationAdapter implements DictationAdapter {
  static isSupported(): boolean {
    return (
      StudioWebSpeechDictationAdapter.isSupported() ||
      StudioModelDictationAdapter.isSupported()
    );
  }

  listen(): StudioDictationSession {
    const { dictationEngine } = useVoiceSettingsStore.getState();
    const preferModel =
      dictationEngine === "model" && StudioModelDictationAdapter.isSupported();

    if (preferModel) {
      return new StudioModelDictationAdapter().listen();
    }
    if (StudioWebSpeechDictationAdapter.isSupported()) {
      return new StudioWebSpeechDictationAdapter().listen();
    }
    // "browser" was requested but is unsupported here (e.g. Firefox): use the
    // model engine when it is available before giving up.
    if (StudioModelDictationAdapter.isSupported()) {
      return new StudioModelDictationAdapter().listen();
    }
    throw new Error("Dictation is not supported in this browser.");
  }
}
