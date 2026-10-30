// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export {
  DEFAULT_VOICE_TYPING_SHORTCUT,
  chordFromEvent,
  formatShortcut,
  isConflictingShortcut,
  isModifierCode,
} from "./shortcut";
export { useVoiceTyping } from "./use-voice-typing";
export {
  MIC_WARM_ALWAYS,
  VOICE_TYPING_HUD_POSITIONS,
  VOICE_TYPING_PASTE_METHODS,
  useVoiceTypingStore,
  type VoiceTypingHudPosition,
  type VoiceTypingMode,
  type VoiceTypingPasteMethod,
  type VoiceTypingReadiness,
} from "./voice-typing-store";
export {
  previewVoiceTypingSound,
  VOICE_TYPING_SOUND_THEMES,
  type VoiceTypingCue,
  type VoiceTypingSoundTheme,
} from "./voice-typing-sounds";
export {
  requestVoiceTypingModelDownload,
  type VoiceTypingModelState,
} from "./voice-typing-preflight";
