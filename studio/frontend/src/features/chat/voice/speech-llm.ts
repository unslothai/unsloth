// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TTS_AUDIO_TYPES } from "./tts-audio-types.ts";

// A speech-LLM chat model (Orpheus, CSM, Spark...) produces its own voice, so a
// separate TTS voice slot doesn't apply and must not be auto-loaded. Shared by the
// render-time greying, the voice-slot ensure-load effect, and the self-heal
// reloader in chat-page.tsx so they cannot disagree.
//
// Name pattern, kept as the FALLBACK only: a checkpoint not yet in the inventory
// has no record to read, and its id is all there is to go on.
const SPEECH_LLM_CHECKPOINT_RE =
  /(?:orpheus|csm|spark|bark|parler|musicgen|text-to-speech|[-_]tts)/i;

/**
 * Whether the chat checkpoint speaks for itself. The model record's detected
 * codec (`audioType`) is the authority: a renamed fine-tune whose id carries none
 * of the keywords still reports "snac" or "csm", and treating it as a text model
 * would leave a previously picked voice slot loaded and preferred by
 * /audio/speech over the model's own voice.
 */
export function isSpeechLLMCheckpoint(
  checkpoint: string | null | undefined,
  audioType?: string | null,
): boolean {
  return (
    TTS_AUDIO_TYPES.has(audioType ?? "") ||
    SPEECH_LLM_CHECKPOINT_RE.test(checkpoint ?? "")
  );
}

/** The store-state form: the active checkpoint plus its inventory record. */
export function chatModelOwnsItsVoice(s: {
  params: { checkpoint: string | null | undefined };
  models: ReadonlyArray<{ id: string; audioType?: string | null }>;
}): boolean {
  const checkpoint = s.params.checkpoint;
  const audioType =
    s.models.find((m) => m.id === checkpoint)?.audioType ?? null;
  return isSpeechLLMCheckpoint(checkpoint, audioType);
}
