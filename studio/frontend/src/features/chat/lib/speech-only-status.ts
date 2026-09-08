// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Its own module so the node suite can drive it: apply-inference-status-to-store pulls in the whole chat runtime store.

export type SpeechOnlyStatusInput = {
  active_model?: string | null;
  is_audio?: boolean;
  audio_type?: string | null;
};

/** Whether the resident model emits speech. The Audio page loads these into the single slot chat
 *  reads, and openai_chat_completions answers a turn on one by SYNTHESIZING the prompt, so chat
 *  must never adopt one. The same condition that route branches on, plus audio_vlm: Gemma 3n
 *  should never set `is_audio`, and misreading one would lock a real chat model out of chat. */
export function isSpeechOnlyStatus(status: SpeechOnlyStatusInput): boolean {
  return (
    Boolean(status.is_audio) &&
    status.audio_type !== "whisper" &&
    status.audio_type !== "audio_vlm"
  );
}

export function isIdleUnloadedStatus(
  status: SpeechOnlyStatusInput,
  idleUnloadArmed: boolean,
): boolean {
  return idleUnloadArmed && !status.active_model;
}
