// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Its own plain module so the node suite can drive it: apply-inference-status-to-store
// pulls in the whole chat runtime store.

export type SpeechOnlyStatusInput = {
  is_audio?: boolean;
  audio_type?: string | null;
};

/**
 * Whether the resident model emits speech rather than text. The Audio page loads
 * these into the same single inference slot chat reads, and openai_chat_completions
 * answers a chat turn on one by synthesizing the prompt instead of replying to it,
 * so chat must never adopt one as its own selection.
 *
 * The condition is the one that route itself branches on. whisper is excluded the
 * same way it excludes it; audio_vlm (Gemma 3n) is named too, though it should never
 * set `is_audio` -- it takes audio in and answers in text, so misreading one as
 * speech-only would lock a real chat model out of chat.
 */
export function isSpeechOnlyStatus(status: SpeechOnlyStatusInput): boolean {
  return (
    Boolean(status.is_audio) &&
    status.audio_type !== "whisper" &&
    status.audio_type !== "audio_vlm"
  );
}
