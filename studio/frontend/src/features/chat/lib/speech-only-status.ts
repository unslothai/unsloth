// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Own module so node tests can import it without the chat runtime store.

export type SpeechOnlyStatusInput = {
  active_model?: string | null;
  is_audio?: boolean;
  audio_type?: string | null;
};

/** Speech models are answered by synthesis, so chat must never adopt one. */
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
