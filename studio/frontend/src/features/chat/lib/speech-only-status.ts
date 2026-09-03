// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Its own module so the node suite can drive it: apply-inference-status-to-store pulls
// in the whole chat runtime store.

export type SpeechOnlyStatusInput = {
  active_model?: string | null;
  is_audio?: boolean;
  audio_type?: string | null;
};

/**
 * Whether the resident audio model cannot answer a normal text-chat turn. The Audio
 * page loads both speech generators and Whisper/STT checkpoints into the single slot
 * chat reads, so chat must never adopt either kind.
 *
 * Audio VLMs remain eligible: Gemma 3n should never set `is_audio`, and treating a
 * defensive true as non-chat would lock a real multimodal model out of chat.
 */
export function isSpeechOnlyStatus(status: SpeechOnlyStatusInput): boolean {
  return Boolean(status.is_audio) && status.audio_type !== "audio_vlm";
}

export function isIdleUnloadedStatus(
  status: SpeechOnlyStatusInput,
  idleUnloadArmed: boolean,
): boolean {
  return idleUnloadArmed && !status.active_model;
}
