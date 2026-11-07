// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MicTarget } from "@/features/settings/stores/voice-settings-store";
import {
  MAX_AUDIO_SIZE,
  getAudioAddError,
  isAudioAttachmentFile,
} from "../../../lib/audio-utils.ts";

/** The mic reaches the model only while the loaded one takes audio; otherwise it transcribes. */
export function micSendsAudioToModel(
  micTarget: MicTarget,
  modelHearsAudio: boolean,
): boolean {
  return micTarget === "model" && modelHearsAudio;
}

export function micClipFilename(now: Date = new Date()): string {
  const stamp = now.toISOString().slice(0, 19).replaceAll(":", "-");
  return `recording-${stamp}.wav`;
}

// One second at the recorder's preferred 16 kHz; less would hold no usable clip.
export const MIN_MIC_CLIP_BYTES = 44 + 16_000 * 2;

export type MicClipRoom = { maxBytes: number } | { reason: string };

/** What the composer can still take for one more clip, by the audio attachment adapter's own limits. */
export function micClipRoom(
  attached: readonly { name: string; type: string; size: number }[],
  maxFiles: number,
  stagedClip: boolean,
): MicClipRoom {
  if (stagedClip) {
    return {
      reason: "Send or remove the staged audio clip before attaching more audio.",
    };
  }
  const audio = attached.filter(isAudioAttachmentFile);
  const totalSize = audio.reduce((sum, file) => sum + file.size, 0);
  const reason = getAudioAddError(
    audio.length,
    totalSize,
    MIN_MIC_CLIP_BYTES,
    maxFiles,
  );
  return reason ? { reason } : { maxBytes: MAX_AUDIO_SIZE - totalSize };
}
