// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Why the loaded model cannot take the audio or video attached to this turn, or null. A file
 *  attached before any model was loaded skips the attach-time check, so it meets its model here. */
export function attachedMediaUnavailableReason({
  activeModel,
  checkpoint,
  modelLabel,
  audio,
  video,
}: {
  activeModel: { hasAudioInput?: boolean; hasVideoInput?: boolean } | undefined;
  checkpoint: string | null | undefined;
  /** What the toast calls the model: its name, else the connection's model id, else the id. */
  modelLabel: string;
  audio: boolean;
  video: boolean;
}): string | null {
  if (!checkpoint || (!audio && !video)) return null;
  if (audio && !activeModel?.hasAudioInput) {
    return `${modelLabel} cannot accept audio. Load an audio-input model, or remove the audio file.`;
  }
  if (video && !activeModel?.hasVideoInput) {
    return `${modelLabel} cannot accept video. Load a model that reads video, or remove the video.`;
  }
  return null;
}
