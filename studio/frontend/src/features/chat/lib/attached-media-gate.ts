// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function attachedMediaUnavailableReason({
  activeModel,
  checkpoint,
  modelLabel,
  audio,
  audioCount = audio ? 1 : 0,
  video,
}: {
  activeModel:
    | { hasAudioInput?: boolean; hasVideoInput?: boolean; isMlx?: boolean }
    | undefined;
  checkpoint: string | null | undefined;
  modelLabel: string;
  audio: boolean;
  /** Number of clips on the turn. */
  audioCount?: number;
  video: boolean;
}): string | null {
  if (!checkpoint || (!audio && !video)) return null;
  if (audio && !activeModel?.hasAudioInput) {
    return `${modelLabel} cannot accept audio. Load an audio-input model, or remove the audio file.`;
  }
  // MLX takes one clip per message (see maxAudioFilesFor).
  if (audio && audioCount > 1 && activeModel?.isMlx) {
    return `${modelLabel} takes one audio file per message. Remove the extra audio files, or load a GGUF model to send several.`;
  }
  if (video && !activeModel?.hasVideoInput) {
    return `${modelLabel} cannot accept video. Load a model that reads video, or remove the video.`;
  }
  return null;
}
