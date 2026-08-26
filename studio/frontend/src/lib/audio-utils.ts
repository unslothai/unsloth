// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Containers the backend can reach wav/mp3 from: wav and mp3 pass through, the
// rest decode via libsndfile, PyAV's bundled FFmpeg, or finally librosa.
export const AUDIO_ACCEPT =
  "audio/wav,audio/mpeg,audio/webm,audio/ogg,audio/opus,audio/flac,audio/mp4,audio/aac,audio/aiff,audio/x-aiff,audio/x-caf,audio/x-ms-wma,audio/amr,audio/3gpp";
// Keep in sync with STT_AUDIO_RAW_MAX_BYTES in the backend upload limits.
const MAX_AUDIO_SIZE_MB = 25;
export const MAX_AUDIO_SIZE = MAX_AUDIO_SIZE_MB * 1024 * 1024;
export const MAX_AUDIO_SIZE_LABEL = `${MAX_AUDIO_SIZE_MB}MB`;

export function getAudioSizeError(size: number): string | null {
  return size > MAX_AUDIO_SIZE
    ? `Audio size exceeds ${MAX_AUDIO_SIZE_LABEL} limit`
    : null;
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const commaIndex = result.indexOf(",");
      resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}
