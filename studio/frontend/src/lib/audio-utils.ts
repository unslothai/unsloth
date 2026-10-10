// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// wav and mp3 pass through; the rest decode via libsndfile, PyAV's FFmpeg, or librosa.
export const AUDIO_ACCEPT =
  "audio/wav,audio/mpeg,audio/webm,audio/ogg,audio/opus,audio/flac,audio/mp4,audio/aac,audio/aiff,audio/x-aiff,audio/x-caf,audio/x-ms-wma,audio/amr,audio/3gpp";
// Browsers report empty or wrong types for some of these, so match names too.
export const AUDIO_ACCEPT_EXTENSIONS =
  ".wav,.mp3,.m4a,.ogg,.oga,.opus,.flac,.aac,.aiff,.aif,.aifc,.caf,.wma,.amr,.mp2";
/** Keep .3gp out: the audio adapter is matched before video, so it would claim 3GP clips. */
export const AUDIO_ATTACHMENT_ACCEPT = `${AUDIO_ACCEPT},audio/x-m4a,${AUDIO_ACCEPT_EXTENSIONS}`;
/** Wider than the accept, since platforms map .3gp to video or nothing; clips are refused once
 * the tracks are read. */
export const AUDIO_PICKER_ACCEPT = `${AUDIO_ATTACHMENT_ACCEPT},.3gp`;

export function isAudioAttachmentFile(file: {
  name: string;
  type: string;
}): boolean {
  if (/^audio\//i.test(file.type)) {
    return true;
  }
  const name = file.name.toLowerCase();
  return AUDIO_ACCEPT_EXTENSIONS.split(",").some((ext) => name.endsWith(ext));
}

// Keep in sync with STT_AUDIO_RAW_MAX_BYTES in the backend upload limits.
const MAX_AUDIO_SIZE_MB = 25;
export const MAX_AUDIO_SIZE = MAX_AUDIO_SIZE_MB * 1024 * 1024;
export const MAX_AUDIO_SIZE_LABEL = `${MAX_AUDIO_SIZE_MB}MB`;

export function getAudioSizeError(size: number): string | null {
  return size > MAX_AUDIO_SIZE
    ? `Audio size exceeds ${MAX_AUDIO_SIZE_LABEL} limit`
    : null;
}

// Keep in sync with _MAX_AUDIO_CLIPS_PER_REQUEST. MAX_AUDIO_SIZE covers all clips together.
export const MAX_AUDIO_FILES = 8;

/** MLX takes one. */
export function maxAudioFilesFor(model: { isMlx?: boolean } | undefined): number {
  return model?.isMlx ? 1 : MAX_AUDIO_FILES;
}

export function getAudioAddError(
  count: number,
  totalSize: number,
  size: number,
  maxFiles: number = MAX_AUDIO_FILES,
): string | null {
  if (count >= maxFiles) {
    return maxFiles === 1
      ? "This model takes one audio file per message. Load a GGUF model to send several."
      : `Up to ${maxFiles} audio files can be attached per message.`;
  }
  if (count > 0 && totalSize + size > MAX_AUDIO_SIZE) {
    return `Audio files together exceed the ${MAX_AUDIO_SIZE_LABEL} per-message limit`;
  }
  return getAudioSizeError(size);
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
