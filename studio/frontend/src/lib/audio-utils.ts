// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Containers the backend can reach wav/mp3 from: wav and mp3 pass through, the
// rest decode via libsndfile, PyAV's bundled FFmpeg, or finally librosa.
export const AUDIO_ACCEPT =
  "audio/wav,audio/mpeg,audio/webm,audio/ogg,audio/opus,audio/flac,audio/mp4,audio/aac,audio/aiff,audio/x-aiff,audio/x-caf,audio/x-ms-wma,audio/amr,audio/3gpp";
// Browsers report an empty or wrong type for several of these containers, so
// every surface that takes audio matches the name as well as the MIME.
export const AUDIO_ACCEPT_EXTENSIONS =
  ".wav,.mp3,.m4a,.ogg,.oga,.opus,.flac,.aac,.aiff,.aif,.aifc,.caf,.wma,.amr,.mp2";
/** What claims a file as audio: the MIME list plus those extensions. Keep .3gp
 *  out of it. A recording and a clip share that extension, and the composer's
 *  audio adapter is matched before its video one, so claiming the name here
 *  would take every 3GP video as audio. */
export const AUDIO_ATTACHMENT_ACCEPT = `${AUDIO_ACCEPT},audio/x-m4a,${AUDIO_ACCEPT_EXTENSIONS}`;
/**
 * What a file dialog should offer.
 *
 * Wider than the accept above, because a dialog only decides what is selectable
 * and a platform that maps .3gp to video/3gpp, or to nothing, greys out a voice
 * recording the surface would otherwise take. The tracks are read once the file
 * is in hand, and a real clip is refused then.
 */
export const AUDIO_PICKER_ACCEPT = `${AUDIO_ATTACHMENT_ACCEPT},.3gp`;

/** Whether a dropped or pasted file is audio, by MIME or, failing that, name. */
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
