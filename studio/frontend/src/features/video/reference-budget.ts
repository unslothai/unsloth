// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** MiniMax-H3's combined budget across picture, video and standalone-audio references. */
export const MAX_H3_REFERENCES = 12;

export function hasReferenceCapacity(
  images: number,
  videos: number,
  audios: number,
): boolean {
  return images + videos + audios < MAX_H3_REFERENCES;
}

export type ReferenceKind = "video" | "audio";

/**
 * The largest reference file of each kind the backend will take.
 *
 * The caps in models/inference.py bound the BASE64 STRING, not the file: 96 MiB for a reference
 * video and 32 MiB for its soundtrack. Base64 costs 4 bytes per 3, so the raw file limits are
 * three quarters of those.
 */
export const MAX_REFERENCE_BYTES: Record<ReferenceKind, number> = {
  video: (96 * 1024 * 1024 * 3) / 4,
  audio: (32 * 1024 * 1024 * 3) / 4,
};

/** Why this file cannot be staged as a reference, or null when it can. */
export function referenceFileRejection(
  kind: ReferenceKind,
  file: { type: string; size: number },
): string | null {
  if (!file.type.startsWith(`${kind}/`)) {
    return `Please choose ${kind === "video" ? "a video" : "an audio"} file`;
  }
  if (file.size > MAX_REFERENCE_BYTES[kind]) {
    const limitMb = Math.round(MAX_REFERENCE_BYTES[kind] / (1024 * 1024));
    return `This ${kind} is too large (limit ${limitMb} MB)`;
  }
  return null;
}

/**
 * Read one reference file into the data URL the request carries.
 *
 * The size check happens BEFORE the FileReader exists, because a data URL costs roughly 2.33x the
 * file in renderer memory and it is built long before the backend's 422 can arrive. H3 reference
 * clips are 2 to 15 seconds by spec, so a 15 second 4K phone clip clears the cap routinely; chat
 * attachments, OpenDocument imports and the seed dialog all guard the same way.
 */
export function readReferenceFile(
  kind: ReferenceKind,
  file: File | undefined | null,
  handlers: {
    onLoaded: (dataUrl: string | null) => void;
    onError: (message: string) => void;
  },
): void {
  if (!file) return;
  const rejection = referenceFileRejection(kind, file);
  if (rejection !== null) {
    handlers.onError(rejection);
    return;
  }
  const reader = new FileReader();
  reader.onload = () =>
    handlers.onLoaded(typeof reader.result === "string" ? reader.result : null);
  reader.onerror = () => handlers.onError(`Could not read the ${kind} file`);
  reader.readAsDataURL(file);
}
