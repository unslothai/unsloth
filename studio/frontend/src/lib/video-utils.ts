// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Containers llama-server can decode. It shells out to ffmpeg, so this is
 * what ffmpeg reads, not what the webview can play. Extensions ride along
 * because MIME is unreliable for mkv and some mov files. */
export const VIDEO_ACCEPT =
  "video/mp4,video/quicktime,video/webm,video/x-matroska,video/x-msvideo,.mp4,.mov,.webm,.mkv,.avi";

// Matches _MAX_VIDEO_B64_CHARS in the backend, so the composer does not accept
// a clip the route refuses. The native reader's cap is a higher backstop.
const MAX_VIDEO_SIZE_MB = 64;
export const MAX_VIDEO_SIZE = MAX_VIDEO_SIZE_MB * 1024 * 1024;
export const MAX_VIDEO_SIZE_LABEL = `${MAX_VIDEO_SIZE_MB}MB`;

export function getVideoSizeError(size: number): string | null {
  return size > MAX_VIDEO_SIZE
    ? `Video size exceeds ${MAX_VIDEO_SIZE_LABEL} limit`
    : null;
}

// Mirrors the extension table in native_intents.rs, which the parity test keeps
// in step: a clip that arrives through the desktop reader and one picked in the
// browser must reach the route as the same mime type.
const VIDEO_MIME_BY_EXTENSION: Record<string, string> = {
  ".mp4": "video/mp4",
  ".mov": "video/quicktime",
  ".webm": "video/webm",
  ".mkv": "video/x-matroska",
  ".avi": "video/x-msvideo",
};

const VIDEO_EXTENSIONS = Object.keys(VIDEO_MIME_BY_EXTENSION);
const VIDEO_MIME_RE = /^video\//i;

/** The mime type to send a picked clip under.
 *
 * The accept list carries extensions as well as mime types because the browser's
 * answer is unreliable for mkv and some mov files, so a file the picker took on
 * its extension can arrive as "" or as application/octet-stream. Both are then
 * carried into the attachment, and the request builder only recognises a file
 * part whose mimeType matches ^video/, so the clip is dropped and the model
 * answers as though it were never attached. Trust the extension whenever the
 * browser did not say video.
 */
export function videoMimeForFile(file: File): string {
  if (VIDEO_MIME_RE.test(file.type)) return file.type;
  const name = file.name.toLowerCase();
  for (const [ext, mime] of Object.entries(VIDEO_MIME_BY_EXTENSION)) {
    if (name.endsWith(ext)) return mime;
  }
  return "video/mp4";
}

/** Whether a picked file is a video. mkv and some mov files arrive with an
 * empty MIME type, hence the extension fallback. */
export function isVideoFile(file: File): boolean {
  if (VIDEO_MIME_RE.test(file.type)) {
    return true;
  }
  const name = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext));
}
