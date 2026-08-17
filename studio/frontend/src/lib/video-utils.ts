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

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".webm", ".mkv", ".avi"];
const VIDEO_MIME_RE = /^video\//i;

/** Whether a picked file is a video. mkv and some mov files arrive with an
 * empty MIME type, hence the extension fallback. */
export function isVideoFile(file: File): boolean {
  if (VIDEO_MIME_RE.test(file.type)) {
    return true;
  }
  const name = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => name.endsWith(ext));
}
