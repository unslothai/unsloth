// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors extract_video_id in core/youtube_transcript.py. The backend re-parses the URL, so this
// only decides whether to offer the prompt.

const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
// `www.` is stripped before the lookup, so only the bare forms are listed.
const WATCH_HOSTS = new Set([
  "youtube.com",
  "m.youtube.com",
  "music.youtube.com",
  "youtube-nocookie.com",
]);
const SHORT_HOSTS = new Set(["youtu.be"]);
const ID_PATH_PREFIXES = ["/shorts/", "/embed/", "/live/", "/v/"];
const MAX_CLIPBOARD_TEXT_LENGTH = 8192;
const WHITESPACE = /\s+/;
const LINE_BREAK = /\r?\n/;

/** The 11-character video id in a YouTube URL, or null if it is not one. */
export function extractYoutubeVideoId(value: string): string | null {
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > 2048) return null;

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return null;
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return null;

  const host = parsed.hostname.toLowerCase().replace(/^www\./, "");
  let candidate = "";
  if (SHORT_HOSTS.has(host)) {
    candidate = parsed.pathname.replace(/^\//, "").split("/", 1)[0] ?? "";
  } else if (!WATCH_HOSTS.has(host)) {
    return null;
  } else if (parsed.pathname.replace(/\/+$/, "") === "/watch") {
    candidate = parsed.searchParams.get("v") ?? "";
  } else {
    const prefix = ID_PATH_PREFIXES.find((p) => parsed.pathname.startsWith(p));
    if (prefix) {
      candidate = parsed.pathname.slice(prefix.length).split("/", 1)[0] ?? "";
    }
  }
  return VIDEO_ID.test(candidate) ? candidate : null;
}

function findYoutubeVideoUrl(value: string, uriList = false): string | null {
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > MAX_CLIPBOARD_TEXT_LENGTH) {
    return null;
  }
  for (const part of trimmed.split(uriList ? LINE_BREAK : WHITESPACE)) {
    const candidate = part.trim();
    if (uriList && candidate.startsWith("#")) continue;
    if (extractYoutubeVideoId(candidate)) {
      return candidate;
    }
  }
  return null;
}

export function extractYoutubeVideoUrlFromClipboard(
  clipboardData: { getData(type: string): string } | null,
): string | null {
  if (!clipboardData) return null;
  return (
    findYoutubeVideoUrl(clipboardData.getData("text/plain")) ??
    findYoutubeVideoUrl(clipboardData.getData("text/uri-list"), true)
  );
}
