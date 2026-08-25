// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mirrors extract_video_id in studio/backend/core/youtube_transcript.py. The
// backend re-parses the URL, so this only decides whether to offer the prompt.

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
