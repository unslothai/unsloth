// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";

export interface YoutubeTranscript {
  videoId: string;
  url: string;
  title: string;
  author: string;
  lengthSeconds: number;
  language: string;
  languageCode: string;
  isGenerated: boolean;
  text: string;
  truncated: boolean;
}

/** Fetch a video's captions as plain text. `languages` is preference order. */
export async function fetchYoutubeTranscript(
  url: string,
  languages: string[],
  signal?: AbortSignal,
): Promise<YoutubeTranscript> {
  const response = await authFetch("/api/youtube/transcript", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, languages }),
    signal,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      body && typeof body === "object"
        ? formatFastApiDetail((body as { detail?: unknown }).detail)
        : null;
    throw new Error(
      detail ?? `Could not fetch the transcript (${response.status})`,
    );
  }
  return body as YoutubeTranscript;
}
