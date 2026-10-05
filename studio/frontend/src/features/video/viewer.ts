// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export type Playback = { time: number; playing: boolean; muted: boolean; volume: number };

type PlayerState = Pick<
  HTMLMediaElement,
  "currentTime" | "paused" | "ended" | "muted" | "volume" | "readyState"
>;

// HTMLMediaElement.HAVE_METADATA, spelled out so this also runs where there is no DOM.
const HAVE_METADATA = 1;

export function readPlayback(
  video: PlayerState | null,
  fallback: Playback,
  positioned = video !== null && video.readyState >= HAVE_METADATA,
): Playback {
  if (!video) return fallback;
  if (!positioned) return { ...fallback, muted: video.muted };
  return {
    time: video.currentTime,
    playing: !video.paused && !video.ended,
    muted: video.muted,
    volume: video.volume,
  };
}

export function playWithMutedFallback(video: Pick<HTMLMediaElement, "play" | "muted">): Promise<void> {
  return video.play().catch((error: unknown) => {
    // An AbortError is a newer load or pause taking over, not a refusal.
    if (video.muted || (error as { name?: unknown } | null)?.name !== "NotAllowedError") return;
    video.muted = true;
    return video.play().catch(() => undefined);
  });
}

export async function fetchWithFreshLink(url: string, mint: () => Promise<string>): Promise<Response> {
  const response = await fetch(url);
  if (response.status !== 401 && response.status !== 403) return response;
  void response.body?.cancel();
  return fetch(await mint());
}
