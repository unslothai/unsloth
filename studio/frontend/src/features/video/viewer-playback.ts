// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What passes between the inline player and the viewer's, each way. */
export type Playback = {
  time: number;
  playing: boolean;
  muted: boolean;
  volume: number;
};

type PlayerState = Pick<
  HTMLMediaElement,
  "currentTime" | "paused" | "ended" | "muted" | "volume" | "readyState"
>;

// HTMLMediaElement.HAVE_METADATA, spelled out so this also runs where there is no DOM.
const HAVE_METADATA = 1;

/**
 * Where `video` is. Until it is `positioned` (by default, until it has metadata) its currentTime
 * reads 0 whatever it was asked to start at, it has not started playing, and its volume has not been
 * set, so those stay `fallback`'s. Whether it is muted is its own throughout: that is set as it
 * mounts, and the controls can change it before then.
 */
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

/**
 * Play, and if the browser refuses sound without a gesture (Safari and WKWebView do on a freshly
 * mounted element), play muted instead of not at all.
 */
export function playWithMutedFallback(video: Pick<HTMLMediaElement, "play" | "muted">): Promise<void> {
  return video.play().catch((error: unknown) => {
    // An AbortError is a newer load or pause taking over, not a refusal.
    if (video.muted || (error as { name?: unknown } | null)?.name !== "NotAllowedError") return;
    video.muted = true;
    return video.play().catch(() => undefined);
  });
}
