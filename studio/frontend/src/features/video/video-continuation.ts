// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VideoContinuationKeep, VideoStatus } from "./api.ts";

// The pipeline keeps 5 frames of the clip (two latent frames); the rest of the length is generated.
export const CONTINUATION_KEPT_FRAMES = 5;
// Mirrors the backend's CONTINUATION_MAX_SECONDS; the backend refuses a longer clip.
export const CONTINUATION_MAX_SECONDS = 15;

export const CONTINUATION_KEEP_OPTIONS: ReadonlyArray<{
  value: VideoContinuationKeep;
  label: string;
}> = [
  { value: "last", label: "From its last frames" },
  { value: "first", label: "From its first frames" },
];

export function supportsVideoContinuation(
  status: Pick<VideoStatus, "supports_video_continuation"> | null | undefined,
): boolean {
  return status?.supports_video_continuation === true;
}

export function continuationLengthError(numFrames: number): string | null {
  return numFrames > CONTINUATION_KEPT_FRAMES
    ? null
    : `Continuing a clip keeps ${CONTINUATION_KEPT_FRAMES} of its frames; pick a longer length so there is something to generate`;
}

/** A note for a picked clip the backend will refuse as too long, or null (unknown durations pass). */
export function continuationDurationWarning(durationSeconds: number | undefined): string | null {
  if (durationSeconds === undefined || durationSeconds <= CONTINUATION_MAX_SECONDS) return null;
  return `This clip is ${durationSeconds.toFixed(1)} s; clips to continue can be up to ${CONTINUATION_MAX_SECONDS} s. Trim it first.`;
}

export function continuationRequest(
  supported: boolean,
  clipDataUrl: string | null | undefined,
  keep: VideoContinuationKeep,
): { continuation_video?: string; continuation_keep?: VideoContinuationKeep } {
  if (!supported || !clipDataUrl) return {};
  return { continuation_video: clipDataUrl, continuation_keep: keep };
}
