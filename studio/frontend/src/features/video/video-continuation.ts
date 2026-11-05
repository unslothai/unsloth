// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VideoContinuationKeep, VideoStatus } from "./api.ts";

// The pipeline keeps 5 frames of the clip (two latent frames); the rest of the length is generated.
export const CONTINUATION_KEPT_FRAMES = 5;

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

/** Why a staged clip cannot be continued at this length, or null. */
export function continuationLengthError(numFrames: number): string | null {
  return numFrames > CONTINUATION_KEPT_FRAMES
    ? null
    : `Continuing a clip keeps ${CONTINUATION_KEPT_FRAMES} of its frames; pick a longer length so there is something to generate`;
}

/** The request fields for a staged clip; nothing when the model cannot take one or none is staged. */
export function continuationRequest(
  supported: boolean,
  clipDataUrl: string | null | undefined,
  keep: VideoContinuationKeep,
): { continuation_video?: string; continuation_keep?: VideoContinuationKeep } {
  if (!supported || !clipDataUrl) return {};
  return { continuation_video: clipDataUrl, continuation_keep: keep };
}
