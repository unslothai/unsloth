// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const VIDEO_GENERATION_FAILED = "Video generation failed";

/** Toast text for a failed generation, or null for the user's own Cancel. */
export function videoFailureToastText(error: string | null | undefined): string | null {
  const msg = error || VIDEO_GENERATION_FAILED;
  // A logged failure carries the exception's own text, which can say "cancelled" (CancelledError).
  if (msg.startsWith(VIDEO_GENERATION_FAILED)) return msg;
  return msg.toLowerCase().includes("cancelled") ? null : msg;
}
