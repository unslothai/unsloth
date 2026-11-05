// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Same text as the backend's VIDEO_GENERATION_FAILED_MSG, so the "View logs" action matches it.
const VIDEO_GENERATION_FAILED = "Video generation failed.";

export interface VideoFailureToast {
  title: string;
  /** Exception class, e.g. "RuntimeError", when the backend named one. */
  errorType: string | null;
  detail: string | null;
}

/** Toast parts for a failed generation, or null for the user's own Cancel. */
export function videoFailureToast(error: string | null | undefined): VideoFailureToast | null {
  const msg = error?.trim() || VIDEO_GENERATION_FAILED;
  if (!msg.startsWith(VIDEO_GENERATION_FAILED)) {
    if (msg.toLowerCase().includes("cancelled")) return null;
    return { title: msg, errorType: null, detail: null };
  }
  // A logged failure carries the exception's own text, which can say "cancelled" (CancelledError).
  const reason = msg.slice(VIDEO_GENERATION_FAILED.length).trim();
  if (!reason) return { title: VIDEO_GENERATION_FAILED, errorType: null, detail: null };
  // "RuntimeError: first line" or a bare "RuntimeError"; fixed backend text has spaces before any colon.
  const named = /^([A-Za-z_][\w.]*)(?::\s*([\s\S]*))?$/.exec(reason);
  if (named) {
    return { title: VIDEO_GENERATION_FAILED, errorType: named[1], detail: named[2]?.trim() || null };
  }
  return { title: VIDEO_GENERATION_FAILED, errorType: null, detail: reason };
}
