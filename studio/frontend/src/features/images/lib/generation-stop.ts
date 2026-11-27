// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The backend cancel only reaches the current denoise, so the page must also stop issuing runs. */

export function shouldContinueGenerating(input: {
  mounted: boolean;
  stopRequested: boolean;
}): boolean {
  return input.mounted && !input.stopRequested;
}

export const GENERATION_CANCELLED_SENTINEL =
  "Diffusion generation was cancelled.";

/** A user's own Stop returns the cancelled sentinel on a 409, which is not an error. */
export function shouldReportGenerateError(input: {
  message: string;
  stopRequested: boolean;
}): boolean {
  if (input.stopRequested) {
    return false;
  }
  return !input.message.toLowerCase().includes("cancelled");
}

export function stopButtonLabel(input: {
  stopping: boolean;
  done: number | null;
  count: number;
  idle?: string;
}): string {
  if (input.stopping) return "Stopping…";
  const idle = input.idle ?? "Stop";
  return input.done != null && input.count > 1 ? `${idle} (${input.done}/${input.count})` : idle;
}
