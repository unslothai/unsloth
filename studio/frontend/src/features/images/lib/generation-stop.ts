// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Stop semantics for an in-flight image generation, kept out of the page so they are testable. A
 *  generate request is a LOOP of `count` backend calls, but the backend cancel only reaches the
 *  denoise running right now. Stopping therefore has two halves: ask the backend to break out of
 *  the sampler, and stop the page issuing the runs that have not started yet. */

/** Whether the run loop should issue another backend generation. */
export function shouldContinueGenerating(input: {
  mounted: boolean;
  stopRequested: boolean;
}): boolean {
  return input.mounted && !input.stopRequested;
}

/** The exact sentinel both image engines raise for a user cancellation. */
export const GENERATION_CANCELLED_SENTINEL =
  "Diffusion generation was cancelled.";

/** Whether a failed generation should be toasted as an error. A user's own Stop comes back as the
 *  cancelled sentinel on a 409, which is the requested outcome. The `stopRequested` latch covers
 *  the case where the message never reaches the page verbatim. */
export function shouldReportGenerateError(input: {
  message: string;
  stopRequested: boolean;
}): boolean {
  if (input.stopRequested) {
    return false;
  }
  return !input.message.toLowerCase().includes("cancelled");
}
