// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the viewer shows during and just after a generation, kept out of the page so it is testable.
 *
 * A run ends in three stages, not one: the denoise stops, the record persists (progress still
 * reports active, now with no preview and step back at 0), then the finished PNG loads. Reading
 * either signal straight off the live progress drops the viewer to a spinner between them. */

/** The preview frame to show, or null to fall through to the finished image / spinner. */
export function previewFrame(input: {
  held: string | null;
  generating: boolean;
  hasSelection: boolean;
  finishedLoaded: boolean;
}): string | null {
  const { held, generating, hasSelection, finishedLoaded } = input;
  if (held === null) {
    return null;
  }
  if (generating) {
    return held;
  }
  return hasSelection && !finishedLoaded ? held : null;
}

/** The progress reading to render, ignoring the step-0 report the persist window emits. */
export function nextProgress<T extends { step: number }>(
  previous: T | null,
  incoming: T,
): T {
  return previous && previous.step > 0 && incoming.step === 0
    ? previous
    : incoming;
}

/** Whether the held frame has finished its handoff and must be dropped.
 *
 * The frame belongs to the run that produced it. Holding it past that run leaves it able to
 * cover an unrelated gallery image whose blob has not loaded -- an eviction from the blob
 * budget, or a page of results still streaming -- which shows the previous run's preview
 * where a different image belongs. */
export function releaseHeldPreview(input: {
  held: string | null;
  generating: boolean;
  finishedLoaded: boolean;
  selectionMatchesRun: boolean;
}): boolean {
  const { held, generating, finishedLoaded, selectionMatchesRun } = input;
  if (held === null || generating) {
    return false;
  }
  return finishedLoaded || !selectionMatchesRun;
}
