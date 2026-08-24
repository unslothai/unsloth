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
  browsing: boolean;
}): string | null {
  const { held, generating, hasSelection, finishedLoaded, browsing } = input;
  if (held === null) {
    return null;
  }
  if (generating) {
    // Browsing the gallery mid-run gives the viewer back to the picked image. Without
    // this the preview owned the viewer for the whole denoise and clicking a thumbnail
    // changed only the highlight, losing the ability to look at past images during a
    // long run that the page had before previews existed.
    return browsing ? null : held;
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
  producedImage: boolean;
}): boolean {
  const { held, generating, finishedLoaded, selectionMatchesRun, producedImage } = input;
  if (held === null || generating) {
    return false;
  }
  // A cancelled or failed run has no image coming, so neither release condition below can
  // ever be met: the selection keeps matching the run and no blob ever loads. Left alone
  // the last frame sits over an unrelated gallery image for as long as that blob is
  // missing -- forever, if its fetch fails.
  if (!producedImage) {
    return true;
  }
  return finishedLoaded || !selectionMatchesRun;
}
