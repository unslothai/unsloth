// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hidden or heavily janked windows may not produce frames, so bound the wait.
const UNPAINTED_REOPEN_MS = 500;

/**
 * Text a closed gate may hold before it publishes regardless.
 *
 * assistant-ui drops whatever a run yields after an abort, so Stop keeps only the
 * last published text. Frames bound the hold in time, not in volume, and the
 * fallback above is what reopens a starved window, which is where the stream runs
 * fastest. This bounds it in characters instead: several frames of a fast local
 * stream, so it binds only once frames are already far apart.
 */
export const MAX_HELD_CHARS = 256;

/**
 * Paces streamed publishes to one per frame while capping what a stop would discard.
 *
 * `length` is the caller's accumulated reply length. The gate closes on a publish and
 * reopens on the next frame, or on UNPAINTED_REOPEN_MS when no frame comes. A closed
 * gate publishes anyway once MAX_HELD_CHARS have arrived since the last publish.
 */
export function createStreamPublishGate(): (length: number) => boolean {
  let open = true;
  let publishedLength = 0;
  return (length: number) => {
    if (!open && length - publishedLength < MAX_HELD_CHARS) {
      return false;
    }
    publishedLength = length;
    if (open) {
      open = false;
      // Per cycle, so the loser of the previous race cannot reopen this one.
      let reopened = false;
      const reopen = () => {
        if (reopened) {
          return;
        }
        reopened = true;
        cancelAnimationFrame(frame);
        clearTimeout(timer);
        open = true;
      };
      const frame = requestAnimationFrame(reopen);
      const timer = setTimeout(reopen, UNPAINTED_REOPEN_MS);
    }
    return true;
  };
}
