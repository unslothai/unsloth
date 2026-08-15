// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A janked window may not paint for a long time, so do not wait on frames
// alone. This is a scheduled fallback, not a deadline: a hidden page throttles
// this timer too (>=1s, and ~1/min once Chrome's intensive throttling kicks in),
// and pauses frames outright. There the cap below is what paces publishes,
// which is why the cap is counted in arrivals rather than in time.
export const UNPAINTED_REOPEN_MS = 500;

/**
 * Text a closed gate may hold before it publishes regardless.
 *
 * assistant-ui drops whatever a run yields after an abort, so Stop keeps only the
 * last published text, and frames bound that hold in time rather than in volume:
 * the fallback above reopens a starved window, which is where the stream runs
 * fastest. Several frames' worth, so it binds only once frames are far apart.
 */
export const MAX_HELD_CHARS = 256;

/**
 * Paces streamed publishes to one per frame while capping what a stop would discard.
 *
 * `streamed` counts the characters the caller has received. It must only grow, since
 * the cap is on the arrivals a publish has yet to carry. The gate closes on a publish
 * and reopens on the next frame, or on UNPAINTED_REOPEN_MS when no frame comes; a
 * closed gate publishes anyway once MAX_HELD_CHARS have arrived since the last one.
 */
export function createStreamPublishGate(): (streamed: number) => boolean {
  let open = true;
  let publishedAt = 0;
  return (streamed: number) => {
    if (!open && streamed - publishedAt < MAX_HELD_CHARS) {
      return false;
    }
    publishedAt = streamed;
    if (open) {
      open = false;
      // Per cycle, so the loser of the previous race cannot reopen this one.
      let reopened = false;
      // Initialised before reopen closes over them, so a scheduler that runs
      // its callback synchronously cannot hit the temporal dead zone and throw
      // out of the stream loop.
      const handles: {
        frame?: number;
        timer?: ReturnType<typeof setTimeout>;
      } = {};
      const reopen = () => {
        if (reopened) {
          return;
        }
        reopened = true;
        if (handles.frame !== undefined) {
          cancelAnimationFrame(handles.frame);
        }
        if (handles.timer !== undefined) {
          clearTimeout(handles.timer);
        }
        open = true;
      };
      handles.frame = requestAnimationFrame(reopen);
      handles.timer = setTimeout(reopen, UNPAINTED_REOPEN_MS);
    }
    return true;
  };
}
