// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A janked window may not paint for a while, so do not wait on frames alone. A scheduled
// fallback, not a deadline: a hidden page pauses frames and throttles this timer too, which is
// why the cap below is in arrivals.
export const UNPAINTED_REOPEN_MS = 500;

/** Text a closed gate may hold before it publishes regardless. assistant-ui drops whatever a run
 *  yields after an abort, so Stop keeps only the last published text. Frames bound that hold in
 *  time, not in volume, so this bounds it in volume. */
export const MAX_HELD_CHARS = 256;

/** Paces streamed publishes to one per frame while capping what a stop would discard. `streamed`
 *  counts the characters the caller has received and must only grow, since the cap is on the
 *  arrivals a publish has yet to carry. The gate closes on a publish and reopens on the next
 *  frame, or on UNPAINTED_REOPEN_MS when no frame comes; a closed gate publishes anyway once
 *  MAX_HELD_CHARS have arrived. */
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
      // Assigned before reopen closes over them, so a synchronous scheduler cannot hit the temporal dead zone.
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
