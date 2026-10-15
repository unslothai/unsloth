// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fallback when no frame paints; hidden pages throttle this timer too, hence the arrival cap.
export const UNPAINTED_REOPEN_MS = 500;

/** assistant-ui drops output after an abort, so this bounds what Stop can discard. */
export const MAX_HELD_CHARS = 256;

/** One publish per frame, but publishes anyway once MAX_HELD_CHARS have arrived. */
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
      // Assigned before reopen closes over them, avoiding the temporal dead zone.
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
