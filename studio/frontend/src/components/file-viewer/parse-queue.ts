// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A started parse cannot be cancelled, so thumbnails run a few at a time.
export const MAX_QUEUED_PARSES = 2;

let active = 0;
const waiting: (() => void)[] = [];

function next(): void {
  while (active < MAX_QUEUED_PARSES && waiting.length > 0) waiting.shift()!();
}

/** Runs `task` once a slot is free. Resolves null, without running it, if cancelled while waiting. */
export function queueParse<T>(task: () => Promise<T>, cancelled: () => boolean): Promise<T | null> {
  return new Promise<T | null>((resolve, reject) => {
    waiting.push(() => {
      if (cancelled()) {
        resolve(null);
        return;
      }
      active += 1;
      task()
        .then(resolve, reject)
        .finally(() => {
          active -= 1;
          next();
        });
    });
    next();
  });
}
