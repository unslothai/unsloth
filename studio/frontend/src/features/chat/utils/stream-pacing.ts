// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hidden or heavily janked windows may not produce frames, so bound the wait.
const UNPAINTED_REOPEN_MS = 500;

/**
 * Limits streamed publishes to one per browser frame.
 *
 * Reopening does not flush pending text. Stop can therefore omit text received
 * during the current interval.
 */
export function createFrameGate(
  schedule: (cb: () => void) => void = (cb) => {
    let woken = false;
    const wake = () => {
      if (woken) {
        return;
      }
      woken = true;
      cb();
    };
    requestAnimationFrame(wake);
    setTimeout(wake, UNPAINTED_REOPEN_MS);
  },
): () => boolean {
  let painted = true;
  return () => {
    if (!painted) {
      return false;
    }
    painted = false;
    schedule(() => {
      painted = true;
    });
    return true;
  };
}
