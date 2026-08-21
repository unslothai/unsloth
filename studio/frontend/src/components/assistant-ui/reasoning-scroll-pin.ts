// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type FrameScheduler = (callback: FrameRequestCallback) => number;
type FrameCanceller = (handle: number) => void;

/** Coalesce mutation-driven reasoning scroll pins to one layout operation per frame. */
export function createReasoningScrollPin(
  shouldPin: () => boolean,
  pin: () => void,
  requestFrame: FrameScheduler = requestAnimationFrame,
  cancelFrame: FrameCanceller = cancelAnimationFrame,
) {
  let frame: number | null = null;

  return {
    cancel(): void {
      if (frame !== null) {
        cancelFrame(frame);
        frame = null;
      }
    },
    schedule(): void {
      if (!shouldPin() || frame !== null) {
        return;
      }
      frame = requestFrame(() => {
        frame = null;
        // A wheel/scroll can detach the viewport after the mutation but before
        // the frame. Re-check before performing the scrollHeight read/write.
        if (shouldPin()) {
          pin();
        }
      });
    },
  };
}
