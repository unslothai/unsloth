// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Keeping the mic alive. WebKit can hand back a capture track that reads as open but
 * delivers only zeros, or no audio callbacks at all (seen after the floating orb hid the
 * main window), and the loop then listens to nothing until the app restarts. A session
 * that hears no non-zero sample for MIC_SILENT_MS re-acquires the mic; real silence in a
 * room is never exactly zero, so this does not fire in a quiet room.
 */
export const MIC_SILENT_MS = 1500;

export type SilenceWatch = {
  /** A frame arrived; `live` when any sample in it was non-zero. */
  frame: (now: number, live: boolean) => void;
  /** Muted by the user, or just unmuted: the gap is not the device's fault. */
  reset: (now: number) => void;
  /** True once, the first time the mic has been dead for `silentMs`. */
  due: (now: number) => boolean;
};

export function createSilenceWatch(startedAt: number, silentMs: number = MIC_SILENT_MS): SilenceWatch {
  let lastLive = startedAt;
  let fired = false;
  return {
    frame: (now, live) => {
      if (live) lastLive = now;
    },
    reset: (now) => {
      lastLive = now;
    },
    due: (now) => {
      if (fired || now - lastLive < silentMs) return false;
      fired = true;
      return true;
    },
  };
}

/**
 * Re-acquiring cannot help while the window is hidden (WebKit mutes capture there), so
 * after `limit` tries in a row with nothing heard it waits `slowMs` between tries.
 */
export function createReacquireBackoff(limit = 3, slowMs = 10000) {
  let tries = 0;
  let lastAt = -Infinity;
  return {
    allow: (now: number): boolean => tries < limit || now - lastAt >= slowMs,
    tried: (now: number): void => {
      tries++;
      lastAt = now;
    },
    heard: (): void => {
      tries = 0;
    },
  };
}

/**
 * One mic open at a time. Two sessions racing (a cancel and a re-arm in the same tick)
 * each called getUserMedia, and the second track came back dead. Each open waits for the
 * one before it to settle, its own clean-up included.
 */
export function createSingleFlight() {
  let tail: Promise<unknown> = Promise.resolve();
  return function run<T>(open: () => Promise<T>): Promise<T> {
    const next = tail.then(open, open);
    tail = next.catch(() => undefined);
    return next;
  };
}
