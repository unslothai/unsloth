// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MIC_SILENT_MS = 1500;

export type SilenceWatch = {
  frame: (now: number, live: boolean) => void;
  reset: (now: number) => void;
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

export function createSingleFlight() {
  let tail: Promise<unknown> = Promise.resolve();
  return function run<T>(open: () => Promise<T>): Promise<T> {
    const next = tail.then(open, open);
    tail = next.catch(() => undefined);
    return next;
  };
}
