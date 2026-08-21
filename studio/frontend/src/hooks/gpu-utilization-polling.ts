// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PollingTimer {
  setTimeout(callback: () => void, delayMs: number): ReturnType<typeof setTimeout>;
  clearTimeout(timer: ReturnType<typeof setTimeout>): void;
}

const DEFAULT_TIMER: PollingTimer = {
  setTimeout: (callback, delayMs) => setTimeout(callback, delayMs),
  clearTimeout: (timer) => clearTimeout(timer),
};

export function createSingleFlight<T>(poll: () => Promise<T>): () => Promise<T> {
  let inFlight: Promise<T> | null = null;

  return () => {
    if (inFlight !== null) return inFlight;

    const request = poll();
    inFlight = request;
    const clear = () => {
      if (inFlight === request) inFlight = null;
    };
    void request.then(clear, clear);
    return request;
  };
}

export function startSerialPolling(
  poll: () => Promise<void>,
  intervalMs: number,
  timer: PollingTimer = DEFAULT_TIMER,
): () => void {
  let cancelled = false;
  let timeout: ReturnType<typeof setTimeout> | null = null;

  const run = async () => {
    if (cancelled) return;

    try {
      await poll();
    } finally {
      if (!cancelled) {
        timeout = timer.setTimeout(() => {
          timeout = null;
          void run();
        }, intervalMs);
      }
    }
  };

  void run();

  return () => {
    cancelled = true;
    if (timeout !== null) timer.clearTimeout(timeout);
  };
}
