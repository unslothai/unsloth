// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Periodically persists partial thread state between runStart and runEnd. */

/** Quiet time after a checkpoint settles before the next one is taken. */
export const RUN_CHECKPOINT_INTERVAL_MS = 8_000;

/** Injectable so a test can drive the schedule without real time passing. */
export type RunCheckpointTimers = {
  setTimeout: (callback: () => void, ms: number) => number;
  clearTimeout: (handle: number) => void;
};

export type RunCheckpointScheduler = {
  /** Begin checkpointing a thread. A second call for a thread already running is a no-op. */
  start: (threadId: string) => void;
  /** Stop checkpointing a thread and drop any pending timer. */
  stop: (threadId: string) => void;
  /** Stop every thread, for unmount. */
  stopAll: () => void;
};

type ThreadState = {
  handle: number | null;
  stopped: boolean;
};

const defaultTimers: RunCheckpointTimers = {
  setTimeout: (callback, ms) => window.setTimeout(callback, ms),
  clearTimeout: (handle) => {
    window.clearTimeout(handle);
  },
};

export function createRunCheckpointScheduler(
  save: (threadId: string) => Promise<unknown>,
  options: {
    intervalMs?: number;
    timers?: RunCheckpointTimers;
  } = {},
): RunCheckpointScheduler {
  const intervalMs = options.intervalMs ?? RUN_CHECKPOINT_INTERVAL_MS;
  const timers = options.timers ?? defaultTimers;
  const threads = new Map<string, ThreadState>();

  const schedule = (threadId: string, state: ThreadState): void => {
    state.handle = timers.setTimeout(() => {
      state.handle = null;
      if (state.stopped) {
        return;
      }
      // Retry after transient failures rather than disabling later checkpoints.
      const reschedule = () => {
        if (!state.stopped) {
          schedule(threadId, state);
        }
      };
      save(threadId).then(reschedule, reschedule);
    }, intervalMs);
  };

  const stop = (threadId: string): void => {
    const state = threads.get(threadId);
    if (!state) {
      return;
    }
    // Also stops an in-flight checkpoint from rescheduling when it settles.
    state.stopped = true;
    if (state.handle !== null) {
      timers.clearTimeout(state.handle);
      state.handle = null;
    }
    threads.delete(threadId);
  };

  return {
    start(threadId) {
      // Repeated runStart events for one thread must not stack timers.
      if (threads.has(threadId)) {
        return;
      }
      const state: ThreadState = { handle: null, stopped: false };
      threads.set(threadId, state);
      schedule(threadId, state);
    },
    stop,
    stopAll() {
      for (const threadId of [...threads.keys()]) {
        stop(threadId);
      }
    },
  };
}
