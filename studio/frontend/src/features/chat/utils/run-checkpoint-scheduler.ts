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
  /**
   * Checkpoint every live thread right now, leaving the schedule untouched. For the page
   * transitions that precede a renderer going away or being throttled.
   */
  flushAll: () => void;
};

type ThreadState = {
  handle: number | null;
  stopped: boolean;
};

const noop = (): void => {};

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
    /**
     * Whether the thread's run is still going. runEnd is the fast path out, but it only
     * reaches whichever thread is main, so one that stops being main mid-run would
     * checkpoint for the life of the page. Omit it to treat every thread as active.
     */
    isActive?: (threadId: string) => boolean;
  } = {},
): RunCheckpointScheduler {
  const intervalMs = options.intervalMs ?? RUN_CHECKPOINT_INTERVAL_MS;
  const timers = options.timers ?? defaultTimers;
  const isActive = options.isActive;
  const threads = new Map<string, ThreadState>();

  /** Never let a caller's throw escape the timer: that would strand the Map entry. */
  const runSave = (threadId: string): Promise<unknown> => {
    try {
      return Promise.resolve(save(threadId));
    } catch (error) {
      return Promise.reject(error);
    }
  };

  /** A thread the runtime has dropped throws rather than reporting itself idle. */
  const isRunning = (threadId: string): boolean => {
    try {
      return isActive?.(threadId) ?? true;
    } catch {
      return false;
    }
  };

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
      if (!isRunning(threadId)) {
        // A missed runEnd also lost the final save, so take one on the way out.
        stop(threadId);
        void runSave(threadId).then(noop, noop);
        return;
      }
      runSave(threadId).then(reschedule, reschedule);
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
    flushAll() {
      // The pending timer stays armed: a flush is an extra checkpoint, not a reschedule.
      for (const threadId of [...threads.keys()]) {
        void runSave(threadId).then(noop, noop);
      }
    },
  };
}
