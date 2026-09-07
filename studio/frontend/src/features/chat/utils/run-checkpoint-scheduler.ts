// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Periodically persists partial thread state between runStart and runEnd. */

/** Quiet time after a checkpoint settles before the next one is taken. */
export const RUN_CHECKPOINT_INTERVAL_MS = 8_000;

/**
 * How long one thread may be checkpointed before the schedule gives up on it. A run that
 * never terminalises would otherwise checkpoint for the life of the page.
 *
 * Never rearmed, unlike the follow deadline in chat-generation-api.ts. That is safe only
 * because the cap applies to durable runs alone (see `isBounded`), which the server
 * persists; a subscriber-owned stream, whose only persistence IS these saves, is never
 * capped.
 */
export const RUN_CHECKPOINT_MAX_DURATION_MS = 30 * 60_000;

/** Injectable so a test can drive the schedule without real time passing. */
export type RunCheckpointTimers = {
  setTimeout: (callback: () => void, ms: number) => number;
  clearTimeout: (handle: number) => void;
  /** Omit to read the wall clock. Only the staleness bound consults it. */
  now?: () => number;
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
  startedAt: number;
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
    /**
     * Wall-clock cap on one thread's schedule. `isActive` cannot serve as one: it reports
     * the runtime's own `isRunning`, which a run that never terminalises holds true.
     */
    maxDurationMs?: number;
    /**
     * Whether `maxDurationMs` applies to this thread. A subscriber-owned stream persists
     * ONLY through these checkpoints, so capping one would lose everything it streamed
     * after the cap if the page went away before `runEnd`. Omit to cap every thread.
     */
    isBounded?: (threadId: string) => boolean;
  } = {},
): RunCheckpointScheduler {
  const intervalMs = options.intervalMs ?? RUN_CHECKPOINT_INTERVAL_MS;
  const timers = options.timers ?? defaultTimers;
  const isActive = options.isActive;
  const maxDurationMs = options.maxDurationMs ?? RUN_CHECKPOINT_MAX_DURATION_MS;
  const isBounded = options.isBounded;
  const now = timers.now ?? (() => Date.now());
  const threads = new Map<string, ThreadState>();

  /** Never let a caller's throw escape the timer: that would strand the Map entry. */
  const runSave = (threadId: string): Promise<unknown> => {
    try {
      return Promise.resolve(save(threadId));
    } catch (error) {
      return Promise.reject(error);
    }
  };

  /** Only durable runs are capped; see `isBounded`. Throwing means "not durable". */
  const isBoundedRun = (threadId: string): boolean => {
    try {
      return isBounded?.(threadId) ?? true;
    } catch {
      return false;
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
      // Both exits take a final save: a schedule that ends without one has lost
      // whatever the last interval produced, exactly as a missed runEnd would.
      const capped =
        now() - state.startedAt >= maxDurationMs && isBoundedRun(threadId);
      if (!isRunning(threadId) || capped) {
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
      const state: ThreadState = {
        handle: null,
        stopped: false,
        startedAt: now(),
      };
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
