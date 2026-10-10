// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const RUN_CHECKPOINT_INTERVAL_MS = 8_000;

/** Cap per thread; applies only to durable runs (see isBounded), never to subscriber streams. */
export const RUN_CHECKPOINT_MAX_DURATION_MS = 30 * 60_000;

export type RunCheckpointTimers = {
  setTimeout: (callback: () => void, ms: number) => number;
  clearTimeout: (handle: number) => void;
  now?: () => number;
};

export type RunCheckpointScheduler = {
  start: (threadId: string) => void;
  stop: (threadId: string) => void;
  stopAll: () => void;
  /** Checkpoint every live thread now without touching the schedule, e.g. before unload. */
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
    /** runEnd only reaches the main thread, so this stops threads that stopped being main. */
    isActive?: (threadId: string) => boolean;
    /** isActive cannot cap: a run that never terminalises reports isRunning forever. */
    maxDurationMs?: number;
    /** Subscriber-owned streams persist only through checkpoints, so they must not be capped. */
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
      const reschedule = () => {
        if (!state.stopped) {
          schedule(threadId, state);
        }
      };
      // Both exits take a final save, or the last interval's output is lost.
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
