// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Drains on real observables (counted mocked timers plus a caller barrier), not a fixed
// round count, which under-drained on Node 22 and surfaced as stale reads.
// Do not drain on store-row quiescence: rows only change when the write lands, so it stops early.

import type { TestContext } from "node:test";

type TimerHandle = ReturnType<typeof setTimeout>;
type SetTimeoutFn = typeof globalThis.setTimeout;
type ClearTimeoutFn = typeof globalThis.clearTimeout;

/** Longer than any debounce here, short enough to fire one chained debounce per round. */
const TICK_MS = 1000;
const MICROTASK_TURNS = 6;
/** Rounds, not turns: a round also waits out the loader, which varies by runtime. */
const QUIET_ROUNDS = 3;
/** Backstop only. Reaching it is a failure, not the normal exit. */
const MAX_ROUNDS = 600;
const BARRIER_AFTER_TURN = 2;

/** Marks a counted wrapper so a second enable() in the same test is not double-wrapped. */
const COUNTED = Symbol("unsloth.mockTimerDrain.counted");

interface TimerCounter {
  outstanding: number;
  activity: number;
  entries: Map<TimerHandle, { done: boolean; handle?: TimerHandle }>;
}

let counter: TimerCounter | null = null;

function countedSetTimeout(): (SetTimeoutFn & { [COUNTED]?: true }) | null {
  const current = globalThis.setTimeout as SetTimeoutFn & { [COUNTED]?: true };
  return current[COUNTED] === true ? current : null;
}

/** Copy own symbols (util.promisify.custom) onto the wrapper. */
function inheritSymbols(from: object, to: object): void {
  for (const key of Object.getOwnPropertySymbols(from)) {
    const descriptor = Object.getOwnPropertyDescriptor(from, key);
    if (descriptor !== undefined) Object.defineProperty(to, key, descriptor);
  }
}

/** Use instead of `t.mock.timers.enable`: the counter must wrap the mocked setTimeout first. */
export function enableCountedTimers(t: TestContext): (ms: number) => void {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  if (countedSetTimeout() === null) {
    const mockedSetTimeout = globalThis.setTimeout as SetTimeoutFn;
    const mockedClearTimeout = globalThis.clearTimeout as ClearTimeoutFn;
    const state: TimerCounter = {
      outstanding: 0,
      activity: 0,
      entries: new Map(),
    };
    counter = state;

    const wrappedSetTimeout = ((
      callback: (...args: unknown[]) => void,
      ms?: number,
      ...args: unknown[]
    ) => {
      // The callback closes over the entry before mockedSetTimeout returns the handle.
      const entry: { done: boolean; handle?: TimerHandle } = { done: false };
      state.outstanding += 1;
      state.activity += 1;
      const run = (...callbackArgs: unknown[]): void => {
        if (!entry.done) {
          entry.done = true;
          state.outstanding -= 1;
          state.activity += 1;
          if (entry.handle !== undefined) state.entries.delete(entry.handle);
        }
        callback(...callbackArgs);
      };
      entry.handle = mockedSetTimeout(
        run as never,
        ms as never,
        ...(args as never[]),
      ) as TimerHandle;
      if (!entry.done) state.entries.set(entry.handle, entry);
      return entry.handle;
    }) as SetTimeoutFn & { [COUNTED]?: true };
    inheritSymbols(mockedSetTimeout, wrappedSetTimeout);
    wrappedSetTimeout[COUNTED] = true;

    const wrappedClearTimeout = ((handle?: TimerHandle) => {
      if (handle !== undefined) {
        const entry = state.entries.get(handle);
        if (entry !== undefined && !entry.done) {
          entry.done = true;
          state.outstanding -= 1;
          state.activity += 1;
        }
        state.entries.delete(handle);
      }
      return mockedClearTimeout(handle as never);
    }) as ClearTimeoutFn;
    inheritSymbols(mockedClearTimeout, wrappedClearTimeout);

    globalThis.setTimeout = wrappedSetTimeout;
    globalThis.clearTimeout = wrappedClearTimeout;
  }
  return (ms: number) => t.mock.timers.tick(ms);
}

/** Caller-supplied wait for pending work with no timer, such as hooks-thread imports. */
export type Barrier = () => Promise<unknown>;

export function pendingTimerCount(): number {
  return countedSetTimeout() === null || counter === null
    ? 0
    : counter.outstanding;
}

export interface DrainOptions {
  barrier?: Barrier;
  /** Extra exit condition; quiescence is still required with it. */
  until?: () => boolean;
  label?: string;
  maxRounds?: number;
}

/** Advance the mocked clock until no timers remain; throws if the backstop runs out. */
export async function drainMockedTimers(
  tick: (ms: number) => void,
  options: DrainOptions = {},
): Promise<void> {
  const { until, barrier, label = "drain", maxRounds = MAX_ROUNDS } = options;
  const state = counter;
  if (countedSetTimeout() === null || state === null) {
    throw new Error(
      `${label}: the timer counter is not installed, so there is nothing to drain ON. ` +
        "Enable the mocked clock with enableCountedTimers(t) rather than " +
        "t.mock.timers.enable, and do it before the code under test schedules anything.",
    );
  }
  let quiet = 0;
  for (let round = 0; round < maxRounds; round += 1) {
    const activityBefore = state.activity;
    tick(TICK_MS);
    for (let turn = 0; turn < MICROTASK_TURNS; turn += 1) {
      await new Promise((resolve) => setImmediate(resolve));
      if (turn === BARRIER_AFTER_TURN && barrier !== undefined) await barrier();
    }
    quiet =
      state.outstanding === 0 && state.activity === activityBefore
        ? quiet + 1
        : 0;
    if (quiet >= QUIET_ROUNDS && (until === undefined || until())) return;
  }
  const pending = state.outstanding;
  if (pending > 0 || quiet < QUIET_ROUNDS) {
    throw new Error(
      `${label}: drain exhausted after ${maxRounds} rounds, ` +
        (pending > 0
          ? `with ${pending} timer(s) still pending`
          : `with no timer pending but work still scheduling or firing within the ` +
            `last ${QUIET_ROUNDS} rounds`) +
        ". Nothing read after this point is trustworthy: a queued write has not landed, " +
        "so the store still shows the PREVIOUS value, which reads as a wrong value " +
        "rather than a missing one. Fix the work or raise the backstop; do not read this " +
        "as the store losing an edit.",
    );
  }
  throw new Error(
    `${label}: drain exhausted after ${maxRounds} rounds. The timers all settled, but ` +
      "the caller's condition never held, so the work either never ran or is not the " +
      "work this was waiting for. This is not the assertion below failing on a wrong value.",
  );
}
