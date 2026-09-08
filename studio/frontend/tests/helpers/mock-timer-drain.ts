// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Waiting out the debounced writes the chat store schedules, on the store's own pending
// work rather than on a round count.
//
// The shape every caller needs is the same: advance the mocked clock, then let the promise
// continuations that firing released run, and repeat -- the writes these suites assert on
// sit behind a 400ms debounce whose callback then awaits, so one tick is never enough, and
// a timer scheduled from inside a continuation only fires on the NEXT tick.
//
// A FIXED round count was what this used to be, and it is a guess that fails in one
// direction silently. Measured on v22.23.2, the version `setup-node: 22` resolved to, the
// sibling compat suite scored 7 failures at 3 rounds, 5 at 10, 1 at 30 and 0 at 60; the
// simulation, which runs many more steps per scenario, still lost 2 of 120 orderings at
// 200 and needed 600 (107s against 52s) -- and even 600 was only load-dependently enough,
// since the same commit at 600 passed one Windows CI run and failed the next. The failure
// mode is what makes this expensive: a drain that returns with a write still queued leaves
// the caller reading a stale value, which the simulation reports as an ORDERING violation
// ("chat A minP: owed 0, shows 0.01"), indistinguishable from the store genuinely losing
// the edit. Hours go into the store before anyone suspects the drain.
//
// So drain on the real observables instead. There are two, because the work has two halves:
//
//   1. The debounce. `t.mock.timers.enable` replaces globalThis.setTimeout with its own;
//      wrapping THAT (and clearTimeout) with a counter gives an exact count of timers
//      scheduled and not yet fired or cleared, which is the pending state itself rather than
//      a proxy for it. Stop only when the count is zero AND a further QUIET_ROUNDS rounds
//      neither scheduled a timer nor ran one, since a promise continuation is free to
//      schedule the next debounce a turn after the last one fired.
//   2. The module loader. The write ends in an `await import()` (see probeModuleLoader),
//      which these suites route through a registered resolver hook and therefore through
//      the hooks thread. Nothing counts that, and on v22.23.2 it takes 3 to 35 macrotask
//      turns where v24 takes 1 -- measured, three repeat imports of an already-loaded
//      module. That is the whole green-locally / red-on-CI split, and quiet rounds alone do
//      not cover it: at 25 quiet rounds per drain (150 turns of nothing) v22 still lost 7
//      orderings across A1 and A3, all of them writes that had not landed. Each round
//      issues its own import and waits for it, so the wait scales with the loader instead
//      of guessing at it. With that, 3 quiet rounds is green on v22 and v24 alike.
//
// The generous bound stays, as a BACKSTOP that THROWS and names what was still outstanding.
// It is not the exit path any more, so it costs nothing to leave high, and a suite that
// hits it is told the drain gave up rather than handed a stale read to misattribute.
//
// Not to be retried: quiescence on the store ROWS. The rows only change WHEN the write
// lands, so "rows have stopped changing" is precisely the pending state being waited
// through, and it stops early by construction -- it scored 4 failures where the fixed bound
// scored 0. Timers and the loader are the opposite kind of observable: they exist while the
// work is outstanding, and are gone when it is done.

import type { TestContext } from "node:test";

type TimerHandle = ReturnType<typeof setTimeout>;
type SetTimeoutFn = typeof globalThis.setTimeout;
type ClearTimeoutFn = typeof globalThis.clearTimeout;

/** How far each round advances the mocked clock. Longer than any debounce here, short
 * enough that a chain of sequential debounces still fires one per round. */
const TICK_MS = 1000;
/** Macrotask turns per round. Each one drains the whole microtask queue behind it. */
const MICROTASK_TURNS = 6;
/** Consecutive rounds of no timer scheduled, none fired and none outstanding. Rounds, not
 * turns: a round also waits out the loader, which is the part that varies by runtime. */
const QUIET_ROUNDS = 3;
/** Backstop only. Reaching it is a failure, not the normal exit. */
const MAX_ROUNDS = 600;
/** Which turn of a round awaits the caller's barrier, once this tick's callbacks have run. */
const BARRIER_AFTER_TURN = 2;

/** Set on the wrapper so a second enable() in the same test is not double-wrapped, and so
 * a drain can tell it is looking at a counted setTimeout rather than a raw mocked one.
 * `enable()` installs a fresh mocked setTimeout each test, which drops this and makes the
 * next install re-wrap with fresh counts. */
const COUNTED = Symbol("unsloth.mockTimerDrain.counted");

interface TimerCounter {
  /** Scheduled and not yet fired or cleared. */
  outstanding: number;
  /** Every schedule, fire and clear since install: the "did anything happen" signal. */
  activity: number;
  /** Live handles, so clearTimeout knows whether it is cancelling real pending work. */
  entries: Map<TimerHandle, { done: boolean; handle?: TimerHandle }>;
}

let counter: TimerCounter | null = null;

function countedSetTimeout(): (SetTimeoutFn & { [COUNTED]?: true }) | null {
  const current = globalThis.setTimeout as SetTimeoutFn & { [COUNTED]?: true };
  return current[COUNTED] === true ? current : null;
}

/** Copy the own symbols (util.promisify.custom above all) onto the wrapper, so code that
 * reaches for them still finds them on the function the tests installed. */
function inheritSymbols(from: object, to: object): void {
  for (const key of Object.getOwnPropertySymbols(from)) {
    const descriptor = Object.getOwnPropertyDescriptor(from, key);
    if (descriptor !== undefined) Object.defineProperty(to, key, descriptor);
  }
}

/**
 * Enable node's mocked setTimeout for this test and count what the code under test
 * schedules on it. Call this instead of `t.mock.timers.enable` -- the counter has to be
 * installed over the MOCKED setTimeout, and before any product code runs, or the timers
 * scheduled in between are invisible and a drain reports quiet while they are pending.
 *
 * Returns the tick the drain wants, for callers that would rather not repeat it.
 */
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
      // The handle lives on the entry because the callback closes over it before
      // mockedSetTimeout has returned it.
      const entry: { done: boolean; handle?: TimerHandle } = { done: false };
      state.outstanding += 1;
      state.activity += 1;
      const run = (...callbackArgs: unknown[]): void => {
        // Guard the arithmetic rather than the call: a mocked timer only fires from
        // tick(), but clearTimeout may have retired this entry first.
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

/**
 * Wait out the work that has no timer behind it.
 *
 * The store's thread-scoped write ends in `await import("../utils/chat-history-storage")`,
 * and these suites `register()` a resolver hook, which puts that import on the hooks
 * thread. That is the second kind of pending work here, and unlike a debounce it has no
 * timer to count: on v24 it settles in ONE macrotask turn, but measured on v22.23.2 with a
 * hook registered, three repeat imports of an already-loaded module settled in 6, 3 and 35
 * turns. That is the whole node-24-green / node-22-red split, and the reason a Windows
 * runner under load fails what the same commit passed an hour earlier: the pending work is
 * a message to another thread, so its cost is scheduling latency, not instructions.
 *
 * The caller supplies the wait, because only the caller knows what its subject has
 * outstanding. For the sampling suites that is the store's own
 * `awaitStartedThreadScopedSettingsWrites`, which awaits the write chains themselves.
 *
 * An earlier version of this raced the loader instead, by issuing an import of its own each
 * round and assuming the hooks thread serves requests in order, so a reply to a later
 * request could not arrive first. It worked, and it is still the wrong thing to assert: it
 * is a claim about node's loader internals standing in for a claim about the store, and if
 * that ordering ever fails it fails as a stale read again rather than as a throw. A barrier
 * on the actual chains has no such assumption behind it.
 */
export type Barrier = () => Promise<unknown>;

/** Debounce timers scheduled and not yet fired or cleared. */
export function pendingTimerCount(): number {
  return countedSetTimeout() === null || counter === null
    ? 0
    : counter.outstanding;
}

export interface DrainOptions {
  /** Awaited each round, for pending work that has no timer to count. See Barrier. */
  barrier?: Barrier;
  /** An extra condition the caller needs true before the drain may return. Quiescence is
   * still required with it: a condition that flips mid-chain must not cut the rest off. */
  until?: () => boolean;
  /** Named in the exhaustion message, so a failure says which wait gave up. */
  label?: string;
  /** Backstop override. For proving the throw fires; not for tuning a wait. */
  maxRounds?: number;
}

/**
 * Advance the mocked clock until the code under test has no scheduled timer left and has
 * stopped scheduling new ones, then return. Throws rather than returning early if the
 * backstop runs out, because everything a caller reads afterwards would otherwise be a
 * stale value wearing the costume of a wrong one.
 */
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
      // After the timer callbacks fired by this tick have had a turn to start their work,
      // wait for that work rather than for a number of turns.
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
