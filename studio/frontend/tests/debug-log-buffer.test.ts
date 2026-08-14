// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_REFRESH_MODE,
  EMPTY_BUFFER,
  MAX_CLIENT_LINES,
  applyLogChunk,
  isPageStale,
  nextDroppedState,
  parseRefreshMode,
  pollDelayMs,
  trimBuffer,
  isRequestTimeout,
  withRequestTimeout,
} from "../src/features/settings/lib/debug-log-buffer.ts";
import { isAbort } from "../src/features/settings/lib/debug-log-error.ts";

test("three seconds is the default refresh mode", () => {
  assert.equal(DEFAULT_REFRESH_MODE, "3s");
  assert.equal(parseRefreshMode(null), "3s");
  assert.equal(parseRefreshMode("nonsense"), "3s");
  assert.equal(parseRefreshMode("live"), "live");
  assert.equal(parseRefreshMode("manual"), "manual");
});

test("each mode maps to its poll delay, manual to none", () => {
  assert.equal(pollDelayMs("live"), 1000);
  assert.equal(pollDelayMs("3s"), 3000);
  assert.equal(pollDelayMs("manual"), null);
});

test("a chunk appends to what is already there", () => {
  const first = applyLogChunk(EMPTY_BUFFER, {
    lines: ["a", "b"],
    cursor: "c1",
    reset: true,
  });
  const second = applyLogChunk(first, {
    lines: ["c"],
    cursor: "c2",
    reset: false,
  });
  assert.deepEqual(second.lines, ["a", "b", "c"]);
  assert.equal(second.cursor, "c2");
});

test("a reset replaces the buffer rather than appending to it", () => {
  const first = applyLogChunk(EMPTY_BUFFER, {
    lines: ["old"],
    cursor: "c1",
    reset: true,
  });
  const second = applyLogChunk(first, {
    lines: ["fresh"],
    cursor: "c2",
    reset: true,
  });
  assert.deepEqual(second.lines, ["fresh"]);
});

test("an empty chunk returns the same object so React can skip the render", () => {
  const first = applyLogChunk(EMPTY_BUFFER, {
    lines: ["a"],
    cursor: "c1",
    reset: true,
  });
  const second = applyLogChunk(first, {
    lines: [],
    cursor: "c1",
    reset: false,
  });
  assert.equal(second, first);
});

test("the buffer is capped and keeps the newest lines", () => {
  const lines = Array.from(
    { length: MAX_CLIENT_LINES + 500 },
    (_, i) => `line${i}`,
  );
  const trimmed = trimBuffer(lines);
  assert.equal(trimmed.length, MAX_CLIENT_LINES);
  assert.equal(trimmed[trimmed.length - 1], `line${MAX_CLIENT_LINES + 499}`);
});

test("a few enormous lines are capped by characters, not just by count", () => {
  const lines = Array.from({ length: 40 }, () => "x".repeat(20_000));
  const trimmed = trimBuffer(lines);
  const chars = trimmed.reduce((total, line) => total + line.length + 1, 0);
  assert.ok(
    chars <= 400_000,
    `expected the buffer under the char cap, got ${chars}`,
  );
  assert.ok(trimmed.length < lines.length);
});

test("appending past the cap still keeps the tail", () => {
  let state = applyLogChunk(EMPTY_BUFFER, {
    lines: ["first"],
    cursor: "c1",
    reset: true,
  });
  for (let i = 0; i < MAX_CLIENT_LINES + 10; i += 1) {
    state = applyLogChunk(state, {
      lines: [`n${i}`],
      cursor: `c${i}`,
      reset: false,
    });
  }
  assert.equal(state.lines.length, MAX_CLIENT_LINES);
  assert.equal(state.lines[state.lines.length - 1], `n${MAX_CLIENT_LINES + 9}`);
  assert.ok(!state.lines.includes("first"));
});

// A request that opens and never answers is the failure the viewer has to
// survive: the auth client hands `init` to fetch and adds no timeout, so every
// awaited request needs the backstop, not just the tail read.
function neverAnswers(signal: AbortSignal): Promise<never> {
  return new Promise((_resolve, reject) => {
    const fail = () => {
      const error = new Error("aborted");
      error.name = "AbortError";
      reject(error);
    };
    // fetch rejects straight away when handed an already aborted signal.
    if (signal.aborted) fail();
    else signal.addEventListener("abort", fail);
  });
}

test("a request that never answers is cut off by the backstop", async () => {
  const started = Date.now();
  await assert.rejects(
    () => withRequestTimeout(neverAnswers, 20),
    (error: Error) => error.name === "DebugLogTimeoutError",
  );
  assert.ok(Date.now() - started < 2000);
});

test("a backstop rejection is not mistaken for a caller cancellation", async () => {
  // The timer aborts the SAME controller an unmount uses, so both arrived as an
  // AbortError and the poll loop swallowed them alike. A hung tunnel then left
  // the pane stale with no notice at all, which is the failure this viewer is
  // supposed to make visible.
  await assert.rejects(
    () => withRequestTimeout(neverAnswers, 20),
    (error: Error) => isRequestTimeout(error) && !isAbort(error),
  );
});

test("a caller abort stays silent even when the backstop races it", async () => {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), 10);
  await assert.rejects(
    () => withRequestTimeout(neverAnswers, 10, controller.signal),
    (error: Error) => isAbort(error) && !isRequestTimeout(error),
  );
});

test("a request that answers in time is untouched by the backstop", async () => {
  assert.equal(await withRequestTimeout(async () => "done", 1000), "done");
});

test("the source rescan cannot freeze the poll loop behind it", async () => {
  // The loop awaits the rescan BEFORE the tail read, so an unanswered /sources
  // used to hang the whole tick: no poll, no reschedule, a pane that stops
  // updating while still looking live.
  let polls = 0;
  let ticks = 0;
  const rescan = async () => {
    try {
      await withRequestTimeout(neverAnswers, 20);
    } catch {
      // What refreshSources does: a failed list just leaves the picker be.
    }
  };
  const poll = async () => {
    polls += 1;
  };
  await new Promise<void>((resolve) => {
    const tick = async () => {
      ticks += 1;
      await rescan();
      await poll();
      if (ticks < 2) setTimeout(tick, 1);
      else resolve();
    };
    void tick();
  });
  assert.equal(polls, 2);
});

test("the caller's signal still cancels, and the timer does not outlive a win", async () => {
  const controller = new AbortController();
  const cancelled = withRequestTimeout(neverAnswers, 60_000, controller.signal);
  controller.abort();
  await assert.rejects(
    () => cancelled,
    (error: Error) => error.name === "AbortError",
  );

  // An already aborted caller signal must not let the request start unguarded.
  const alreadyGone = new AbortController();
  alreadyGone.abort();
  await assert.rejects(
    () => withRequestTimeout(neverAnswers, 60_000, alreadyGone.signal),
    (error: Error) => error.name === "AbortError",
  );

  // A request that wins leaves nothing behind that could abort a later one.
  let seen: AbortSignal | null = null;
  const value = await withRequestTimeout(async (signal) => {
    seen = signal;
    return "ok";
  }, 20);
  assert.equal(value, "ok");
  await new Promise((resolve) => setTimeout(resolve, 40));
  assert.equal((seen as unknown as AbortSignal).aborted, false);
});

test("a response for the source the user just left is dropped", () => {
  // A manual refresh of A, answered after the picker moved to B.
  assert.equal(
    isPageStale({
      requestSelection: 1,
      currentSelection: 2,
      requestSourceId: "a",
      pageSourceId: "a",
    }),
    true,
  );
  // A -> B -> A: the id matches again, but the cursor and buffer were reset.
  assert.equal(
    isPageStale({
      requestSelection: 1,
      currentSelection: 3,
      requestSourceId: "a",
      pageSourceId: "a",
    }),
    true,
  );
  // The ordinary poll, and the unset source the server answers with its default.
  assert.equal(
    isPageStale({
      requestSelection: 2,
      currentSelection: 2,
      requestSourceId: "a",
      pageSourceId: "a",
    }),
    false,
  );
  assert.equal(
    isPageStale({
      requestSelection: 2,
      currentSelection: 2,
      requestSourceId: null,
      pageSourceId: "server-default",
    }),
    false,
  );
  // A server that answered with a different file than the one asked for.
  assert.equal(
    isPageStale({
      requestSelection: 2,
      currentSelection: 2,
      requestSourceId: "a",
      pageSourceId: "b",
    }),
    true,
  );
});

test("the skipped-lines warning outlives the poll that raised it", () => {
  const dropped = nextDroppedState(false, { droppedBytes: 4096, reset: false });
  assert.equal(dropped, true);
  // The next quiet poll: the gap is still in the buffer, so the warning stays.
  assert.equal(
    nextDroppedState(dropped, { droppedBytes: 0, reset: false }),
    true,
  );
  // A reset replaces everything on screen with a fresh tail.
  assert.equal(
    nextDroppedState(dropped, { droppedBytes: 0, reset: true }),
    false,
  );
  assert.equal(
    nextDroppedState(false, { droppedBytes: 0, reset: false }),
    false,
  );
});

test("the deadline fires even when the work ignores the abort", async () => {
  // authFetch awaits refreshSession() on a 401 and hands it no signal, so
  // aborting settled nothing: the promise stayed pending, the caller's
  // in-flight guard stayed pinned and the pane froze. Racing the deadline is
  // what makes the backstop a backstop.
  const deaf = () => new Promise<never>(() => {});
  const started = Date.now();
  await assert.rejects(
    () => withRequestTimeout(deaf, 20),
    (error: Error) => isRequestTimeout(error),
  );
  assert.ok(Date.now() - started < 2000);
});

test("a caller abort still reads as one when the work ignores it too", async () => {
  const deaf = () => new Promise<never>(() => {});
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    () => withRequestTimeout(deaf, 20, controller.signal),
    (error: Error) => isAbort(error) && !isRequestTimeout(error),
  );
});
