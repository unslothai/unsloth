// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createFrameGate } from "../src/features/chat/utils/stream-pacing.ts";

test("the gate reopens once per painted frame", () => {
  const frames: Array<() => void> = [];
  const canPublish = createFrameGate((cb) => frames.push(cb));

  assert.equal(canPublish(), true, "the first chunk always publishes");
  assert.equal(canPublish(), false);
  assert.equal(canPublish(), false);
  assert.equal(frames.length, 1, "one pending frame, not one per chunk");

  frames.shift()?.();
  assert.equal(canPublish(), true);
  assert.equal(canPublish(), false);
});

test("a burst between frames collapses into one update", () => {
  const frames: Array<() => void> = [];
  const canPublish = createFrameGate((cb) => frames.push(cb));
  const chunks = ["a", "b", "c", "d", "e", "f"];

  let cumulative = "";
  const published: string[] = [];
  chunks.forEach((chunk, index) => {
    cumulative += chunk;
    // A frame lands after the third chunk.
    if (index === 3) {
      frames.shift()?.();
    }
    if (canPublish()) {
      published.push(cumulative);
    }
  });

  assert.deepEqual(published, ["a", "abcd"]);
  // Skipped chunks remain accumulated for the next publish.
  assert.equal(cumulative, "abcdef");
});

test("a quiet tail waits for another chunk or the caller's final update", () => {
  const frames: Array<() => void> = [];
  const canPublish = createFrameGate((cb) => frames.push(cb));
  let cumulative = "";
  const published: string[] = [];
  const feed = (chunk: string) => {
    cumulative += chunk;
    if (canPublish()) {
      published.push(cumulative);
    }
  };

  feed("a");
  feed("b");
  feed("c");
  // Reopening alone cannot publish the quiet tail.
  frames.shift()?.();
  assert.deepEqual(published, ["a"]);
  assert.equal(
    published.at(-1),
    "a",
    "Stop here would retain only the previous streamed update",
  );

  // The next chunk carries everything withheld.
  feed("d");
  assert.deepEqual(published, ["a", "abcd"]);
});

test("a stream slower than the frame rate is never held back", () => {
  const frames: Array<() => void> = [];
  const canPublish = createFrameGate((cb) => frames.push(cb));

  for (let i = 0; i < 5; i += 1) {
    assert.equal(canPublish(), true, `chunk ${i} publishes`);
    frames.shift()?.();
  }
});

/** Stub the default frame and timer schedulers. */
function withStubbedScheduling(
  body: (frames: Array<() => void>, timers: [() => void, number][]) => void,
): void {
  const globals = globalThis as unknown as {
    requestAnimationFrame: (cb: () => void) => number;
    setTimeout: (cb: () => void, ms: number) => number;
  };
  const realFrame = globals.requestAnimationFrame;
  const realTimeout = globals.setTimeout;
  const frames: Array<() => void> = [];
  const timers: [() => void, number][] = [];
  globals.requestAnimationFrame = (cb) => frames.push(cb);
  globals.setTimeout = (cb, ms) => timers.push([cb, ms]);
  try {
    body(frames, timers);
  } finally {
    globals.requestAnimationFrame = realFrame;
    globals.setTimeout = realTimeout;
  }
}

test("the default gate waits for a frame", () => {
  withStubbedScheduling((frames, timers) => {
    const canPublish = createFrameGate();
    assert.equal(canPublish(), true);
    assert.equal(frames.length, 1, "the wait is on a frame, not a timer alone");
    assert.equal(canPublish(), false);
    frames[0]?.();
    assert.equal(canPublish(), true);
    assert.ok(timers.length > 0);
  });
});

test("the default gate reopens even when no frame ever comes", () => {
  withStubbedScheduling((frames, timers) => {
    const canPublish = createFrameGate();
    assert.equal(canPublish(), true);
    assert.equal(canPublish(), false);
    assert.equal(frames.length, 1);
    const [wake, delay] = timers[0] ?? [];
    assert.equal(delay, 500, "the fallback bounds one unpainted interval");
    wake?.();
    assert.equal(canPublish(), true, "a stream off screen still lands updates");
  });
});

test("a frame arriving after the timer already reopened grants nothing extra", () => {
  withStubbedScheduling((frames, timers) => {
    const canPublish = createFrameGate();
    canPublish();
    timers[0]?.[0]?.();
    assert.equal(canPublish(), true);
    // The frame from the first cycle is late; it must not open the second one.
    frames[0]?.();
    assert.equal(canPublish(), false);
  });
});

test("the chat stream loop gates the publish, not just the yield", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  const loopStart = source.indexOf("const canPublish = createFrameGate()");
  assert.ok(loopStart > 0, "the run creates one gate per request");

  const gate = source.indexOf("if (!canPublish()) {", loopStart);
  const rebuild = source.indexOf(
    "const assistantContent = liveAssistantContent()",
    loopStart,
  );
  assert.ok(
    gate > 0 && rebuild > gate,
    "the gate precedes the message rebuild",
  );
  const skip = source.indexOf("continue;", gate);
  assert.ok(
    skip > gate && skip < rebuild,
    "a closed gate skips the rebuild instead of becoming a no-op",
  );

  // Accumulate text before the gate so skipped chunks are retained.
  const append = source.indexOf("cumulativeText += delta", loopStart);
  assert.ok(append > 0 && append < gate);
});
