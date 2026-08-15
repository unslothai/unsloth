// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MAX_HELD_CHARS,
  createStreamPublishGate,
} from "../src/features/chat/utils/stream-pacing.ts";

type Scheduled = {
  frames: Array<() => void>;
  timers: { run: () => void; ms: number }[];
  cancelledFrames: number[];
  clearedTimers: number[];
};

/** Run `body` with the frame and timer schedulers replaced by recording stubs. */
function withStubbedScheduling(body: (scheduled: Scheduled) => void): void {
  const globals = globalThis as unknown as {
    requestAnimationFrame: (cb: () => void) => number;
    cancelAnimationFrame: (handle: number) => void;
    setTimeout: (cb: () => void, ms: number) => number;
    clearTimeout: (handle: number | undefined) => void;
  };
  const real = {
    requestAnimationFrame: globals.requestAnimationFrame,
    cancelAnimationFrame: globals.cancelAnimationFrame,
    setTimeout: globals.setTimeout,
    clearTimeout: globals.clearTimeout,
  };
  const scheduled: Scheduled = {
    frames: [],
    timers: [],
    cancelledFrames: [],
    clearedTimers: [],
  };
  globals.requestAnimationFrame = (cb) => scheduled.frames.push(cb);
  globals.cancelAnimationFrame = (handle) => {
    scheduled.cancelledFrames.push(handle);
  };
  globals.setTimeout = (run, ms) => scheduled.timers.push({ run, ms });
  globals.clearTimeout = (handle) => {
    scheduled.clearedTimers.push(handle as number);
  };
  try {
    body(scheduled);
  } finally {
    Object.assign(globals, real);
  }
}

/** A stream feeding one gate, recording what each publish would carry. */
function streamThrough(canPublish: (length: number) => boolean) {
  let cumulative = "";
  const published: string[] = [];
  return {
    feed(chunk: string) {
      cumulative += chunk;
      if (canPublish(cumulative.length)) {
        published.push(cumulative);
      }
    },
    published,
    get cumulative() {
      return cumulative;
    },
    /** What a stop right now would discard, since only published text survives. */
    get held() {
      return cumulative.length - (published.at(-1)?.length ?? 0);
    },
  };
}

test("the gate reopens once per frame", () => {
  withStubbedScheduling(({ frames }) => {
    const canPublish = createStreamPublishGate();

    assert.equal(canPublish(1), true, "the first chunk always publishes");
    assert.equal(canPublish(2), false);
    assert.equal(canPublish(3), false);
    assert.equal(frames.length, 1, "one pending frame, not one per chunk");

    frames.shift()?.();
    assert.equal(canPublish(4), true);
    assert.equal(canPublish(5), false);
  });
});

test("a burst between frames collapses into one update", () => {
  withStubbedScheduling(({ frames }) => {
    const stream = streamThrough(createStreamPublishGate());

    ["a", "b", "c", "d", "e", "f"].forEach((chunk, index) => {
      // A frame lands before the fourth chunk.
      if (index === 3) {
        frames.shift()?.();
      }
      stream.feed(chunk);
    });

    assert.deepEqual(stream.published, ["a", "abcd"]);
    // Skipped chunks remain accumulated for the next publish.
    assert.equal(stream.cumulative, "abcdef");
  });
});

test("a quiet tail waits for another chunk or the caller's final update", () => {
  withStubbedScheduling(({ frames }) => {
    const stream = streamThrough(createStreamPublishGate());

    stream.feed("a");
    stream.feed("b");
    stream.feed("c");
    frames.shift()?.();
    assert.deepEqual(
      stream.published,
      ["a"],
      "reopening alone cannot publish the quiet tail",
    );

    // The next chunk carries everything withheld.
    stream.feed("d");
    assert.deepEqual(stream.published, ["a", "abcd"]);
  });
});

test("a stream slower than the frame rate is never held back", () => {
  withStubbedScheduling(({ frames }) => {
    const canPublish = createStreamPublishGate();

    for (let length = 1; length <= 5; length += 1) {
      assert.equal(canPublish(length), true, `chunk ${length} publishes`);
      frames.shift()?.();
    }
  });
});

test("the gate reopens even when no frame ever comes", () => {
  withStubbedScheduling(({ frames, timers }) => {
    const canPublish = createStreamPublishGate();

    assert.equal(canPublish(1), true);
    assert.equal(canPublish(2), false);
    assert.equal(frames.length, 1, "the wait is on a frame, not a timer alone");
    assert.equal(
      timers[0]?.ms,
      500,
      "the fallback bounds one unpainted interval",
    );

    timers[0]?.run();
    assert.equal(
      canPublish(3),
      true,
      "a stream off screen still lands updates",
    );
  });
});

test("whichever wakes the gate cancels the other", () => {
  withStubbedScheduling(({ frames, cancelledFrames, clearedTimers }) => {
    const canPublish = createStreamPublishGate();
    canPublish(1);

    frames[0]?.();
    assert.deepEqual(cancelledFrames, [1], "the frame handle is released");
    assert.equal(
      clearedTimers.length,
      1,
      "the 500ms fallback is not left pending",
    );
  });
});

test("a frame arriving after the timer already reopened grants nothing extra", () => {
  withStubbedScheduling(({ frames, timers }) => {
    const canPublish = createStreamPublishGate();
    canPublish(1);

    timers[0]?.run();
    assert.equal(canPublish(2), true);
    // The frame from the first cycle is late; it must not open the second one.
    frames[0]?.();
    assert.equal(canPublish(3), false);
  });
});

test("a closed gate publishes rather than hold more than the cap", () => {
  // No frame and no timer ever fires here, so only the cap can publish again.
  withStubbedScheduling(() => {
    const stream = streamThrough(createStreamPublishGate());

    stream.feed("a");
    assert.deepEqual(stream.published, ["a"]);

    stream.feed("b".repeat(MAX_HELD_CHARS - 2));
    assert.equal(
      stream.published.length,
      1,
      "a hold under the cap still coalesces",
    );

    stream.feed("cc");
    assert.equal(stream.published.length, 2, "reaching the cap publishes");
    assert.equal(stream.published.at(-1), stream.cumulative);
  });
});

test("an unpainted burst never holds more than the cap", () => {
  withStubbedScheduling(() => {
    const stream = streamThrough(createStreamPublishGate());

    for (let chunk = 0; chunk < 400; chunk += 1) {
      stream.feed("0123456789");
      assert.ok(
        stream.held < MAX_HELD_CHARS,
        `a stop after chunk ${chunk} would discard ${stream.held} characters`,
      );
    }
  });
});

test("a painting window publishes on frames, never on the cap", () => {
  withStubbedScheduling(({ frames }) => {
    const stream = streamThrough(createStreamPublishGate());

    // Two chunks well under the cap per frame is what a painting window looks like.
    for (let frame = 0; frame < 20; frame += 1) {
      stream.feed("x".repeat(30));
      stream.feed("x".repeat(30));
      frames.shift()?.();
    }

    assert.equal(stream.published.length, 20, "one publish per frame");
  });
});

const ADAPTER = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

/** Drop comments, so a commented-out gate cannot satisfy a search. */
function withoutComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .split("\n")
    .map((line) => {
      const at = line.indexOf("//");
      if (at === -1) {
        return line;
      }
      // Keep a line whose "//" sits inside a string literal, as in "https://".
      const quotes = line.slice(0, at).match(/["'`]/g)?.length ?? 0;
      return quotes % 2 === 1 ? line : line.slice(0, at);
    })
    .join("\n");
}

/** The adapter between two anchors, without its comments. */
function regionOf(from: string, to: string): string {
  const start = ADAPTER.indexOf(from);
  assert.notEqual(start, -1, `"${from}" is gone; this test needs rewriting`);
  const end = ADAPTER.indexOf(to, start);
  assert.notEqual(end, -1, `"${to}" is gone; this test needs rewriting`);
  return withoutComments(ADAPTER.slice(start, end));
}

test("the stream loop gates the text publish, not just the yield", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  const gate = loop.indexOf("if (!canPublish(streamedChars)) {");
  assert.notEqual(gate, -1, "the text publish is not gated");
  const rebuild = loop.indexOf(
    "const assistantContent = liveAssistantContent()",
  );
  assert.ok(rebuild > gate, "the gate must precede the message rebuild");
  const skip = loop.indexOf("continue;", gate);
  assert.ok(
    skip > gate && skip < rebuild,
    "a closed gate must skip the rebuild instead of becoming a no-op",
  );

  // Accumulate before the gate, so the next publish still carries a skipped chunk.
  const append = loop.indexOf("cumulativeText += delta");
  assert.ok(append !== -1 && append < gate);
});

test("the run creates one gate, not one per chunk", () => {
  const source = withoutComments(ADAPTER);

  const construction = source.indexOf(
    "const canPublish = createStreamPublishGate()",
  );
  assert.notEqual(construction, -1, "the gate is gone");
  const loop = source.indexOf("for await (const chunk of stream) {");
  assert.notEqual(loop, -1, "the stream loop is gone");
  assert.ok(
    construction < loop,
    "a gate built inside the loop is new for every chunk, so it coalesces nothing",
  );
});

test("the live tool-argument preview shares the gate", () => {
  const preview = regionOf(
    'if (toolEvent.type === "tool_args") {',
    "\n                closeReasoningContent();",
  );

  const gate = preview.indexOf("if (canPublish(streamedChars)) {");
  assert.notEqual(gate, -1, "the per-argument-delta preview is not gated");
  const rebuild = preview.indexOf("content: liveAssistantContent()");
  assert.ok(rebuild > gate, "the gate must precede the message rebuild");

  // Argument deltas never reach cumulativeText, so without this the cap can
  // never fire on a turn that is only streaming a tool call's arguments.
  const count = preview.indexOf("streamedChars += fragment.length;");
  assert.ok(
    count !== -1 && count < gate,
    "arguments must count toward the cap",
  );
});

test("the gate is fed a counter that only grows", () => {
  const source = withoutComments(ADAPTER);

  // cumulativeText shrinks when the ${...} strip fires, which would let a closed
  // gate hold the removed length on top of the cap before publishing again.
  assert.equal(
    source.indexOf("canPublish(cumulativeText"),
    -1,
    "the cap must not be measured against the mutable reply length",
  );
  assert.ok(source.includes("let streamedChars = 0;"), "the counter is gone");
  const writes = source.match(/streamedChars\s*[+^*/-]?=[^=]/g) ?? [];
  assert.equal(
    writes.filter((write) => write.startsWith("streamedChars +=")).length,
    writes.length - 1,
    "the counter may only be initialised once and incremented after",
  );
});

test("a reasoning group the gate skipped is adopted before the final metadata", () => {
  const source = withoutComments(ADAPTER);

  const adopt = source.indexOf(
    "if (finalReasoningGroups > reasoningDurationTracker.groupCount) {",
  );
  assert.notEqual(adopt, -1, "a group revealed only by skipped chunks is lost");
  const finalize = source.indexOf(
    "reasoningDurationTracker.finishGroup();",
    adopt,
  );
  assert.ok(
    finalize > adopt,
    "the group must be adopted before the run finalizes durations",
  );
});
