// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import * as pacing from "../src/features/chat/utils/stream-pacing.ts";
import { readSrc } from "./helpers/kit.ts";

const { MAX_HELD_CHARS, createStreamPublishGate } = pacing;
// Looked up rather than imported by name, so a missing export fails the test instead of the file.
const withTrailingPublish = (pacing as Record<string, unknown>)
  .withTrailingPublish as typeof pacing.withTrailingPublish | undefined;
const TRAILING_PUBLISH = (pacing as Record<string, unknown>).TRAILING_PUBLISH as
  | typeof pacing.TRAILING_PUBLISH
  | undefined;

type Clock = {
  /** Runs every frame callback queued so far, like one painted frame. */
  frame(): void;
  framesQueued(): number;
  cancelledFrames: number[];
  timersArmed: number;
  timersCleared: number;
};

/** Deterministic rAF and timers: nothing fires unless the test says so. */
async function withFakeClock(body: (clock: Clock) => Promise<void>) {
  const globals = globalThis as unknown as Record<string, unknown>;
  const real = {
    requestAnimationFrame: globals.requestAnimationFrame,
    cancelAnimationFrame: globals.cancelAnimationFrame,
    setTimeout: globals.setTimeout,
    clearTimeout: globals.clearTimeout,
  };
  let nextHandle = 1;
  let frames = new Map<number, () => void>();
  const clock: Clock = {
    frame() {
      const due = frames;
      frames = new Map();
      for (const cb of due.values()) cb();
    },
    framesQueued: () => frames.size,
    cancelledFrames: [],
    timersArmed: 0,
    timersCleared: 0,
  };
  globals.requestAnimationFrame = (cb: () => void) => {
    const handle = nextHandle++;
    frames.set(handle, cb);
    return handle;
  };
  globals.cancelAnimationFrame = (handle: number) => {
    clock.cancelledFrames.push(handle);
    frames.delete(handle);
  };
  globals.setTimeout = () => {
    clock.timersArmed += 1;
    return nextHandle++;
  };
  globals.clearTimeout = () => {
    clock.timersCleared += 1;
  };
  try {
    await body(clock);
  } finally {
    Object.assign(globals, real);
  }
}

const settle = () => new Promise<void>((resolve) => setImmediate(resolve));

/** An async source the test feeds by hand, like a network reader. */
function channel<T>() {
  const queue: T[] = [];
  let ended = false;
  let failure: unknown = null;
  let waiter: ((r: IteratorResult<T>) => void) | null = null;
  let rejecter: ((e: unknown) => void) | null = null;
  let returned = 0;
  const flush = () => {
    if (!waiter) return;
    const resolve = waiter;
    const reject = rejecter!;
    if (queue.length > 0) {
      waiter = rejecter = null;
      resolve({ value: queue.shift()!, done: false });
    } else if (failure !== null) {
      waiter = rejecter = null;
      reject(failure);
    } else if (ended) {
      waiter = rejecter = null;
      resolve({ value: undefined, done: true });
    }
  };
  const iterable: AsyncIterable<T> = {
    [Symbol.asyncIterator]() {
      return {
        next: () =>
          new Promise<IteratorResult<T>>((resolve, reject) => {
            waiter = resolve;
            rejecter = reject;
            flush();
          }),
        return: async () => {
          returned += 1;
          return { value: undefined, done: true };
        },
      };
    },
  };
  return {
    iterable,
    push(...items: T[]) {
      queue.push(...items);
      flush();
    },
    end() {
      ended = true;
      flush();
    },
    fail(error: unknown) {
      failure = error;
      flush();
    },
    get returned() {
      return returned;
    },
  };
}

/** The adapter's loop in miniature: text publishes through the gate, held text on TRAILING_PUBLISH. */
function consume(source: AsyncIterable<string>, clock: Clock) {
  assert.ok(withTrailingPublish, "stream-pacing has no trailing publish");
  const gate = createStreamPublishGate();
  const published: { text: string; frame: number }[] = [];
  let frameNo = 0;
  const realFrame = clock.frame;
  clock.frame = () => {
    frameNo += 1;
    realFrame();
  };
  let text = "";
  const done = (async () => {
    for await (const item of withTrailingPublish(source, gate)) {
      if (item === TRAILING_PUBLISH) {
        if (gate(text.length)) published.push({ text, frame: frameNo });
        continue;
      }
      text += item as string;
      if (gate(text.length)) published.push({ text, frame: frameNo });
    }
    published.push({ text, frame: frameNo }); // the adapter's final yield
  })();
  return {
    published,
    done,
    get text() {
      return text;
    },
  };
}

test("held text is published on the next frame without waiting for another chunk", async () => {
  await withFakeClock(async (clock) => {
    const source = channel<string>();
    const run = consume(source.iterable, clock);

    source.push("a", "b", "c");
    await settle();
    assert.deepEqual(
      run.published.map((p) => p.text),
      ["a"],
    );

    clock.frame();
    await settle();
    assert.deepEqual(
      run.published.map((p) => p.text),
      ["a", "abc"],
      "the held tail waited for the next chunk instead of the next frame",
    );

    source.end();
    await run.done;
    assert.equal(run.published.at(-1)?.text, "abc");
  });
});

test("at most one paced publish per frame, and text and order are unchanged", async () => {
  await withFakeClock(async (clock) => {
    const source = channel<string>();
    const run = consume(source.iterable, clock);
    // Deterministic bursty arrival pattern: 0-4 chunks between frames, some frames idle.
    const pattern = [3, 0, 1, 4, 0, 0, 2, 1, 3, 0, 4, 1, 0, 2, 2, 0, 1, 3];
    const sent: string[] = [];
    let n = 0;
    for (const burst of pattern) {
      for (let i = 0; i < burst; i += 1) {
        const token = `t${n++} `;
        sent.push(token);
        source.push(token);
        await settle();
      }
      clock.frame();
      await settle();
      // Every burst is fully visible by the frame after it: nothing waits for a later chunk.
      assert.equal(
        run.published.at(-1)?.text ?? "",
        sent.join(""),
        `text sent before frame ${run.published.length} is still held after it`,
      );
    }
    source.end();
    await run.done;

    const paced = run.published.slice(0, -1);
    const perFrame = new Map<number, number>();
    for (const p of paced)
      perFrame.set(p.frame, (perFrame.get(p.frame) ?? 0) + 1);
    for (const [frame, count] of perFrame) {
      assert.ok(count <= 1, `frame ${frame} published ${count} times`);
    }
    // Each publish is a prefix of the next, ending with everything sent, in order.
    const full = sent.join("");
    let previous = "";
    for (const p of run.published) {
      assert.ok(p.text.startsWith(previous) && full.startsWith(p.text));
      previous = p.text;
    }
    assert.equal(run.text, full);
    assert.equal(run.published.at(-1)?.text, full);
  });
});

test("nothing held means no trailing publish and no extra render", async () => {
  await withFakeClock(async (clock) => {
    const source = channel<string>();
    const run = consume(source.iterable, clock);
    // Slower than the frame rate: every chunk lands on an open gate.
    for (let i = 0; i < 5; i += 1) {
      source.push(`w${i}`);
      await settle();
      clock.frame();
      await settle();
      clock.frame();
      await settle();
    }
    source.end();
    await run.done;
    assert.deepEqual(
      run.published.map((p) => p.text),
      ["w0", "w0w1", "w0w1w2", "w0w1w2w3", "w0w1w2w3w4", "w0w1w2w3w4"],
    );
  });
});

test("a publish the gate did not pace clears the held text", async () => {
  await withFakeClock(async () => {
    assert.ok(withTrailingPublish);
    const gate = createStreamPublishGate();
    assert.equal(gate(1), true);
    assert.equal(gate(2), false);
    assert.ok(gate.heldUntilReopen(), "refused text is held");
    gate.published();
    assert.equal(
      gate.heldUntilReopen(),
      null,
      "a forced publish already showed it",
    );
  });
});

test("the cap still publishes mid-frame and leaves nothing held", async () => {
  await withFakeClock(async () => {
    const gate = createStreamPublishGate();
    gate(0);
    assert.equal(gate(1), false);
    assert.equal(gate(MAX_HELD_CHARS), true);
    assert.equal(gate.heldUntilReopen(), null);
  });
});

test("an early exit cancels the pending frame and does not wait on an in-flight read", async () => {
  await withFakeClock(async (clock) => {
    assert.ok(withTrailingPublish && TRAILING_PUBLISH);
    const source = channel<string>();
    const gate = createStreamPublishGate();
    const iterator = withTrailingPublish(source.iterable, gate);
    source.push("a", "b");
    assert.equal((await iterator.next()).value, "a");
    gate(1);
    assert.equal((await iterator.next()).value, "b");
    assert.equal(gate(2), false, "held");
    const trailing = iterator.next();
    clock.frame();
    assert.equal((await trailing).value, TRAILING_PUBLISH);
    gate(2); // the trailing publish closes the gate and arms a new frame
    assert.equal(clock.framesQueued(), 1);
    // The consumer leaves (Stop / thread switch) while a read is in flight.
    await iterator.return(undefined);
    assert.equal(
      clock.framesQueued(),
      0,
      "the pending frame was not cancelled",
    );
    assert.equal(
      source.returned,
      0,
      "return waits for the in-flight read, off the exit path",
    );
    source.push("c");
    await settle();
    assert.equal(
      source.returned,
      1,
      "the source is closed once its read settles",
    );
  });
});

test("a source error and a clean end pass through unchanged", async () => {
  await withFakeClock(async (clock) => {
    const source = channel<string>();
    const run = consume(source.iterable, clock);
    source.push("x", "y");
    await settle();
    source.fail(new Error("boom"));
    await assert.rejects(run.done, /boom/);
    assert.equal(clock.framesQueued(), 0, "the frame outlived the stream");
  });
});

test("the adapter loop publishes held text when the gate reopens", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const gate = adapter.indexOf("const canPublish = createStreamPublishGate();");
  const wrap = adapter.indexOf(
    "const stream = withTrailingPublish(sourceStream, canPublish);",
  );
  const loop = adapter.indexOf("for await (const chunk of stream) {");
  assert.ok(
    gate !== -1 && wrap > gate && loop > wrap,
    "the stream is not wrapped",
  );
  const head = adapter.slice(loop, loop + 700);
  assert.match(
    head,
    /if \(chunk === TRAILING_PUBLISH\) \{\s*if \(canPublish\(streamedChars\)\) \{\s*yield \{\s*content: liveAssistantContent\(\)/,
    "the trailing publish must go through the gate",
  );
});

test("every live publish in the loop is paced or clears the held text", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const start = adapter.indexOf("for await (const chunk of stream) {");
  const end = adapter.indexOf("} catch (streamError) {", start);
  assert.ok(start !== -1 && end > start, "the stream loop moved");
  const loop = adapter.slice(start, end);
  const yields = [...loop.matchAll(/yield \{\s*content:/g)];
  assert.ok(
    yields.length >= 7,
    `expected the live yields, found ${yields.length}`,
  );
  for (const match of yields) {
    const before = loop.slice(Math.max(0, match.index - 240), match.index);
    // An unpaced publish that leaves `held` set is repeated, unchanged, on the next frame.
    assert.ok(
      before.includes("canPublish.published();") ||
        /if \(canPublish\(streamedChars\)\) \{\s*$/.test(before),
      `a live yield neither paced nor marked published: ${loop.slice(match.index, match.index + 60)}`,
    );
  }
});
