// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  MAX_HELD_CHARS,
  createStreamPublishGate,
} from "../src/features/chat/utils/stream-pacing.ts";
import {
  countReasoningGroups,
  createReasoningDurationTracker,
  lastReasoningGroupTextLength,
} from "../src/features/chat/utils/reasoning-duration.ts";
import { parseAssistantContent } from "../src/features/chat/utils/parse-assistant-content.ts";

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
function regionOf(from: string, to: string, maxChars = 75_000): string {
  const start = ADAPTER.indexOf(from);
  assert.notEqual(start, -1, `"${from}" is gone; this test needs rewriting`);
  const end = ADAPTER.indexOf(to, start);
  assert.notEqual(end, -1, `"${to}" is gone; this test needs rewriting`);
  // Without this, editing the end anchor's line (even adding a space) silently
  // slides the region to the next match hundreds of lines away, and the
  // ordering assertions below go on passing against the wrong slice. A ceiling
  // on drift, not a budget: raise it when the loop legitimately grows, after
  // checking the anchors still land where they should.
  assert.ok(
    end - start < maxChars,
    `the region from "${from}" to "${to}" is ${end - start} chars; ` +
      "an anchor has drifted and this test needs rewriting",
  );
  return withoutComments(ADAPTER.slice(start, end));
}

test("the gate paces the publish, not the bookkeeping before it", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // The whole shape of this change. Everything that interprets the stream --
  // the content rebuild and the reasoning tracker -- runs on EVERY arrival, and
  // only the yield to assistant-ui is coalesced. Pacing the interpretation too
  // is what dragged reasoning timing, split tags, server summaries and replay
  // metadata into a change that is about paint cost.
  // The append goes through `appendCumulative`, which is what keeps the
  // delta-fed think tracker, placeholder watch and incremental parse in step
  // with the reply. What this test cares about is unchanged: it happens in the
  // loop, on every arrival, before the rebuild reads it.
  const append = loop.indexOf("appendCumulative(delta)");
  const rebuild = loop.indexOf(
    "const assistantContent = liveAssistantContent()",
  );
  const track = loop.indexOf("countReasoningGroups(assistantContent)");
  const finish = loop.indexOf("reasoningDurationTracker.finishGroup()");
  const gate = loop.search(/if \([^)]*!canPublish\(streamedChars\)\) \{/);
  const publish = loop.indexOf("content: assistantContent,");

  for (const [name, at] of [
    ["the text append", append],
    ["the content rebuild", rebuild],
    ["the reasoning tracker", track],
    ["the group finish", finish],
    ["the gate", gate],
    ["the publish", publish],
  ] as const) {
    assert.notEqual(at, -1, `${name} is gone from the loop`);
  }

  assert.ok(append < rebuild, "the chunk must be accumulated before the rebuild");
  assert.ok(rebuild < track, "the tracker reads the rebuilt content");
  assert.ok(
    finish < gate,
    "the reasoning tracker must observe every arrival, not only publishing ones",
  );
  assert.ok(gate < publish, "the publish must be paced");

  const skip = loop.indexOf("continue;", gate);
  assert.ok(
    skip > gate && skip < publish,
    "a closed gate must skip the publish instead of becoming a no-op",
  );
});

test("pacing cannot change a reasoning duration", () => {
  // The property the placement buys, stated directly: run the loop's
  // interpretation over a set of arrivals, publish on every one, then publish
  // on only the last, and require identical durations. Under the old placement
  // each of these cases measured differently depending on which arrivals the
  // gate let through, and each one cost a review round.
  const hasUnclosed = (text: string) =>
    text.lastIndexOf("<think>") > text.lastIndexOf("</think>");

  const run = (
    arrivals: Array<[number, string]>,
    publishes: ReadonlySet<number>,
  ) => {
    let clock = 0;
    const tracker = createReasoningDurationTracker(() => clock);
    let cumulative = "";
    let lastPublished = "";

    arrivals.forEach(([at, delta], index) => {
      clock = at;
      cumulative += delta;
      // Everything here is what the loop does before it consults the gate.
      const content = parseAssistantContent(cumulative);
      const groups = countReasoningGroups(content);
      if (groups > tracker.groupCount) {
        tracker.startGroup(groups - 1);
      }
      if (groups > 0) {
        tracker.resumeGroup(groups - 1, lastReasoningGroupTextLength(content));
      }
      if (tracker.hasActiveGroup && !hasUnclosed(cumulative)) {
        tracker.finishGroup();
      }
      if (publishes.has(index)) {
        lastPublished = cumulative;
      }
    });
    return { meta: tracker.metadata(), lastPublished };
  };

  const cases: Array<[string, Array<[number, string]>]> = [
    [
      "an opening tag split across two arrivals",
      [[0, "Hello "], [1000, "<thi"], [1000, "nk>why"], [9000, "</think>"], [9500, "answer"]],
    ],
    [
      "the opening tag as its own delta",
      [[0, "<think>"], [1000, "body"], [30000, "</think>"], [30100, "answer"]],
    ],
    [
      "several complete blocks in a row",
      [[1000, "<think>a</think>"], [5000, "<think>b</think>"], [9000, "<think>c</think>"], [12000, "answer"]],
    ],
    [
      "a later pass that opens bodyless",
      [[1000, "<think>first</think>"], [2000, "text "], [3000, "<think>"], [4000, "second "], [30000, "more</think>"], [31000, "end"]],
    ],
    [
      "a long pause between the reasoning and the answer",
      [[1000, "<think>x"], [2000, "y</think>"], [32000, "answer"]],
    ],
  ];

  for (const [name, arrivals] of cases) {
    const everyChunk = run(
      arrivals,
      new Set(arrivals.map((_, index) => index)),
    );
    const onlyTheLast = run(arrivals, new Set([arrivals.length - 1]));
    assert.deepEqual(
      onlyTheLast.meta,
      everyChunk.meta,
      `pacing changed the durations for ${name}`,
    );
    assert.equal(
      onlyTheLast.lastPublished,
      everyChunk.lastPublished,
      `pacing changed the final text for ${name}`,
    );
  }
});

test("no gate timestamp is threaded through the reasoning tracker", () => {
  const source = withoutComments(ADAPTER);

  // The tracker sees every arrival, so it never has to be told when something
  // it missed happened. These are the names the deferred-parse design needed;
  // if any comes back, the coalescing has leaked into the bookkeeping again.
  for (const leaked of [
    "gateHeldSince",
    "gateReasoningEndedAt",
    "reconcileReasoning",
    "adoptGatedReasoningGroups",
  ]) {
    assert.ok(
      !source.includes(leaked),
      `${leaked} is back: the gate is deferring interpretation again`,
    );
  }

  // And the tracker's own API stays free of the back-dating arguments.
  assert.ok(
    source.includes("reasoningDurationTracker.finishGroup()"),
    "finishGroup is being given a timestamp again",
  );
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
    // This branch is short. Bounding it keeps a whitespace edit to the end
    // anchor's line from sliding the region hundreds of lines down and leaving
    // the ordering assertions below to pass against the wrong slice.
    6_000,
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

test("streaming tool-call argument deltas are paced like the text path", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // Both branches of the OpenAI delta.tool_calls accumulator feed the counter,
  // so a turn that only streams a call's arguments is still capped.
  const count = loop.indexOf(
    "streamedChars +=\n                    argsFragment.length",
  );
  assert.notEqual(count, -1, "tool-call argument deltas are not counted");

  // Tolerant of extra forcing conditions (replay state), intolerant of the
  // gate call going away.
  const gate = loop.search(/addedToolCall \|\|[\s\S]{0,80}?canPublish\(streamedChars\)/);
  assert.notEqual(gate, -1, "the tool-call delta publish is not gated");
  assert.ok(gate > count, "the fragment must be counted before the gate");

  // A fragment that introduces a call must never be coalesced away: that part
  // is state an aborted turn would otherwise lose.
  assert.ok(
    loop.includes("addedToolCall = true;"),
    "a newly created tool call no longer forces a publish",
  );
});

test("the gate is fed every arrival, not only the tool-call ones", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // Without this the cap can never bind on a plain-text reply, which silently
  // restores the unbounded stop loss the cap exists to prevent.
  assert.ok(
    loop.includes("streamedChars += reasoning.length + delta.length;"),
    "text and reasoning arrivals are not counted toward the cap",
  );
});

test("a cap-forced publish resets the baseline for the next one", () => {
  withStubbedScheduling(() => {
    const canPublish = createStreamPublishGate();
    assert.equal(canPublish(0), true, "the first chunk publishes");

    // No frame and no timer ever fire, so only the cap can publish. Each cycle
    // must measure from the last publish; if the baseline only moved while the
    // gate was open, the second cycle would publish on every single chunk.
    for (let cycle = 1; cycle <= 4; cycle += 1) {
      const at = cycle * MAX_HELD_CHARS;
      assert.equal(canPublish(at - 1), false, `cycle ${cycle} held below cap`);
      assert.equal(canPublish(at), true, `cycle ${cycle} published at the cap`);
    }
  });
});

test("the backend tool events publish ungated", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // tool_start / tool_end carry the card's state -- name, result, approval,
  // provenance -- not a preview of it, and they are rare. Pacing them would
  // let a Stop persist a card that never got its result. Only the per-delta
  // argument preview above them is paced.
  const preview = loop.indexOf('if (toolEvent.type === "tool_args") {');
  const events = loop.indexOf("const toolProvenance = parseToolProvenance(");
  assert.ok(preview !== -1 && events > preview, "the tool-event branch moved");

  const between = loop.slice(preview, events);
  const previewGate = between.indexOf("if (canPublish(streamedChars)) {");
  assert.notEqual(previewGate, -1, "the argument preview is not paced");

  // Exactly one gate call between the preview and the events: the preview's.
  const gates = between.match(/canPublish\(/g) ?? [];
  assert.equal(
    gates.length,
    1,
    "a state-bearing tool event is being paced along with the preview",
  );

  // And none after them either, up to the publish they share.
  const publish = loop.indexOf("yield {", events);
  const after = loop.slice(events, publish);
  assert.ok(
    !after.includes("canPublish("),
    "the tool-event publish itself is paced",
  );
});

test("a state-bearing provider delta is never held by the gate", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // A thought signature or reasoning ledger reaches the message only through a
  // yield, so holding one behind the gate loses it outright on Stop.
  assert.ok(
    loop.includes("let replayStateChanged = false;"),
    "replay state changes are not tracked",
  );
  const gate = loop.search(/if \(!replayStateChanged && !canPublish\(/);
  assert.notEqual(gate, -1, "replay state does not force a publish");
});

test("a content-free replay delta still reaches the message", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // Gemini 3 ships a fragment whose only payload is a thoughtSignature, and the
  // Codex client puts its reasoning ledger on a text-free terminal delta. The
  // empty-content skip runs before the gate, so forcing a publish at the gate
  // alone never sees either of them.
  const replaySkip = loop.indexOf(
    "if (replayStateChanged && !delta && !reasoning) {",
  );
  const emptySkip = loop.indexOf("if (!delta && !reasoning) {");
  assert.notEqual(replaySkip, -1, "content-free replay metadata is dropped");
  assert.ok(
    replaySkip < emptySkip,
    "replay state must be handled before the empty-content skip",
  );
});

test("a per-call thought signature forces a publish", () => {
  const source = withoutComments(ADAPTER);

  // Gemini carries the signature on the tool call itself, not only at message
  // level, and the next turn is rejected outright without it. Updating an
  // EXISTING call adds no part, so addedToolCall is false and the message-level
  // latch never sees it; a Stop while the gate holds it persists a turn that
  // cannot be replayed.
  const update = source.indexOf("const prevExtra =");
  assert.notEqual(update, -1, "the existing-call update path is gone");
  const window = source.slice(update, update + 700);
  assert.ok(
    window.includes("replayStateChanged = true"),
    "a changed per-call extra_content does not force a publish",
  );
  // Read off `incomingExtra`, which is `call.extra_content` except when the
  // delta announced the NEXT call and parked its metadata for it.
  assert.ok(
    window.includes("incomingExtra !== undefined"),
    "the latch fires on calls that carry no extra_content at all",
  );
  assert.ok(
    window.includes("? undefined\n                      : call.extra_content"),
    "the latch no longer reads the delta's own extra_content",
  );

  // And the latch has to be honoured where the tool-call publish is decided.
  const decide = source.indexOf("addedToolCall ||", update);
  assert.ok(
    decide !== -1 &&
      source.slice(decide, decide + 120).includes("replayStateChanged"),
    "the tool-call publish ignores the replay latch",
  );
});

test("a chunk with nothing new to show does not spend a gate cycle", () => {
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // Two shapes, one guard. The reply can be empty, and the ${...} strip can
  // return a nonempty reply to exactly its previous length -- the Mistral case.
  // Either way the publish would be identical to the last one, and asking the
  // gate would spend the open cycle on it and hold the next real token until a
  // frame, the timer or the cap.
  const emptied = loop.indexOf("assistantContent.length === 0");
  const unchanged = loop.indexOf(
    "cumulativeText.length === textLenBeforeChunk",
  );
  const gate = loop.search(/if \([^)]*!canPublish\(streamedChars\)\) \{/);
  assert.notEqual(emptied, -1, "an emptied reply still reaches the gate");
  assert.notEqual(unchanged, -1, "an unchanged reply still reaches the gate");
  assert.ok(
    emptied < gate && unchanged < gate,
    "the skip must come before the gate is asked",
  );

  // Skipping must never swallow a publish that carries replay state. The latch
  // guards the condition, so look back from it rather than forward.
  const start = loop.lastIndexOf("if (", Math.min(emptied, unchanged));
  const skip = loop.slice(start, gate);
  assert.ok(
    skip.includes("!replayStateChanged"),
    "the skip can drop a state-bearing publish",
  );
});

test("a scheduler that calls back synchronously does not throw", () => {
  const globals = globalThis as unknown as {
    requestAnimationFrame: (cb: () => void) => number;
  };
  const real = globals.requestAnimationFrame;
  // No browser does this, but a polyfill or a test double can, and reopen runs
  // before the handles it cancels would have been assigned. Throwing here would
  // escape the stream loop and surface as a failed generation.
  globals.requestAnimationFrame = (cb) => {
    cb();
    return 1;
  };
  try {
    const canPublish = createStreamPublishGate();
    assert.equal(canPublish(0), true);
    assert.equal(canPublish(1), true, "the synchronous frame reopened the gate");
  } finally {
    globals.requestAnimationFrame = real;
  }
});

test("a cap-forced publish does not arm a second frame or timer", () => {
  withStubbedScheduling((scheduled) => {
    const canPublish = createStreamPublishGate();
    canPublish(0);
    assert.equal(scheduled.frames.length, 1);
    assert.equal(scheduled.timers.length, 1);

    canPublish(MAX_HELD_CHARS);
    canPublish(MAX_HELD_CHARS * 2);
    assert.equal(scheduled.frames.length, 1, "the cap re-armed a frame");
    assert.equal(scheduled.timers.length, 1, "the cap re-armed a timer");
  });
});
