// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Issue #9807: a backend streaming parallel tool calls as id-less, index-based
// deltas reuses one slot for several calls, and the adapter used to append
// their arguments into one unparsable `{"url":"a"}{"url":"b"}`.
//
// The boundary is the end of a top-level JSON object, not a change of function
// name: the reported stream calls the same tool three times. These cover the
// scanner, the replay guard behind it, and the accumulation loop itself.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "typescript";

import {
  createBoundaryScan,
  splitTopLevelJsonObjects,
  toolCallReplayArguments,
} from "../src/features/chat/tool-call-arguments.ts";
import { findStreamedToolCallPartIndex } from "../src/features/chat/tool-call-id.ts";

// ---------------------------------------------------------------------------
// A. The scanner
// ---------------------------------------------------------------------------

test("a slot holding one object is left as one object", () => {
  assert.deepEqual(splitTopLevelJsonObjects('{"url":"a"}'), {
    complete: ['{"url":"a"}'],
    tail: "",
  });
  assert.deepEqual(splitTopLevelJsonObjects("{}"), {
    complete: ["{}"],
    tail: "",
  });
  assert.deepEqual(splitTopLevelJsonObjects(""), { complete: [], tail: "" });
});

test("adjacent objects are cut apart, however they are spaced", () => {
  assert.deepEqual(
    splitTopLevelJsonObjects('{"a":1}{"b":2} {"c":3}\n{"d":4}\r\n{"e":5}'),
    {
      complete: ['{"a":1}', '{"b":2}', '{"c":3}', '{"d":4}', '{"e":5}'],
      tail: "",
    },
  );
});

test("the object still being written is the tail, not a call", () => {
  assert.deepEqual(splitTopLevelJsonObjects('{"a":1}{"b":'), {
    complete: ['{"a":1}'],
    tail: '{"b":',
  });
  assert.deepEqual(splitTopLevelJsonObjects('{"a":'), {
    complete: [],
    tail: '{"a":',
  });
});

test("braces that are data, not structure, are not boundaries", () => {
  // A brace in a string, an escaped quote that does not end it, an even run
  // of backslashes that does, a Windows path, and nesting.
  assert.deepEqual(splitTopLevelJsonObjects('{"a":"}{"}{"b":2}'), {
    complete: ['{"a":"}{"}', '{"b":2}'],
    tail: "",
  });
  assert.deepEqual(splitTopLevelJsonObjects('{"a":"say \\"}{\\" ok"}{"b":2}'), {
    complete: ['{"a":"say \\"}{\\" ok"}', '{"b":2}'],
    tail: "",
  });
  assert.deepEqual(
    splitTopLevelJsonObjects('{"p":"C:\\\\Users\\\\me"}{"b":2}'),
    {
      complete: ['{"p":"C:\\\\Users\\\\me"}', '{"b":2}'],
      tail: "",
    },
  );
  assert.deepEqual(splitTopLevelJsonObjects('{"a":{"b":{"c":1}}}'), {
    complete: ['{"a":{"b":{"c":1}}}'],
    tail: "",
  });
  assert.deepEqual(splitTopLevelJsonObjects('{"a":[{"b":1},{"c":2}]}'), {
    complete: ['{"a":[{"b":1},{"c":2}]}'],
    tail: "",
  });
});

test("text that is not a run of objects is handed back untouched", () => {
  // Cutting any of these would invent a call the model never made.
  for (const text of [
    '[{"a":1}]',
    '"hello"',
    "42",
    "null",
    '{"a":1}junk{"b":2}',
    '{"a":1}}',
    '{"a":1,}{"b":2}',
    '{"a":"unterminated',
  ]) {
    assert.deepEqual(
      splitTopLevelJsonObjects(text),
      { complete: [], tail: text },
      text,
    );
  }
});

// ---------------------------------------------------------------------------
// B. Replay
// ---------------------------------------------------------------------------

test("a healthy call replays byte for byte", () => {
  assert.equal(
    toolCallReplayArguments('{"query":"first"}', { query: "first" }),
    '{"query":"first"}',
  );
  assert.equal(
    toolCallReplayArguments("", { query: "first" }),
    '{"query":"first"}',
  );
});

test("the _raw marker never reaches a backend as a tool parameter", () => {
  // Threads stored before the split still hold these, as do threads imported
  // through chat-import.ts. `_raw` is the adapter's own marker for text it
  // could not parse; no tool declares it.
  assert.equal(
    toolCallReplayArguments('{"query":"a"}{"query":"b"}', {
      _raw: '{"query":"a"}{"query":"b"}',
    }),
    "{}",
  );
  assert.equal(
    toolCallReplayArguments('{"query":', { _raw: '{"query":' }),
    "{}",
  );
});

test("arguments that are not one JSON object fall back rather than replay", () => {
  assert.equal(toolCallReplayArguments("[1,2]", { url: "a" }), '{"url":"a"}');
  assert.equal(toolCallReplayArguments(undefined, [1, 2]), "{}");
  assert.equal(toolCallReplayArguments(undefined, "nope"), "{}");
  assert.equal(toolCallReplayArguments(undefined, null), "{}");
  assert.equal(toolCallReplayArguments(undefined, undefined), "{}");
});

// ---------------------------------------------------------------------------
// C. The accumulation loop, as shipped
// ---------------------------------------------------------------------------

// chat-adapter.ts reaches the stores and a JSX barrel and cannot be imported,
// so lift the loop the way tests/pr9057-video-simulation.test.ts lifts its
// extractor. A re-implementation would pass while the adapter stayed broken,
// which is how this defect survived tool-call-delta-index.test.ts.
const adapterSource = readFileSync(
  fileURLToPath(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  ),
  "utf8",
);

function liftBetween(what: string, from: string, to: string): string {
  const start = adapterSource.indexOf(from);
  assert.ok(start >= 0, `${what}: "${from}" is gone from chat-adapter.ts`);
  const end = adapterSource.indexOf(to, start);
  assert.ok(end > start, `${what}: "${to}" is gone from chat-adapter.ts`);
  return adapterSource.slice(start, end);
}

/** The two helpers the loop calls, declared beside `toolCallParts`. */
function liftSplitHelpers(): string {
  const lifted = liftBetween(
    "split helpers",
    "let splitToolCallSeq = 0;",
    "// Raw tool_args accumulator per card",
  );
  assert.ok(
    lifted.includes("bornSplitToolCalls"),
    "the split helpers moved in chat-adapter.ts",
  );
  return lifted;
}

function liftDeltaLoop(): string {
  const loopStart = adapterSource.indexOf(
    "for (const tc of rawDeltaToolCalls) {",
  );
  assert.ok(
    loopStart >= 0,
    "the delta.tool_calls loop moved in chat-adapter.ts",
  );
  const gate = adapterSource.lastIndexOf(
    "if (",
    adapterSource.indexOf("addedToolCall ||", loopStart),
  );
  assert.ok(gate > loopStart, "the publish gate moved in chat-adapter.ts");
  const lifted = adapterSource.slice(loopStart, gate);
  assert.ok(
    lifted.includes("splitTopLevelJsonObjects"),
    "the loop no longer splits on JSON object boundaries",
  );
  return lifted;
}

interface DeltaCall {
  id?: string;
  index?: number;
  function?: { name?: string; arguments?: string };
  extra_content?: unknown;
}

interface LoopPart {
  toolCallId: string;
  toolName?: string;
  argsText?: string;
  args?: Record<string, unknown>;
  _delta_index?: number;
  _has_stable_id?: boolean;
  extra_content?: unknown;
}

/** The lifted loop, with the locals it closes over supplied by hand. */
function makeStream(): {
  feed: (batch: DeltaCall[], finished?: boolean) => boolean;
  parts: LoopPart[];
} {
  const body = `
    const toolCallParts = [];
    let codexRoundToolCallIds = [];
    const toolPartIdByBackendId = new Map();
    const cumulativeText = "";
    let streamedChars = 0;
    ${liftSplitHelpers()}
    const resolveToolPartId = (backendId) => {
      const seen = toolPartIdByBackendId.get(backendId);
      if (seen) return seen;
      toolPartIdByBackendId.set(backendId, backendId);
      return backendId;
    };
    let addedToolCall = false;
    let replayStateChanged = false;
    // The chunk the lifted loop reads finish_reason off, so the provider-turn
    // reset it performs is the shipped one.
    let chunk = { choices: [{}] };
    function feed(rawDeltaToolCalls, finished) {
      chunk = { choices: [finished ? { finish_reason: "tool_calls" } : {}] };
      addedToolCall = false;
      replayStateChanged = false;
      ${liftDeltaLoop()}
      return addedToolCall;
    }
    return { feed, parts: toolCallParts };
  `;
  const js = ts.transpileModule(body, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return new Function(
    "splitTopLevelJsonObjects",
    "createBoundaryScan",
    "findStreamedToolCallPartIndex",
    js,
  )(
    splitTopLevelJsonObjects,
    createBoundaryScan,
    findStreamedToolCallPartIndex,
  ) as {
    feed: (batch: DeltaCall[], finished?: boolean) => boolean;
    parts: LoopPart[];
  };
}

function run(batches: DeltaCall[][]): LoopPart[] {
  const stream = makeStream();
  for (const batch of batches) stream.feed(batch);
  return stream.parts;
}

const shape = (parts: LoopPart[]) => parts.map((p) => [p.toolName, p.argsText]);

test("the stream from #9807 becomes one call per JSON object", () => {
  // Three fetches and a search, all id-less at index 0. The same tool repeats,
  // so there is no name change to cut on.
  const parts = run([
    [{ index: 0, function: { name: "url", arguments: '{"url":"a"}' } }],
    [{ index: 0, function: { name: "url", arguments: '{"url":"b"}' } }],
    [{ index: 0, function: { name: "query", arguments: '{"q":"c"}' } }],
    [{ index: 0, function: { name: "url", arguments: '{"url":"d"}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["url", '{"url":"a"}'],
    ["url", '{"url":"b"}'],
    ["query", '{"q":"c"}'],
    ["url", '{"url":"d"}'],
  ]);
  const ids = parts.map((p) => p.toolCallId);
  assert.equal(new Set(ids).size, ids.length, `ids collide: ${ids.join(",")}`);
});

test("a call at another index is not overwritten when a slot splits", () => {
  // The split slot sits before the neighbour, so removing it and writing back
  // through the old position would delete that neighbour.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 1, function: { name: "beta", arguments: '{"b":2}' } }],
    [{ index: 0, function: { name: "gamma", arguments: '{"c":3}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
    ["gamma", '{"c":3}'],
  ]);
});

test("a call opened third reads third, whichever index it reused", () => {
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 1, function: { name: "beta", arguments: '{"b":2}' } }],
    [{ index: 1, function: { name: "gamma", arguments: '{"c":3}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
    ["gamma", '{"c":3}'],
  ]);
});

test("several objects inside one fragment split just the same", () => {
  const parts = run([
    [
      {
        index: 0,
        function: { name: "url", arguments: '{"url":"a"}{"url":"b"}' },
      },
    ],
  ]);

  assert.deepEqual(shape(parts), [
    ["url", '{"url":"a"}'],
    ["url", '{"url":"b"}'],
  ]);
});

test("a call born from a split is state, so it does not wait to publish", () => {
  const stream = makeStream();
  assert.equal(
    stream.feed([
      { index: 0, function: { name: "alpha", arguments: '{"a":1}' } },
    ]),
    true,
  );
  // Otherwise the pacing gate holds the second call back and Stop persists a
  // snapshot without it.
  assert.equal(
    stream.feed([
      { index: 0, function: { name: "beta", arguments: '{"b":2}' } },
    ]),
    true,
  );
});

test("an ordinary fragmented call is still one call", () => {
  assert.deepEqual(
    shape(
      run([
        [{ index: 0, function: { name: "alpha", arguments: '{"a":' } }],
        [{ index: 0, function: { arguments: "1" } }],
        [{ index: 0, function: { arguments: "}" } }],
      ]),
    ),
    [["alpha", '{"a":1}']],
  );
});

test("a stream that carries ids is left exactly as it was", () => {
  const parts = run([
    [
      {
        id: "call_a",
        index: 0,
        function: { name: "alpha", arguments: '{"a":' },
      },
      {
        id: "call_b",
        index: 1,
        function: { name: "beta", arguments: '{"b":' },
      },
    ],
    [
      { id: "call_a", index: 0, function: { arguments: "1}" } },
      { id: "call_b", index: 1, function: { arguments: "2}" } },
    ],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.toolName, p.argsText]),
    [
      ["call_a", "alpha", '{"a":1}'],
      ["call_b", "beta", '{"b":2}'],
    ],
  );
});

test("an id stamped on a later fragment still claims its own slot", () => {
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":' } }],
    [{ id: "call_a", index: 0, function: { arguments: "1}" } }],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.argsText]),
    [["call_a", '{"a":1}']],
  );
});

test("a fragment repeating the slot's id continues that call", () => {
  // llama-server grows the name across deltas. An id names its call, so
  // opening a new one here would give two cards one id, and tool_end files a
  // result against the first that carries it.
  const parts = run([
    [{ id: "call_a", index: 0, function: { name: "web", arguments: '{"q":"x"}' } }],
    [{ id: "call_a", index: 0, function: { name: "web_search" } }],
  ]);

  assert.deepEqual(shape(parts), [["web_search", '{"q":"x"}']]);
  assert.equal(parts.length, 1);
});

test("whitespace chunked after a closing brace is not a new call", () => {
  // Trailing whitespace is legal JSON and says nothing about another call.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "alpha", arguments: " " } }],
    [{ index: 0, function: { name: "alpha", arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1} '],
    ["alpha", '{"b":2}'],
  ]);
});

test("a late id claims the call still being written, never a closed one", () => {
  // The slot splits, leaving a half-written third call, then an id arrives. It
  // has to land on that unfinished call: appending to either of the two that
  // closed is the gluing this whole change is about.
  const parts = run([
    [
      {
        index: 0,
        function: { name: "alpha", arguments: '{"a":1}{"b":2}{"c":' },
      },
    ],
    [{ id: "call_c", index: 0, function: { arguments: "3}" } }],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.argsText]),
    [
      ["tool_call_0", '{"a":1}'],
      ["tool_call_0_1", '{"b":2}'],
      ["call_c", '{"c":3}'],
    ],
  );
});

test("a name-only delta for the next call does not rename the finished one", () => {
  // The name arrives before its arguments, so the accumulated text is still one
  // whole object and there is nothing to split on yet.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "beta" } }],
    [{ index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
  const ids = parts.map((p) => p.toolCallId);
  assert.equal(new Set(ids).size, ids.length, `ids collide: ${ids.join(",")}`);
});

test("an id arriving after one closed object opens a call, not a claim", () => {
  // A late id claims the slot its id-less opening fragment created, but only
  // while that call is still being written. This one has closed.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ id: "call_b", index: 0, function: { name: "beta", arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
});

test("a late id opens its own call when every call in the slot has closed", () => {
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}{"b":2}' } }],
    [
      {
        id: "call_c",
        index: 0,
        function: { name: "gamma", arguments: '{"c":3}' },
      },
    ],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.argsText]),
    [
      ["tool_call_0", '{"a":1}'],
      ["tool_call_0_1", '{"b":2}'],
      ["call_c", '{"c":3}'],
    ],
  );
});

test("an opening delta after a closed call does not claim it", () => {
  // The conventional opening delta carries the id and the name with empty
  // arguments. Landing it on the finished card stamps that card with the id, so
  // the arguments delta that follows matches and glues on, losing the call the
  // delta was announcing.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ id: "call_b", index: 0, function: { name: "beta", arguments: "" } }],
    [{ id: "call_b", index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
  const ids = parts.map((p) => p.toolCallId);
  assert.equal(ids[1], "call_b");
  assert.equal(new Set(ids).size, ids.length, `ids collide: ${ids.join(",")}`);
});

test("a name held for the next call grows across deltas", () => {
  // OpenAI streams a name as "web" then "_search"; llama-server resends "web"
  // then "web_search". Last-write-wins opens the call as "_search", which
  // matches no enabled tool and silently never runs.
  for (const fragments of [
    ["web", "_search"],
    ["web", "web_search"],
  ]) {
    const parts = run([
      [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
      ...fragments.map((name) => [{ index: 0, function: { name } }]),
      [{ index: 0, function: { arguments: '{"q":"x"}' } }],
    ]);

    assert.deepEqual(shape(parts), [
      ["alpha", '{"a":1}'],
      ["web_search", '{"q":"x"}'],
    ]);
  }
});

test("whitespace carrying the repeated name is not the next call", () => {
  // A provider that repeats the name on every delta and chunks the trailing
  // whitespace separately is still writing to the call that closed, so its name
  // is that call's resent. Parking it merged the two into "alphabeta".
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "alpha", arguments: " " } }],
    [{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1} '],
    ["beta", '{"b":2}'],
  ]);
});

test("metadata announced with a name waits for that call", () => {
  // Gemini stows the thought signature for the call being announced, so a
  // name-only delta carrying one describes the next call, not the closed one.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [
      {
        index: 0,
        function: { name: "beta" },
        extra_content: { google: { thought_signature: "SIG" } },
      },
    ],
    [{ index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
  assert.equal(parts[0].extra_content, undefined);
  assert.deepEqual(parts[1].extra_content, {
    google: { thought_signature: "SIG" },
  });
});

test("the resumable scan agrees with scanning from the start", () => {
  // The accumulator resumes the boundary scan instead of restarting it, which
  // is only safe while the two agree for every string and every set of chunk
  // boundaries.
  const pieces = [
    ..."{}\"\\ abc:,1[]".split(""),
    '\\"',
    '"a"',
    "NaN",
    "\n",
    "\r\n",
    "\t",
    '{"a":1}',
    "}{",
  ];
  // Deterministic, so a failure names one string rather than a mood.
  let seed = 20260827;
  const next = (n: number) => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed % n;
  };
  for (let trial = 0; trial < 4000; trial += 1) {
    let text = "";
    for (let i = next(17); i > 0; i -= 1) text += pieces[next(pieces.length)];
    const scan = createBoundaryScan();
    let cut = 0;
    let result = scan.feed("");
    while (cut < text.length) {
      cut = Math.min(text.length, cut + 1 + next(4));
      result = scan.feed(text.slice(0, cut));
    }
    assert.deepEqual(result, splitTopLevelJsonObjects(text), text);
  }
});

test("one argument streamed a character at a time stays linear", () => {
  // Rescanning the whole accumulation per fragment made a 20 KB argument cost
  // over a second on the thread that also paints the stream. Timed as a ratio
  // between two payload sizes rather than against a deadline: what has to hold
  // is that four times the argument costs four times the work, and a slow or
  // contended runner moves both measurements together instead of failing.
  const timeFeed = (size: number): number => {
    const payload = '{"code":"' + "x".repeat(size) + '"}';
    const stream = makeStream();
    stream.feed([{ index: 0, function: { name: "write", arguments: "" } }]);
    const started = performance.now();
    for (const ch of payload) {
      stream.feed([{ index: 0, function: { arguments: ch } }]);
    }
    const elapsed = performance.now() - started;
    assert.deepEqual(shape(stream.parts), [["write", payload]]);
    return elapsed;
  };
  timeFeed(4000);
  const small = timeFeed(4000);
  const large = timeFeed(16000);
  // Linear is 4x and quadratic 16x, so the line between them is 8x.
  assert.ok(
    large < small * 8,
    `four times the payload cost ${(large / small).toFixed(1)}x the time`,
  );
});

test("metadata arriving alone stays on the call that closed", () => {
  // No name, so nothing announces another call: the signature is this card's.
  // Parking it lost it outright when no further call followed.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 0, extra_content: { google: { thought_signature: "SIG" } } }],
  ]);

  assert.deepEqual(shape(parts), [["alpha", '{"a":1}']]);
  assert.deepEqual(parts[0].extra_content, {
    google: { thought_signature: "SIG" },
  });
});

test("a name resent or grown after a call closed invents nothing", () => {
  // Indistinguishable from a second no-argument call to the same tool, so the
  // conservative reading is the one that does not run a tool twice.
  for (const resent of ["alpha", "alpha_long"]) {
    const parts = run([
      [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
      [{ index: 0, function: { name: resent } }],
    ]);
    assert.deepEqual(shape(parts), [["alpha", '{"a":1}']]);
  }
});

test("an argument fragment that is not a string does not abort the stream", () => {
  // llama-server has shipped `arguments` as a decoded object rather than the
  // string the API specifies, and the chunk is cast rather than validated.
  const parts = run([
    [
      {
        index: 0,
        function: {
          name: "alpha",
          arguments: { a: 1 } as unknown as string,
        },
      },
    ],
    [{ index: 0, function: { arguments: '{"a":1}' } }],
  ]);

  assert.deepEqual(shape(parts), [["alpha", '{"a":1}']]);
});

test("a fragment that does not open an object does not open a call", () => {
  // A next call begins with the "{" of its own arguments object. Forking on any
  // non-whitespace text cut where the scanner deliberately leaves the text
  // whole, so a stray scalar suffix ran the tool a second time.
  const parts = run([
    [{ index: 0, function: { name: "q", arguments: '{"query":"a"}' } }],
    [{ index: 0, function: { name: "q", arguments: '"b"' } }],
  ]);

  assert.deepEqual(shape(parts), [["q", '{"query":"a"}"b"']]);
});

test("metadata parked with a name is merged, not replaced", () => {
  // A signature parked with the name and metadata arriving with the arguments
  // are different fields of one call, and the replayed turn is rejected
  // without either.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [
      {
        index: 0,
        function: { name: "beta" },
        extra_content: { google: { thought_signature: "SIG" } },
      },
    ],
    [
      {
        index: 0,
        function: { arguments: '{"b":2}' },
        extra_content: { openai: { x: 1 } },
      },
    ],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
  assert.deepEqual(parts[1].extra_content, {
    google: { thought_signature: "SIG" },
    openai: { x: 1 },
  });
});

test("an MCP tool that really takes _raw keeps it", () => {
  // The adapter writes `{ _raw }` holding the exact text it could not parse, so
  // that pairing is the marker. `_raw` is not reserved, and an MCP server's
  // schema is its own, so a tool that declares one must still be callable.
  assert.equal(
    toolCallReplayArguments('{"url":"a"}{"url":"b"}', {
      _raw: '{"url":"a"}{"url":"b"}',
    }),
    "{}",
  );
  assert.equal(
    toolCallReplayArguments(undefined, { _raw: "a legitimate value" }),
    '{"_raw":"a legitimate value"}',
  );
  assert.equal(
    toolCallReplayArguments('{"url":"a"}{"url":"b"}', { _raw: 42 }),
    '{"_raw":42}',
  );
});

test("a name waiting for arguments does not cross a turn boundary", () => {
  // The backend starts the next provider turn on the same response with the
  // delta index restarted at 0, so a name left over would be prepended to
  // whatever opens there.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "A", arguments: '{"a":1}' } }]);
  // The name-only delta rides the same chunk as finish_reason, which is how a
  // clear placed before the deltas let the name through.
  stream.feed([{ index: 0, function: { name: "B" } }], true);
  stream.feed([{ index: 0, function: { name: "C", arguments: '{"c":3}' } }]);

  assert.deepEqual(shape(stream.parts), [
    ["A", '{"a":1}'],
    ["C", '{"c":3}'],
  ]);
});

test("metadata on several name fragments is merged, not replaced", () => {
  // The name is accumulated across the fragments, so the metadata they carry
  // has to be too: replacing drops the signature an earlier fragment brought.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [
      {
        index: 0,
        function: { name: "web" },
        extra_content: { google: { thought_signature: "SIG" } },
      },
    ],
    [{ index: 0, function: { name: "_search" }, extra_content: { seq: 2 } }],
    [{ index: 0, function: { arguments: '{"q":1}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["web_search", '{"q":1}'],
  ]);
  assert.deepEqual(parts[1].extra_content, {
    google: { thought_signature: "SIG" },
    seq: 2,
  });
});

test("an opening name beats a parked resend, and keeps its own metadata", () => {
  // Concatenating gave "alphabeta", a name the backend never executes, so the
  // card disagreed with the call that ran. The metadata parked with that name
  // is the closed call's too, once the name is read as its.
  const parts = run([
    [
      {
        index: 0,
        function: { name: "alpha", arguments: '{"a":1}' },
        extra_content: { own: "A" },
      },
    ],
    [{ index: 0, function: { name: "alpha_long" }, extra_content: { resent: 1 } }],
    [{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
  assert.deepEqual(parts[0].extra_content, { own: "A", resent: 1 });
  assert.equal(parts[1].extra_content, undefined);
});

test("an id stamped after the object closed claims that call", () => {
  // A provider that opens id-less and stamps the real id on a later delta.
  // Reading the id as proof of another call left the finished one under its
  // provisional id beside an empty second card.
  for (const late of [
    { id: "call_a", index: 0 },
    { id: "call_a", index: 0, function: { name: "alpha" } },
  ]) {
    const parts = run([
      [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
      [late],
    ]);
    assert.deepEqual(
      parts.map((p) => [p.toolCallId, p.toolName, p.argsText]),
      [["call_a", "alpha", '{"a":1}']],
    );
  }

  // An id that names a different call is still the next call opening.
  const opened = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ id: "call_b", index: 0, function: { name: "beta", arguments: "" } }],
    [{ id: "call_b", index: 0, function: { arguments: '{"b":2}' } }],
  ]);
  assert.deepEqual(shape(opened), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
});

test("a new name arriving with whitespace is still held", () => {
  // The whitespace belongs to the object that just closed, but the name on that
  // delta may be the next call announced early. Merging it left the closed card
  // named "alphabeta" and the new one unnamed.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "beta", arguments: " " } }],
    [{ index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1} '],
    ["beta", '{"b":2}'],
  ]);
});

test("a call announced by name is placed where it was announced", () => {
  // The backend orders by when a call was announced, so a card appended where
  // its arguments turned up reads C before B while the backend runs B first.
  const parts = run([
    [{ index: 0, function: { name: "A", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "B" } }],
    [{ index: 1, function: { name: "C", arguments: '{"c":3}' } }],
    [{ index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["A", '{"a":1}'],
    ["B", '{"b":2}'],
    ["C", '{"c":3}'],
  ]);
});

test("a late id claims the last call a bundled delta opened", () => {
  // A provider that writes several calls in one delta can stamp the last one's
  // real id in a delta of its own. Marking that card owned sent the id to a
  // third empty card, while the backend put it on the split call.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}{"b":2}' } }],
    [{ id: "call_b", index: 0 }],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.toolName, p.argsText]),
    [
      ["tool_call_0", "alpha", '{"a":1}'],
      ["call_b", "alpha", '{"b":2}'],
    ],
  );
});

test("two calls announced at once keep the order they were announced in", () => {
  // Both announcements record the same position, so splicing each at that one
  // mark put the later one first.
  const parts = run([
    [{ index: 0, function: { name: "A", arguments: '{"a":1}' } }],
    [{ index: 1, function: { name: "B", arguments: '{"b":1}' } }],
    [{ index: 0, function: { name: "C" } }],
    [{ index: 1, function: { name: "D" } }],
    [{ index: 0, function: { arguments: '{"c":1}' } }],
    [{ index: 1, function: { arguments: '{"d":1}' } }],
  ]);

  assert.deepEqual(
    parts.map((p) => p.toolName),
    ["A", "B", "C", "D"],
  );
});

test("a rejected resend gives up its place too", () => {
  // The place goes with the announcement: a name read as the closed call's
  // resent announced nothing, so the call that opens takes its arrival
  // position, which is what the backend records for the same stream.
  const parts = run([
    [{ index: 0, function: { name: "A", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "A_long" } }],
    [{ index: 1, function: { name: "C", arguments: '{"c":1}' } }],
    [{ index: 0, function: { name: "B", arguments: '{"b":1}' } }],
  ]);

  assert.deepEqual(
    parts.map((p) => p.toolName),
    ["A", "C", "B"],
  );
});

test("a stable id naming a longer tool opens its own call", () => {
  // A catalog can hold both "web" and "web_search". Reading the second name as
  // a growth of the first gave the id to the completed card, and the arguments
  // that followed glued onto it.
  const parts = run([
    [{ index: 0, function: { name: "web", arguments: '{"a":1}' } }],
    [{ id: "call_b", index: 0, function: { name: "web_search", arguments: "" } }],
    [{ id: "call_b", index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.toolName, p.argsText]),
    [
      ["tool_call_0", "web", '{"a":1}'],
      ["call_b", "web_search", '{"b":2}'],
    ],
  );
});

test("a fork whose object never closed is dropped when the turn ends", () => {
  // A stream that stops after `{"a":1}{` is not marked truncated, so the lone
  // brace would persist as a card no execution event can complete and replay
  // would send as `{}`. The backend holds the same fork back in open_tail_keys.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "a", arguments: '{"a":1}{' } }]);
  assert.deepEqual(shape(stream.parts), [
    ["a", '{"a":1}'],
    ["a", "{"],
  ]);
  stream.feed([], true);
  assert.deepEqual(shape(stream.parts), [["a", '{"a":1}']]);
});

test("a fork that does close its object is kept", () => {
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "a", arguments: '{"a":1}{' } }]);
  stream.feed([{ index: 0, function: { arguments: '"b":2}' } }]);
  stream.feed([], true);
  assert.deepEqual(shape(stream.parts), [
    ["a", '{"a":1}'],
    ["a", '{"b":2}'],
  ]);
});

test("metadata from a resent name goes to the call that runs", () => {
  // "alpha_long" at a closed slot reads as "alpha" grown, so no second call is
  // invented for it and _announced_but_unopened hangs the signature on the held
  // call. Queued under the fragment it would never be read: the tool_start that
  // follows is named "alpha", and the card would replay without it.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
  stream.feed([
    { index: 0, function: { name: "alpha_long" }, extra_content: { sig: "s" } },
  ]);
  stream.feed([], true);

  assert.deepEqual(shape(stream.parts), [["alpha", '{"a":1}']]);
  assert.deepEqual(stream.parts[0].extra_content, { sig: "s" });
});

test("a provider-hosted tool event does not end the provider turn", () => {
  // The backend records those with note_hosted_tool_event and leaves its _Turn
  // open, so an argument-only delta after one still opens the parked name.
  // Hosted events ride a whole chunk, `choices` and all; Unsloth's own tool
  // frames arrive as a bare {"type": "tool_start"} that chat-api wraps alone.
  const guarded = liftBetween(
    "the hosted-event guard",
    "const toolEvent = (",
    "// Deep Research is an ordinary tool",
  );
  assert.match(guarded, /if \(!chunk\.choices\) \{\s*endProviderTurn\(\);/);
});

test("a late id does not rescue a fork whose object never closed", () => {
  // _call_is_finished holds the fork back on its arguments alone, so a card
  // kept because an id reached it is one no tool_start or tool_end can reach.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "a", arguments: '{}{"x":' } }]);
  stream.feed([{ id: "call_z", index: 0, function: { arguments: "" } }]);
  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.argsText]),
    [
      ["tool_call_0", "{}"],
      ["call_z", '{"x":'],
    ],
  );
  stream.feed([], true);
  assert.deepEqual(shape(stream.parts), [["a", "{}"]]);
});

test("only the announced call takes the place reserved for it", () => {
  // B was announced before C opened, so B goes ahead of C. The calls B's
  // arguments introduce were never announced: _fork_glued_arguments numbers
  // them as the arguments arrive, which is after C.
  const parts = run([
    [{ index: 0, function: { name: "A", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "B" } }],
    [{ index: 1, function: { name: "C", arguments: '{"c":1}' } }],
    [{ index: 0, function: { arguments: '{"b":1}{"b2":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["A", '{"a":1}'],
    ["B", '{"b":1}'],
    ["C", '{"c":1}'],
    ["B", '{"b2":2}'],
  ]);
});

test("the empty status between rounds ends the provider turn", () => {
  // A round whose calls were all rejected as disabled emits no tool_start or
  // tool_end, and an upstream ending its turns with [DONE] sends no
  // finish_reason, so the empty tool_status is the only boundary there is.
  const branch = liftBetween(
    "the tool_status branch",
    "const toolStatusText = (",
    "if (chunk.context_truncated) {",
  );
  assert.match(branch, /if \(!toolStatusText\) \{\s*endProviderTurn\(\);/);
  // And an announcement still queued at that point was never matched by a
  // card, because the backend emits none for a call it rejected as disabled.
  // Kept, the next call of the same name would replay a signature it never
  // carried.
  assert.ok(
    branch.indexOf("announcedExtraByName.clear();") >
      branch.indexOf("endProviderTurn();"),
    "the queue outlives the round that filled it",
  );
});

test("the announced call keeps its own metadata when its delta splits", () => {
  // The signature parked with the announcement is B's; the one riding the
  // arguments belongs to the call that delta closes, which is the last of
  // them. _take_parked and _fork_glued_arguments divide them the same way.
  const parts = run([
    [{ index: 0, function: { name: "A", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "B" }, extra_content: { sig: "parked" } }],
    [
      {
        index: 0,
        function: { arguments: '{"b":1}{"b2":2}' },
        extra_content: { sig: "incoming" },
      },
    ],
  ]);

  assert.deepEqual(shape(parts), [
    ["A", '{"a":1}'],
    ["B", '{"b":1}'],
    ["B", '{"b2":2}'],
  ]);
  assert.deepEqual(parts[1].extra_content, { sig: "parked" });
  assert.deepEqual(parts[2].extra_content, { sig: "incoming" });
});

test("a second call to the same tool keeps that tool's name", () => {
  // The provider gave the name once and reused the index for the next call
  // with arguments alone. A blank name is no tool: _normalized_call drops the
  // call on the backend and the card here would name nothing.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 0, function: { arguments: '{"a":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["alpha", '{"a":2}'],
  ]);
});

test("a snapshot repeated to carry the id claims the call", () => {
  // Snapshot-style servers resend the whole call rather than fragments of it,
  // so the id arrives on a verbatim repeat. Opening a second call there runs a
  // side-effecting tool twice.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ id: "call_a", index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.toolName, p.argsText]),
    [["call_a", "alpha", '{"a":1}']],
  );
});

test("a second call that differs anywhere still opens its own", () => {
  // The claim above is an exact repeat only, so genuine parallel calls with
  // ids of their own are not collapsed into one.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ id: "call_b", index: 0, function: { name: "alpha", arguments: '{"a":2}' } }],
  ]);

  assert.deepEqual(
    parts.map((p) => [p.toolCallId, p.argsText]),
    [
      ["tool_call_0", '{"a":1}'],
      ["call_b", '{"a":2}'],
    ],
  );
});
