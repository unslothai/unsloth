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
}

interface LoopPart {
  toolCallId: string;
  toolName?: string;
  argsText?: string;
  args?: Record<string, unknown>;
  _delta_index?: number;
  _has_stable_id?: boolean;
}

/** The lifted loop, with the locals it closes over supplied by hand. */
function makeStream(): {
  feed: (batch: DeltaCall[]) => boolean;
  parts: LoopPart[];
} {
  const body = `
    const toolCallParts = [];
    const codexRoundToolCallIds = [];
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
    function feed(rawDeltaToolCalls) {
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
    "findStreamedToolCallPartIndex",
    js,
  )(splitTopLevelJsonObjects, findStreamedToolCallPartIndex) as {
    feed: (batch: DeltaCall[]) => boolean;
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
