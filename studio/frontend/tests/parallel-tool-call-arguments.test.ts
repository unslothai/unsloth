// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Issue #9807: a backend streaming id-less, index-based deltas reuses one slot
// for several calls, so the adapter glued their arguments into an unparsable
// `{"url":"a"}{"url":"b"}`. The boundary is the end of a top-level JSON object,
// not a change of name: the reported stream calls one tool three times.

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
import {
  bindStreamedToolCallCard,
  findStreamedToolCallPartIndex,
  mintStreamedToolCallId,
  resolveToolCallPartId,
} from "../src/features/chat/tool-call-id.ts";
import {
  discardAuthoritativeExecutionRecord,
  stripUntrustedExecutionMetadata,
} from "../src/features/chat/types/api.ts";

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
  // `_raw` is the adapter's marker for text it could not parse.
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

// chat-adapter.ts cannot be imported, so lift the loop as
// tests/pr9057-video-simulation.test.ts does: a re-implementation passes while
// the adapter stays broken, which is how this defect survived
// tool-call-delta-index.test.ts.
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
    "const reservedToolCallIds = new Set<string>();",
    "const toolPartIdByBackendId = new Map<string, string>();",
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

/**
 * The lifted loop, with the locals it closes over supplied by hand.
 *
 * `mintPartIds` picks which `resolveToolPartId` the loop closes over. The
 * accumulation tests want the identity, because they assert which call an id
 * lands on and the spelling is noise. The card tests want the shipped mint,
 * `<backend id>:<uuid>`, because whether the backend's id finds the card the
 * deltas drew is the whole question there, and under the identity every id
 * finds one whether or not the two halves ever agreed.
 */
function makeStream(mintPartIds = false): {
  feed: (batch: DeltaCall[], finished?: boolean) => boolean;
  parts: LoopPart[];
  resolveToolPartId: (backendId: string) => string;
  endRound: () => void;
} {
  const body = `
    const toolCallParts = [];
    let codexRoundToolCallIds = [];
    const toolPartIdByBackendId = new Map();
    const cumulativeText = "";
    let streamedChars = 0;
    ${liftSplitHelpers()}
    let mintedPartIds = 0;
    const resolveToolPartId = (backendId) =>
      resolveToolCallPartId(
        toolPartIdByBackendId,
        backendId,
        undefined,
        toolCallParts[toolCallParts.length - 1]?.toolCallId ?? "",
        () =>
          mintPartIds ? backendId + ":uuid-" + (mintedPartIds += 1) : backendId,
      );
    let addedToolCall = false;
    let replayStateChanged = false;
    // What the lifted loop reads finish_reason off.
    let chunk = { choices: [{}] };
    function feed(rawDeltaToolCalls, finished) {
      chunk = { choices: [finished ? { finish_reason: "tool_calls" } : {}] };
      addedToolCall = false;
      replayStateChanged = false;
      ${liftDeltaLoop()}
      return addedToolCall;
    }
    // Production drops a backend id's binding on tool_end, so an id the
    // provider reuses in a later round reaches a card of its own. That branch
    // is outside the lifted loop, so a test that spans rounds does it here.
    const endRound = () => {
      for (const part of toolCallParts) {
        toolPartIdByBackendId.delete(part.toolCallId);
      }
    };
    return { feed, parts: toolCallParts, resolveToolPartId, endRound };
  `;
  const js = ts.transpileModule(`const mintPartIds = ${mintPartIds};` + body, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return new Function(
    "splitTopLevelJsonObjects",
    "createBoundaryScan",
    "findStreamedToolCallPartIndex",
    "mintStreamedToolCallId",
    "bindStreamedToolCallCard",
    "resolveToolCallPartId",
    "discardAuthoritativeExecutionRecord",
    "stripUntrustedExecutionMetadata",
    js,
  )(
    splitTopLevelJsonObjects,
    createBoundaryScan,
    findStreamedToolCallPartIndex,
    mintStreamedToolCallId,
    bindStreamedToolCallCard,
    resolveToolCallPartId,
    discardAuthoritativeExecutionRecord,
    stripUntrustedExecutionMetadata,
  ) as {
    feed: (batch: DeltaCall[], finished?: boolean) => boolean;
    parts: LoopPart[];
    resolveToolPartId: (backendId: string) => string;
    endRound: () => void;
  };
}

function run(batches: DeltaCall[][]): LoopPart[] {
  const stream = makeStream();
  for (const batch of batches) stream.feed(batch);
  return stream.parts;
}

const shape = (parts: LoopPart[]) => parts.map((p) => [p.toolName, p.argsText]);

test("the stream from #9807 becomes one call per JSON object", () => {
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
  // Writing back through the old position deleted the neighbour.
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
  // Or the pacing gate holds it back and Stop persists without it.
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
  // llama-server grows the name across deltas, so opening a call here gives
  // two cards one id.
  const parts = run([
    [{ id: "call_a", index: 0, function: { name: "web", arguments: '{"q":"x"}' } }],
    [{ id: "call_a", index: 0, function: { name: "web_search" } }],
  ]);

  assert.deepEqual(shape(parts), [["web_search", '{"q":"x"}']]);
  assert.equal(parts.length, 1);
});

test("whitespace chunked after a closing brace is not a new call", () => {
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
  // The id has to land on the half-written third call; appending to either of
  // the closed ones is the gluing this change is about.
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
      ["tool_call_1", '{"b":2}'],
      ["call_c", '{"c":3}'],
    ],
  );
});

test("a name-only delta for the next call does not rename the finished one", () => {
  // The name arrives before its arguments, so there is nothing to split on.
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
  // A late id claims its slot only while that call is still being written.
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
      ["tool_call_1", '{"b":2}'],
      ["call_c", '{"c":3}'],
    ],
  );
});

test("an opening delta after a closed call does not claim it", () => {
  // The conventional opening delta: id and name, empty arguments. Landing it
  // on the finished card glues the arguments that follow onto that one.
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
  // OpenAI streams "web" then "_search"; llama-server resends "web" then
  // "web_search". Last-write-wins opens the call as "_search".
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
  // A repeated name with the trailing whitespace chunked separately is still
  // that call's; parking it merged the two into "alphabeta".
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
  // Only safe while it agrees with restarting, for every chunking.
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

test("one argument streamed a character at a time is scanned once", () => {
  // Rescanning per fragment made a 20 KB argument cost over a second on the
  // thread that paints the stream. Counted, not timed: a resumable scan
  // parses once per object however many deltas it arrived in, a restarting
  // one parses every closed object again on every delta. A wall-clock ratio
  // agreed, with too little margin for a loaded runner.
  const parses = (size: number, feed: (text: string) => unknown): number => {
    const real = JSON.parse;
    let calls = 0;
    (JSON as { parse: typeof JSON.parse }).parse = ((
      text: string,
      reviver?: unknown,
    ) => {
      calls += 1;
      return (real as (t: string, r?: unknown) => unknown)(text, reviver);
    }) as typeof JSON.parse;
    try {
      const payload = '{"a":1}{"code":"' + "x".repeat(size) + '"}';
      let text = "";
      for (const ch of payload) {
        text += ch;
        feed(text);
      }
      return calls;
    } finally {
      (JSON as { parse: typeof JSON.parse }).parse = real;
    }
  };

  for (const size of [500, 2000]) {
    const scan = createBoundaryScan();
    assert.equal(
      parses(size, (text) => scan.feed(text)),
      2,
      "the scan is parsing more than once per object it closes",
    );
  }

  // The shape it replaces, measured the same way, so the test says what it
  // guards against rather than only asserting a number.
  const restarting = parses(2000, splitTopLevelJsonObjects);
  assert.ok(
    restarting > 2000,
    `restarting from the beginning parsed ${restarting} times, so this test is no longer measuring the difference it was written for`,
  );

  // And the loop still reads it as one call, whatever the scan costs.
  const stream = makeStream();
  const payload = '{"code":"' + "x".repeat(400) + '"}';
  stream.feed([{ index: 0, function: { name: "write", arguments: "" } }]);
  for (const ch of payload) {
    stream.feed([{ index: 0, function: { arguments: ch } }]);
  }
  assert.deepEqual(shape(stream.parts), [["write", payload]]);
});

test("metadata arriving alone stays on the call that closed", () => {
  // No name, so nothing announces another call: the signature is this one's.
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
  // Indistinguishable from a second no-argument call to the same tool, so
  // take the reading that does not run a tool twice. A grown name announces
  // the next call, so its provisional card is reaped unfilled at the boundary.
  for (const resent of ["alpha", "alpha_long"]) {
    const stream = makeStream();
    stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
    stream.feed([{ index: 0, function: { name: resent } }]);
    stream.feed([], true);
    assert.deepEqual(shape(stream.parts), [["alpha", '{"a":1}']]);
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
  // A next call begins with its own "{"; forking on anything else made a stray
  // scalar suffix run the tool twice.
  const parts = run([
    [{ index: 0, function: { name: "q", arguments: '{"query":"a"}' } }],
    [{ index: 0, function: { name: "q", arguments: '"b"' } }],
  ]);

  assert.deepEqual(shape(parts), [["q", '{"query":"a"}"b"']]);
});

test("metadata announced with a name is merged, not replaced", () => {
  // Two fields of one call, and replay needs both.
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
  // The marker is the pairing, not the key: `_raw` is not reserved.
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
  // The next turn restarts the delta index at 0.
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
  // The name accumulates across fragments, so the metadata has to as well.
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

test("a resent name does not rename the call it closed", () => {
  // Concatenating gave "alphabeta", which the backend never executes. Once the
  // name is read as the closed call's, so is the metadata beside it.
  const stream = makeStream();
  stream.feed([
    {
      index: 0,
      function: { name: "alpha", arguments: '{"a":1}' },
      extra_content: { own: "A" },
    },
  ]);
  stream.feed([
    { index: 0, function: { name: "alpha_long" }, extra_content: { resent: 1 } },
  ]);
  stream.feed([{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }]);
  stream.feed([], true);
  const parts = stream.parts;

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", '{"b":2}'],
  ]);
  assert.deepEqual(parts[0].extra_content, { own: "A", resent: 1 });
  assert.equal(parts[1].extra_content, undefined);
});

test("an id stamped after the object closed claims that call", () => {
  // Reading the late id as another call left the finished one provisional.
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

test("a new name arriving with whitespace opens its own call", () => {
  // A different name opens the next call rather than renaming the finished
  // one, which gave "alphabeta" and an unnamed second card. The whitespace
  // rides the announcing delta and is valid JSON, so both still parse.
  const parts = run([
    [{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }],
    [{ index: 0, function: { name: "beta", arguments: " " } }],
    [{ index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["beta", ' {"b":2}'],
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
  // A provider bundling several calls can stamp the last one's real id later;
  // marking that card owned sent the id to a third, empty card.
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
  // Both record the same position, so splicing at it put the later first.
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
  // The place goes with the announcement, and a name read as a resent
  // announced nothing, so the call that opens takes its own arrival.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "A", arguments: '{"a":1}' } }]);
  stream.feed([{ index: 0, function: { name: "A_long" } }]);
  stream.feed([{ index: 1, function: { name: "C", arguments: '{"c":1}' } }]);
  stream.feed([{ index: 0, function: { name: "B", arguments: '{"b":1}' } }]);
  stream.feed([], true);
  const parts = stream.parts;

  assert.deepEqual(
    parts.map((p) => p.toolName),
    ["A", "C", "B"],
  );
});

test("a catalog holding both web and web_search splits either way round", () => {
  // Both are in Studio's own catalog, so a shared prefix is no evidence either
  // way; reading it as evidence swallowed the second announcement.
  for (const [first, second] of [
    ["web_search", "web"],
    ["web", "web_search"],
  ]) {
    const stream = makeStream();
    stream.feed([{ index: 0, function: { name: first, arguments: '{"a":1}' } }]);
    stream.feed([{ index: 0, function: { name: second } }]);
    stream.feed([{ index: 0, function: { arguments: '{"b":2}' } }]);
    stream.feed([], true);
    assert.deepEqual(shape(stream.parts), [
      [first, '{"a":1}'],
      [second, '{"b":2}'],
    ]);
  }
});

test("a name bringing an object over an announcement is the next call", () => {
  // An announcement has no object to close, so the next call's name grew into
  // it: "alpha_longbeta" and "zetabeta" match no tool.
  for (const announced of ["alpha_long", "zeta"]) {
    const stream = makeStream();
    stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
    stream.feed([{ index: 0, function: { name: announced } }]);
    stream.feed([{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }]);
    stream.feed([], true);
    assert.deepEqual(shape(stream.parts), [
      ["alpha", '{"a":1}'],
      ["beta", '{"b":2}'],
    ]);
  }
});

test("a dropped card gives its id back to the next round", () => {
  // The backend never reserves the filtered fork's card id, so holding it here
  // made the next round mint tool_call_2 against the backend's tool_call_1.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}{' } }]);
  stream.feed([], true);
  stream.feed([{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }]);

  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName]),
    [
      ["tool_call_0", "alpha"],
      ["tool_call_1", "beta"],
    ],
  );
});

test("a provider claiming a minted id displaces the card holding it", () => {
  // tool_call_<n> is not reserved to Unsloth. Resolving the claim through the
  // id-less call's binding merged the two into one card, losing a call. The
  // backend reserves provider ids before minting; do the same on the claim.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
  stream.feed([
    { id: "tool_call_0", index: 1, function: { name: "beta", arguments: '{"b":2}' } },
  ]);

  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName, p.argsText]),
    [
      ["tool_call_1", "alpha", '{"a":1}'],
      ["tool_call_0", "beta", '{"b":2}'],
    ],
  );
});

test("a card taking a late provider id gives its minted id back", () => {
  // The backend never reserves a minted id for a call the provider went on to
  // name, so holding it made the next call at that index mint tool_call_1
  // against the backend's tool_call_0.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
  stream.feed([{ id: "call_a", index: 0, function: { arguments: "" } }]);
  stream.feed([{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }]);

  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName]),
    [
      ["call_a", "alpha"],
      ["tool_call_0", "beta"],
    ],
  );
});

test("a claim on a split-born card renumbers every minted card", () => {
  // A split marks every born card but the last _has_stable_id, so reading
  // that as provider-owned let the claim merge them: "alphabeta", arguments
  // glued. The backend reserves the claim then numbers the id-less calls in
  // order, so matching it means renumbering all of them.
  const stream = makeStream();
  stream.feed([
    { index: 0, function: { name: "alpha", arguments: '{"a":1}{"b":2}{"c":3}' } },
  ]);
  stream.feed([
    { id: "tool_call_1", index: 1, function: { name: "beta", arguments: '{"d":4}' } },
  ]);

  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName, p.argsText]),
    [
      ["tool_call_0", "alpha", '{"a":1}'],
      ["tool_call_2", "alpha", '{"b":2}'],
      ["tool_call_3", "alpha", '{"c":3}'],
      ["tool_call_1", "beta", '{"d":4}'],
    ],
  );
});

test("a born call carries only the metadata of the delta that opened it", () => {
  // The merged fields belong to the call the slot was holding; carried onto a
  // born call they put one call's signature on another, and Gemini validates a
  // signature against the functionCall part it was returned on.
  const stream = makeStream();
  stream.feed([
    { index: 0, function: { name: "alpha" }, extra_content: { parked: 1 } },
  ]);
  stream.feed([
    {
      index: 0,
      function: { arguments: '{"a":1}{"b":2}' },
      extra_content: { delta: 2 },
    },
  ]);

  assert.deepEqual(shape(stream.parts), [
    ["alpha", '{"a":1}'],
    ["alpha", '{"b":2}'],
  ]);
  assert.deepEqual(stream.parts[0].extra_content, { parked: 1 });
  assert.deepEqual(stream.parts[1].extra_content, { delta: 2 });
});

test("a claim in a later round leaves an earlier round's card alone", () => {
  // The backend's card ledger is append-only, so a card from a finished round
  // keeps its number however a later round spells its ids. Renumbering it put
  // the two sides one apart from the third round on, and the backend's events
  // for the renamed card reached whatever had taken its old id.
  const stream = makeStream(true);
  stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
  stream.feed([], true);
  stream.endRound();
  stream.feed([
    { id: "tool_call_0", index: 0, function: { name: "beta", arguments: '{"b":2}' } },
  ]);
  stream.feed([], true);
  stream.endRound();
  stream.feed([{ index: 0, function: { name: "gamma", arguments: '{"c":3}' } }]);

  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName]),
    [
      ["tool_call_0", "alpha"],
      ["tool_call_0:uuid-1", "beta"],
      ["tool_call_1", "gamma"],
    ],
  );
});

test("a dropped card gives back the provider id that aliased it", () => {
  // A card that took a late id answers to a run-unique part id, so the id the
  // provider sent is a second key pointing at it. Releasing only the part id
  // left tool_call_1 reserved, and the next round minted tool_call_2 where
  // the backend minted tool_call_1.
  const stream = makeStream(true);
  stream.feed([{ index: 0, function: { name: "alpha", arguments: '{}{"x":' } }]);
  stream.feed([{ id: "tool_call_1", index: 0, function: { arguments: "" } }]);
  stream.feed([], true);
  stream.endRound();
  stream.feed([{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }]);

  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName]),
    [
      ["tool_call_0", "alpha"],
      ["tool_call_1", "beta"],
    ],
  );
});

test("a card that never got a name is dropped when the turn ends", () => {
  // _normalized_call rejects a nameless call before it reserves a card id, so
  // a card kept for one holds a number the backend gives the next round, and
  // that round's events land on the blank card instead.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { arguments: '{"a":1}' } }]);
  assert.equal(stream.parts.length, 1);
  stream.feed([], true);
  assert.equal(stream.parts.length, 0);

  stream.feed([{ index: 0, function: { name: "beta", arguments: '{"b":2}' } }]);
  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName]),
    [["tool_call_0", "beta"]],
  );
});

test("a claim that turns out not to be a call gives the number back", () => {
  // The displacement happens as the claim lands, but a nameless call is not a
  // call: the backend reserves nothing for it and keeps tool_call_0 for the
  // valid one, so a card left at tool_call_1 is one its execution events miss.
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
  stream.feed([{ id: "tool_call_0", index: 1, function: { arguments: '{"b":2}' } }]);
  assert.equal(stream.parts.length, 2);
  stream.feed([], true);

  assert.deepEqual(
    stream.parts.map((p) => [p.toolCallId, p.toolName]),
    [["tool_call_0", "alpha"]],
  );
});

test("a repeated name's metadata waits for the call it announced", () => {
  // The same tool twice on one slot, the second announced by a name-only
  // delta carrying its own signature. Merging it where it landed overwrote
  // the closed call's and left the new one unsigned, and Gemini validates a
  // signature against the call it is replayed on.
  const stream = makeStream();
  stream.feed([
    {
      index: 0,
      function: { name: "lookup", arguments: '{"q":"a"}' },
      extra_content: { sig: "A" },
    },
  ]);
  stream.feed([
    { index: 0, function: { name: "lookup" }, extra_content: { sig: "B" } },
  ]);
  stream.feed([{ index: 0, function: { arguments: '{"q":"b"}' } }]);
  stream.feed([], true);

  assert.deepEqual(shape(stream.parts), [
    ["lookup", '{"q":"a"}'],
    ["lookup", '{"q":"b"}'],
  ]);
  assert.deepEqual(stream.parts[0].extra_content, { sig: "A" });
  assert.deepEqual(stream.parts[1].extra_content, { sig: "B" });
});

test("a repeated name that announced nothing keeps its metadata", () => {
  // No object followed, so the repeated name really was that call's resent and
  // the signature riding it is that call's too.
  const stream = makeStream();
  stream.feed([
    {
      index: 0,
      function: { name: "lookup", arguments: '{"q":"a"}' },
      extra_content: { own: 1 },
    },
  ]);
  stream.feed([
    { index: 0, function: { name: "lookup" }, extra_content: { sig: "B" } },
  ]);
  stream.feed([], true);

  assert.deepEqual(shape(stream.parts), [["lookup", '{"q":"a"}']]);
  assert.deepEqual(stream.parts[0].extra_content, { own: 1, sig: "B" });
});

test("parked metadata follows the card a late id renames", () => {
  // The signature waits under the id the card was minted with, so it has to
  // move with the card or the turn-end sweep drops what the backend keeps.
  const stream = makeStream(true);
  stream.feed([
    {
      index: 0,
      function: { name: "lookup", arguments: '{"q":"a"}' },
      extra_content: { sig: "A" },
    },
  ]);
  stream.feed([
    { index: 0, function: { name: "lookup" }, extra_content: { sig: "B" } },
  ]);
  stream.feed([{ index: 0, id: "call_x", function: { arguments: "" } }], true);

  assert.deepEqual(shape(stream.parts), [["lookup", '{"q":"a"}']]);
  assert.deepEqual(stream.parts[0].extra_content, { sig: "B" });
  assert.equal(stream.parts[0].toolCallId, "call_x:uuid-1");
});

test("parked metadata follows the card a claim renumbers", () => {
  // Same entry, other mover: a provider claiming the spelling a minted card
  // holds renumbers every minted card in the round.
  const stream = makeStream(true);
  stream.feed([
    {
      index: 0,
      function: { name: "lookup", arguments: '{"q":"a"}' },
      extra_content: { sig: "A" },
    },
  ]);
  stream.feed([
    { index: 0, function: { name: "lookup" }, extra_content: { sig: "B" } },
  ]);
  stream.feed(
    [{ index: 1, id: "tool_call_0", function: { name: "beta", arguments: '{"b":2}' } }],
    true,
  );

  assert.deepEqual(shape(stream.parts), [
    ["lookup", '{"q":"a"}'],
    ["beta", '{"b":2}'],
  ]);
  assert.deepEqual(stream.parts[0].extra_content, { sig: "B" });
  assert.equal(stream.parts[1].extra_content, undefined);
});

test("a stable id naming a longer tool opens its own call", () => {
  // Reading "web_search" as "web" grown gave the id to the completed card.
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
  // A stream stopping after `{"a":1}{` is not marked truncated, so the lone
  // brace persists as a card nothing completes. The backend holds it too.
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
  // No call is invented for a name read as a resent, so its signature has
  // nowhere else to go and the card would replay without it.
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
  // Hosted events ride a whole chunk, `choices` and all, and leave the turn
  // open; Unsloth's are bare {"type": "tool_start"}.
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
  // B was announced before C opened. The calls B's arguments introduce were
  // never announced, so they are numbered as the arguments arrive.
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
  // A round of only disabled calls emits no tool_start, and a [DONE] upstream
  // sends no finish_reason, so this is the only boundary there is.
  const branch = liftBetween(
    "the tool_status branch",
    "const toolStatusText = (",
    "if (chunk.context_truncated) {",
  );
  assert.match(branch, /if \(!toolStatusText\) \{\s*endProviderTurn\(\);/);
});

test("the announced call keeps its own metadata when its delta splits", () => {
  // The parked signature is B's; the one riding the arguments belongs to the
  // call that delta closes, the last. The backend divides them the same way.
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
  // The index is reused with arguments alone, and a blank name is no tool:
  // the backend drops the call and the card here names nothing.
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
  // Snapshot servers resend the whole call, so the id arrives on a verbatim
  // repeat and opening a second call runs the tool twice.
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
  // The claim above is exact repeats only.
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

test("an id-less card answers to the id the backend mints for it", () => {
  // The backend addresses tool_start at tool_call_<n>. Without the binding,
  // resolveToolPartId mints "<id>:<uuid>" and no card is found: #9807's four
  // calls ended the turn holding eight cards.
  const stream = makeStream(true);
  for (const url of ["a", "b", "c", "d"]) {
    stream.feed([{ index: 0, function: { name: "fetch", arguments: `{"url":"${url}"}` } }]);
  }

  assert.deepEqual(
    stream.parts.map((p) => p.toolCallId),
    ["tool_call_0", "tool_call_1", "tool_call_2", "tool_call_3"],
  );

  const painted = stream.parts.length;
  for (const backendId of ["tool_call_0", "tool_call_1", "tool_call_2", "tool_call_3"]) {
    const partId = stream.resolveToolPartId(backendId);
    assert.equal(partId, backendId, `${backendId} did not resolve to its own card`);
    assert.ok(
      stream.parts.some((p) => p.toolCallId === partId),
      `${backendId} found no card to update`,
    );
  }
  assert.equal(stream.parts.length, painted, "a backend event opened a second card");
});

test("a call the provider named still resolves through the minted part id", () => {
  // Unchanged for streams that carry ids: still keyed on "<id>:<uuid>".
  const stream = makeStream(true);
  stream.feed([{ id: "call_a", index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);

  const partId = stream.resolveToolPartId("call_a");
  assert.match(partId, /^call_a:uuid-\d+$/);
  assert.deepEqual(
    stream.parts.map((p) => p.toolCallId),
    [partId],
  );
});

test("a provider id spelled tool_call_0 keeps its own card", () => {
  // tool_call_<n> is not reserved to Unsloth. A provider using that spelling
  // must not have the card taken from it by the id-less call beside it.
  const stream = makeStream(true);
  stream.feed([
    { id: "tool_call_0", index: 0, function: { name: "alpha", arguments: '{"a":1}' } },
  ]);
  stream.feed([{ index: 1, function: { name: "beta", arguments: '{"b":2}' } }]);

  const ids = stream.parts.map((p) => p.toolCallId);
  assert.equal(new Set(ids).size, ids.length, "two cards share one id");
  assert.ok(!ids.includes("tool_call_0"), "the minted id took the provider's spelling");
});

test("several calls opened by one delta each get their own card id", () => {
  // The cards are minted before any joins the parts array, so the ids have to
  // be reserved as they are handed out or all three collide.
  const stream = makeStream(true);
  stream.feed([
    { index: 0, function: { name: "fetch", arguments: '{"a":1}{"b":2}{"c":3}' } },
  ]);

  const ids = stream.parts.map((p) => p.toolCallId);
  assert.deepEqual(ids, ["tool_call_0", "tool_call_1", "tool_call_2"]);
  assert.equal(new Set(ids).size, 3);
});

test("the marker is only recognised by the text that proves it", () => {
  // Threads written before argsText was kept carry { _raw } with nothing to
  // compare against. Reading the value's shape instead would discard a real
  // _raw argument and gain nothing: the wrapped form raises no Extra data.
  const glued = '{"url":"https://example.com/1"}{"query":"search"}';
  assert.equal(toolCallReplayArguments(glued, { _raw: glued }), "{}");
  assert.equal(
    toolCallReplayArguments(undefined, { _raw: glued }),
    JSON.stringify({ _raw: glued }),
  );
  assert.equal(
    toolCallReplayArguments("", { _raw: glued }),
    JSON.stringify({ _raw: glued }),
  );
});

test("an empty _raw is an argument, not the marker", () => {
  // Both writers are guarded on non-empty text, so `{ _raw: "" }` beside an
  // empty argsText is a real argument. Equality alone read it as the marker.
  assert.equal(toolCallReplayArguments("", { _raw: "" }), '{"_raw":""}');
  assert.equal(toolCallReplayArguments(undefined, { _raw: "" }), '{"_raw":""}');
});

test("a tool that really takes a _raw parameter keeps it either way", () => {
  // _raw is not reserved and an MCP server's schema is its own.
  assert.equal(
    toolCallReplayArguments('{"_raw":"hello"}', { _raw: "hello" }),
    '{"_raw":"hello"}',
  );
  assert.equal(
    toolCallReplayArguments(undefined, { _raw: "hello" }),
    '{"_raw":"hello"}',
  );
  assert.equal(
    toolCallReplayArguments(undefined, { _raw: '{"one":1}' }),
    '{"_raw":"{\\"one\\":1}"}',
  );
});
