// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Id-less index-based deltas reuse one slot; the call boundary is the end of a top-level
// JSON object, not a name change.

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import {
  createBoundaryScan,
  splitTopLevelJsonObjects,
  streamedToolCallArguments,
  toolCallReplayArguments,
} from "../src/features/chat/tool-call-arguments.ts";
import {
  bindStreamedToolCallCard,
  findStreamedToolCallPartIndex,
  mintStreamedToolCallId,
  resolveToolCallPartId,
} from "../src/features/chat/tool-call-id.ts";

import { readSrc } from "./helpers/kit.ts";

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

// chat-adapter.ts cannot be imported, so lift the real loop: a re-implementation would pass
// while the adapter stays broken.
const adapterSource = readSrc("features/chat/api/chat-adapter.ts");

function liftBetween(what: string, from: string, to: string): string {
  const start = adapterSource.indexOf(from);
  assert.ok(start >= 0, `${what}: "${from}" is gone from chat-adapter.ts`);
  const end = adapterSource.indexOf(to, start);
  assert.ok(end > start, `${what}: "${to}" is gone from chat-adapter.ts`);
  return adapterSource.slice(start, end);
}

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
  const gate = adapterSource.indexOf(
    "if (forcePublish || canPublish(",
    loopStart,
  );
  assert.ok(gate > loopStart, "the publish gate moved in chat-adapter.ts");
  const lifted = adapterSource.slice(loopStart, gate);
  assert.ok(
    lifted.includes("splitTopLevelJsonObjects"),
    "the loop no longer splits on JSON object boundaries",
  );
  assert.ok(
    lifted.includes("endProviderTurn()"),
    "the lifted loop stops before the turn ends, so it is not the loop production runs",
  );
  assert.ok(
    lifted.includes("const forcePublish ="),
    "the lifted loop stops before the publish decision it is supposed to reach",
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
 * `mintPartIds`: accumulation tests use identity ids; card tests use the shipped
 * `<backend id>:<uuid>` mint, under which mismatched halves actually fail to match.
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
    "streamedToolCallArguments",
    "findStreamedToolCallPartIndex",
    "mintStreamedToolCallId",
    "bindStreamedToolCallCard",
    "resolveToolCallPartId",
    js,
  )(
    splitTopLevelJsonObjects,
    createBoundaryScan,
    streamedToolCallArguments,
    findStreamedToolCallPartIndex,
    mintStreamedToolCallId,
    bindStreamedToolCallCard,
    resolveToolCallPartId,
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
  // Gemini stows the signature for the call being announced, i.e. the next one.
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
  // Counted, not timed: wall-clock ratios are too noisy on a loaded runner.
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

  const restarting = parses(2000, splitTopLevelJsonObjects);
  assert.ok(
    restarting > 2000,
    `restarting from the beginning parsed ${restarting} times, so this test is no longer measuring the difference it was written for`,
  );

  const stream = makeStream();
  const payload = '{"code":"' + "x".repeat(400) + '"}';
  stream.feed([{ index: 0, function: { name: "write", arguments: "" } }]);
  for (const ch of payload) {
    stream.feed([{ index: 0, function: { arguments: ch } }]);
  }
  assert.deepEqual(shape(stream.parts), [["write", payload]]);
});

test("metadata arriving alone stays on the call that closed", () => {
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
  // Ambiguous with a second no-arg call; prefer the reading that does not run a tool twice.
  for (const resent of ["alpha", "alpha_long"]) {
    const stream = makeStream();
    stream.feed([{ index: 0, function: { name: "alpha", arguments: '{"a":1}' } }]);
    stream.feed([{ index: 0, function: { name: resent } }]);
    stream.feed([], true);
    assert.deepEqual(shape(stream.parts), [["alpha", '{"a":1}']]);
  }
});

test("an argument fragment that is not a string does not abort the stream", () => {
  // llama-server has shipped `arguments` as a decoded object; one object is a whole document.
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
    [{ index: 0, function: { arguments: '{"b":2}' } }],
  ]);

  assert.deepEqual(shape(parts), [
    ["alpha", '{"a":1}'],
    ["alpha", '{"b":2}'],
  ]);
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
  const stream = makeStream();
  stream.feed([{ index: 0, function: { name: "A", arguments: '{"a":1}' } }]);
  stream.feed([{ index: 0, function: { name: "B" } }], true);
  stream.feed([{ index: 0, function: { name: "C", arguments: '{"c":3}' } }]);

  assert.deepEqual(shape(stream.parts), [
    ["A", '{"a":1}'],
    ["C", '{"c":3}'],
  ]);
});

test("metadata on several name fragments is merged, not replaced", () => {
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
  // The backend orders by announcement, not by where arguments turned up.
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
  // The backend never reserves the filtered fork's card id.
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
  // tool_call_<n> is not Unsloth-reserved; the backend reserves provider ids before minting.
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
  // The backend never reserves a minted id for a call the provider later named.
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
  // The backend reserves the claim, then numbers id-less calls in order.
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
  // Gemini validates a signature against the functionCall part it was returned on.
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
  // The backend card ledger is append-only, so a finished round's card keeps its number.
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
  // The provider id is a second key for a late-id card, so release both.
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
  // _normalized_call rejects a nameless call before reserving a card id.
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
  // Gemini validates a signature against the call it is replayed on.
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
  // The signature is keyed by the minted id, so it must move with the card.
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
  // A stream stopping after `{"a":1}{` is not marked truncated; the backend keeps it too.
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
  // Hosted events ride whole chunks; Unsloth tool events and skill preloads are bare.
  const guarded = liftBetween(
    "the hosted-event guard",
    "const toolEvent = (",
    "// Deep Research is an ordinary tool",
  );
  assert.match(
    guarded,
    /if \(!chunk\.choices && toolEvent\.tool_name !== "studio_load_skill"\) \{\s*endProviderTurn\(\);/,
  );
  const js = ts.transpileModule(`${guarded}\n}`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const applyGuard = new Function("chunk", "endProviderTurn", js);
  for (const [chunk, expectedEnds] of [
    [{ _toolEvent: { tool_name: "web_search" } }, 1],
    [{ choices: [{}], _toolEvent: { tool_name: "web_search" } }, 0],
    [{ _toolEvent: { tool_name: "studio_load_skill" } }, 0],
    [{ choices: [{}], _toolEvent: { tool_name: "studio_load_skill" } }, 0],
    [{ choices: [{}] }, 0],
  ] as const) {
    let ends = 0;
    applyGuard(chunk, () => { ends += 1; });
    assert.equal(ends, expectedEnds, JSON.stringify(chunk));
  }
});

test("a late id does not rescue a fork whose object never closed", () => {
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
  // Disabled-only rounds emit no tool_start and [DONE] sends no finish_reason.
  const branch = liftBetween(
    "the tool_status branch",
    "const toolStatusText = (",
    "if (chunk.context_truncated) {",
  );
  assert.match(branch, /if \(!toolStatusText\) \{\s*endProviderTurn\(\);/);
});

test("the announced call keeps its own metadata when its delta splits", () => {
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
  // Snapshot servers resend the whole call, so the id may arrive on a verbatim repeat.
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
  // The backend addresses tool_start at tool_call_<n>; without the binding no card matches.
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
  const stream = makeStream(true);
  stream.feed([
    { index: 0, function: { name: "fetch", arguments: '{"a":1}{"b":2}{"c":3}' } },
  ]);

  const ids = stream.parts.map((p) => p.toolCallId);
  assert.deepEqual(ids, ["tool_call_0", "tool_call_1", "tool_call_2"]);
  assert.equal(new Set(ids).size, 3);
});

test("the marker is only recognised by the text that proves it", () => {
  // Older threads carry { _raw } with no argsText to compare against.
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

test("decoded object arguments preserve their JSON payload", () => {
  assert.equal(
    streamedToolCallArguments({ query: "雪", nested: [1, { ok: true }] }),
    '{"query":"雪","nested":[1,{"ok":true}]}',
  );
});

test("string fragments pass through byte-exact and junk contributes nothing", () => {
  assert.equal(streamedToolCallArguments('{"partial'), '{"partial');
  assert.equal(streamedToolCallArguments(""), "");
  assert.equal(streamedToolCallArguments(undefined), "");
  assert.equal(streamedToolCallArguments(null), "");
  assert.equal(streamedToolCallArguments(7), "");
});

test("the adapter's delta site reads arguments through the helper", () => {
  assert.ok(
    adapterSource.includes(
      "const deltaArgs = streamedToolCallArguments(",
    ),
    "chat-adapter.ts no longer routes delta arguments through streamedToolCallArguments",
  );
});

test("a delta whose arguments arrive as a decoded object keeps its payload", () => {
  const parts = run([
    [
      {
        id: "call-obj",
        index: 0,
        function: {
          name: "web_search",
          arguments: { query: "value" } as unknown as string,
        },
      },
    ],
  ]);
  assert.deepEqual(
    parts.map((part) => part.argsText),
    ['{"query":"value"}'],
  );
});

test("a decoded object lands exactly where its string spelling would", () => {
  // Object and string arguments must get identical answers downstream; pinned as a pair.
  const asObject = (a: unknown) => a as unknown as string;
  for (const [label, objectStream, stringStream] of [
    [
      "one payload",
      [[{ id: "c1", index: 0, function: { name: "s", arguments: asObject({ query: "v" }) } }]],
      [[{ id: "c1", index: 0, function: { name: "s", arguments: '{"query":"v"}' } }]],
    ],
    [
      "an empty opening, then the rest as fragments",
      [
        [{ id: "c1", index: 0, function: { name: "s", arguments: asObject({}) } }],
        [{ index: 0, function: { arguments: '{"query":' } }],
        [{ index: 0, function: { arguments: '"v"}' } }],
      ],
      [
        [{ id: "c1", index: 0, function: { name: "s", arguments: "{}" } }],
        [{ index: 0, function: { arguments: '{"query":' } }],
        [{ index: 0, function: { arguments: '"v"}' } }],
      ],
    ],
    [
      "two snapshots under one id",
      [
        [{ id: "c1", index: 0, function: { name: "s", arguments: asObject({ query: "a" }) } }],
        [{ id: "c1", index: 0, function: { name: "s", arguments: asObject({ query: "ab" }) } }],
      ],
      [
        [{ id: "c1", index: 0, function: { name: "s", arguments: '{"query":"a"}' } }],
        [{ id: "c1", index: 0, function: { name: "s", arguments: '{"query":"ab"}' } }],
      ],
    ],
  ] as const) {
    const shapeOf = (deltas: unknown) =>
      run(deltas as never).map((part) => [part.toolName, part.argsText]);
    assert.deepEqual(shapeOf(objectStream), shapeOf(stringStream), label);
  }
});
