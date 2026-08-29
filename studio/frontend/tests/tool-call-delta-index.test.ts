// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { toolCallReplayArguments } from "../src/features/chat/tool-call-arguments.ts";
import {
  type StreamedToolCallPart,
  findStreamedToolCallPartIndex,
  fragmentStartsNewToolCall,
  mintStreamedToolCallId,
  splitTopLevelJsonDocuments,
} from "../src/features/chat/tool-call-id.ts";

interface DeltaFragment {
  id?: string;
  index?: number;
  arguments: string;
}

/** Accumulate `delta.tool_calls[]` fragments the way the chat adapter does. */
function accumulate(
  fragments: DeltaFragment[],
): (StreamedToolCallPart & { argsText: string })[] {
  const parts: (StreamedToolCallPart & { argsText: string })[] = [];
  for (const fragment of fragments) {
    const matched = findStreamedToolCallPartIndex(
      parts,
      fragment.id,
      fragment.index,
    );
    // Mirrors the adapter: an id-less opening landing on a slot whose
    // arguments are already complete is the next parallel call (#9807).
    const target =
      matched !== -1 &&
      !fragment.id &&
      fragmentStartsNewToolCall(parts[matched].argsText, fragment.arguments)
        ? -1
        : matched;
    if (target === -1) {
      parts.push({
        toolCallId:
          fragment.id ?? mintStreamedToolCallId(parts, fragment.index),
        argsText: fragment.arguments,
        ...(fragment.id ? { _has_stable_id: true } : {}),
        ...(fragment.index !== undefined
          ? { _delta_index: fragment.index }
          : {}),
      });
      continue;
    }
    parts[target] = {
      ...parts[target],
      ...(fragment.id ? { toolCallId: fragment.id, _has_stable_id: true } : {}),
      argsText: parts[target].argsText + fragment.arguments,
    };
  }
  return parts;
}

test("tool rounds that both reuse index 0 stay separate calls", () => {
  const parts = accumulate([
    { id: "call-A", index: 0, arguments: '{"query":' },
    { index: 0, arguments: '"first"}' },
    { id: "call-B", index: 0, arguments: '{"query":' },
    { index: 0, arguments: '"second"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-A", '{"query":"first"}'],
      ["call-B", '{"query":"second"}'],
    ],
  );
});

test("a fragment with a fresh id never merges into a slot another id owns", () => {
  assert.equal(
    findStreamedToolCallPartIndex(
      [{ toolCallId: "call-A", _delta_index: 0, _has_stable_id: true }],
      "call-B",
      0,
    ),
    -1,
  );
});

test("an id stamped on a later fragment adopts its own opening slot", () => {
  const parts = accumulate([
    { index: 0, arguments: "" },
    { id: "call-A", index: 0, arguments: '{"query":' },
    { index: 0, arguments: '"first"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [["call-A", '{"query":"first"}']],
  );
});

test("a late id still keeps the next round's call separate", () => {
  const parts = accumulate([
    { index: 0, arguments: "" },
    { id: "call-A", index: 0, arguments: '{"query":"first"}' },
    { id: "call-B", index: 0, arguments: '{"query":"second"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-A", '{"query":"first"}'],
      ["call-B", '{"query":"second"}'],
    ],
  );
});

test("parallel calls in one round accumulate per index", () => {
  const parts = accumulate([
    { id: "call-A", index: 0, arguments: '{"a":' },
    { id: "call-B", index: 1, arguments: '{"b":' },
    { index: 0, arguments: "1}" },
    { index: 1, arguments: "2}" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["call-A", '{"a":1}'],
      ["call-B", '{"b":2}'],
    ],
  );
});

test("an id-less stream still accumulates by index alone", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"q":' },
    { index: 0, arguments: '"x"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [["tool_call_0", '{"q":"x"}']],
  );
});

test("a fragment carrying neither id nor index starts its own call", () => {
  assert.equal(
    findStreamedToolCallPartIndex(
      [{ toolCallId: "call-A", _delta_index: 0 }],
      undefined,
      undefined,
    ),
    -1,
  );
});

test("replayed arguments keep parsable text and fall back otherwise", () => {
  assert.equal(
    toolCallReplayArguments('{"query":"first"}', { query: "first" }),
    '{"query":"first"}',
  );
  assert.equal(
    toolCallReplayArguments('{"query":"first"}{"query":"second"}', {
      _raw: '{"query":"first"}{"query":"second"}',
    }),
    '{"_raw":"{\\"query\\":\\"first\\"}{\\"query\\":\\"second\\"}"}',
  );
  assert.equal(
    toolCallReplayArguments("", { query: "first" }),
    '{"query":"first"}',
  );
  assert.equal(toolCallReplayArguments(undefined, undefined), "{}");
});

test("id-less parallel calls renumbered to index 0 stay separate calls", () => {
  // LiteLLM's proxy can strip ids and renumber every parallel call's index to
  // 0. Each opening fragment then "continues" the newest slot-0 part and the
  // argument JSONs concatenate into one malformed string that poisons the
  // thread on replay (#9807).
  const parts = accumulate([
    { index: 0, arguments: '{"url":"https://example.com/1"}' },
    { index: 0, arguments: '{"url":"https://example.com/2"}' },
    { index: 0, arguments: '{"query":"example search"}' },
    { index: 0, arguments: '{"url":"https://example.com/3"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    [
      '{"url":"https://example.com/1"}',
      '{"url":"https://example.com/2"}',
      '{"query":"example search"}',
      '{"url":"https://example.com/3"}',
    ],
  );
  assert.equal(new Set(parts.map((part) => part.toolCallId)).size, 4);
});

test("an id-less chunked continuation still merges into its own call", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"url":' },
    { index: 0, arguments: '"https://example.com/1"}' },
    { index: 0, arguments: '{"url":"https://example.com/2"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    ['{"url":"https://example.com/1"}', '{"url":"https://example.com/2"}'],
  );
});

test("JSON document boundaries survive strings, nesting, whitespace, and Unicode", () => {
  const documents = [
    JSON.stringify({ nested: { items: [1, { text: '} ] { " \\\\ 雪' }] } }),
    "[]",
    "{}",
    '[{"emoji":"🦥"}]',
  ];
  const split = splitTopLevelJsonDocuments(` \n${documents.join(" \t\r\n")}  `);

  assert.deepEqual(split.complete, documents);
  assert.equal(split.tail, "");
});

test("JSON document boundaries keep an incomplete final document as the tail", () => {
  assert.deepEqual(splitTopLevelJsonDocuments('{"a":1} \n ["open"'), {
    complete: ['{"a":1}'],
    tail: '["open"',
  });
});

test("JSON document boundaries reject mismatched or non-document text", () => {
  for (const text of ['{"a":1]','{"a":1}suffix','"{\\"a\\":1}"','true{"a":1}']) {
    assert.deepEqual(splitTopLevelJsonDocuments(text), {
      complete: [],
      tail: text,
    });
  }
});

test("new-call detection works when the SSE boundary lands inside either document", () => {
  const first = '{"path":"C:\\\\tmp","nested":[{"text":"{x}"}]}';
  const second = '{"query":"雪 🦥"}';
  for (let firstCut = 1; firstCut <= first.length; firstCut += 1) {
    const existing = first.slice(0, firstCut);
    const remainder = first.slice(firstCut) + second;
    assert.equal(
      fragmentStartsNewToolCall(existing, remainder),
      true,
      `missed boundary after first character ${firstCut}`,
    );
  }
  for (let secondCut = 1; secondCut <= second.length; secondCut += 1) {
    assert.equal(
      fragmentStartsNewToolCall(first, second.slice(0, secondCut)),
      true,
      `missed partial second document at character ${secondCut}`,
    );
  }
});

test("a document-looking continuation is not split while the first document is open", () => {
  assert.equal(
    fragmentStartsNewToolCall('{"text":"prefix', '{\\"nested\\":true}"}'),
    false,
  );
});
