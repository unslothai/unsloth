// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { toolCallReplayArguments } from "../src/features/chat/tool-call-arguments.ts";
import {
  type StreamedToolCallPart,
  findStreamedToolCallPartIndex,
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
    const target = findStreamedToolCallPartIndex(
      parts,
      fragment.id,
      fragment.index,
    );
    if (target === -1) {
      parts.push({
        toolCallId:
          fragment.id ?? `tool_call_${fragment.index ?? parts.length}`,
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
  // Two calls glued into one slot. The stream adapter no longer produces this
  // (see tests/parallel-tool-call-arguments.test.ts), but threads stored before
  // it split them still hold it, and `_raw` is the adapter's own marker rather
  // than a parameter any tool declares, so it must not go back on the wire.
  assert.equal(
    toolCallReplayArguments('{"query":"first"}{"query":"second"}', {
      _raw: '{"query":"first"}{"query":"second"}',
    }),
    "{}",
  );
  assert.equal(
    toolCallReplayArguments("", { query: "first" }),
    '{"query":"first"}',
  );
  assert.equal(toolCallReplayArguments(undefined, undefined), "{}");
});
