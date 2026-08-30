// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { toolCallReplayArguments } from "../src/features/chat/tool-call-arguments.ts";
import {
  type StreamedToolCallPart,
  ToolCallArgumentBoundaries,
  bindStreamedToolCallBackendId,
  findStreamedToolCallPartIndex,
  findUnclaimedToolCallPartIndex,
  findUnnamedToolCallPartIndex,
  mintStreamedToolCallId,
  resolveToolCallPartId,
  toolCallArgumentSegments,
} from "../src/features/chat/tool-call-id.ts";

interface DeltaFragment {
  id?: string;
  index?: number;
  name?: string;
  arguments: string;
  extra?: unknown;
}

/** A tool_start event mirrored from the adapter. */
interface ToolStartEvent {
  toolStart: string;
  arguments?: unknown;
}

/** The adapter's tool_end branch, which puts a result on the card. */
interface ToolEndEvent {
  toolEnd: string;
}

type StreamEvent = DeltaFragment | ToolStartEvent | ToolEndEvent;

/** The adapter's parseStreamedToolCallArgs, used by the tool_start mirror. */
function parseArgs(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return { _raw: raw };
  }
}

type AccumulatedPart = StreamedToolCallPart & {
  argsText: string;
  toolName: string;
  extra_content?: unknown;
  result?: unknown;
};

/** Mirror the adapter's tool-call accumulator and optional final sweep. */
function accumulate(
  fragments: StreamEvent[],
  endOfStream = false,
): AccumulatedPart[] {
  const parts: AccumulatedPart[] = [];
  const scans = new Map<string, ToolCallArgumentBoundaries>();
  const providerIds = new Set<string>();
  const splitTails = new Set<string>();
  const withheld = (part: AccumulatedPart) =>
    splitTails.has(part.toolCallId) &&
    part.result === undefined &&
    scans.get(part.toolCallId)?.isUnfinished() === true;
  for (const event of fragments) {
    if ("toolStart" in event) {
      const at = parts.findIndex((part) => part.toolCallId === event.toolStart);
      if (at !== -1) {
        parts[at] = {
          ...parts[at],
          argsText: JSON.stringify(
            "arguments" in event ? event.arguments : parseArgs(parts[at].argsText),
          ),
        };
      }
      continue;
    }
    if ("toolEnd" in event) {
      const at = parts.findIndex((part) => part.toolCallId === event.toolEnd);
      if (at !== -1) parts[at] = { ...parts[at], result: "done" };
      continue;
    }
    const fragment = event;
    if (fragment.id) providerIds.add(fragment.id);
    let target = findStreamedToolCallPartIndex(
      parts,
      fragment.id,
      fragment.index,
    );
    if (
      fragment.id &&
      fragment.name &&
      target !== -1 &&
      parts[target].toolCallId !== fragment.id
    ) {
      const unclaimed = findUnclaimedToolCallPartIndex(parts, fragment.index);
      if (unclaimed !== -1) target = unclaimed;
    }
    const slotPart = target === -1 ? undefined : parts[target];
    const matched = slotPart?.result === undefined ? slotPart : undefined;
    const name = fragment.name ?? "";
    let scan = matched ? scans.get(matched.toolCallId) : undefined;
    if (!scan) {
      scan = new ToolCallArgumentBoundaries();
      if (matched) scans.set(matched.toolCallId, scan);
    }
    const cuts = fragment.arguments ? scan.feed(fragment.arguments) : [];
    const segments =
      cuts.length > 0 && !fragment.id
        ? toolCallArgumentSegments(scan.text(), cuts)
        : [];
    const namedIndex =
      name && !fragment.id
        ? findUnnamedToolCallPartIndex(parts, fragment.index)
        : -1;
    const namedNextCall =
      !fragment.id &&
      !fragment.arguments &&
      Boolean(name) &&
      Boolean(matched?.toolName) &&
      namedIndex === -1 &&
      !name.startsWith(matched?.toolName ?? "") &&
      scan.holdsOneCompleteDocument();

    if (segments.length > 1) {
      const splitName = name || matched?.toolName || "";
      let lastSplitId = "";
      for (const [position, argsText] of segments.entries()) {
        if (position === 0 && matched) {
          parts[target] = {
            ...matched,
            toolName: matched.toolName || splitName,
            argsText,
          };
          scans.set(matched.toolCallId, new ToolCallArgumentBoundaries());
          continue;
        }
        const splitId = mintStreamedToolCallId(
          parts,
          `tool_call_${fragment.index ?? parts.length}`,
          providerIds,
        );
        parts.push({
          toolCallId: splitId,
          toolName: splitName,
          argsText,
          ...(fragment.extra !== undefined && position === segments.length - 1
            ? { extra_content: fragment.extra }
            : {}),
          ...(fragment.index !== undefined
            ? { _delta_index: fragment.index }
            : {}),
        });
        lastSplitId = splitId;
      }
      if (lastSplitId) {
        scan.rebase(segments[segments.length - 1]);
        scans.set(lastSplitId, scan);
        splitTails.add(lastSplitId);
      }
      continue;
    }

    if (matched && !namedNextCall) {
      parts[target] = {
        ...matched,
        ...(fragment.id
          ? { toolCallId: fragment.id, _has_stable_id: true }
          : {}),
        toolName:
          namedIndex === -1 || namedIndex === target
            ? (fragment.name ?? matched.toolName)
            : matched.toolName,
        argsText: matched.argsText + fragment.arguments,
      };
      if (namedIndex !== -1 && namedIndex !== target) {
        parts[namedIndex] = { ...parts[namedIndex], toolName: name };
      }
      if (fragment.id && fragment.id !== matched.toolCallId) {
        scans.delete(matched.toolCallId);
        scans.set(fragment.id, scan);
        if (splitTails.delete(matched.toolCallId)) splitTails.add(fragment.id);
      }
      continue;
    }

    const callId =
      fragment.id ??
      mintStreamedToolCallId(
        parts,
        `tool_call_${fragment.index ?? parts.length}`,
        providerIds,
      );
    scans.set(callId, matched ? new ToolCallArgumentBoundaries() : scan);
    if (matched) splitTails.add(callId);
    parts.push({
      toolCallId: callId,
      toolName: name,
      argsText: fragment.arguments,
      ...(fragment.id ? { _has_stable_id: true } : {}),
      ...(fragment.index !== undefined ? { _delta_index: fragment.index } : {}),
    });
  }
  // Drop split calls that the backend withheld.
  if (endOfStream) {
    for (let i = parts.length - 1; i >= 0; i -= 1) {
      if (withheld(parts[i])) parts.splice(i, 1);
    }
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

test("id-less parallel calls at one index stay separate calls", () => {
  // The #9807 stream reports every id-less call at index 0.
  const parts = accumulate([
    {
      index: 0,
      name: "web_fetch",
      arguments: '{"url":"https://example.com/1"}',
    },
    {
      index: 0,
      name: "web_fetch",
      arguments: '{"url":"https://example.com/2"}',
    },
    { index: 0, name: "web_search", arguments: '{"query":"example search"}' },
    {
      index: 0,
      name: "web_fetch",
      arguments: '{"url":"https://example.com/3"}',
    },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_fetch", '{"url":"https://example.com/1"}'],
      ["web_fetch", '{"url":"https://example.com/2"}'],
      ["web_search", '{"query":"example search"}'],
      ["web_fetch", '{"url":"https://example.com/3"}'],
    ],
  );
  assert.equal(new Set(parts.map((part) => part.toolCallId)).size, 4);
  for (const part of parts) {
    assert.doesNotThrow(() => JSON.parse(part.argsText));
  }
});

test("an id-less call still accumulates across argument fragments", () => {
  const parts = accumulate([
    { index: 0, name: "web_fetch", arguments: '{"url":' },
    { index: 0, arguments: '"https://example.com/1"}' },
    { index: 0, name: "web_search", arguments: '{"query":"example"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_fetch", '{"url":"https://example.com/1"}'],
      ["web_search", '{"query":"example"}'],
    ],
  );
});

test("one fragment holding two documents is two calls", () => {
  const parts = accumulate([
    { index: 0, name: "web_fetch", arguments: '{"url":"a"}{"url":"b"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    ['{"url":"a"}', '{"url":"b"}'],
  );
});

test("an id-less name lands on the next call once the slot's document closed", () => {
  const parts = accumulate([
    { index: 0, name: "web_fetch", arguments: '{"url":"a"}' },
    { index: 0, name: "web_search", arguments: "" },
    { index: 0, arguments: '{"query":"b"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_fetch", '{"url":"a"}'],
      ["web_search", '{"query":"b"}'],
    ],
  );
});

test("a name growing over several fragments still names one call", () => {
  const parts = accumulate([
    { index: 0, name: "web", arguments: "" },
    { index: 0, name: "web_search", arguments: "" },
    { index: 0, arguments: '{"query":"b"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [["web_search", '{"query":"b"}']],
  );
});

test("a split names the call it opened and the one it closed", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"url":"a"}' },
    { index: 0, name: "web_fetch", arguments: '{"url":"b"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_fetch", '{"url":"a"}'],
      ["web_fetch", '{"url":"b"}'],
    ],
  );
});

test("a complete document followed by an open one keeps both calls", () => {
  const parts = accumulate([
    { index: 0, name: "web_fetch", arguments: '{"url":"a"}{"url":' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_fetch", '{"url":"a"}'],
      ["web_fetch", '{"url":'],
    ],
  );
});

test("balanced text that is not JSON is never split", () => {
  const parts = accumulate([
    { index: 0, name: "web_fetch", arguments: "{bad}{worse}" },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    ["{bad}{worse}"],
  );
});

test("arguments that are not JSON are left whole", () => {
  const parts = accumulate([
    { index: 0, name: "web_search", arguments: "query=" },
    { index: 0, arguments: "example" },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    ["query=example"],
  );
});

test("a third id-less call still splits off the second", () => {
  const parts = accumulate([
    { index: 0, name: "web_fetch", arguments: '{"url":"1"}' },
    { index: 0, name: "web_search", arguments: '{"q":"2"}' },
    { index: 0, name: "web_fetch", arguments: '{"url":"3"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_fetch", '{"url":"1"}'],
      ["web_search", '{"q":"2"}'],
      ["web_fetch", '{"url":"3"}'],
    ],
  );
});

test("a call split off mid-document keeps accumulating its own arguments", () => {
  const parts = accumulate([
    { index: 0, name: "web_fetch", arguments: '{"url":"1"}{"url":' },
    { index: 0, arguments: '"2"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.argsText),
    ['{"url":"1"}', '{"url":"2"}'],
  );
});

test("an id-less call's backend events find the card it painted", () => {
  const ids = new Map<string, string>();
  let minted = 0;
  const createId = () => `run-scoped-${(minted += 1)}`;

  bindStreamedToolCallBackendId(ids, "tool_call_0");

  assert.equal(
    resolveToolCallPartId(ids, "tool_call_0", undefined, "last", createId),
    "tool_call_0",
  );
  assert.equal(minted, 0, "a bound id must not mint a run-scoped one");

  assert.equal(
    resolveToolCallPartId(ids, "call_from_nowhere", undefined, "last", createId),
    "run-scoped-1",
  );
});

test("a provider id stays reserved after its call has finished", () => {
  const parts = accumulate([
    { id: "tool_call_0", index: 0, name: "web_search", arguments: '{"q":"a"}' },
    { index: 0, name: "web_search", arguments: '{"q":"b"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.toolCallId),
    ["tool_call_0", "tool_call_1"],
  );
});

test("an id adopted late frees the spelling its call was painted under", () => {
  const parts = accumulate([
    { index: 0, name: "web_search", arguments: '{"q":' },
    { id: "call_a", index: 0, arguments: '"a"}' },
    { index: 0, name: "web_search", arguments: '{"q":"b"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => part.toolCallId),
    ["call_a", "tool_call_0"],
  );
});

test("minting skips a spelling the backend already reserved", () => {
  const parts = [
    { toolCallId: "tool_call_0:run-1", _delta_index: 0, _has_stable_id: true },
  ];
  const claimed = new Map<string, string>([["tool_call_0", "tool_call_0:run-1"]]);

  assert.equal(
    mintStreamedToolCallId(parts, "tool_call_0", claimed.keys()),
    "tool_call_1",
  );
  assert.equal(mintStreamedToolCallId(parts, "tool_call_0"), "tool_call_0");
});

test("binding never steals an id another part already owns", () => {
  const ids = new Map<string, string>([["tool_call_0", "already-mine"]]);
  bindStreamedToolCallBackendId(ids, "tool_call_0");

  assert.equal(ids.get("tool_call_0"), "already-mine");
});

test("a long argument is read once, not once per fragment", () => {
  const opening = '{"code":"';
  const body = "x".repeat(64 * 1024);
  const scan = new ToolCallArgumentBoundaries();
  scan.feed(opening);
  for (let at = 0; at < body.length; at += 1024) {
    scan.feed(body.slice(at, at + 1024));
  }

  assert.equal(scan.scanned, opening.length + body.length);
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

// Boundary offsets must survive tool_start rewriting argsText.
test("a next round splits correctly after tool_start rewrites the arguments", () => {
  const parts = accumulate([
    { index: 0, name: "web_search", arguments: '{"query": "first"}' },
    { toolStart: "tool_call_0" },
    { index: 0, name: "web_search", arguments: '{"query":"second"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, JSON.parse(part.argsText)]),
    [
      ["tool_call_0", { query: "first" }],
      ["tool_call_1", { query: "second" }],
    ],
  );
});

test("boundary offsets index the scan's own text, not the part's", () => {
  const scan = new ToolCallArgumentBoundaries();
  assert.deepEqual(scan.feed('{"query": "first"}'), []);
  const cuts = scan.feed('{"query":"second"}');

  assert.deepEqual(cuts, [18]);
  assert.deepEqual(toolCallArgumentSegments(scan.text(), cuts), [
    '{"query": "first"}',
    '{"query":"second"}',
  ]);
});

// A late provider id must not hide an unfinished split tail from cleanup.
test("a split tail renamed by a late id is still dropped when its JSON never closes", () => {
  const parts = accumulate(
    [
      { index: 0, name: "web_search", arguments: '{"a":1}{"b":' },
      { id: "call-A", index: 0, arguments: "" },
    ],
    true,
  );

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [["tool_call_0", '{"a":1}']],
  );
});

// Per-call metadata belongs only to the last call opened by a split.
test("split metadata rides only the last call the delta opened", () => {
  const parts = accumulate([
    {
      index: 0,
      name: "web_search",
      arguments: '{"a":1}{"b":2}',
      extra: { thought_signature: "S" },
    },
  ]);

  assert.deepEqual(
    parts.map((part) => part.extra_content),
    [undefined, { thought_signature: "S" }],
  );
});

// A closed malformed call must not absorb the next round.
test("a finished malformed call does not swallow the next round", () => {
  const parts = accumulate([
    { index: 0, name: "web_search", arguments: "{bad}" },
    { toolStart: "tool_call_0" },
    { toolEnd: "tool_call_0" },
    { index: 0, name: "web_search", arguments: '{"query":"second"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["tool_call_0", '{"_raw":"{bad}"}'],
      ["tool_call_1", '{"query":"second"}'],
    ],
  );
});

// A resent whole name stays on the completed call.
test("a name resent after its arguments closed opens no second call", () => {
  const parts = accumulate([
    { index: 0, name: "web_search", arguments: '{"query":"first"}' },
    { index: 0, name: "web_search", arguments: "" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [["web_search", '{"query":"first"}']],
  );
});

test("a second call on the same tool still splits on its arguments", () => {
  const parts = accumulate([
    { index: 0, name: "web_search", arguments: '{"query":"first"}' },
    { index: 0, name: "web_search", arguments: "" },
    { index: 0, arguments: '{"query":"second"}' },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_search", '{"query":"first"}'],
      ["web_search", '{"query":"second"}'],
    ],
  );
});

// Cleanup must include unfinished name-only forks.
test("a name fork left unfinished is swept like any other split tail", () => {
  const parts = accumulate(
    [
      { index: 0, name: "web_fetch", arguments: '{"url":"a"}' },
      { index: 0, name: "web_search", arguments: "" },
      { index: 0, arguments: '{"query":' },
    ],
    true,
  );

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [["web_fetch", '{"url":"a"}']],
  );
});

// A closed split tail must not absorb the next round.
test("a withheld split tail does not take the next round's arguments", () => {
  const parts = accumulate(
    [
      { index: 0, name: "web_search", arguments: '{"a":1}{"b":' },
      { toolStart: "tool_call_1", arguments: {} },
      { toolEnd: "tool_call_1" },
      { toolStart: "tool_call_0", arguments: { a: 1 } },
      { toolEnd: "tool_call_0" },
      { index: 0, name: "web_search", arguments: '{"query":"second"}' },
    ],
    true,
  );

  // The closed tail keeps tool_call_1 reserved.
  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.argsText]),
    [
      ["tool_call_0", '{"a":1}'],
      ["tool_call_1", "{}"],
      ["tool_call_2", '{"query":"second"}'],
    ],
  );
});

// Early arguments and a resent growing name must remain one call.
test("a growing name after early arguments still names one call", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"query":"first"}' },
    { index: 0, name: "web", arguments: "" },
    { index: 0, name: "web_search", arguments: "" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [["web_search", '{"query":"first"}']],
  );
});

// Late names fill unnamed split calls in order.
test("late names fill the calls a split left waiting, in order", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"a":1}{"b":2}' },
    { index: 0, name: "web_search", arguments: "" },
    { index: 0, name: "web_fetch", arguments: "" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolName, part.argsText]),
    [
      ["web_search", '{"a":1}'],
      ["web_fetch", '{"b":2}'],
    ],
  );
});

// The boundary is recorded when the next `{` arrives, before that document has
// parsed, so a split is already permanent when a later fragment closes it into
// something malformed. The tail must not then look finished.
test("a split whose document never parses counts as unfinished", () => {
  const scan = new ToolCallArgumentBoundaries();
  assert.deepEqual(scan.feed('{"q":"ok"}{bad'), [10]);
  scan.feed("}");

  assert.equal(scan.isOpen(), false);
  assert.equal(scan.isUnfinished(), true);
});

// Two documents arrive before any identity, then late ids name them. Targeting
// the open slot bound the first id to the second document, left the first
// unnamed for the backend to drop, and opened an empty third call.
test("a late id claims the first call a split left waiting", () => {
  const parts = accumulate([
    { index: 0, arguments: '{"a":1}{"b":2}' },
    { id: "A", index: 0, name: "web_search", arguments: "" },
    { id: "B", index: 0, name: "web_fetch", arguments: "" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.toolName, part.argsText]),
    [
      ["A", "web_search", '{"a":1}'],
      ["B", "web_fetch", '{"b":2}'],
    ],
  );
});

// The split hands its own name down to every segment, so a named segment can
// still be waiting for the id that belongs to it. Requiring the name to be
// missing bound the first id to the second document and opened an empty third.
test("a late id claims a split call that inherited a name", () => {
  const parts = accumulate([
    { index: 0, name: "web_search", arguments: '{"a":1}{"b":2}' },
    { id: "A", index: 0, name: "web_search", arguments: "" },
    { id: "B", index: 0, name: "web_fetch", arguments: "" },
  ]);

  assert.deepEqual(
    parts.map((part) => [part.toolCallId, part.toolName, part.argsText]),
    [
      ["A", "web_search", '{"a":1}'],
      ["B", "web_fetch", '{"b":2}'],
    ],
  );
});
