// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A user turn is identified by its id, never by its text (#9984). Neither function here is
// exported, so these read source. They count the ways a turn can be dropped rather than
// pinning the text, so a rename or a reformat is fine and a new drop is not.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const runtimeProvider = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  "utf8",
);

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start);
  assert.ok(start >= 0 && end > start, `could not slice ${from} .. ${to}`);
  return source.slice(start, end);
}

function count(source: string, pattern: RegExp): number {
  return source.match(pattern)?.length ?? 0;
}

// From the signature, so a filter folded into the input is in scope too.
const outboundPrune = slice(
  adapter,
  "function pruneOutboundHistory(",
  "function extractImageBase64(",
);
const historyLoad = slice(
  runtimeProvider,
  "let msgs: MessageRecord[];",
  "const hasParentIds",
);
const historyAppend = slice(
  runtimeProvider,
  "append({ parentId, message }: ExportedMessageRepositoryItem) {",
  "return trackHistoryAppend(",
);

test("the outbound prune drops a turn only for the abandoned-turn guard", () => {
  // The input is copied whole; filtering here drops a turn without touching the loop.
  assert.match(outboundPrune, /const history = \[\.\.\.messages\];/);
  // Inside the loop a turn can only be lost three ways, and the guard already owns one of
  // each. A second is a dedupe, whatever it is called.
  assert.equal(count(outboundPrune, /\bcontinue;/g), 1);
  assert.equal(count(outboundPrune, /surviving\.pop\(\)/g), 1);
  // Unconditional: `if (...) surviving.push(message);` would skip turns silently.
  assert.equal(count(outboundPrune, /surviving\.push\(message\);/g), 1);
  assert.match(outboundPrune, /\n\s*surviving\.push\(message\);/);
});

test("a message is persisted under the id the runtime gave it", () => {
  // A different id leaves the next assistant parented to a row nothing wrote.
  assert.match(historyAppend, /id: message\.id,/);
  assert.doesNotMatch(historyAppend, /\bid:\s*(?!message\.id\b)\w+,/);
});

test("appending a message does not read the whole thread", () => {
  // A whole-thread GET here is the per-message cost #9865 removed.
  assert.doesNotMatch(historyAppend, /listStoredChatMessages/);
});

test("loading a thread passes on every stored message", () => {
  // A dangling parent breaks the thread, so cover both names and in-place removal.
  // Not reordering: the existing msgs.sort keeps the set.
  assert.deepEqual(
    historyLoad.match(/\b(?:msgs|snapshot\.messages)\s*=\s*[^=][^;\n]*/g),
    ["msgs = snapshot.messages", "msgs = []"],
  );
  assert.doesNotMatch(
    historyLoad,
    /\b(?:msgs|snapshot\.messages)(?:\.length\s*=|\.(?:filter|splice|shift|pop|slice)\()/,
  );
});
