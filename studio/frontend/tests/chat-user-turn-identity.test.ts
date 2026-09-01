// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A user turn is identified by its id, never by its text (#9984). Neither function here is
// exported and both files pull in React, so these assert on structure rather than behaviour:
// they must fail for ANY reinserted dedupe, not just one that reuses the old names.

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

const outboundPrune = slice(
  adapter,
  "const surviving: RunMessage[] = [];",
  "return surviving;",
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

test("the outbound prune keeps every turn the abandoned-turn guard did not drop", () => {
  // The guard's `continue` runs straight into the push, so there is nowhere for a content
  // comparison to live. Renaming a helper or inlining JSON.stringify cannot get past this.
  assert.match(
    outboundPrune,
    /if \(refused \|\| abandoned\[index\]\) \{[\s\S]*?\n {6}continue;\n {4}\}\n {4}surviving\.push\(message\);\n {2}\}/,
  );
});

test("the outbound prune pops a prompt only for the abandoned turn it belongs to", () => {
  // A second, unguarded pop would delete a real prompt from the request.
  assert.equal(outboundPrune.match(/surviving\.pop\(\)/g)?.length, 1);
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
  // msgs is assigned once per branch and never rewritten, so nothing can filter the tree
  // before it is rebuilt. A dangling parent makes the whole thread unimportable.
  assert.deepEqual(
    historyLoad.match(/\bmsgs\s*=\s*[^=][^;\n]*/g),
    ["msgs = snapshot.messages", "msgs = []"],
  );
  assert.doesNotMatch(historyLoad, /\bmsgs\.filter\(/);
});
