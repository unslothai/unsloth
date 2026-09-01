// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A user turn is identified by its id, never by its text (#9984). Neither function here is
// exported, so these assert structure: they must fail for ANY reinserted dedupe, not just
// one that reuses the old names.

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

/** Whitespace-insensitive so reformatting does not fail the pin, but nothing else is. */
function normalise(source: string): string {
  return source.trim().replace(/\s+/g, " ");
}

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start);
  assert.ok(start >= 0 && end > start, `could not slice ${from} .. ${to}`);
  return source.slice(start, end);
}

// The WHOLE function, not just the loop: a dedupe folded into `const history = ...`
// would sit above the loop and escape a narrower slice.
const LOOP_START = "for (let index = 0; index < history.length; index += 1) {";
const LOOP_BODY = normalise(`
  for (let index = 0; index < history.length; index += 1) {
    const message = history[index];
    const refused = isAnthropicRefusalMessage(message);
    if (refused || abandoned[index]) {
      if (refused || index < lastSurviving) {
        const last = surviving.at(-1);
        if (last && last.role === "user") surviving.pop();
      }
      continue;
    }
    surviving.push(message);
  }
`);

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

test("the outbound prune keeps every turn the abandoned-turn guard did not drop", () => {
  // The input is copied whole. Folding a filter into this line is the one way to drop a
  // turn without touching the loop below.
  assert.match(outboundPrune, /\n {2}const history = \[\.\.\.messages\];\n/);
  // The loop body is pinned exactly, not pattern-matched. Three rounds of review each found
  // another spelling a regex let through, the last being an early `continue` above the
  // refusal check, so the only assertion that holds is that nothing was inserted at all.
  // A deliberate refactor here has to update this string, which is the point.
  assert.equal(normalise(slice(outboundPrune, LOOP_START, "return surviving;")), LOOP_BODY);
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
  // Nothing may drop a turn before the tree is rebuilt: a dangling parent breaks the thread.
  // Both the binding and the array behind it are covered, since either can be narrowed, and
  // in-place removal counts. Reordering does not: msgs.sort is already there and keeps the set.
  assert.deepEqual(
    historyLoad.match(/\b(?:msgs|snapshot\.messages)\s*=\s*[^=][^;\n]*/g),
    ["msgs = snapshot.messages", "msgs = []"],
  );
  assert.doesNotMatch(
    historyLoad,
    /\b(?:msgs|snapshot\.messages)(?:\.length\s*=|\.(?:filter|splice|shift|pop|slice)\()/,
  );
});
