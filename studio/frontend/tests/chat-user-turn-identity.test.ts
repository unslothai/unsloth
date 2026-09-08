// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A user turn is identified by its id, never by its text (#9984).
//
// Scope, because the names below would otherwise overclaim. These read source, so they catch
// the shape the reverted change actually had, a comparison written inline at one of the three
// sites. They are NOT a proof that no dedupe can return: source matching has no closed set of
// patterns, and review found several ways past it, among them replacing `history` in place,
// binding the filtered array to a new name, and reassigning the id after the payload is built.
// Proving the negative needs the pure logic extracted and called, which is a change to
// production code rather than to this file. Behaviour is covered where the damage lands, in
// studio/backend/tests/test_chat_message_identity.py, against the real studio_db.

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

/** Brace nesting at `needle`, counted from the start of `source`. */
function depthAt(source: string, needle: string): number {
  const upto = source.slice(0, source.indexOf(needle));
  return count(upto, /{/g) - count(upto, /}/g);
}

// From the signature, so a filter folded into the input is in scope too.
const outboundPrune = slice(
  adapter,
  "function pruneOutboundHistory(",
  "function extractImageBase64(",
);
// Through the reconstruction: the branches below build the repository from msgs.
const historyLoad = slice(
  runtimeProvider,
  "let msgs: MessageRecord[];",
  "append({ parentId, message }: ExportedMessageRepositoryItem) {",
);
const historyAppend = slice(
  runtimeProvider,
  "append({ parentId, message }: ExportedMessageRepositoryItem) {",
  "return trackHistoryAppend(",
);

test("the outbound prune has no second way to drop a turn", () => {
  // The input is copied whole; filtering here drops a turn without touching the loop.
  assert.match(outboundPrune, /const history = \[\.\.\.messages\];/);
  // The guard owns one of each already, so a second is a dedupe under any name.
  assert.equal(count(outboundPrune, /\bcontinue;/g), 1);
  assert.equal(count(outboundPrune, /surviving\.pop\(\)/g), 1);
  assert.equal(count(outboundPrune, /surviving\.push\(message\);/g), 1);
  // Depth catches a block wrapper, line start catches an inline one.
  assert.equal(
    depthAt(outboundPrune, "surviving.push(message);"),
    depthAt(outboundPrune, "const message = history[index];"),
  );
  assert.match(outboundPrune, /\n\s*surviving\.push\(message\);/);
});

test("the append payload is built with the id the runtime gave it", () => {
  // A different id leaves the next assistant parented to a row nothing wrote.
  assert.match(historyAppend, /id: message\.id,/);
  assert.doesNotMatch(historyAppend, /\bid:\s*(?!message\.id\b)\w+,/);
});

test("appending a message does not read the whole thread", () => {
  // A whole-thread GET here is the per-message cost #9865 removed.
  assert.doesNotMatch(historyAppend, /listStoredChatMessages/);
});

test("nothing between the load and the rebuild narrows msgs", () => {
  // Both names and in-place removal. Not reordering: the existing msgs.sort keeps the set.
  assert.deepEqual(
    historyLoad.match(/\b(?:msgs|snapshot\.messages)\s*=\s*[^=][^;\n]*/g),
    ["msgs = snapshot.messages", "msgs = []"],
  );
  assert.doesNotMatch(
    historyLoad,
    /\b(?:msgs|snapshot\.messages)(?:\.length\s*=|\.(?:filter|splice|shift|pop|slice)\()/,
  );
});
