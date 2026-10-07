// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A user turn is identified by its id, never its text. Source checks only catch the inline shape;
// behaviour is covered in studio/backend/tests/test_chat_message_identity.py.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const adapter = readSrc("features/chat/api/chat-adapter.ts");
const runtimeProvider = readSrc("features/chat/runtime-provider.tsx");

function slice(source: string, from: string, to: string): string {
  const start = source.indexOf(from);
  const end = source.indexOf(to, start);
  assert.ok(start >= 0 && end > start, `could not slice ${from} .. ${to}`);
  return source.slice(start, end);
}

function count(source: string, pattern: RegExp): number {
  return source.match(pattern)?.length ?? 0;
}

function depthAt(source: string, needle: string): number {
  const upto = source.slice(0, source.indexOf(needle));
  return count(upto, /{/g) - count(upto, /}/g);
}

const outboundPrune = slice(
  adapter,
  "function pruneOutboundHistory(",
  "function extractImageBase64(",
);
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
  assert.match(outboundPrune, /const history = \[\.\.\.messages\];/);
  assert.equal(count(outboundPrune, /\bcontinue;/g), 1);
  assert.equal(count(outboundPrune, /surviving\.pop\(\)/g), 1);
  assert.equal(count(outboundPrune, /surviving\.push\(message\);/g), 1);
  assert.equal(
    depthAt(outboundPrune, "surviving.push(message);"),
    depthAt(outboundPrune, "const message = history[index];"),
  );
  assert.match(outboundPrune, /\n\s*surviving\.push\(message\);/);
});

test("the append payload is built with the id the runtime gave it", () => {
  assert.match(historyAppend, /id: message\.id,/);
  assert.doesNotMatch(historyAppend, /\bid:\s*(?!message\.id\b)\w+,/);
});

test("appending a message does not read the whole thread", () => {
  assert.doesNotMatch(historyAppend, /listStoredChatMessages/);
});

test("nothing between the load and the rebuild narrows msgs", () => {
  assert.deepEqual(
    historyLoad.match(/\b(?:msgs|snapshot\.messages)\s*=\s*[^=][^;\n]*/g),
    ["msgs = snapshot.messages", "msgs = []"],
  );
  assert.doesNotMatch(
    historyLoad,
    /\b(?:msgs|snapshot\.messages)(?:\.length\s*=|\.(?:filter|splice|shift|pop|slice)\()/,
  );
});
