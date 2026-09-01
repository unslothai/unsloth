// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A user turn is identified by its id, never by its text (#9984).

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
const historyAppend = slice(
  runtimeProvider,
  "append({ parentId, message }: ExportedMessageRepositoryItem) {",
  "return trackHistoryAppend(",
);

test("the outbound prune drops turns by what they are, not by what they say", () => {
  // collectTextParts reads text only: matching on it drops a turn's attachments off the wire.
  assert.doesNotMatch(outboundPrune, /collectTextParts/);
});

test("the outbound prune pops a prompt only for the abandoned turn it belongs to", () => {
  // A second, unguarded pop would delete a real prompt from the request.
  assert.equal(outboundPrune.match(/surviving\.pop\(\)/g)?.length, 1);
  assert.match(
    outboundPrune,
    /if \(refused \|\| abandoned\[index\]\) \{[\s\S]*?surviving\.pop\(\);[\s\S]*?continue;/,
  );
});

test("a message is persisted under the id the runtime gave it", () => {
  // A different id leaves the next assistant parented to a row nothing wrote.
  assert.match(historyAppend, /id: message\.id,/);
  assert.doesNotMatch(historyAppend, /finalMessageId/);
});

test("appending a message does not read the whole thread", () => {
  // A whole-thread GET here is the per-message cost #9865 removed.
  assert.doesNotMatch(historyAppend, /listStoredChatMessages/);
});

test("loading a thread returns every stored message", () => {
  // A dangling parent makes the whole thread unimportable, not just that turn.
  assert.doesNotMatch(runtimeProvider, /seenUserTurns/);
  assert.doesNotMatch(runtimeProvider, /dedupedMsgs/);
});
