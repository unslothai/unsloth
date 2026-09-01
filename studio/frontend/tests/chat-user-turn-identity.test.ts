// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A user turn is identified by its id, never by its text (#9984). Regenerate reloads through
// startRun({ parentId }), which mints a new assistant message and leaves the user turn alone,
// so nothing on these paths may collapse turns by comparing what they say.

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
  // collectTextParts reads text only, so an equality test on it would also collapse two turns
  // carrying different documents, dropping one turn's attachments off the wire.
  assert.doesNotMatch(outboundPrune, /collectTextParts/);
});

test("the outbound prune pops a prompt only for the abandoned turn it belongs to", () => {
  // One pop, guarded by the abandoned/refused branch. A second, unguarded pop would silently
  // delete a real prompt from the request.
  assert.equal(outboundPrune.match(/surviving\.pop\(\)/g)?.length, 1);
  assert.match(
    outboundPrune,
    /if \(refused \|\| abandoned\[index\]\) \{[\s\S]*?surviving\.pop\(\);[\s\S]*?continue;/,
  );
});

test("a message is persisted under the id the runtime gave it", () => {
  // Writing a different id leaves the assistant that follows parented to a row that was never
  // written, and no caller reads the id back: chat-history-storage discards the response.
  assert.match(historyAppend, /id: message\.id,/);
  assert.doesNotMatch(historyAppend, /finalMessageId/);
});

test("appending a message does not read the whole thread", () => {
  // The point read above it is the one the model run waits on. A whole-thread GET here is the
  // per-chunk cost #9865 removed.
  assert.doesNotMatch(historyAppend, /listStoredChatMessages/);
});

test("loading a thread returns every stored message", () => {
  // Filtering user turns out of the tree orphans the assistant messages parented to them, and
  // a dangling parent makes the whole thread unimportable rather than just that turn.
  assert.doesNotMatch(runtimeProvider, /seenUserTurns/);
  assert.doesNotMatch(runtimeProvider, /dedupedMsgs/);
});
