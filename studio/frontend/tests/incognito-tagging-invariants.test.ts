// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// ensureThreadRecord's incognito shortcut used to key on a `__LOCALID_` id, which is the
// permanent primary key of every chat the app creates, not a "fresh thread" marker. With the
// toggle on, a caller passing the OPEN chat's id therefore tagged that SAVED chat incognito
// for the session, and a tagged chat stops persisting, loses its settings snapshot and loses
// its fork badge.
//
// Structural, like thread-scoped-pairing-invariants.test.ts: runtime-provider.tsx cannot be
// loaded under stubs, so what is pinned here is the shape of the decision.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const provider = readFileSync(
  new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
  "utf8",
);

/** ensureThreadRecord's body, code only. */
function ensureThreadRecordBody(): string {
  const start = provider.indexOf("export async function ensureThreadRecord({");
  assert.ok(start > 0, "ensureThreadRecord not found");
  const end = provider.indexOf("const record: ThreadRecord = {", start);
  assert.ok(end > start, "end of ensureThreadRecord's guards not found");
  return provider
    .slice(start, end)
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^[ \t]*\/\/.*$/gm, "");
}

test("the incognito shortcut is gated on the thread being new, not on its id", () => {
  const body = ensureThreadRecordBody();

  assert.match(
    body,
    /if \(incognitoAtInit && neverSent\) \{/,
    "the shortcut must ask whether the thread has ever been sent to",
  );
  assert.doesNotMatch(
    body,
    /isAssistantLocalThreadId/,
    "a `__LOCALID_` prefix says nothing about whether a row exists: every chat the app " +
      "creates keeps that id after it is saved",
  );
});

test("only initialize() claims a thread has never been sent to", () => {
  // Every other caller hands in whatever chat is open, which may well be saved.
  const claims = provider.match(/neverSent: true/g) ?? [];
  assert.equal(
    claims.length,
    1,
    `neverSent: true is claimed ${claims.length} times; only initialize() may claim it`,
  );

  const initialize = provider.slice(
    provider.indexOf("initialize(threadId: string) {"),
    provider.indexOf("adoptDefaultThreadRun"),
  );
  assert.match(initialize, /neverSent: true/);
});

test("a saved chat still reaches the existing-row check before it can be tagged", () => {
  // Without this ordering the tag is unconditional for anything the shortcut misses.
  const body = ensureThreadRecordBody();
  const shortcut = body.indexOf("incognitoAtInit && neverSent");
  const lookup = body.indexOf("await getStoredChatThread(threadId)");
  const lateTag = body.lastIndexOf("if (incognitoAtInit)");

  assert.ok(shortcut > 0 && lookup > shortcut, "the row lookup must follow the shortcut");
  assert.ok(
    lateTag > lookup,
    "every path that did not take the shortcut must consult the stored row first",
  );
});
