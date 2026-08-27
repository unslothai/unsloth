// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The per-chunk autosave must not write back a message the server owns.
 *
 * When `preserveServerManaged` is true the backend owns the message and refuses to let a client
 * edit it, and every field the autosave would send is one it just read from the backend. So the
 * request cannot change anything and is answered 409.
 *
 * Measured on a real session before this guard existed: one 43.6 s generation issued 265 PUTs of
 * which 256 were rejected with "server-managed generation messages cannot be edited", a median
 * 166 ms apart, plus 353 whole-thread GETs a median 339 ms apart from the `ensureStoredChatThread`
 * inside `saveStoredChatMessage`. All against the backend the model is saturating.
 *
 * This is a SOURCE guard rather than a behavioural one because the call site is an inline closure
 * inside the history adapter, with no seam to stub. It therefore pins ORDERING, not spelling: that
 * the `preserveServerManaged` early return appears before the `saveStoredChatMessage` call it
 * guards. A rename of either keeps working; moving the save above the guard does not.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROVIDER = path.join(
  HERE,
  "..",
  "src",
  "features",
  "chat",
  "runtime-provider.tsx",
);

const source = readFileSync(PROVIDER, "utf8");

/** The history adapter's append path, which is the one that autosaves per chunk. */
function appendWindow(): string {
  const anchor = source.indexOf("const preserveServerManaged =");
  assert.notEqual(
    anchor,
    -1,
    "the autosave no longer computes preserveServerManaged, so this guard is measuring nothing",
  );
  const end = source.indexOf("trackHistoryAppend(", anchor);
  assert.notEqual(end, -1, "could not find the end of the append path");
  return source.slice(anchor, end);
}

test("the autosave returns early instead of writing a server-managed message", () => {
  const window = appendWindow();
  const guard = window.search(/if\s*\(\s*preserveServerManaged\s*\)/);
  assert.notEqual(
    guard,
    -1,
    "no `if (preserveServerManaged)` guard in the append path: the per-chunk autosave will PUT a " +
      "message the server owns and take a 409 on every chunk",
  );
  const save = window.indexOf("saveStoredChatMessage(");
  assert.notEqual(save, -1, "the append path no longer saves at all, which is not the fix");
  assert.ok(
    guard < save,
    "the `preserveServerManaged` guard must come BEFORE saveStoredChatMessage. Below it, the " +
      "write still goes out and the 409 storm is unchanged",
  );
});

test("the guard actually returns rather than only skipping fields", () => {
  const window = appendWindow();
  const guard = window.search(/if\s*\(\s*preserveServerManaged\s*\)/);
  const save = window.indexOf("saveStoredChatMessage(");
  const body = window.slice(guard, save);
  assert.match(
    body,
    /\breturn\b/,
    "the guard must return. Skipping only the changed fields still issues the PUT, and still " +
      "runs the ensureStoredChatThread whole-thread GET inside saveStoredChatMessage",
  );
});

test("the write no longer echoes the server's own content back at it", () => {
  const window = appendWindow();
  assert.doesNotMatch(
    window,
    /preserveServerManaged\s*\?\s*existingMessage/,
    "the save still passes existingMessage.content when preserveServerManaged, which is the " +
      "read-it-then-write-it-back shape this change removed",
  );
});
