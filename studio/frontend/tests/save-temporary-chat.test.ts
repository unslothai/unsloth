// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Saving a temporary chat to history. runtime-provider.tsx cannot load under stubs, so the
// ordering helper is run on its own and the rest is pinned on source.

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const provider = readSrc("features/chat/runtime-provider.tsx");
const storage = readSrc("features/chat/utils/chat-history-storage.ts");
const button = readSrc("features/chat/components/temporary-chat-save.tsx");
const page = readSrc("features/chat/chat-page.tsx");

type Item = { parentId: string | null; message: { id: string } };
const parentsFirst = new Function(
  `${stripTypeScriptTypes(provider.slice(provider.indexOf("function parentsFirst("), provider.indexOf("/** Save a temporary chat to history")))}\nreturn parentsFirst;`,
)() as (items: Item[]) => Item[];

const item = (id: string, parentId: string | null = null): Item => ({ parentId, message: { id } });
const ids = (items: Item[]) => items.map((i) => i.message.id);

function persistBody(): string {
  const start = provider.indexOf("export async function persistTemporaryThread(");
  assert.ok(start > 0, "persistTemporaryThread not found");
  return provider.slice(start, provider.indexOf("function createStudioDbAdapter(", start));
}

test("every message is saved after its parent, on every branch", () => {
  // Two replies to u1 (a regenerate), and a child listed before its parent.
  const ordered = ids(
    parentsFirst([item("a2", "u2"), item("u1"), item("a1", "u1"), item("u2", "a1"), item("b1", "u1")]),
  );
  assert.deepEqual([...ordered].sort(), ["a1", "a2", "b1", "u1", "u2"]);
  for (const [child, parent] of [["a1", "u1"], ["b1", "u1"], ["u2", "a1"], ["a2", "u2"]]) {
    assert.ok(ordered.indexOf(parent) < ordered.indexOf(child), `${parent} before ${child}`);
  }
});

test("a parent missing from the export does not drop the message", () => {
  assert.deepEqual(ids(parentsFirst([item("x", "gone"), item("y")])), ["x", "y"]);
});

test("the thread stops being temporary before writing, and goes back if the save fails", () => {
  const body = persistBody();
  assert.match(storage, /export function unmarkThreadIncognito\(threadId: string\): void \{\s*incognitoThreadIds\.delete\(threadId\);/);
  assert.ok(body.indexOf("unmarkThreadIncognito(threadId)") < body.indexOf("await ensureThreadRecord("));
  assert.match(body, /\} catch \(error\) \{\s*markThreadIncognito\(threadId\);\s*throw error;/);
  assert.match(body, /incognito: false,/);
});

test("the save covers the whole branch tree, not only the visible path", () => {
  assert.match(button, /messages: aui\.thread\(\)\.export\(\)\.messages/);
  assert.match(persistBody(), /parentId: parentId \?\? null,/);
});

test("the button shows only in a temporary single chat and waits for a finished reply", () => {
  assert.match(page, /view\.mode === "single" && incognito \? \(\s*<SaveTemporaryChatButton/);
  assert.match(button, /\? "Nothing to save yet"\s*: target\.running\s*\? "Wait for the response to finish"/);
});

test("Don't show this again is remembered only once a save succeeds", () => {
  const save = button.slice(button.indexOf("const save = async () => {"), button.indexOf("return (\n"));
  assert.ok(save.indexOf("await target.save()") < save.indexOf("if (dontShowAgain) rememberSkipConfirm()"));
  assert.doesNotMatch(save.slice(save.indexOf("catch")), /rememberSkipConfirm/);
  assert.match(button, /if \(skipConfirm\(\)\) void save\(\);\s*else setOpen\(true\);/);
});
