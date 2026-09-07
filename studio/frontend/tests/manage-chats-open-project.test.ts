// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Opening a managed row must carry the row's project the way the sidebar does.
// The .tsx view pulls in the whole app, so it is read as text.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function openChatSource(): Promise<string> {
  const src = await readFile(
    new URL(
      "../src/features/settings/components/manage-chats-view.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const start = src.indexOf("function openChat(");
  assert.ok(start !== -1, "openChat not found in manage-chats-view.tsx");
  const end = src.indexOf("\n  }", start);
  assert.ok(end !== -1, "openChat body not delimited");
  return src.slice(start, end);
}

test("openChat passes the row's project so ChatPage does not reuse the old one", async () => {
  const body = await openChatSource();
  assert.match(body, /item\.projectId/);
  assert.match(body, /project:/);
});

test("openChat carries the project for both single and compare rows", async () => {
  const body = await openChatSource();
  for (const key of ["thread:", "compare:"]) {
    const at = body.indexOf(key);
    assert.ok(at !== -1, `${key} branch not found`);
    // The spread that adds the project sits in the same object literal.
    assert.match(body.slice(at, at + 120), /\.\.\.project/);
  }
});
