// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc, readText } from "./helpers/kit.ts";

const RUNNER = readSrc("features/chat/utils/import-chats.ts");
const ACCEPT = /export const CHAT_IMPORT_ACCEPT = "([^"]+)";/.exec(RUNNER)?.[1] ?? "";

test("every chat import picker offers the same file types", () => {
  assert.match(RUNNER, /input\.accept = CHAT_IMPORT_ACCEPT;/);
  // A literal list here went stale once: the Projects page hid Markdown exports.
  for (const page of ["features/chat/projects-page.tsx", "features/settings/tabs/data-tab.tsx"]) {
    const accepts = [...readSrc(page).matchAll(/\baccept=(\{[^}]*\}|"[^"]*")/g)].map((match) => match[1]);
    assert.deepEqual(accepts, ["{CHAT_IMPORT_ACCEPT}"], page);
  }
});

test("the browser pickers offer what the desktop picker and the importer take", () => {
  const extensions = ACCEPT.split(",").map((entry) => entry.replace(/^\./, ""));
  const native = /const CHAT_IMPORT_EXTENSIONS: &\[&str\] = &\[([^\]]*)\];/.exec(
    readText("../../src-tauri/src/native_file_dialogs.rs"),
  );
  assert.ok(native, "desktop chat import extensions not found");
  assert.deepEqual(extensions, [...native[1].matchAll(/"([^"]+)"/g)].map((match) => match[1]));
  // The importer routes these by extension; the rest stream as JSON records.
  const importer = readSrc("features/chat/utils/chat-import.ts");
  assert.match(importer, /\/\\.\(\?:csv\|md\|markdown\)\$\/i\.test\(source\.name\)/);
  for (const extension of ["csv", "md", "markdown"]) assert.ok(extensions.includes(extension), extension);
});
