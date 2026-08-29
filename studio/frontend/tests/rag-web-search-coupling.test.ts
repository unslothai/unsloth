// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

// #9947: Project Sources must pre-retrieve even when the Search pill is off.
const SOURCE = readFileSync(
  fileURLToPath(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url)),
  "utf8",
);

test("project sources force rag autoinject on in the local chat body", () => {
  assert.match(
    SOURCE,
    /function resolveRagAutoinject\([\s\S]*?if \(projectRagEnabled\) return true;/,
  );
  assert.match(
    SOURCE,
    /autoinject: resolveRagAutoinject\(\s*ragAutoInject,\s*params\.checkpoint,\s*projectRagEnabled,/,
  );
});

test("local enabled_tools still lists search_knowledge_base without web_search", () => {
  const start = SOURCE.indexOf("...(ragEnabled || projectRagEnabled");
  assert.ok(start > 0, "enabled_tools list moved");
  const slice = SOURCE.slice(start, start + 400);
  assert.match(slice, /search_knowledge_base/);
  assert.match(slice, /toolsEnabled \? \["web_search"\]/);
});
