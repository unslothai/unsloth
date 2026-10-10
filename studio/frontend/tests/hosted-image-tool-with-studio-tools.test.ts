// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The body is built deep in the adapter's run closure, so this reads the branch out of the source.
const SOURCE = readSrc("features/chat/api/chat-adapter.ts");

function studioToolsBranch(): string {
  const start = SOURCE.indexOf('...(ragEnabled || projectRagEnabled\n');
  assert.ok(start > 0, "the Unsloth-tools enabled_tools list moved");
  const end = SOURCE.indexOf("mcp_enabled:", start);
  assert.ok(end > start, "the Unsloth-tools branch moved");
  return SOURCE.slice(start, end);
}

// Images and Fetch have no local implementation, so the Unsloth branch must still request them.
test("the Unsloth-tools branch still asks for the hosted tools Unsloth cannot run", () => {
  const branch = studioToolsBranch();

  assert.match(branch, /imageGenerationEnabledForThisTurn/);
  assert.match(branch, /"image_generation"/);
  assert.match(branch, /webFetchEnabledForThisTurn/);
  assert.match(branch, /"web_fetch"/);
});

// Code is not in this set: code-tool-placement.ts decides between provider sandbox and local.
test("it does not ask the provider for the tools Unsloth is running locally", () => {
  const branch = studioToolsBranch();

  assert.match(branch, /toolsEnabled \? \["web_search"\]/);
  assert.doesNotMatch(branch, /webSearchEnabledForThisTurn/);
  assert.doesNotMatch(branch, /codeExecEnabledForThisTurn/);
});
