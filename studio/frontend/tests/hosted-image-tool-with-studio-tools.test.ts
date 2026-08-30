// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

// The request body is built deep inside the adapter's run closure, which needs
// a live runtime, a provider store and an encryption key to reach. The property
// that regressed is structural though: which names the Unsloth-tools branch puts
// in enabled_tools. So this reads that branch out of the source, the same way
// the backend's route tests read the gate out of routes/inference.py.
const SOURCE = readFileSync(
  fileURLToPath(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url)),
  "utf8",
);

// The branch taken when the provider runs Unsloth's tools. Bounded by the
// hosted-only branch that follows it, so the two cannot be confused.
function studioToolsBranch(): string {
  const start = SOURCE.indexOf('...(ragEnabled || projectRagEnabled\n');
  assert.ok(start > 0, "the Unsloth-tools enabled_tools list moved");
  const end = SOURCE.indexOf("mcp_enabled:", start);
  assert.ok(end > start, "the Unsloth-tools branch moved");
  return SOURCE.slice(start, end);
}

// Images and Fetch have their own toggles and no local implementation. Before
// this PR an OpenAI or Gemini connection never took the Unsloth branch, so the
// hosted branch below always carried them; now that Search, Code, MCP or a
// project's automatic RAG selects the Unsloth branch instead, a list of purely
// local names would leave a lit Images pill out of the request entirely.
test("the Unsloth-tools branch still asks for the hosted tools Unsloth cannot run", () => {
  const branch = studioToolsBranch();

  assert.match(branch, /imageGenerationEnabledForThisTurn/);
  assert.match(branch, /"image_generation"/);
  assert.match(branch, /webFetchEnabledForThisTurn/);
  assert.match(branch, /"web_fetch"/);
});

// The other half of the same rule: Unsloth runs search itself on this path, so
// asking the provider for its own would run both and bill for theirs. Code is
// NOT in that set -- `code_execution` is the provider's sandbox and
// python/terminal are this machine, so the two are never both requested and
// which one a turn asks for is decided in code-tool-placement.ts (see
// tests/code-tool-placement.test.ts).
test("it does not ask the provider for the tools Unsloth is running locally", () => {
  const branch = studioToolsBranch();

  assert.match(branch, /toolsEnabled \? \["web_search"\]/);
  assert.doesNotMatch(branch, /webSearchEnabledForThisTurn/);
  assert.doesNotMatch(branch, /codeExecEnabledForThisTurn/);
});
