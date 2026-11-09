// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (rel: string) => readFileSync(path.join(HERE, "..", rel), "utf8");
const ADAPTER = read("src/features/chat/api/chat-adapter.ts");
const RUNTIME = read("src/features/chat/hooks/use-chat-model-runtime.ts");

test("a startup auto-load sends the remembered split for the GPUs it resolved", () => {
  assert.match(
    ADAPTER,
    /reconcileTensorSplit\(\s*config\.tensorSplit,\s*config\.selectedGpuIds,\s*effectiveGpuIds,\s*\)/,
  );
  assert.match(ADAPTER, /tensor_split: effectiveTensorSplit \?\? undefined,/);
  assert.doesNotMatch(ADAPTER, /split ratio is never remembered/);
});

test("a picker load sanitizes its stored extra args before sending them", () => {
  const start = RUNTIME.indexOf(
    "let loadLlamaExtraArgs = pendingLoadConfig?.llamaExtraArgs;",
  );
  assert.ok(start > 0);
  const block = RUNTIME.slice(start, start + 1400);
  assert.match(block, /isGguf &&\s*!targetIsDiffusion &&/);
  assert.match(
    block,
    /loadLlamaExtraArgs != null &&\s*loadLlamaExtraArgs\.length > 0/,
  );
  assert.match(block, /await loadManagedLlamaFlags\(\)/);
  assert.match(
    block,
    /sanitizeStoredExtraArgs\(\s*loadLlamaExtraArgs,\s*managed\?\.managed \?\? new Set<string>\(\),\s*\{\s*maxBytes: managed\?\.maxBytes,\s*windowsCommandBudget: managed\?\.windowsCommandBudget,/,
  );
  assert.match(block, /loadLlamaExtraArgs = cleaned;/);
});
