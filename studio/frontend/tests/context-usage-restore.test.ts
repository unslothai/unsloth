// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { restorableContextUsage } from "../src/features/chat/utils/context-usage-restore.ts";

const LAST_ASSISTANT = /lastRunMessage\?\.role === "assistant"/;
const EARLY_RETURN = /published = true;\s*return;/;

const saved = {
  contextUsage: {
    promptTokens: 2336,
    completionTokens: 490,
    totalTokens: 2826,
    cachedTokens: 0,
    modelId: "local/model",
  },
};

test("saved usage waits for local-model hydration, then restores exactly", () => {
  // Local checkpoints intentionally start empty after a restart. The history load must not
  // misattribute the snapshot, but the post-status path must retry it instead of recounting the
  // whole visible transcript.
  assert.equal(restorableContextUsage(saved, "", 32_000), null);
  assert.deepEqual(
    restorableContextUsage(saved, "local/model", 32_000),
    saved.contextUsage,
  );
});

test("saved usage is scoped to its model and active context window", () => {
  assert.equal(restorableContextUsage(saved, "another/model", 32_000), null);
  assert.equal(restorableContextUsage(saved, "local/model", 2_000), null);
});

test("legacy unscoped usage is trusted only for a known local window", () => {
  const legacy = {
    contextUsage: {
      promptTokens: 900,
      completionTokens: 100,
      totalTokens: 1000,
      cachedTokens: 0,
    },
  };
  assert.deepEqual(
    restorableContextUsage(legacy, "local/model", 32_000),
    legacy.contextUsage,
  );
  assert.equal(restorableContextUsage(legacy, "external/model", null), null);
});

test("post-load recount restores the last completed turn before tokenizing history", () => {
  const source = readFileSync(
    new URL(
      "../src/features/chat/utils/refresh-context-usage.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const restoreAt = source.indexOf("restorableContextUsage(");
  const countAt = source.indexOf("await countChatInputTokens(");
  assert.ok(restoreAt >= 0, "refresh must inspect persisted context usage");
  assert.ok(
    countAt > restoreAt,
    "exact persisted usage must win over a full recount",
  );
  assert.match(source, LAST_ASSISTANT);
  assert.match(source.slice(restoreAt, countAt), EARLY_RETURN);
});
