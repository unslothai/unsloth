// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { ModelType, ThreadRecord } from "../src/features/chat/types.ts";
import { resolveComparePaneThreadId } from "../src/features/chat/utils/compare-pane-threads.ts";

const chatPageSource = readFileSync(
  new URL("../src/features/chat/chat-page.tsx", import.meta.url),
  "utf8",
);

const INLINE_THREAD_FIND = /threads\.find\(/;

/** One pair as `listStoredChatThreads({ pairId })` hands it over: updatedAt descending. */
function storedPair(rows: [ModelType, number][]): ThreadRecord[] {
  return rows
    .map(([modelType, updatedAt]) => ({
      id: `${modelType}-thread`,
      title: modelType,
      modelType,
      pairId: "pair-1",
      archived: false,
      createdAt: updatedAt,
      updatedAt,
    }))
    .sort((a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0));
}

test("the LoRA compare rehydrates a pair the generalized compare saved", () => {
  const threads = storedPair([
    ["model1", 20],
    ["model2", 10],
  ]);
  assert.equal(
    resolveComparePaneThreadId(threads, "base", "model1"),
    "model1-thread",
  );
  assert.equal(
    resolveComparePaneThreadId(threads, "lora", "model2"),
    "model2-thread",
  );
});

test("the generalized compare rehydrates a pair the LoRA compare saved", () => {
  const threads = storedPair([
    ["base", 20],
    ["lora", 10],
  ]);
  assert.equal(
    resolveComparePaneThreadId(threads, "model1", "base"),
    "base-thread",
  );
  assert.equal(
    resolveComparePaneThreadId(threads, "model2", "lora"),
    "lora-thread",
  );
});

test("a pair holding both shapes resolves each pane from its own shape, not the freshest row", () => {
  // Reachable: the pre-fix blank LoRA panes wrote base/lora into a model1/model2 pair.
  const threads = storedPair([
    ["model2", 40],
    ["lora", 30],
    ["base", 20],
    ["model1", 10],
  ]);
  assert.equal(
    resolveComparePaneThreadId(threads, "base", "model1"),
    "base-thread",
  );
  assert.equal(
    resolveComparePaneThreadId(threads, "lora", "model2"),
    "lora-thread",
  );
  assert.equal(
    resolveComparePaneThreadId(threads, "model1", "base"),
    "model1-thread",
  );
  assert.equal(
    resolveComparePaneThreadId(threads, "model2", "lora"),
    "model2-thread",
  );
});

test("both compare resolvers go through the helper", () => {
  for (const call of [
    'resolveComparePaneThreadId(threads, "base", "model1")',
    'resolveComparePaneThreadId(threads, "lora", "model2")',
    'resolveComparePaneThreadId(threads, "model1", "base")',
    'resolveComparePaneThreadId(threads, "model2", "lora")',
  ]) {
    assert.ok(chatPageSource.includes(call), call);
  }
  // An inlined find is how both the blanking and the mixing got in.
  assert.doesNotMatch(chatPageSource, INLINE_THREAD_FIND);
});
