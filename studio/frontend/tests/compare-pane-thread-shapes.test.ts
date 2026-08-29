// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { ModelType, ThreadRecord } from "../src/features/chat/types.ts";
import {
  compareVariantForPair,
  pairHasGeneralThreads,
  resolveComparePaneThreadId,
} from "../src/features/chat/utils/compare-pane-threads.ts";

const chatPageSource = readFileSync(
  new URL("../src/features/chat/chat-page.tsx", import.meta.url),
  "utf8",
);

const LORA_PANE_SOURCE = chatPageSource.slice(
  chatPageSource.indexOf("const LoraCompareContent"),
  chatPageSource.indexOf("function GeneralCompareHeader"),
);

const ADOPTS_GENERALIZED = /modelType === "model[12]"/;
const CHECKPOINT_PICKS_RENDERER = /const isLoraCompare = useIsLoraCompare\(\)/;

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

test("a generalized pair never reaches the adapter-toggle path", () => {
  const threads = storedPair([
    ["model1", 20],
    ["model2", 10],
  ]);
  assert.equal(pairHasGeneralThreads(threads), true);
  // Loading model 2, a LoRA export, is what used to swap the component mid-session.
  assert.equal(compareVariantForPair(true, true), "general");
  assert.equal(compareVariantForPair(true, false), "general");
});

test("a LoRA pair follows the loaded checkpoint, as it always has", () => {
  const threads = storedPair([
    ["base", 20],
    ["lora", 10],
  ]);
  assert.equal(pairHasGeneralThreads(threads), false);
  assert.equal(compareVariantForPair(false, true), "lora");
  // No LoRA loaded: the adapter toggle would answer both panes from the same
  // weights, so the generalized panes take it and show their model selectors.
  assert.equal(compareVariantForPair(false, false), "general");
});

test("a pair with no threads yet follows the loaded checkpoint", () => {
  assert.equal(pairHasGeneralThreads([]), false);
  assert.equal(compareVariantForPair(false, true), "lora");
  assert.equal(compareVariantForPair(false, false), "general");
});

test("the generalized panes recover a LoRA pair", () => {
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

test("a pair holding both shapes resolves from its own shape, not the freshest row", () => {
  // Reachable: the pre-fix blank LoRA panes wrote base/lora into a model1/model2 pair.
  const threads = storedPair([
    ["model2", 40],
    ["lora", 30],
    ["base", 20],
    ["model1", 10],
  ]);
  assert.equal(
    resolveComparePaneThreadId(threads, "model1", "base"),
    "model1-thread",
  );
  assert.equal(
    resolveComparePaneThreadId(threads, "model2", "lora"),
    "model2-thread",
  );
});

test("the LoRA panes adopt no generalized thread", () => {
  assert.ok(
    LORA_PANE_SOURCE.includes('threads.find((t) => t.modelType === "base")'),
  );
  assert.ok(
    LORA_PANE_SOURCE.includes('threads.find((t) => t.modelType === "lora")'),
  );
  // Adopting one is what mislabelled it and wrote adapter answers into its history.
  assert.doesNotMatch(LORA_PANE_SOURCE, ADOPTS_GENERALIZED);
});

test("the renderer is chosen from the pair, not straight from the checkpoint", () => {
  assert.ok(
    chatPageSource.includes(
      "const compareVariant = useCompareVariant(pairId);",
    ),
  );
  assert.ok(
    chatPageSource.includes("if (compareVariant === null) return <></>;"),
  );
  assert.ok(chatPageSource.includes('return compareVariant === "lora" ? ('));
  assert.doesNotMatch(chatPageSource, CHECKPOINT_PICKS_RENDERER);
  // Structural because the failure is a dropped dependency, not a wrong return: a
  // generalized pair's first send writes its rows and loads model 2 in one go, so
  // without the re-read the pair is still "empty" when the flip arrives and the
  // adapter path takes it mid-run.
  assert.ok(chatPageSource.includes("}, [pairId, checkpointIsLora]);"));
});
