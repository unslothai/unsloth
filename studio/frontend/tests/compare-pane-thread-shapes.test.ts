// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { ModelType, ThreadRecord } from "../src/features/chat/types.ts";
import {
  compareVariantForPair,
  resolveComparePaneThreadIds,
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
  // Loading model 2, a LoRA export, is what used to swap the component mid-session.
  assert.equal(compareVariantForPair(threads, true), "general");
  assert.equal(compareVariantForPair(threads, false), "general");
});

test("a LoRA pair follows the loaded checkpoint, as it always has", () => {
  const threads = storedPair([
    ["base", 20],
    ["lora", 10],
  ]);
  assert.equal(compareVariantForPair(threads, true), "lora");
  // No LoRA loaded: the adapter toggle would answer both panes from the same
  // weights, so the generalized panes take it and show their model selectors.
  assert.equal(compareVariantForPair(threads, false), "general");
});

test("a pair with no threads yet follows the loaded checkpoint", () => {
  assert.equal(compareVariantForPair([], true), "lora");
  assert.equal(compareVariantForPair([], false), "general");
});

test("the generalized panes recover a LoRA pair", () => {
  const threads = storedPair([
    ["base", 20],
    ["lora", 10],
  ]);
  assert.deepEqual(resolveComparePaneThreadIds(threads), {
    shape: "lora",
    first: "base-thread",
    second: "lora-thread",
  });
});

test("a pair holding both shapes resolves from its own shape, not the freshest row", () => {
  // Reachable: the pre-fix blank LoRA panes wrote base/lora into a model1/model2 pair.
  const threads = storedPair([
    ["model2", 40],
    ["lora", 30],
    ["base", 20],
    ["model1", 10],
  ]);
  assert.deepEqual(resolveComparePaneThreadIds(threads), {
    shape: "general",
    first: "model1-thread",
    second: "model2-thread",
  });
  assert.equal(compareVariantForPair(threads, true), "general");
  assert.equal(compareVariantForPair(threads, false), "general");
});

test("a complete LoRA shape wins over an interrupted generalized write", () => {
  const threads = storedPair([
    ["model1", 30],
    ["base", 20],
    ["lora", 10],
  ]);
  assert.deepEqual(resolveComparePaneThreadIds(threads), {
    shape: "lora",
    first: "base-thread",
    second: "lora-thread",
  });
  assert.equal(compareVariantForPair(threads, true), "lora");
});

test("two partial shapes never splice into a complete comparison", () => {
  const threads = storedPair([
    ["model1", 20],
    ["lora", 10],
  ]);
  assert.deepEqual(resolveComparePaneThreadIds(threads), {
    shape: "general",
    first: "model1-thread",
    second: undefined,
  });
});

test("one surviving legacy pane stays in its persisted shape", () => {
  assert.deepEqual(resolveComparePaneThreadIds(storedPair([["lora", 10]])), {
    shape: "lora",
    first: undefined,
    second: "lora-thread",
  });
  assert.deepEqual(resolveComparePaneThreadIds(storedPair([["model2", 10]])), {
    shape: "general",
    first: undefined,
    second: "model2-thread",
  });
});

test("the freshest row wins only within the selected shape", () => {
  const [model2] = storedPair([["model2", 10]]);
  const threads: ThreadRecord[] = [
    {
      ...model2,
      id: "fresh-model1",
      modelType: "model1",
      updatedAt: 30,
    },
    {
      ...model2,
      id: "old-model1",
      modelType: "model1",
      updatedAt: 20,
    },
    model2,
  ];
  assert.deepEqual(resolveComparePaneThreadIds(threads), {
    shape: "general",
    first: "fresh-model1",
    second: "model2-thread",
  });
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
  assert.ok(
    chatPageSource.includes(
      "if (checkpointIsLora === null || stored?.pairId === pairId) return;",
    ),
  );
  assert.ok(chatPageSource.includes("s.residentCheckpoint === undefined"));
  assert.ok(chatPageSource.includes("activeModel.isLora"));
  assert.ok(chatPageSource.includes("  return stored.variant;"));
});
