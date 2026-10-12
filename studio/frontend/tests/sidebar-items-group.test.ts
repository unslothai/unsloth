// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Library reads models and last-modified times off the sidebar rows, not a second thread read.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./sidebar-items-resolver.mjs", import.meta.url);

const { groupThreads } = await import("../src/features/chat/hooks/use-chat-sidebar-items.ts");

const base = { title: "T", modelType: "base", archived: false, createdAt: 1 } as const;

test("a compare row carries each pane's model once and its latest edit", () => {
  const [row] = groupThreads([
    { ...base, id: "a", pairId: "pair", modelId: "m1", modifiedAt: 5 },
    { ...base, id: "b", pairId: "pair", modelId: "m2", modifiedAt: 9 },
    { ...base, id: "c", pairId: "pair", modelId: "m1" },
  ]);
  assert.deepEqual(row?.modelIds, ["m1", "m2"]);
  assert.equal(row?.modifiedAt, 9);
});

test("a never edited chat has no modified time", () => {
  const [row] = groupThreads([{ ...base, id: "a", modelId: "m1", modifiedAt: null }]);
  assert.deepEqual(row?.modelIds, ["m1"]);
  assert.equal("modifiedAt" in (row ?? {}), false);
});
