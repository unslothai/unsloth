// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_LIBRARY_SETTINGS,
  compareBySort,
  includedBySettings,
} from "../src/features/library/settings-store.ts";

test("hidden sources drop out, Library uploads always stay", () => {
  const settings = { ...DEFAULT_LIBRARY_SETTINGS, showChatAttachments: false, showFineTunes: false };
  assert.equal(includedBySettings("upload:abc", settings), true);
  assert.equal(includedBySettings("attachment:m:a", settings), false);
  assert.equal(includedBySettings("model:training:/runs/x", settings), false);
  assert.equal(includedBySettings("sandbox:t:out.csv", settings), true);
  assert.equal(includedBySettings("image:1", { ...settings, showGeneratedMedia: false }), false);
});

test("sort orders by recency, name and size", () => {
  const items = [
    { name: "b 10", updatedAt: 2, sizeBytes: 5 },
    { name: "b 9", updatedAt: 3, sizeBytes: null },
    { name: "a", updatedAt: 1, sizeBytes: 50 },
  ];
  const names = (sort: Parameters<typeof compareBySort>[0]) =>
    [...items].sort(compareBySort(sort)).map((item) => item.name);
  assert.deepEqual(names("recent"), ["b 9", "b 10", "a"]);
  assert.deepEqual(names("oldest"), ["a", "b 10", "b 9"]);
  assert.deepEqual(names("name"), ["a", "b 9", "b 10"]);
  assert.deepEqual(names("size"), ["a", "b 10", "b 9"]);
});
