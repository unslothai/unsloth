// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { promptBoundaryIds } from "../src/features/images/lib/prompt-boundaries.ts";

test("marks changes between full prompts without splitting identical neighboring images", () => {
  const prefix = "a detailed image ".repeat(30);
  const images = [
    { id: "a1", prompt: `${prefix}at sunrise` },
    { id: "a2", prompt: `${prefix}at sunrise` },
    { id: "b1", prompt: `${prefix}at sunset` },
    { id: "b2", prompt: `${prefix}at sunset` },
    { id: "a3", prompt: `${prefix}at sunrise` },
  ];
  assert.deepEqual([...promptBoundaryIds(images)], ["b1", "a3"]);
});

test("does not infer a change across missing prompts or before the first image", () => {
  const images = [
    { id: "a", prompt: "first" },
    { id: "unknown", prompt: "" },
    { id: "b", prompt: "second" },
    { id: "c", prompt: "third" },
  ];
  assert.deepEqual([...promptBoundaryIds(images)], ["c"]);
  assert.equal(promptBoundaryIds([]).size, 0);
  assert.equal(promptBoundaryIds(images.slice(0, 1)).size, 0);
});

test("uses current display order as images are reordered or more are loaded", () => {
  const a1 = { id: "a1", prompt: "A" };
  const a2 = { id: "a2", prompt: "A" };
  const b = { id: "b", prompt: "B" };
  assert.deepEqual([...promptBoundaryIds([a1, a2])], []);
  assert.deepEqual([...promptBoundaryIds([a1, a2, b])], ["b"]);
  assert.deepEqual([...promptBoundaryIds([a1, b, a2])], ["b", "a2"]);
});
