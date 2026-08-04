// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Rows drop the "unsloth/" prefix but keep every other owner, which is what
// tells the two apart.

import assert from "node:assert/strict";
import test from "node:test";
import {
  isUnslothOwner,
  splitRepoLabel,
} from "../src/features/model-picker/components/model-selector/row-meta.ts";

test("unsloth is the owner whose prefix rows hide", () => {
  assert.equal(
    isUnslothOwner(splitRepoLabel("unsloth/gemma-4-26b").owner),
    true,
  );
  // Case as the Hub listing returns it.
  assert.equal(isUnslothOwner("Unsloth"), true);
});

test("other owners keep their prefix", () => {
  assert.equal(isUnslothOwner(splitRepoLabel("Qwen/Qwen3-8B").owner), false);
  assert.equal(isUnslothOwner("unsloth-community"), false);
  // A bare model name has no owner to hide.
  assert.equal(isUnslothOwner(splitRepoLabel("gemma-4-26b").owner), false);
});
