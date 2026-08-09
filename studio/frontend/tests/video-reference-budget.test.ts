// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_H3_REFERENCES,
  hasReferenceCapacity,
} from "../src/features/video/reference-budget.ts";

test("the combined reference budget closes every add slot at twelve items", () => {
  assert.equal(MAX_H3_REFERENCES, 12);
  assert.equal(hasReferenceCapacity(8, 2, 1), true);
  assert.equal(hasReferenceCapacity(9, 2, 1), false);
  assert.equal(hasReferenceCapacity(7, 3, 2), false);
});
