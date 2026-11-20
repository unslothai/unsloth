// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { adjacentGalleryId } from "../src/features/images/gallery-keyboard-navigation.ts";

const ids = ["first", "second", "third"];

test("moves to the previous and next visible gallery tile", () => {
  assert.equal(adjacentGalleryId(ids, "second", "previous"), "first");
  assert.equal(adjacentGalleryId(ids, "second", "next"), "third");
});

test("stops at either edge and ignores an unknown tile", () => {
  assert.equal(adjacentGalleryId(ids, "first", "previous"), null);
  assert.equal(adjacentGalleryId(ids, "third", "next"), null);
  assert.equal(adjacentGalleryId(ids, "missing", "next"), null);
});
