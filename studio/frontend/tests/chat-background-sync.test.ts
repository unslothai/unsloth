// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { planBackgroundPull } from "../src/features/chat-background/sync-plan.ts";

const image = { sha256: "abc", type: "image/png", size: 10 };
const clean = { updatedAt: 1, sha256: "abc", dirty: false };

test("a new browser or origin takes the account's background", () => {
  const remote = { saved: true, updatedAt: 5, image };
  assert.equal(planBackgroundPull(remote, { updatedAt: null, sha256: null, dirty: false }, false), "download");
});

test("the first browser after the update uploads its existing background", () => {
  assert.equal(planBackgroundPull({ saved: false, updatedAt: null, image: null }, clean, true), "push");
});

test("unsent local edits are sent instead of being overwritten", () => {
  const remote = { saved: true, updatedAt: 9, image };
  assert.equal(planBackgroundPull(remote, { ...clean, dirty: true }, true), "push");
});

test("an unchanged account copy leaves this browser alone", () => {
  assert.equal(planBackgroundPull({ saved: true, updatedAt: 1, image }, clean, true), "keep");
});

test("adjustments from another browser reuse the image already here", () => {
  assert.equal(planBackgroundPull({ saved: true, updatedAt: 7, image }, clean, true), "adjust");
  const other = { ...image, sha256: "def" };
  assert.equal(planBackgroundPull({ saved: true, updatedAt: 7, image: other }, clean, true), "download");
});

test("a background removed elsewhere is removed here instead of uploaded back", () => {
  assert.equal(planBackgroundPull({ saved: true, updatedAt: 8, image: null }, clean, true), "clear");
});
