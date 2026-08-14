// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { isPartialOffload } from "../src/features/chat/lib/partial-offload.ts";

test("a split load is reported", () => {
  assert.equal(isPartialOffload({ offloaded: 38, total: 60 }), true);
  assert.equal(isPartialOffload({ offloaded: 1, total: 60 }), true);
});

test("a full offload is the normal case and says nothing", () => {
  assert.equal(isPartialOffload({ offloaded: 60, total: 60 }), false);
});

test("no layers on the GPU is the CPU-fallback path, not this one", () => {
  // That case has its own reporting; folding it in here would double-warn.
  assert.equal(isPartialOffload({ offloaded: 0, total: 60 }), false);
});

test("a load that reported no counts says nothing", () => {
  assert.equal(isPartialOffload({}), false);
  assert.equal(isPartialOffload({ offloaded: null, total: null }), false);
  assert.equal(isPartialOffload({ offloaded: 38, total: null }), false);
  assert.equal(isPartialOffload({ offloaded: undefined, total: 60 }), false);
});

test("a nonsense total cannot produce a warning", () => {
  assert.equal(isPartialOffload({ offloaded: 5, total: 0 }), false);
  assert.equal(isPartialOffload({ offloaded: 5, total: -1 }), false);
});
