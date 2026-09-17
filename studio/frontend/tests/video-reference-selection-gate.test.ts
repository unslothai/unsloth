// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createReferenceSelectionGate } from "../src/features/video/reference-budget.ts";

test("the newest read is the only one allowed to write back", () => {
  const gate = createReferenceSelectionGate();
  const first = gate.begin();
  assert.equal(first.isCurrent(), true);

  const second = gate.begin();
  assert.equal(first.isCurrent(), false);
  assert.equal(second.isCurrent(), true);

  assert.equal(first.isCurrent(), false);
});

test("a value this picker did not set retires the read in flight", () => {
  const gate = createReferenceSelectionGate();
  const claim = gate.begin();

  gate.invalidate();
  assert.equal(claim.isCurrent(), false);

  assert.equal(gate.begin().isCurrent(), true);
});

test("unmounting revokes a read, and a remount does not revive it", () => {
  const gate = createReferenceSelectionGate();
  const unmount = gate.mount();
  const claim = gate.begin();
  assert.equal(claim.isCurrent(), true);

  unmount();
  assert.equal(claim.isCurrent(), false);

  gate.mount();
  assert.equal(claim.isCurrent(), false);
  assert.equal(gate.begin().isCurrent(), true);
});
