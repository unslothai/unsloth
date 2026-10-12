// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  CompareRunOwnership,
  isCompareCancellation,
  throwIfCompareCancelled,
} from "../src/features/chat/compare-run-ownership.ts";

test("late cleanup from a superseded run cannot release the current run", () => {
  const ownership = new CompareRunOwnership<string>();
  const first = ownership.begin();
  const second = ownership.begin();

  assert.equal(first.controller.signal.aborted, true);
  assert.equal(ownership.release(first), false);
  assert.equal(ownership.current(), second);
  assert.equal(ownership.release(second), true);
  assert.equal(ownership.current(), null);
});

test("superseded runs cannot replace the current backend load target", () => {
  const ownership = new CompareRunOwnership<string>();
  const first = ownership.begin();
  assert.equal(ownership.setLoadingModel(first, "model-1"), true);

  const second = ownership.begin();
  assert.equal(ownership.setLoadingModel(first, null), false);
  assert.equal(ownership.setLoadingModel(second, "model-2"), true);
  assert.equal(second.loadingModel, "model-2");
});

test("cancelCurrent aborts the owner without releasing it early", () => {
  const ownership = new CompareRunOwnership<string>();
  const run = ownership.begin();

  assert.equal(ownership.cancelCurrent(), run);
  assert.equal(run.controller.signal.aborted, true);
  assert.equal(ownership.current(), run);
  assert.throws(
    () => throwIfCompareCancelled(run.controller.signal),
    (error) =>
      error instanceof DOMException &&
      error.name === "AbortError" &&
      isCompareCancellation(error, run.controller.signal),
  );
});

test("cleanup promises are attached only by the current owner", async () => {
  const ownership = new CompareRunOwnership<string>();
  const first = ownership.begin();
  const second = ownership.begin();
  const firstCleanup = Promise.resolve();
  const secondCleanup = Promise.resolve();

  assert.equal(ownership.setCleanup(first, firstCleanup), false);
  assert.equal(ownership.setCleanup(second, secondCleanup), true);
  assert.equal(second.cleanup, secondCleanup);
  await second.cleanup;
});
