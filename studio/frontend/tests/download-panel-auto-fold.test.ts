// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Downloads panel folds into its icon while downloads run steadily (#11497), and must open
// again for anything the user needs to see: a new download, a finish, a failure, a cancel.

import assert from "node:assert/strict";
import test from "node:test";

import type { ManagedDownload } from "../src/features/hub/download-manager/download-manager-types.ts";
import {
  aggregateFraction,
  foldSnapshot,
  needsAttention,
  shouldPopOpen,
} from "../src/features/hub/download-manager/download-panel-fold.ts";

function job(
  key: string,
  patch: Partial<ManagedDownload> = {},
): ManagedDownload {
  return {
    key,
    state: "running",
    error: null,
    downloadedBytes: 0,
    expectedBytes: 100,
    fraction: 0,
    ...patch,
  } as ManagedDownload;
}

const jobsOf = (...list: ManagedDownload[]) =>
  Object.fromEntries(list.map((j) => [j.key, j]));

test("only a plain running row is steady", () => {
  assert.equal(needsAttention(job("a")), false);
  for (const state of [
    "complete",
    "error",
    "cancelled",
    "cancelling",
  ] as const) {
    assert.equal(needsAttention(job("a", { state })), true, state);
  }
  assert.equal(needsAttention(job("a", { error: "stalled" })), true);
});

test("a new download pops the panel open", () => {
  const before = foldSnapshot(["a"], jobsOf(job("a")), 0);
  const after = foldSnapshot(["a", "b"], jobsOf(job("a"), job("b")), 0);
  assert.equal(shouldPopOpen(before, after), true);
  assert.equal(
    shouldPopOpen(before, foldSnapshot(["a"], jobsOf(job("a")), 1)),
    true,
  );
});

test("progress alone, or a queued entry starting, keeps it folded", () => {
  const before = foldSnapshot(["a"], jobsOf(job("a")), 2);
  const ticked = foldSnapshot(
    ["a"],
    jobsOf(job("a", { downloadedBytes: 50 })),
    2,
  );
  assert.equal(shouldPopOpen(before, ticked), false);
  const started = foldSnapshot(["a", "b"], jobsOf(job("a"), job("b")), 1);
  assert.equal(shouldPopOpen(before, started), false);
});

test("a finish, failure or cancel pops it open once", () => {
  const before = foldSnapshot(["a", "b"], jobsOf(job("a"), job("b")), 0);
  for (const state of ["complete", "error", "cancelling"] as const) {
    const after = foldSnapshot(
      ["a", "b"],
      jobsOf(job("a", { state }), job("b")),
      0,
    );
    assert.equal(shouldPopOpen(before, after), true, state);
    assert.equal(shouldPopOpen(after, after), false, `${state} twice`);
  }
});

test("a warning that turns into a failure pops it open again", () => {
  const warned = foldSnapshot(["a"], jobsOf(job("a", { error: "stalled" })), 0);
  const failed = foldSnapshot(
    ["a"],
    jobsOf(job("a", { state: "error", error: "stalled" })),
    0,
  );
  assert.equal(shouldPopOpen(warned, failed), true);
  const cancelling = foldSnapshot(
    ["a"],
    jobsOf(job("a", { state: "cancelling" })),
    0,
  );
  const cancelled = foldSnapshot(
    ["a"],
    jobsOf(job("a", { state: "cancelled" })),
    0,
  );
  assert.equal(shouldPopOpen(cancelling, cancelled), true);
});

test("retrying a failed download under the same key pops it open", () => {
  const failed = foldSnapshot(
    ["a"],
    jobsOf(job("a", { state: "error", startedAt: 1 })),
    0,
  );
  const retried = foldSnapshot(["a"], jobsOf(job("a", { startedAt: 2 })), 0);
  assert.equal(shouldPopOpen(failed, retried), true);
});

test("a finished row leaving does not pop it open", () => {
  const before = foldSnapshot(
    ["a", "b"],
    jobsOf(job("a", { state: "complete" }), job("b")),
    0,
  );
  const after = foldSnapshot(["b"], jobsOf(job("b")), 0);
  assert.equal(shouldPopOpen(before, after), false);
});

test("the icon ring shows combined progress of running downloads", () => {
  assert.equal(
    aggregateFraction([
      job("a", { downloadedBytes: 50, expectedBytes: 100 }),
      job("b", { downloadedBytes: 0, expectedBytes: 300 }),
      job("c", { state: "complete", downloadedBytes: 900, expectedBytes: 900 }),
    ]),
    0.125,
  );
  assert.equal(aggregateFraction([job("a", { expectedBytes: 0 })]), null);
  assert.equal(aggregateFraction([]), null);
});
