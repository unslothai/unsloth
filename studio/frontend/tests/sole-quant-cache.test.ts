// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SoleQuantEntry,
  type SoleQuantTarget,
  partitionSoleQuants,
  soleQuantKey,
} from "../src/features/model-picker/components/model-selector/sole-quant-cache.ts";

const A = "unsloth/Qwen3-8B-GGUF";
const B = "unsloth/Llama-3.1-8B-Instruct-GGUF";

/** Two listed repos, each at its own cache version. */
const targetsAt = (versionA: string, versionB: string): SoleQuantTarget[] => [
  { repoId: A, localSource: null, key: soleQuantKey(versionA, null) },
  { repoId: B, localSource: null, key: soleQuantKey(versionB, null) },
];

const settled = (
  targets: SoleQuantTarget[],
  quants: (string | null)[],
): Map<string, SoleQuantEntry<string>> =>
  new Map(
    targets.map((target, index) => [
      target.repoId,
      { key: target.key, quant: quants[index] ?? null },
    ]),
  );

test("resolved repos are rows, unread repos are pending", () => {
  const targets = targetsAt("1:0", "1:0");
  const { quants, pending, stale } = partitionSoleQuants(
    targets,
    settled([targets[0]], ["Q4_K_M"]),
    { enabled: true },
  );
  assert.deepEqual([...quants], [[A, "Q4_K_M"]]);
  assert.deepEqual([...pending], [B]);
  assert.deepEqual(
    stale.map((target) => target.repoId),
    [B],
  );
});

test("one repo's invalidation leaves the other repo's row alone", () => {
  const before = targetsAt("1:0", "1:0");
  const entries = settled(before, ["Q4_K_M", "Q8_0"]);
  // B is downloaded into, so only B's version moves.
  const after = targetsAt("1:0", "1:7");

  const { quants, pending, stale } = partitionSoleQuants(after, entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], [[A, "Q4_K_M"]]);
  assert.deepEqual([...pending], [B]);
  assert.deepEqual(
    stale.map((target) => target.repoId),
    [B],
  );
});

test("a repo pointed at another directory is re-read", () => {
  const targets = targetsAt("1:0", "1:0");
  const entries = settled(targets, ["Q4_K_M", "Q8_0"]);
  const moved: SoleQuantTarget[] = [
    {
      repoId: A,
      localSource: "/other/cache",
      key: soleQuantKey("1:0", "/other/cache"),
    },
    targets[1],
  ];

  const { quants, pending } = partitionSoleQuants(moved, entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], [[B, "Q8_0"]]);
  assert.deepEqual([...pending], [A]);
});

test("a repo with no single quant is settled, not pending", () => {
  const targets = targetsAt("1:0", "1:0");
  // A holds two quants, or could not be read: either way no row, no re-read.
  const { quants, pending, stale } = partitionSoleQuants(
    targets,
    settled(targets, [null, "Q8_0"]),
    { enabled: true },
  );
  assert.deepEqual([...quants], [[B, "Q8_0"]]);
  assert.deepEqual([...pending], []);
  assert.deepEqual(stale, []);
});

test("disabled reports nothing and asks for nothing", () => {
  const targets = targetsAt("1:0", "1:0");
  const { quants, pending, stale } = partitionSoleQuants(
    targets,
    settled(targets, ["Q4_K_M", "Q8_0"]),
    { enabled: false },
  );
  assert.deepEqual([...quants], []);
  assert.deepEqual([...pending], []);
  assert.deepEqual(stale, []);
});

test("a global invalidation moves every repo's key", () => {
  const before = targetsAt("1:0", "1:0");
  const entries = settled(before, ["Q4_K_M", "Q8_0"]);
  const after = targetsAt("2:0", "2:0");
  const { quants, pending } = partitionSoleQuants(after, entries, {
    enabled: true,
  });
  assert.deepEqual([...quants], []);
  assert.deepEqual([...pending], [A, B]);
});
