// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { createLiveInventoryJobsSelector } =
  await import("../src/features/hub/inventory/use-hub-inventory.ts");
const { createLiveGgufVariantStatesSelector } =
  await import("../src/features/hub/catalog/gguf-live-variant-states.ts");

function failed(over: Record<string, unknown> = {}) {
  return {
    key: "model:org/repo:Q4_K_M",
    kind: "model",
    repoId: "Org/Repo",
    variant: "Q4_K_M",
    state: "error",
    startedAt: 1,
    downloadedBytes: 1024,
    completedBytes: 0,
    expectedBytes: 4096,
    fraction: 0.25,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: "Connection reset",
    completeOnDisk: false,
    ...over,
  } as never;
}

test("a retained failure is a panel row only, not a live inventory or variant row", () => {
  const inventory = createLiveInventoryJobsSelector(false);
  const variants = createLiveGgufVariantStatesSelector("Org/Repo");
  const lingering = { jobs: { a: failed() } };
  assert.equal(inventory(lingering).length, 1);
  assert.equal(variants(lingering).size, 1);
  const retained = { jobs: { a: failed({ retained: true }) } };
  assert.equal(inventory(retained).length, 0);
  assert.equal(variants(retained).size, 0);
});
