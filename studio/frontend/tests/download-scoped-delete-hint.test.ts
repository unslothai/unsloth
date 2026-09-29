// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A scoped `@variant` download publishes a `model` hint, so deleting one while clearing only `gguf` leaves the row to come back until the hint expires.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  discardPendingInventoryHint,
  rememberCompletedInventoryHints,
  useInventoryHintStore,
} = await import("../src/features/hub/inventory/inventory-hint-store.ts");
const { discardDeletedModelInventoryHints } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

function pendingHas(kind: "model" | "gguf", repoId: string): boolean {
  return useInventoryHintStore
    .getState()
    .pending[kind].has(repoId.toLowerCase());
}

function seed(kind: "model" | "gguf", repoId: string) {
  rememberCompletedInventoryHints([{ kind, repoId }]);
  assert.equal(pendingHas(kind, repoId), true, "precondition: hint was stored");
}

test("deleting a scoped model download clears its model hint", () => {
  const repoId = "org/diffusion-model";
  seed("model", repoId);

  discardDeletedModelInventoryHints(repoId, "@diffusion");

  assert.equal(
    pendingHas("model", repoId),
    false,
    "the deleted scoped row keeps its hint and reappears until it expires",
  );
});

test("deleting a quant download still clears its gguf hint", () => {
  const repoId = "org/quant-model";
  seed("gguf", repoId);

  discardDeletedModelInventoryHints(repoId, "Q4_K_M");

  assert.equal(pendingHas("gguf", repoId), false);
});

test("deleting a whole repo clears both hint kinds", () => {
  const repoId = "org/both-model";
  seed("gguf", repoId);
  seed("model", repoId);

  discardDeletedModelInventoryHints(repoId);

  assert.equal(pendingHas("gguf", repoId), false);
  assert.equal(pendingHas("model", repoId), false);
});

test("deleting one variant leaves an unrelated repo's hint alone", () => {
  const kept = "org/untouched";
  seed("model", kept);
  discardPendingInventoryHint("model", "org/somewhere-else");

  discardDeletedModelInventoryHints("org/other-model", "@diffusion");

  assert.equal(pendingHas("model", kept), true);
});
