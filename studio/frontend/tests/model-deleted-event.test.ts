// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

(globalThis as { window?: EventTarget }).window ??= new EventTarget();
const { notifyModelDeleted, subscribeModelDeleted } = await import(
  "../src/lib/model-lifecycle-events.ts"
);

test("a cached-model delete is announced with its repo id", () => {
  const seen: string[] = [];
  const unsubscribe = subscribeModelDeleted((repoId) => seen.push(repoId));
  notifyModelDeleted("unsloth/orpheus-3b-0.1-ft-GGUF");
  unsubscribe();
  notifyModelDeleted("unsloth/after-unsubscribe");
  assert.deepEqual(seen, ["unsloth/orpheus-3b-0.1-ft-GGUF"]);
});

test("only a delete the server accepted is announced", () => {
  const api = readSrc("features/hub/inventory/api.ts");
  assert.match(
    api,
    /await throwIfNotOk\(response\);\s*discardDeletedModelInventoryHints\(repoId, variant\);\s*bumpInventoryVersion\(\);\s*notifyModelDeleted\(repoId\);/,
  );
});
