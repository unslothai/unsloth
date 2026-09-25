// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { pinDropAnchor, usePinnedModelsStore } = await import(
  "../src/features/model-picker/components/model-selector/pinned-models.ts"
);

type Edge = "top" | "bottom";

/** Lands `from` on `edge` of `target` the way the picker's drop does. */
function drop(pinned: string[], from: string, target: string, edge: Edge) {
  usePinnedModelsStore.setState({ pinned });
  const store = usePinnedModelsStore.getState();
  const anchor = pinDropAnchor(store.pinned, from, target, edge);
  if (anchor) {
    store.beginPinnedDrag();
    store.movePinned(from, anchor);
    store.endPinnedDrag(true);
  }
  return usePinnedModelsStore.getState().pinned;
}

test("a row lands above or below its target, moving either way", () => {
  const list = ["a", "b", "c", "d"];
  assert.deepEqual(drop(list, "a", "c", "top"), ["b", "a", "c", "d"]);
  assert.deepEqual(drop(list, "a", "c", "bottom"), ["b", "c", "a", "d"]);
  assert.deepEqual(drop(list, "d", "b", "top"), ["a", "d", "b", "c"]);
  assert.deepEqual(drop(list, "d", "b", "bottom"), ["a", "b", "d", "c"]);
  assert.deepEqual(drop(list, "a", "d", "bottom"), ["b", "c", "d", "a"]);
  assert.deepEqual(drop(list, "d", "a", "top"), ["d", "a", "b", "c"]);
});

test("a drop into the slot a row already holds is a no-op", () => {
  const list = ["a", "b", "c"];
  assert.equal(pinDropAnchor(list, "b", "b", "top"), null);
  assert.equal(pinDropAnchor(list, "b", "c", "top"), null);
  assert.equal(pinDropAnchor(list, "b", "a", "bottom"), null);
  assert.equal(pinDropAnchor(list, "b", "missing", "top"), null);
  assert.equal(pinDropAnchor(list, "missing", "a", "top"), null);
});

test("pins that are not drawn keep their place relative to the target", () => {
  // "h" is pinned but filtered out of view; dropping "c" above "a" still lands it right above "a".
  assert.deepEqual(drop(["a", "h", "b", "c"], "c", "a", "top"), [
    "c",
    "a",
    "h",
    "b",
  ]);
  assert.deepEqual(drop(["a", "h", "b", "c"], "a", "b", "bottom"), [
    "h",
    "b",
    "a",
    "c",
  ]);
});
