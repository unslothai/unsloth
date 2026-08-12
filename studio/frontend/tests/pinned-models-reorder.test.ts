// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { usePinnedModelsStore } = await import(
  "../src/features/model-picker/components/model-selector/pinned-models.ts"
);

function setPinned(pinned: string[]) {
  usePinnedModelsStore.setState({ pinned });
}

test("movePinned moves a key before a later key", () => {
  setPinned(["a", "b", "c", "d"]);
  usePinnedModelsStore.getState().movePinned("a", "c");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "b",
    "c",
    "a",
    "d",
  ]);
});

test("movePinned moves a key back to an earlier position", () => {
  setPinned(["a", "b", "c", "d"]);
  usePinnedModelsStore.getState().movePinned("d", "b");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "a",
    "d",
    "b",
    "c",
  ]);
});

test("movePinned ignores unknown keys and self-moves", () => {
  setPinned(["a", "b"]);
  usePinnedModelsStore.getState().movePinned("missing", "a");
  usePinnedModelsStore.getState().movePinned("a", "missing");
  usePinnedModelsStore.getState().movePinned("a", "a");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b"]);
});

test("movePinned keeps keys not shown in the current view in place", () => {
  // A dataset pin ("hidden") sits between model pins; moving models around it
  // must not lose or reorder it relative to untouched neighbours.
  setPinned(["m1", "hidden", "m2", "m3"]);
  usePinnedModelsStore.getState().movePinned("m3", "m1");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "m3",
    "m1",
    "hidden",
    "m2",
  ]);
});
