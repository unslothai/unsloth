// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { installLocalStorageFake } from "./helpers/kit.ts";

const { store: storage, fireWindowEvent } = installLocalStorageFake();
const { usePinnedConnectedModelsStore: pins } = await import(
  "../src/features/model-picker/components/model-selector/pinned-connected-models.ts"
);
const KEY = "unsloth_pinned_connected_models";
const [A, B, C, D] = ["a", "b", "c", "d"].map((id) => `external::connection::${id}`);

function reset(order = [A, B]) {
  pins.getState().endPinnedConnectedDrag(false);
  storage.clear();
  storage.set(KEY, JSON.stringify(order));
  pins.setState({ pinned: order });
}

function externalWrite(order: string[], notify = true) {
  storage.set(KEY, JSON.stringify(order));
  if (notify) assert.equal(fireWindowEvent("storage", { key: KEY }), 1);
}

function storedOrder(): string[] {
  return JSON.parse(storage.get(KEY)!);
}

function drag() {
  pins.getState().beginPinnedConnectedDrag();
  pins.getState().movePinnedConnected(A, B);
}

test("ordinary drags persist only on drop", () => {
  reset();
  drag();
  assert.deepEqual(pins.getState().pinned, [B, A]);
  assert.deepEqual(storedOrder(), [A, B]);
  pins.getState().endPinnedConnectedDrag(true);
  pins.getState().endPinnedConnectedDrag(false);
  assert.deepEqual(storedOrder(), [B, A]);
  assert.deepEqual(pins.getState().pinned, [B, A]);
});

for (const notify of [true, false]) {
  test(`drop keeps the dragged order and remote additions, storage event=${notify}`, () => {
    reset();
    drag();
    externalWrite([C, A, B], notify);
    assert.deepEqual(pins.getState().pinned, [B, A]);
    pins.getState().endPinnedConnectedDrag(true);
    assert.deepEqual(pins.getState().pinned, [C, B, A]);
    assert.deepEqual(storedOrder(), [C, B, A]);
  });
}

test("drop respects remote removals and the latest additions", () => {
  reset([A, B, C]);
  drag();
  externalWrite([D, A, B, C]);
  externalWrite([D, A, B]);
  pins.getState().endPinnedConnectedDrag(true);
  assert.deepEqual(pins.getState().pinned, [D, B, A]);
  assert.deepEqual(storedOrder(), [D, B, A]);
});

for (const notify of [true, false]) {
  test(`cancel adopts remote changes without persisting the drag, storage event=${notify}`, () => {
    reset();
    drag();
    externalWrite([C, A], notify);
    pins.getState().endPinnedConnectedDrag(false);
    assert.deepEqual(pins.getState().pinned, [C, A]);
    assert.deepEqual(storedOrder(), [C, A]);
  });
}

test("a drag without a move adopts the remote order", () => {
  reset([A, B, C]);
  pins.getState().beginPinnedConnectedDrag();
  externalWrite([C, B, A]);
  pins.getState().endPinnedConnectedDrag(true);
  assert.deepEqual(pins.getState().pinned, [C, B, A]);
  assert.deepEqual(storedOrder(), [C, B, A]);
});

test("storage events outside a drag apply immediately", () => {
  reset();
  externalWrite([C, A]);
  assert.deepEqual(pins.getState().pinned, [C, A]);
});
