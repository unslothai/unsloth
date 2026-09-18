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
const RESET_PROBE = "external::connection::__reset__";

function reset(order = [A, B]) {
  pins.getState().endPinnedConnectedDrag(false);
  storage.clear();
  // Two toggles that WRITE, to clear the module-private failure state a previous test may have
  // left behind. Resetting only storage and the zustand state left `storageWritable` false with
  // ids still in `unpersisted`, which made the test after it pass without testing anything.
  pins.getState().togglePinnedConnected(RESET_PROBE);
  pins.getState().togglePinnedConnected(RESET_PROBE);
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

test("a pin that could not be written survives the next toggle", () => {
  // Quota exhausted with the key already present: reads keep working and return the OLDER list,
  // so basing the next edit on the record drops the pin that never persisted. The store already
  // intends "pins stay session-only" here; this is that intent holding across a second edit.
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
    assert.deepEqual(pins.getState().pinned, [B, A]);
    pins.getState().togglePinnedConnected(C);
    assert.deepEqual(pins.getState().pinned, [C, B, A]);
  } finally {
    storage.set = realSet;
  }
  // A write succeeding again hands the record back its authority.
  pins.getState().togglePinnedConnected(D);
  assert.deepEqual(storedOrder(), [D, C, B, A]);
});

test("a drag after a failed write keeps both sides' pins", () => {
  // The record stays FRESH for other windows while our own writes fail, so it is merged, not
  // dropped: an early return that ignored it let the drop overwrite another window's additions
  // once storage recovered.
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B); // session-only, never persisted
    assert.deepEqual(pins.getState().pinned, [B, A]);
  } finally {
    storage.set = realSet;
  }
  pins.getState().beginPinnedConnectedDrag();
  pins.getState().movePinnedConnected(B, A); // reorder locally
  externalWrite([D, A]); // another window pins D mid-drag
  pins.getState().endPinnedConnectedDrag(true);
  const after = pins.getState().pinned;
  assert.ok(after.includes(B), `this session's unpersisted pin was lost: ${JSON.stringify(after)}`);
  assert.ok(after.includes(D), `the other window's pin was overwritten: ${JSON.stringify(after)}`);
});

test("a peer removal survives a failed write", () => {
  // Only the ids this window failed to persist are its own. Treating everything the record lacks
  // as local re-added what a peer had just unpinned.
  reset([A, C]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B); // ours, never persisted
  } finally {
    storage.set = realSet;
  }
  externalWrite([A]); // a peer unpins C
  pins.getState().togglePinnedConnected(D); // this one lands
  const after = pins.getState().pinned;
  assert.ok(!after.includes(C), `the peer's removal was undone: ${JSON.stringify(after)}`);
  assert.ok(after.includes(B), `this session's unpersisted pin was lost: ${JSON.stringify(after)}`);
  assert.deepEqual(storedOrder(), after);
});

test("undoing a failed pin does not resurrect it", () => {
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
    assert.deepEqual(pins.getState().pinned, [B, A]);
    pins.getState().togglePinnedConnected(B); // undo, still failing
    assert.deepEqual(pins.getState().pinned, [A]);
  } finally {
    storage.set = realSet;
  }
  pins.getState().togglePinnedConnected(D);
  const after = pins.getState().pinned;
  assert.ok(!after.includes(B), `an undone pin came back: ${JSON.stringify(after)}`);
});

test("a failed pin stays on screen when a peer writes", () => {
  // The rendered list decides which way the row's own action toggles, so it has to agree with
  // what every merge below it holds.
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
  } finally {
    storage.set = realSet;
  }
  externalWrite([C, A]);
  assert.ok(
    pins.getState().pinned.includes(B),
    `the row would render unpinned while the merge still holds it: ${JSON.stringify(pins.getState().pinned)}`,
  );
});

test("a peer persisting our failed pin hands it back to the record", () => {
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
  } finally {
    storage.set = realSet;
  }
  externalWrite([B, A]); // a peer pins the same model, so the record carries it now
  externalWrite([A]); // and later unpins it
  assert.ok(
    !pins.getState().pinned.includes(B),
    `a peer's removal was undone by a stale unpersisted entry: ${JSON.stringify(pins.getState().pinned)}`,
  );
});

test("a record read outside the storage handler retires a pin too", () => {
  // The peer's write is observed synchronously by an unchanged drag, before its event is
  // delivered, so retiring only in the handler left the id classified as ours.
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
  } finally {
    storage.set = realSet;
  }
  externalWrite([B, A], false); // a peer persists the same model, event still pending
  pins.getState().beginPinnedConnectedDrag();
  pins.getState().endPinnedConnectedDrag(false); // unchanged drag: reads the record synchronously
  externalWrite([A]); // the peer then unpins it
  assert.ok(
    !pins.getState().pinned.includes(B),
    `a synchronously observed pin stayed classified as ours: ${JSON.stringify(pins.getState().pinned)}`,
  );
});

test("a pin a peer added and removed between events is retired", () => {
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
  } finally {
    storage.set = realSet;
  }
  // Both events arrive after both writes, so each handler reads the SAME final record and only
  // the payloads carry the intermediate pin.
  storage.set(KEY, JSON.stringify([B, A]));
  storage.set(KEY, JSON.stringify([A]));
  fireWindowEvent("storage", { key: KEY, newValue: JSON.stringify([B, A]) });
  fireWindowEvent("storage", { key: KEY, newValue: JSON.stringify([A]) });
  assert.ok(
    !pins.getState().pinned.includes(B),
    `an id the peer pinned then unpinned was resurrected: ${JSON.stringify(pins.getState().pinned)}`,
  );
});

test("a newer peer pin sits above an older unpersisted one", () => {
  reset([A]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
  } finally {
    storage.set = realSet;
  }
  externalWrite([C, A]); // a peer pins C after B was attempted
  assert.deepEqual(pins.getState().pinned, [C, B, A]);
});
