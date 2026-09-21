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
const [A, B, C, D, E] = ["a", "b", "c", "d", "e"].map(
  (id) => `external::connection::${id}`,
);
const RESET_PROBE = "external::connection::__reset__";

function reset(order = [A, B]) {
  pins.getState().endPinnedConnectedDrag(false);
  storage.clear();
  // Two toggles that WRITE, to clear module-private failure state: resetting only storage and the
  // zustand state left a leaked `unpersisted` making the next test pass without testing anything.
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
  // Quota exhausted with the key present: reads still return the OLDER list, so basing the next
  // edit on the record dropped the pin that never persisted.
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
  // The record stays FRESH for other windows, so it is merged: ignoring it let the drop overwrite
  // another window's additions once storage recovered.
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
  // Treating everything the record lacks as local re-added what a peer had just unpinned.
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
  // The rendered list decides which way the row toggles, so it must agree with the merge.
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
  // An unchanged drag sees the peer's write before its event, so retiring only in the handler
  // left the id classified as ours.
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

test("a queued payload does not retire a pin the record never carried", () => {
  // The mirror of the test above and INDISTINGUISHABLE from it: the same two payloads over the
  // same final record. Retiring on the payload is right there and drops the user's own pin here,
  // so only the record retires and this pin survives session-only.
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
  storage.set(KEY, JSON.stringify([B, A]));
  storage.set(KEY, JSON.stringify([A]));
  fireWindowEvent("storage", { key: KEY, newValue: JSON.stringify([B, A]) });
  fireWindowEvent("storage", { key: KEY, newValue: JSON.stringify([A]) });
  assert.ok(
    pins.getState().pinned.includes(B),
    `the pin the user just made was dropped: ${JSON.stringify(pins.getState().pinned)}`,
  );
  assert.deepEqual(storedOrder(), [A]);
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

test("a later successful toggle persists the rendered order", () => {
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
  assert.deepEqual(pins.getState().pinned, [C, B, A]);
  pins.getState().togglePinnedConnected(D); // this one lands
  assert.deepEqual(pins.getState().pinned, [D, C, B, A]);
  assert.deepEqual(storedOrder(), [D, C, B, A]);
});

test("a peer reorder wins once nothing of ours is unwritten", () => {
  // `storageWritable` is about OUR writes, not the record: with nothing of ours left to carry,
  // holding the rendered order let the next toggle that landed overwrite a peer's reorder.
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
  externalWrite([B, A]); // a peer persists the same pin, so nothing of ours is unwritten now
  externalWrite([A, B], false); // and reorders, event not yet delivered
  pins.getState().togglePinnedConnected(D); // this one lands
  assert.deepEqual(storedOrder(), [D, A, B]);
  assert.deepEqual(pins.getState().pinned, [D, A, B]);
});

test("interleaved local and peer pins keep one newest-first order", () => {
  // Two blocks is an order neither window produced: it lifts every pin we could not write above
  // every peer pin on screen, and the next write that lands freezes that.
  reset([A]);
  const realSet = storage.set.bind(storage);
  const failing = () => {
    throw new Error("QuotaExceededError");
  };
  storage.set = failing;
  try {
    pins.getState().togglePinnedConnected(B);
  } finally {
    storage.set = realSet;
  }
  externalWrite([C, A]);
  assert.deepEqual(pins.getState().pinned, [C, B, A]);
  storage.set = failing;
  try {
    pins.getState().togglePinnedConnected(D);
  } finally {
    storage.set = realSet;
  }
  assert.deepEqual(pins.getState().pinned, [D, C, B, A]);
  externalWrite([E, C, A]);
  assert.deepEqual(pins.getState().pinned, [E, D, C, B, A]);
});

function externalRemove() {
  storage.delete(KEY);
  assert.equal(fireWindowEvent("storage", { key: KEY, newValue: null }), 1);
}

test("a peer clearing the record unpins everything it held", () => {
  reset([A, B]);
  externalRemove();
  assert.deepEqual(pins.getState().pinned, []);
  pins.getState().togglePinnedConnected(D);
  // The pins the peer cleared must not come back with the next write that lands.
  assert.deepEqual(storedOrder(), [D]);
});

test("a peer clearing the record keeps a pin that was never written", () => {
  // Only this window ever held it, so a reset of the record says nothing about it.
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
  externalRemove();
  assert.deepEqual(pins.getState().pinned, [B]);
});

test("an unchanged drag keeps a peer pin above an older failed one", () => {
  // Ending on the list the event carried re-added the unpersisted pin at the front, above a
  // newer peer pin.
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
  pins.getState().beginPinnedConnectedDrag();
  externalWrite([C, A]); // a peer pins C mid-drag
  pins.getState().endPinnedConnectedDrag(false);
  assert.deepEqual(pins.getState().pinned, [C, B, A]);
});

test("a peer reordering the same pins reorders this window too", () => {
  // Nothing added or removed, so a merge carrying only our unwritten pins left the old order.
  reset([A, B]);
  externalWrite([B, A]);
  assert.deepEqual(pins.getState().pinned, [B, A]);
});

test("a record this window cannot read changes nothing on screen", () => {
  // Revoked access reads like a deleted key and means the opposite: as a reset it unpinned
  // everything, on a window whose writes still land.
  reset([A]);
  const realGet = storage.get.bind(storage);
  storage.get = () => {
    throw new Error("SecurityError");
  };
  try {
    pins.getState().togglePinnedConnected(B);
    assert.deepEqual(pins.getState().pinned, [B, A]);
  } finally {
    storage.get = realGet;
  }
  assert.deepEqual(storedOrder(), [B, A]);
});

test("an event arriving while the record is unreadable leaves the list alone", () => {
  reset([A, B]);
  const realGet = storage.get.bind(storage);
  storage.get = () => {
    throw new Error("SecurityError");
  };
  try {
    externalWrite([C]);
    assert.deepEqual(pins.getState().pinned, [A, B]);
  } finally {
    storage.get = realGet;
  }
});

test("an unpin stays an unpin when a peer cleared the record first", () => {
  // The row draws itself pinned, so the click means unpin; the emptied base wrote it back in.
  reset([A, B]);
  storage.delete(KEY); // a peer clears, its event not yet delivered
  pins.getState().togglePinnedConnected(A);
  assert.ok(
    !pins.getState().pinned.includes(A),
    `an unpin was turned into a pin: ${JSON.stringify(pins.getState().pinned)}`,
  );
  assert.deepEqual(storedOrder(), []);
});

test("an unpin stays an unpin when a peer unpinned it first", () => {
  // The quiet version of the same thing: one model gone from the record rather than all of them.
  reset([A, B]);
  storage.set(KEY, JSON.stringify([B])); // a peer unpins A, its event not yet delivered
  pins.getState().togglePinnedConnected(A);
  assert.deepEqual(storedOrder(), [B]);
  assert.deepEqual(pins.getState().pinned, [B]);
});

test("a record we cannot read does not make older pins ours", () => {
  // Blind both ways, the only id known unpublished is the one just added: claiming the screen
  // made stored pins ours, so a peer unpinning one was undone by the next write that landed.
  reset([A, B]);
  const realSet = storage.set.bind(storage);
  const realGet = storage.get.bind(storage);
  storage.set = () => {
    throw new Error("QuotaExceededError");
  };
  storage.get = () => {
    throw new Error("SecurityError");
  };
  try {
    pins.getState().togglePinnedConnected(C);
    assert.deepEqual(pins.getState().pinned, [C, A, B]);
  } finally {
    storage.set = realSet;
    storage.get = realGet;
  }
  externalWrite([A]); // a peer unpinned B while this window was blind
  pins.getState().togglePinnedConnected(D); // this one lands
  assert.deepEqual(storedOrder(), [D, C, A]);
});

test("a peer removing a pin as the write fails does not make it ours", () => {
  // The peer's write lands between the read this edit was built on and the setItem that failed,
  // so reading ownership off the difference claimed B and undid the removal.
  reset([A, B]);
  const realSet = storage.set.bind(storage);
  storage.set = () => {
    realSet(KEY, JSON.stringify([A])); // the peer unpins B, and then our write throws
    throw new Error("QuotaExceededError");
  };
  try {
    pins.getState().togglePinnedConnected(C);
  } finally {
    storage.set = realSet;
  }
  assert.equal(fireWindowEvent("storage", { key: KEY }), 1);
  assert.deepEqual(pins.getState().pinned, [C, A]);
  pins.getState().togglePinnedConnected(D); // this one lands
  assert.deepEqual(storedOrder(), [D, C, A]);
});
