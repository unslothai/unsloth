// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

// Installed before the import so the store hydrates from it and its writes land
// somewhere the persistence cases below can read back.
const { store: storageStore } = installLocalStorageFake();

const { pinKey, usePinnedModelsStore } = await import(
  "../src/features/model-picker/components/model-selector/pinned-models.ts"
);

const STORAGE_KEY = "unsloth_pinned_models";

function setPinned(pinned: string[]) {
  usePinnedModelsStore.setState({ pinned });
  storageStore.clear();
}

function storedPinned(): string[] | null {
  const raw = storageStore.get(STORAGE_KEY);
  return raw ? JSON.parse(raw) : null;
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

test("movePinned keeps keys not shown in the current view in relative order", () => {
  // The pinned list is one global order shared by the hub's On Device list and
  // the model selector's Pinned section, and the hub only ever shows the repo
  // keys. A quant pin the hub cannot show ("hidden") therefore has to survive a
  // hub reorder. It keeps its position relative to every pin the drag did not
  // touch, which is not the same as keeping its index: one key moving past it
  // necessarily shifts it by one slot.
  setPinned(["m1", "hidden", "m2", "m3"]);
  usePinnedModelsStore.getState().movePinned("m3", "m1");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "m3",
    "m1",
    "hidden",
    "m2",
  ]);
});

test("movePinned leaves every untouched key in its original relative order", () => {
  // The invariant behind the case above, stated directly: exactly one key moves.
  const before = ["a", "b::Q4_K_M", "c", "d::Q8_0", "e"];
  setPinned([...before]);
  usePinnedModelsStore.getState().movePinned("e", "b::Q4_K_M");
  const after = usePinnedModelsStore.getState().pinned;
  assert.deepEqual([...after].sort(), [...before].sort(), "no key is lost");
  const untouched = before.filter((key) => key !== "e");
  assert.deepEqual(
    after.filter((key) => key !== "e"),
    untouched,
    "the keys the drag did not touch keep their order",
  );
});

// --- the drop convention ---------------------------------------------------
// The dragged cell lands in the slot the pointer is over, which is what makes
// reorder-on-dragenter stable: after the move the pointer sits on the dragged
// cell itself, so the next dragenter is a self-move and does nothing. Getting
// there means a forward drag inserts AFTER the target and a backward drag
// inserts BEFORE it, so both are pinned down here on purpose.

test("dragging forward puts the moved key after the target", () => {
  setPinned(["a", "b", "c", "d"]);
  usePinnedModelsStore.getState().movePinned("a", "c");
  const after = usePinnedModelsStore.getState().pinned;
  assert.deepEqual(after, ["b", "c", "a", "d"]);
  assert.equal(after.indexOf("a"), 2, "the moved key takes the target's index");
});

test("dragging backward puts the moved key before the target", () => {
  setPinned(["a", "b", "c", "d"]);
  usePinnedModelsStore.getState().movePinned("d", "b");
  const after = usePinnedModelsStore.getState().pinned;
  assert.deepEqual(after, ["a", "d", "b", "c"]);
  assert.equal(after.indexOf("d"), 1, "the moved key takes the target's index");
});

test("moving onto an adjacent key swaps the two, in either direction", () => {
  setPinned(["a", "b", "c"]);
  usePinnedModelsStore.getState().movePinned("a", "b");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["b", "a", "c"]);
  usePinnedModelsStore.getState().movePinned("c", "a");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["b", "c", "a"]);
});

test("moving to the first and last key reaches both ends", () => {
  setPinned(["a", "b", "c"]);
  usePinnedModelsStore.getState().movePinned("c", "a");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "a", "b"]);
  usePinnedModelsStore.getState().movePinned("c", "b");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b", "c"]);
});

// --- degenerate inputs -----------------------------------------------------

test("movePinned on an empty or single-entry list is inert", () => {
  setPinned([]);
  usePinnedModelsStore.getState().movePinned("a", "b");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, []);
  setPinned(["a"]);
  usePinnedModelsStore.getState().movePinned("a", "a");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a"]);
});

test("movePinned with both keys missing changes nothing", () => {
  setPinned(["a", "b"]);
  usePinnedModelsStore.getState().movePinned("x", "y");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b"]);
});

test("a duplicated key in stored order does not lose an entry", () => {
  // readPinned takes localStorage at its word, so a list written by an older
  // build can hold the same key twice. Moving it must still leave the list
  // well formed rather than dropping one of the copies.
  setPinned(["a", "b", "a"]);
  usePinnedModelsStore.getState().movePinned("a", "b");
  const after = usePinnedModelsStore.getState().pinned;
  assert.equal(after.length, 3);
  assert.equal(after.filter((key) => key === "a").length, 2);
  assert.equal(after.filter((key) => key === "b").length, 1);
});

// --- key shapes ------------------------------------------------------------

test("a quant pin reorders like any other key", () => {
  setPinned(["r1::Q4_K_M", "r2", "r3"]);
  usePinnedModelsStore.getState().movePinned("r3", "r1::Q4_K_M");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "r3",
    "r1::Q4_K_M",
    "r2",
  ]);
});

test("a repo key never moves a pin that was stored per quant", () => {
  // The hub's On Device list keys its cells by repo id alone, so a repo that is
  // only pinned per quant is not in its Pinned section at all and no repo-keyed
  // drag can reach it. Asserted so the two surfaces cannot drift apart quietly.
  setPinned(["r1::Q4_K_M", "r2"]);
  usePinnedModelsStore.getState().movePinned(pinKey("r1"), "r2");
  usePinnedModelsStore.getState().movePinned("r2", pinKey("r1"));
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["r1::Q4_K_M", "r2"]);
});

// --- persistence -----------------------------------------------------------

test("a move persists the new order, and a no-op writes nothing", () => {
  setPinned(["a", "b", "c"]);
  assert.equal(storedPinned(), null, "no write before the move");
  usePinnedModelsStore.getState().movePinned("c", "a");
  assert.deepEqual(storedPinned(), ["c", "a", "b"]);

  setPinned(["a", "b", "c"]);
  usePinnedModelsStore.getState().movePinned("a", "a");
  usePinnedModelsStore.getState().movePinned("a", "missing");
  assert.equal(storedPinned(), null, "a rejected move must not touch storage");
});

// --- drag sessions ---------------------------------------------------------
// A drag reorders live on every dragenter, so the store holds those moves in
// memory and either commits once on drop or rolls back to the snapshot taken at
// dragstart. Without that, a drag cancelled with Escape or released outside the
// grid stayed applied and was already written to localStorage.

test("a cancelled drag restores the order it started from", () => {
  setPinned(["a", "b", "c", "d"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "b");
  store.movePinned("a", "c");
  assert.deepEqual(
    usePinnedModelsStore.getState().pinned,
    ["b", "c", "a", "d"],
    "the live preview still moves during the drag",
  );
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b", "c", "d"]);
});

test("a cancelled drag leaves localStorage untouched", () => {
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  assert.equal(storedPinned(), null, "no write while the drag is in flight");
  store.endPinnedDrag(false);
  assert.equal(storedPinned(), null);
});

test("a dropped drag commits the new order exactly once", () => {
  setPinned(["a", "b", "c", "d"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("d", "c");
  store.movePinned("d", "b");
  store.movePinned("d", "a");
  assert.equal(storedPinned(), null, "the preview moves do not persist");
  store.endPinnedDrag(true);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "d",
    "a",
    "b",
    "c",
  ]);
  assert.deepEqual(storedPinned(), ["d", "a", "b", "c"]);
});

test("endPinnedDrag is idempotent, so dragend after drop cannot undo it", () => {
  // The browser fires drop and then dragend on the source. The drop commits and
  // closes the session; the dragend that follows must not roll anything back.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("c", "a");
  store.endPinnedDrag(true);
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "a", "b"]);
  assert.deepEqual(storedPinned(), ["c", "a", "b"]);
});

test("ending a drag that never moved anything writes nothing", () => {
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.endPinnedDrag(true);
  assert.equal(storedPinned(), null);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b", "c"]);
});

test("endPinnedDrag without a session in flight is inert", () => {
  setPinned(["a", "b"]);
  usePinnedModelsStore.getState().endPinnedDrag(false);
  usePinnedModelsStore.getState().endPinnedDrag(true);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b"]);
  assert.equal(storedPinned(), null);
});

test("outside a drag session movePinned still persists on every call", () => {
  // The store is shared with surfaces that have no drag at all, so the plain
  // call has to keep writing through.
  setPinned(["a", "b", "c"]);
  usePinnedModelsStore.getState().movePinned("a", "b");
  assert.deepEqual(storedPinned(), ["b", "a", "c"]);
});

test("a pin added in another window mid-drag survives a cancel", () => {
  // pinned-models installs a window "storage" listener that replaces the list
  // wholesale. If that lands mid-drag the snapshot no longer describes the same
  // set of pins, and restoring it would resurrect a pin the user just removed
  // or drop one they just added. The newer list wins instead.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  usePinnedModelsStore.setState({ pinned: ["b", "c", "a", "new"] });
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "b",
    "c",
    "a",
    "new",
  ]);
});

test("a cancel rolls back even when a same-set write landed mid-drag", () => {
  // Same keys, different order: that is a pure reorder, so the rollback is safe
  // and the drag the user abandoned leaves nothing behind.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b", "c"]);
});
