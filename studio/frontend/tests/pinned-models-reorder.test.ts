// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { installLocalStorageFake, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

// Installed before the import so the store hydrates from it and its writes land
// somewhere the persistence cases below can read back.
const { store: storageStore, fireWindowEvent } = installLocalStorageFake();

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

/**
 * Another Unsloth window rewriting the pin list: the record changes underneath
 * this window and a "storage" event follows, which is the only way the list can
 * change without this window's store doing it. Goes through the real listener,
 * so what the store learns is exactly what a second window would teach it.
 */
function externalWrite(pinned: string[], key: string | null = STORAGE_KEY) {
  storageStore.set(STORAGE_KEY, JSON.stringify(pinned));
  if (fireWindowEvent("storage", { key }) !== 1) {
    // Otherwise every case below would pass by doing nothing at all.
    throw new Error("the store did not subscribe to storage on construction");
  }
}

/** Forget the record without touching the store, so a later write is visible. */
function forgetStoredWrites() {
  storageStore.delete(STORAGE_KEY);
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

// --- another window writing mid-drag ---------------------------------------
// pinned-models installs a window "storage" listener that replaces the list
// wholesale, so a second Unsloth window can rewrite the order underneath a drag
// that is still in flight. A snapshot taken before that write no longer
// describes what is in localStorage, so restoring it would put this window out
// of step with the record and the next write from here would clobber the other
// window's change. Whenever a storage event lands mid-drag the order it
// installed is what the session falls back to, whatever it did to the keys.

test("a pin added in another window mid-drag survives a cancel", () => {
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  externalWrite(["b", "c", "a", "new"]);
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "b",
    "c",
    "a",
    "new",
  ]);
  assert.deepEqual(storedPinned(), ["b", "c", "a", "new"], "and no write back");
});

test("a pin removed in another window mid-drag stays removed after a cancel", () => {
  // The mirror image: rolling the snapshot back would resurrect "b".
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  externalWrite(["c", "a"]);
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "a"]);
  assert.deepEqual(storedPinned(), ["c", "a"]);
});

test("a reorder in another window mid-drag survives a cancel", () => {
  // Same keys, different order. The pre-drag snapshot is just as stale here as
  // it is when the key set changed: localStorage holds the other window's
  // order, so restoring the snapshot would leave this window disagreeing with
  // the record while writing nothing to say so.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  externalWrite(["c", "b", "a"]);
  assert.deepEqual(
    usePinnedModelsStore.getState().pinned,
    ["c", "b", "a"],
    "the storage listener replaces the list mid-drag",
  );
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "b", "a"]);
  assert.deepEqual(storedPinned(), ["c", "b", "a"], "and no write back");
});

test("a cancel after another window reordered cannot clobber it on the next pin", () => {
  // Why the case above matters: nothing is lost at the moment of the rollback,
  // since a cancel writes nothing. The damage lands on the next write from this
  // window, which persists the whole list.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  externalWrite(["c", "b", "a"]);
  store.endPinnedDrag(false);
  usePinnedModelsStore.getState().togglePinned("d");
  assert.deepEqual(storedPinned(), ["d", "c", "b", "a"]);
});

test("a drag cancelled after another window wrote drops its own preview too", () => {
  // The moves made after the storage event are still just a preview, and the
  // user abandoned them. Falling back to the order the other window installed
  // discards them without writing anything.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  externalWrite(["c", "b", "a"]);
  store.movePinned("c", "a");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["b", "a", "c"]);
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "b", "a"]);
  assert.deepEqual(storedPinned(), ["c", "b", "a"]);
});

test("a drop after another window wrote mid-drag commits what is on screen", () => {
  // A drop is the user saying they meant it, so the last writer wins, but it
  // wins on top of the other window's list rather than the stale snapshot.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "b");
  externalWrite(["c", "b", "a", "new"]);
  store.movePinned("new", "c");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "new",
    "c",
    "b",
    "a",
  ]);
  store.endPinnedDrag(true);
  assert.deepEqual(storedPinned(), ["new", "c", "b", "a"]);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "new",
    "c",
    "b",
    "a",
  ]);
});

test("a drop that only re-applies another window's order writes nothing", () => {
  // The storage event already put that order in localStorage. "Nothing moved"
  // is measured against it, not against the snapshot from before it landed.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  externalWrite(["c", "b", "a"]);
  forgetStoredWrites();
  store.endPinnedDrag(true);
  assert.equal(storedPinned(), null, "no redundant write");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "b", "a"]);
});

test("a storage event for another key does not look like a cross-window pin write", () => {
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("a", "c");
  fireWindowEvent("storage", { key: "unsloth_something_else" });
  assert.deepEqual(
    usePinnedModelsStore.getState().pinned,
    ["b", "c", "a"],
    "the listener ignores it",
  );
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a", "b", "c"]);
});

test("a cross-window write is only remembered for the drag it landed in", () => {
  // The next drag starts from a clean session, so an ordinary cancel after one
  // that saw a storage event still rolls back.
  setPinned(["a", "b", "c"]);
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  externalWrite(["c", "b", "a"]);
  store.endPinnedDrag(false);
  storageStore.clear();

  store.beginPinnedDrag();
  store.movePinned("c", "a");
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "b", "a"]);
  assert.equal(storedPinned(), null);
});

test("a storage event with no drag in flight just replaces the list", () => {
  setPinned(["a", "b", "c"]);
  externalWrite(["c", "a", "b"]);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "a", "b"]);
  // And it becomes the order the next drag snapshots and rolls back to.
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("c", "b");
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "a", "b"]);
});

// Pin keys carry no repo type, and model and dataset repos are separate
// namespaces on the Hub, so one id can name both (huggingface/documentation-images,
// nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim). An on-device dataset whose
// repoId also names a pinned model therefore lands in the hub's Pinned grid. The
// row menu offers datasets no pin action, so the drag must not offer one either,
// or dragging dataset rows persistently reorders the user's model pins.
test("the hub's pinned grid never makes a dataset row draggable", async () => {
  const lists = await readFile(
    new URL(
      "../src/features/hub/catalog/models-catalog-lists.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    lists,
    /const itemPinKey =\s*!isDataset &&\s*item\.row\.repoId &&\s*pinnedSet\.has\(pinKey\(item\.row\.repoId\)\)/,
  );
  // The invariant the gate keeps: the row menu withholds pin/unpin for datasets.
  const rows = await readFile(
    new URL(
      "../src/features/hub/catalog/models-catalog-rows.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    rows,
    /pin=\{\s*isDataset \|\| !deletableRepoId\s*\?\s*undefined/,
  );
});
