// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  readSrcAsync,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();

// Installed before the import so the store hydrates from it.
const { store: storageStore, fireWindowEvent } = installLocalStorageFake();

const { pinKey, usePinnedModelsStore } = await import(
  "../src/features/model-picker/components/model-selector/pinned-models.ts"
);

const MODELS_CATALOG_ROWS = readSrc("features/hub/catalog/models-catalog-rows.tsx");

const STORAGE_KEY = "unsloth_pinned_models";

function setPinned(pinned: string[]) {
  usePinnedModelsStore.setState({ pinned });
  storageStore.clear();
}

function storedPinned(): string[] | null {
  const raw = storageStore.get(STORAGE_KEY);
  return raw ? JSON.parse(raw) : null;
}

function externalWrite(pinned: string[], key: string | null = STORAGE_KEY) {
  storageStore.set(STORAGE_KEY, JSON.stringify(pinned));
  if (fireWindowEvent("storage", { key }) !== 1) {
    // Otherwise every case below would pass by doing nothing at all.
    throw new Error("the store did not subscribe to storage on construction");
  }
}

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

// Forward drags insert after the target, backward before it, so reorder-on-dragenter is stable.

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
  // Older builds may have stored duplicate keys.
  setPinned(["a", "b", "a"]);
  usePinnedModelsStore.getState().movePinned("a", "b");
  const after = usePinnedModelsStore.getState().pinned;
  assert.equal(after.length, 3);
  assert.equal(after.filter((key) => key === "a").length, 2);
  assert.equal(after.filter((key) => key === "b").length, 1);
});

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
  // The hub keys cells by repo id, so per-quant-only pins are unreachable from it.
  setPinned(["r1::Q4_K_M", "r2"]);
  usePinnedModelsStore.getState().movePinned(pinKey("r1"), "r2");
  usePinnedModelsStore.getState().movePinned("r2", pinKey("r1"));
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["r1::Q4_K_M", "r2"]);
});

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

// Drags reorder live, so the store commits on drop or rolls back to the dragstart snapshot.

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
  // Drop commits, then dragend fires; it must not roll back.
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
  setPinned(["a", "b", "c"]);
  usePinnedModelsStore.getState().movePinned("a", "b");
  assert.deepEqual(storedPinned(), ["b", "a", "c"]);
});

// A storage event mid-drag replaces the snapshot, or the next write clobbers the other window.

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
  // Same keys, different order: localStorage holds the other window's order, so the snapshot is stale.
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
  // A drop wins, but on top of the other window's list rather than the stale snapshot.
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
  const store = usePinnedModelsStore.getState();
  store.beginPinnedDrag();
  store.movePinned("c", "b");
  store.endPinnedDrag(false);
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["c", "a", "b"]);
});

// Model and dataset repos share ids on the Hub, and datasets have no pin action, so no drag.
test("the hub's pinned grid never makes a dataset row draggable", async () => {
  const lists = await readSrcAsync("features/hub/catalog/models-catalog-lists.tsx");
  assert.match(
    lists,
    /const itemPinKey =\s*!isDataset &&\s*item\.row\.repoId &&\s*pinnedSet\.has\(pinKey\(item\.row\.repoId\)\)/,
  );
  assert.match(
    MODELS_CATALOG_ROWS,
    /pin=\{\s*isDataset \|\| !deletableRepoId\s*\?\s*undefined/,
  );
});

test("deleting a repo drops its quant pins, not just the repo's own", () => {
  // Pins are keyed `repoId::quant`, so a whole-repo delete must clear every quant pin.
  setPinned([
    "unsloth/Qwen3-8B-GGUF::Q4_K_M",
    "unsloth/Qwen3-8B-GGUF",
    "unsloth/Qwen3-8B-GGUF::UD-Q4_K_XL",
    "unsloth/Gemma-3-4B-GGUF::Q8_0",
    "unsloth/Gemma-3-4B-GGUF",
  ]);
  usePinnedModelsStore.getState().unpinRepo("unsloth/Qwen3-8B-GGUF");
  const left = usePinnedModelsStore.getState().pinned;
  assert.deepEqual(left, [
    "unsloth/Gemma-3-4B-GGUF::Q8_0",
    "unsloth/Gemma-3-4B-GGUF",
  ]);
  assert.deepEqual(storedPinned(), left);
});

test("unpinning a repo leaves a longer repo id that merely starts the same", () => {
  // The `::` boundary keeps "X-GGUF" from also matching "X-GGUF-128K".
  setPinned([
    "unsloth/Qwen3-8B-GGUF-128K",
    "unsloth/Qwen3-8B-GGUF-128K::Q4_K_M",
    "unsloth/Qwen3-8B-GGUF",
  ]);
  usePinnedModelsStore.getState().unpinRepo("unsloth/Qwen3-8B-GGUF");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "unsloth/Qwen3-8B-GGUF-128K",
    "unsloth/Qwen3-8B-GGUF-128K::Q4_K_M",
  ]);
});

test("unpinning a repo that was never pinned writes nothing", () => {
  // Called on every repo delete; writing on no change would wake every window's listener.
  setPinned(["unsloth/Gemma-3-4B-GGUF"]);
  usePinnedModelsStore.getState().unpinRepo("unsloth/Nothing-Here");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "unsloth/Gemma-3-4B-GGUF",
  ]);
  assert.equal(storedPinned(), null, "no write, so the record is untouched");
});

test("both repo-level deletes clear pins through that one action", async () => {
  // Picker and Hub cache deletes must agree on what a pin outlives.
  const pickers = await readSrcAsync("features/model-picker/components/model-selector/pickers.tsx");
  assert.equal(
    pickers.split("unpinRepo(c.repo_id);").length - 1,
    2,
    "the partial GGUF repo row and the cached model row alike",
  );
  assert.ok(
    !pickers.includes("if (pinnedSet.has(pinKey(c.repo_id))) {"),
    "and neither clears by toggling the bare repo key",
  );
  assert.ok(
    MODELS_CATALOG_ROWS.includes(
      "usePinnedModelsStore.getState().unpinRepo(deletableRepoId);",
    ),
  );
});

// A GGUF export's id is its first file, so a variant delete can change its key.
test("replacePinned moves a pin to its new key in the same slot", () => {
  setPinned(["a", "/exports/run-gguf/x.Q4_K_M.gguf", "b"]);
  usePinnedModelsStore
    .getState()
    .replacePinned("/exports/run-gguf/x.Q4_K_M.gguf", "/exports/run-gguf/x.Q8_0.gguf");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, [
    "a",
    "/exports/run-gguf/x.Q8_0.gguf",
    "b",
  ]);
  assert.deepEqual(storedPinned(), usePinnedModelsStore.getState().pinned);
  setPinned(["a", "b", "c"]);
  usePinnedModelsStore.getState().replacePinned("a", "c");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["b", "c"]);
  setPinned(["a"]);
  usePinnedModelsStore.getState().replacePinned("z", "y");
  assert.deepEqual(usePinnedModelsStore.getState().pinned, ["a"]);
  assert.equal(storedPinned(), null);
});
