// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  applyPin,
  nextSelectedId,
  removeGalleryItem,
  sortGalleryItems,
} from "../src/lib/gallery-flags.ts";

const item = (id: string, created_at: number | string, pinned = false) => ({
  id,
  created_at,
  pinned,
});

const ids = (items: { id: string }[]) => items.map((i) => i.id);

test("unpinned items sort newest first", () => {
  const items = [item("old", 1), item("new", 3), item("mid", 2)];
  assert.deepEqual(ids(sortGalleryItems(items)), ["new", "mid", "old"]);
});

test("pinned items lead, and stay newest-first within their own group", () => {
  const items = [item("new", 3), item("pinOld", 1, true), item("pinNew", 2, true)];
  assert.deepEqual(ids(sortGalleryItems(items)), ["pinNew", "pinOld", "new"]);
});

test("the item just pinned goes to the very front of the pinned group", () => {
  // The backend orders pinned by pin time, which the client cannot know, so the freshly pinned
  // item leads even though an older pin has a newer created_at.
  const items = [item("a", 5, true), item("b", 1)];
  assert.deepEqual(ids(applyPin(items, "b", true)), ["b", "a"]);
});

test("unpinning drops the item back into the newest-first tail", () => {
  const items = [item("pinned", 1, true), item("newer", 9), item("older", 2)];
  assert.deepEqual(ids(applyPin(items, "pinned", false)), ["newer", "older", "pinned"]);
});

test("pinning is applied to the record, not just the order", () => {
  const next = applyPin([item("a", 1), item("b", 2)], "a", true);
  assert.equal(next.find((i) => i.id === "a")?.pinned, true);
  assert.equal(next.find((i) => i.id === "b")?.pinned, false);
});

test("ISO timestamps sort alongside epoch seconds, so videos order like images", () => {
  const items = [
    item("older", "2026-01-01T00:00:00Z"),
    item("newer", "2026-06-01T00:00:00Z"),
  ];
  assert.deepEqual(ids(sortGalleryItems(items)), ["newer", "older"]);
});

test("removing an item leaves the rest in order", () => {
  const items = [item("a", 3), item("b", 2), item("c", 1)];
  assert.deepEqual(ids(removeGalleryItem(items, "b")), ["a", "c"]);
});

test("selection is untouched when some other item leaves the strip", () => {
  const remaining = [item("a", 2), item("c", 1)];
  assert.equal(nextSelectedId(remaining, "b", "a", 1), "a");
});

test("removing the selected item falls to the neighbour that took its place", () => {
  const remaining = [item("a", 3), item("c", 1)];
  assert.equal(nextSelectedId(remaining, "b", "b", 1), "c");
});

test("removing the last item selects the new last, not an empty slot", () => {
  const remaining = [item("a", 3), item("b", 2)];
  assert.equal(nextSelectedId(remaining, "c", "c", 2), "b");
});

test("removing the only item clears the selection", () => {
  assert.equal(nextSelectedId([], "a", "a", 0), null);
});
