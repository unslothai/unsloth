// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  applyPin,
  hasUnknownRecord,
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

test("pinned items lead, keeping the order they arrived in", () => {
  const items = [item("new", 3), item("pinnedFirst", 1, true), item("pinnedSecond", 2, true)];
  assert.deepEqual(ids(sortGalleryItems(items)), ["pinnedFirst", "pinnedSecond", "new"]);
});

test("an unrelated merge does not rearrange the pinned group", () => {
  // The backend orders pins by PIN time, which the client never learns. Sorting them by created_at
  // would flip this pair, since the older item was pinned more recently and so leads on the server.
  const serverOrder = [item("pinnedRecentlyButOld", 1, true), item("pinnedLongAgoButNew", 9, true)];
  const merged = sortGalleryItems([item("fresh", 10), ...serverOrder]);
  assert.deepEqual(ids(merged), ["pinnedRecentlyButOld", "pinnedLongAgoButNew", "fresh"]);
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

test("a merged generation lands after the pinned group, not at the very front", () => {
  // The server sorts pinned first, so prepending a fresh record would disagree with it on reload.
  const existing = [item("pin", 1, true), item("older", 2)];
  const fresh = item("new", 9);
  assert.deepEqual(ids(sortGalleryItems([fresh, ...existing])), ["pin", "new", "older"]);
});

test("with nothing pinned a merged generation still leads", () => {
  const merged = sortGalleryItems([item("new", 9), item("older", 2)]);
  assert.deepEqual(ids(merged), ["new", "older"]);
});

// --- lost-generation probe ---------------------------------------------------------------------

/** A fake gallery listing, already in server order, served in pages. */
const pager =
  (all: ReturnType<typeof item>[], pageSize: number) => async (offset: number) => ({
    items: all.slice(offset, offset + pageSize),
    hasMore: offset + pageSize < all.length,
  });

test("a pinned first row does not mask a newly saved record", async () => {
  // The regression: with a pin present, reading only row 0 saw the pin, which was already known,
  // and reported a finished generation as never submitted.
  const listing = [item("pin", 5, true), item("fresh", 9), item("old", 1)];
  const known = new Set(["pin", "old"]);
  assert.equal(await hasUnknownRecord(known, pager(listing, 50), 50), true);
});

test("no new record is reported when the first unpinned row is already known", async () => {
  const listing = [item("pin", 5, true), item("known", 9), item("old", 1)];
  assert.equal(
    await hasUnknownRecord(new Set(["pin", "known", "old"]), pager(listing, 50), 50),
    false,
  );
});

test("the probe walks past a pinned group that spans more than one page", async () => {
  const listing = [
    item("p1", 5, true),
    item("p2", 4, true),
    item("p3", 3, true),
    item("fresh", 9),
  ];
  const known = new Set(["p1", "p2", "p3"]);
  // Page size 2, so the first unpinned row only appears on the second page.
  assert.equal(await hasUnknownRecord(known, pager(listing, 2), 2), true);
});

test("the probe stops at its page cap instead of scanning the whole gallery", async () => {
  const listing = Array.from({ length: 100 }, (_, i) => item(`p${i}`, 100 - i, true));
  let pages = 0;
  const counted = async (offset: number) => {
    pages += 1;
    return pager(listing, 10)(offset);
  };
  assert.equal(await hasUnknownRecord(new Set(listing.map((i) => i.id)), counted, 10, 3), false);
  assert.equal(pages, 3);
});

test("an empty gallery reports no new record", async () => {
  assert.equal(await hasUnknownRecord(new Set(), pager([], 50), 50), false);
});

test("an unknown pinned row is not proof that a generation landed", async () => {
  // With more pins than the client had loaded, knownIds omits the later ones. Treating such a pin
  // as evidence reported a lost submission as a finished run that produced no image.
  const listing = [item("loadedPin", 5, true), item("unloadedPin", 4, true), item("old", 1)];
  const known = new Set(["loadedPin", "old"]);
  assert.equal(await hasUnknownRecord(known, pager(listing, 50), 50), false);
});

test("a new record is still found past an unknown pinned row", async () => {
  const listing = [item("unloadedPin", 4, true), item("fresh", 9), item("old", 1)];
  assert.equal(await hasUnknownRecord(new Set(["old"]), pager(listing, 50), 50), true);
});
