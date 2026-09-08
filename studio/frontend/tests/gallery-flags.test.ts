// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  PAGE_MAX_ATTEMPTS,
  applyPin,
  fetchNextPage,
  fetchWhileStable,
  hasUnknownRecord,
  mergeGenerated,
  newRecordProbeBaseline,
  nextSelectedId,
  pinnedOrder,
  removeGalleryItem,
  restorePinOrder,
  serializeById,
  sortGalleryItems,
} from "../src/lib/gallery-flags.ts";

const item = (id: string, created_at: number | string, pinned = false) => ({
  id,
  created_at,
  pinned,
});

const ids = (items: { id: string }[]) => items.map((i) => i.id);

const archivedMediaSource = readFileSync(
  new URL(
    "../src/features/settings/components/archived-media-dialog.tsx",
    import.meta.url,
  ),
  "utf8",
);

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

/** A baseline whose loaded window is a sound basis for judging an unpinned row. */
const judging = (knownIds: ReadonlySet<string>) => ({ knownIds, canJudgeUnpinned: true });

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
  assert.equal(await hasUnknownRecord(judging(known), pager(listing, 50), 50), true);
});

test("no new record is reported when the first unpinned row is already known", async () => {
  const listing = [item("pin", 5, true), item("known", 9), item("old", 1)];
  assert.equal(
    await hasUnknownRecord(judging(new Set(["pin", "known", "old"])), pager(listing, 50), 50),
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
  assert.equal(await hasUnknownRecord(judging(known), pager(listing, 2), 2), true);
});

test("the probe stops at its page cap instead of scanning the whole gallery", async () => {
  const listing = Array.from({ length: 100 }, (_, i) => item(`p${i}`, 100 - i, true));
  let pages = 0;
  const counted = async (offset: number) => {
    pages += 1;
    return pager(listing, 10)(offset);
  };
  assert.equal(await hasUnknownRecord(judging(new Set(listing.map((i) => i.id))), counted, 10, 3), false);
  assert.equal(pages, 3);
});

test("an empty gallery reports no new record", async () => {
  assert.equal(await hasUnknownRecord(judging(new Set()), pager([], 50), 50), false);
});

test("an unknown pinned row is not proof that a generation landed", async () => {
  // With more pins than the client had loaded, knownIds omits the later ones. Treating such a pin
  // as evidence reported a lost submission as a finished run that produced no image.
  const listing = [item("loadedPin", 5, true), item("unloadedPin", 4, true), item("old", 1)];
  const known = new Set(["loadedPin", "old"]);
  assert.equal(await hasUnknownRecord(judging(known), pager(listing, 50), 50), false);
});

test("a new record is still found past an unknown pinned row", async () => {
  const listing = [item("unloadedPin", 4, true), item("fresh", 9), item("old", 1)];
  assert.equal(await hasUnknownRecord(judging(new Set(["old"])), pager(listing, 50), 50), true);
});

// --- what the loaded window is allowed to conclude -----------------------------------------------

test("an all-pinned partial window cannot judge an unpinned row", () => {
  // 50 pins loaded, more pages behind them: every unpinned row is unfamiliar just for being
  // unloaded, so unknown stops meaning new.
  const loaded = Array.from({ length: 50 }, (_, i) => item(`p${i}`, 100 - i, true));
  const known = new Set(loaded.map((i) => i.id));
  assert.equal(newRecordProbeBaseline(loaded, true, known).canJudgeUnpinned, false);
});

test("a window holding any unpinned record can judge", () => {
  const loaded = [item("pin", 5, true), item("plain", 4)];
  assert.equal(newRecordProbeBaseline(loaded, true, new Set()).canJudgeUnpinned, true);
});

test("a complete window can judge even with nothing unpinned in it", () => {
  // hasMore false means the client has the whole gallery, so there is nothing it has not seen.
  const loaded = [item("pin", 5, true)];
  assert.equal(newRecordProbeBaseline(loaded, false, new Set()).canJudgeUnpinned, true);
  // The empty gallery of a first-ever generation is the same case.
  assert.equal(newRecordProbeBaseline([], false, new Set()).canJudgeUnpinned, true);
});

test("a window that cannot judge refuses to claim proof", async () => {
  // The regression: with 50+ pins loaded, the first historical unpinned row is necessarily
  // unknown, and treating it as proof suppressed a real submission failure.
  const listing = [
    item("p0", 9, true),
    item("p1", 8, true),
    item("historical", 1),
  ];
  const loaded = [item("p0", 9, true), item("p1", 8, true)];
  const baseline = newRecordProbeBaseline(loaded, true, new Set(["p0", "p1"]));
  assert.equal(await hasUnknownRecord(baseline, pager(listing, 50), 50), false);
});

test("a judging window still proves a genuinely new record", async () => {
  const listing = [item("pin", 9, true), item("fresh", 10), item("older", 1)];
  const loaded = [item("pin", 9, true), item("older", 1)];
  const baseline = newRecordProbeBaseline(loaded, true, new Set(["pin", "older"]));
  assert.equal(await hasUnknownRecord(baseline, pager(listing, 50), 50), true);
});

test("an empty gallery still proves its first generation", async () => {
  const baseline = newRecordProbeBaseline([], false, new Set());
  assert.equal(await hasUnknownRecord(baseline, pager([item("first", 1)], 50), 50), true);
});

// --- per-item serialization ----------------------------------------------------------------------

test("two tasks on the same key run in call order, never overlapping", async () => {
  const events: string[] = [];
  const gate = (name: string, ms: number) => async () => {
    events.push(`${name}:start`);
    await new Promise((r) => setTimeout(r, ms));
    events.push(`${name}:end`);
  };
  // The slow one is queued first, which is exactly the case a plain `await` gets wrong: the fast
  // second PATCH would land first and the server would keep the earlier intent.
  const first = serializeById("img:a", gate("first", 20));
  const second = serializeById("img:a", gate("second", 0));
  await Promise.all([first, second]);
  assert.deepEqual(events, ["first:start", "first:end", "second:start", "second:end"]);
});

test("different keys are not serialized against each other", async () => {
  const events: string[] = [];
  const slow = serializeById("img:a", async () => {
    await new Promise((r) => setTimeout(r, 20));
    events.push("a");
  });
  const fast = serializeById("img:b", async () => {
    events.push("b");
  });
  await Promise.all([slow, fast]);
  assert.deepEqual(events, ["b", "a"]);
});

test("a failure does not break the key's chain", async () => {
  const failed = serializeById("img:c", async () => {
    throw new Error("nope");
  });
  await assert.rejects(failed, /nope/);
  assert.equal(await serializeById("img:c", async () => "ran"), "ran");
});

test("the rejection reaches the caller that queued it", async () => {
  await serializeById("img:d", async () => "ok");
  await assert.rejects(
    serializeById("img:d", async () => {
      throw new Error("second failed");
    }),
    /second failed/,
  );
});

test("per item keys let a later pin be stamped first, one key per gallery does not", async () => {
  // The server stamps pinned_at when it RUNS the PATCH and orders pins by that stamp. Two requests
  // in flight together can therefore be stamped in either order, which is what a per-item key
  // allowed: the strip showed the click order and the next load showed the stamp order.
  const stamped = async (keyFor: (id: string) => string) => {
    const order: string[] = [];
    const patch = (id: string, ms: number) => async () => {
      await new Promise((r) => setTimeout(r, ms));
      order.push(id);
    };
    // "a" is clicked first but is the slower request.
    await Promise.all([
      serializeById(keyFor("a"), patch("a", 20)),
      serializeById(keyFor("b"), patch("b", 0)),
    ]);
    return order;
  };
  assert.deepEqual(await stamped((id) => `per-item:${id}`), ["b", "a"]);
  assert.deepEqual(await stamped(() => "one-gallery"), ["a", "b"]);
});

test("a baseline frozen before the request is not softened by a page loaded during it", async () => {
  // The regression: the ids came from before the POST but the window half was read in the catch.
  // Scrolling while the request was in flight paged in historical unpinned rows, which turned a
  // window that must refuse to judge into one that judged, and the newest historical row then read
  // as proof of a generation that never reached the server.
  const pins = [item("p0", 9, true), item("p1", 8, true)];
  const baseline = newRecordProbeBaseline(pins, true, new Set(["p0", "p1"]));
  // The window the user scrolled to, and the listing, both now hold that historical row.
  const listing = [...pins, item("historical", 1)];
  assert.equal(await hasUnknownRecord(baseline, pager(listing, 50), 50), false);
  // Read after the scroll instead, the same moment claims proof, which is the bug.
  const afterScroll = newRecordProbeBaseline(listing, true, new Set(["p0", "p1"]));
  assert.equal(await hasUnknownRecord(afterScroll, pager(listing, 50), 50), true);
});

test("a page fetch re-reads its offset when the shelf shortens mid request", async () => {
  // The server's list, newest first. The client has the first two loaded.
  let shelf = ["e", "d", "c", "b", "a"];
  let loaded = ["e", "d"];
  let epoch = 0;
  const requested: number[] = [];
  const result = await fetchNextPage(
    () => loaded.length,
    () => epoch,
    () => 0,
    async (offset) => {
      requested.push(offset);
      // The archive lands while the first request is in flight: "d" leaves the shelf, so every
      // record behind it shifts up by one on the server AND locally.
      if (requested.length === 1) {
        shelf = shelf.filter((id) => id !== "d");
        loaded = loaded.filter((id) => id !== "d");
      }
      return shelf.slice(offset, offset + 2);
    },
  );
  assert.ok(result);
  // Without the re-read this asks for offset 2 against the shortened shelf and returns ["b", "a"],
  // skipping "c" entirely -- the record that shifted across the boundary.
  assert.deepEqual(requested, [2, 1]);
  assert.deepEqual(result.page, ["c", "b"]);
});

test("a page fetch gives up rather than spinning when the shelf keeps moving", async () => {
  let loaded = 10;
  let calls = 0;
  const result = await fetchNextPage(
    () => loaded,
    () => 0,
    () => 0,
    async () => {
      calls += 1;
      loaded -= 1; // something mutates the list on every single response
      return [];
    },
  );
  assert.equal(result, null);
  assert.equal(calls, PAGE_MAX_ATTEMPTS);
});

test("a failed unpin puts the image back where it was, not at the front of the pins", () => {
  // Pins are ordered by pin TIME, which the client never learns, so a rollback must replay the
  // order it saw. applyPin(..., true) means "freshly pinned" and would promote this to the head.
  const before = [item("first", 1, true), item("second", 2, true), item("third", 3, true)];
  const order = pinnedOrder(before);
  const optimistic = applyPin(before, "third", false);
  assert.deepEqual(ids(optimistic), ["first", "second", "third"]);
  assert.deepEqual(ids(restorePinOrder(optimistic, "third", order)), ["first", "second", "third"]);
  assert.deepEqual(ids(applyPin(optimistic, "third", true)), ["third", "first", "second"]);
});

test("a rollback leaves an image pinned during the request at the front", () => {
  // Absent from the snapshot means pinned since, and the newest pin leads.
  const before = [item("a", 1, true), item("b", 2, true)];
  const order = pinnedOrder(before);
  const withNewPin = applyPin([...before, item("c", 3)], "c", true);
  assert.deepEqual(ids(restorePinOrder(withNewPin, "a", order)), ["c", "a", "b"]);
});

test("a finished generation does not duplicate a record a resync already loaded", () => {
  const fresh = item("new", 9);
  const alreadyLoaded = [fresh, item("old", 1)];
  const merged = mergeGenerated(alreadyLoaded, [fresh]);
  assert.deepEqual(ids(merged), ["new", "old"]);
  assert.equal(merged.length, 2);
  // And it still merges a record nothing had seen.
  assert.deepEqual(ids(mergeGenerated([item("old", 1)], [fresh])), ["new", "old"]);
});

test("a gallery load is discarded when a pin lands while it is in flight", async () => {
  // The response was snapshotted before the PATCH, so applying it would show the image unpinned
  // while the server has it pinned, with nothing scheduled to correct it.
  let epoch = 0;
  let fetches = 0;
  const result = await fetchWhileStable(
    () => epoch,
    async () => {
      fetches += 1;
      if (fetches === 1) epoch += 1; // the user pins mid request
      return [item("a", 1, fetches > 1)];
    },
  );
  assert.equal(fetches, 2);
  assert.ok(result);
  assert.equal(result[0].pinned, true); // the retry sees the server after the pin
});

test("a gallery load gives up rather than overwriting a strip that keeps changing", async () => {
  let epoch = 0;
  let fetches = 0;
  const result = await fetchWhileStable(
    () => epoch,
    async () => {
      fetches += 1;
      epoch += 1;
      return "stale";
    },
  );
  // Null, so the caller keeps its optimistic state, which is what the server already agreed to.
  assert.equal(result, null);
  assert.equal(fetches, PAGE_MAX_ATTEMPTS);
});

test("a page fetch is refused when an archive is merely IN FLIGHT", async () => {
  // The gap the count cannot see: the server shortens the shelf when it PROCESSES the archive,
  // while the count only moves when the response gets back, so only a token bumped at request
  // START reveals a page read inside that round trip.
  const full = ["e", "d", "c", "b", "a"];
  const shortened = ["e", "c", "b", "a"]; // the server has already archived "d"
  let loaded = 2;
  let epoch = 0;
  const seen: string[][] = [];
  const result = await fetchNextPage(
    () => loaded,
    () => epoch,
    () => 0,
    async (offset) => {
      if (seen.length === 0) {
        epoch += 1; // the user clicks Archive while this request is in flight
        const page = shortened.slice(offset, offset + 2);
        seen.push(page);
        return page; // ["b", "a"] -- "c" would be skipped for good
      }
      loaded = 1; // the archive has landed locally by now
      const page = shortened.slice(offset, offset + 2);
      seen.push(page);
      return page;
    },
  );
  assert.deepEqual(seen[0], ["b", "a"], "the first read did skip the boundary record");
  assert.ok(result);
  assert.deepEqual(result.page, ["c", "b"], "the retry picks the skipped record back up");
  assert.equal(full.length, 5);
});

test("a page fetch is refused while a shelf mutation is still pending", async () => {
  // The token is an EDGE, not a state: a page starting after the bump and landing before the row is
  // dropped sees it and the count hold still, so only "is anything in flight" catches this.
  const shortened = ["e", "c", "b", "a"]; // the server has already archived "d"
  let loaded = 2;
  let pending = 1; // the PATCH is in flight for the whole of the first read
  const epoch = 7; // bumped before this page even started, so it never moves again
  const seen: string[][] = [];
  const result = await fetchNextPage(
    () => loaded,
    () => epoch,
    () => pending,
    async (offset) => {
      const page = shortened.slice(offset, offset + 2);
      seen.push(page);
      if (seen.length === 1) {
        pending = 0; // the archive lands, the row is dropped
        loaded = 1;
      }
      return page;
    },
  );
  assert.deepEqual(seen[0], ["b", "a"], "the first read did skip the boundary record");
  assert.ok(result);
  assert.deepEqual(result.page, ["c", "b"], "the retry, once nothing is pending, recovers it");
});

test("archived audio pages from the stable server cursor", () => {
  assert.match(
    archivedMediaSource,
    /listAudioGallery\(\s*0,\s*ARCHIVED_PAGE_SIZE,\s*before,\s*true,?\s*\)/,
  );
  assert.match(
    archivedMediaSource,
    /const page = await loadPage\(\s*rowsRef\.current\.length,\s*audioCursor\.current,?\s*\);[\s\S]*audioCursor\.current = page\.nextAudioCursor;/,
  );
});
