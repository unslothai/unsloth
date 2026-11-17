// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { GalleryImage } from "../src/features/images/api.ts";
import { collectGalleryWindow } from "../src/features/images/lib/gallery-window.ts";

function fakeGallery(ids: string[]) {
  const records = ids.map((id) => ({ id }) as GalleryImage);
  const server = {
    records,
    calls: 0,
    insertNewest(id: string) {
      records.unshift({ id } as GalleryImage);
    },
    fetchPage: async (offset: number, limit: number) => {
      server.calls += 1;
      return { images: records.slice(offset, offset + limit), has_more: offset + limit < records.length };
    },
  };
  return server;
}

const range = (n: number, prefix = "old") => Array.from({ length: n }, (_, i) => `${prefix}${i + 1}`);
const strip = (ids: string[]) => ids.map((id) => ({ id }) as GalleryImage);
async function collect(...args: Parameters<typeof collectGalleryWindow>) {
  const window = await collectGalleryWindow(...args);
  assert.ok(window);
  return window;
}

test("a finished run that inserts at the front does not push the selected image out of the refreshed window", async () => {
  const gallery = fakeGallery(range(300));
  gallery.insertNewest("new1");
  const refreshed = await collect(gallery.fetchPage, strip(range(100)), 100, 50, () => "old100");
  assert.ok(refreshed.images.some((image) => image.id === "old100"));
  assert.equal(refreshed.images[0].id, "new1");
  assert.equal(refreshed.hasMore, true);
  // Contiguous from the top, so the next loadMore offset follows it.
  assert.deepEqual(
    refreshed.images.map((image) => image.id),
    gallery.records.slice(0, refreshed.images.length).map((image) => image.id),
  );
});

test("without a selection past the window, the refresh fetches the same count as before", async () => {
  const gallery = fakeGallery(range(300));
  gallery.insertNewest("new1");
  const refreshed = await collect(gallery.fetchPage, strip(range(100)), 100, 50, () => "old10");
  assert.equal(refreshed.images.length, 100);
  assert.equal(gallery.calls, 2);
  const noSelection = await collect(gallery.fetchPage, strip(range(100)), 100, 50, () => null);
  assert.equal(noSelection.images.length, 100);
});

test("a selection deleted elsewhere costs one extra page on the next refresh, and none after", async () => {
  const gallery = fakeGallery(range(4000));
  const refreshed = await collect(gallery.fetchPage, strip(["gone", ...range(99)]), 100, 50, () => "gone");
  assert.equal(refreshed.images.length, 150);
  assert.equal(gallery.calls, 3);
  const again = await collect(gallery.fetchPage, refreshed.images, 150, 50, () => "gone");
  assert.equal(again.images.length, 150);
  assert.equal(gallery.calls, 6);
});

test("a refresh superseded during a request stops before the next page", async () => {
  const gallery = fakeGallery(range(5000));
  const superseded = () => gallery.calls < 1;
  const refreshed = await collectGalleryWindow(gallery.fetchPage, strip(range(5000)), 5000, 50, () => "old1", superseded);
  assert.equal(refreshed, null);
  assert.equal(gallery.calls, 1);
});

test("a short gallery stops at its end", async () => {
  const gallery = fakeGallery(range(30));
  const refreshed = await collect(gallery.fetchPage, strip(["gone"]), 50, 50, () => "gone");
  assert.equal(refreshed.images.length, 30);
  assert.equal(refreshed.hasMore, false);
  assert.equal(gallery.calls, 1);
});

test("a pick made while the refresh is fetching is kept, even one pushed past the window by the new record", async () => {
  for (const pickDuringCall of [1, 2]) {
    const gallery = fakeGallery(range(100));
    gallery.insertNewest("new1");
    let selected: string | null = "old5";
    const fetchPage = async (offset: number, limit: number) => {
      const page = await gallery.fetchPage(offset, limit);
      if (gallery.calls === pickDuringCall) selected = "old100";
      return page;
    };
    const refreshed = await collect(fetchPage, strip(range(100)), 100, 50, () => selected);
    assert.ok(
      refreshed.images.some((image) => image.id === "old100"),
      `picked during request ${pickDuringCall}`,
    );
    assert.equal(refreshed.images.length, 101);
    assert.equal(refreshed.hasMore, false);
  }
});
