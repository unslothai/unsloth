// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const { placeNavRows } = await import("../src/components/nav-row-state.ts");

const SHIPPED = [
  { id: "hub", pinned: true },
  { id: "projects", pinned: true },
  { id: "library", pinned: true },
  { id: "images", pinned: true },
  { id: "video", pinned: false },
  { id: "audio", pinned: false },
  { id: "train", pinned: true },
  { id: "recipes", pinned: false },
  { id: "export", pinned: false },
  { id: "api", pinned: false },
];

function onlyUnpinned(...ids: string[]) {
  return SHIPPED.map((row) => ({ id: row.id, pinned: !ids.includes(row.id) }));
}

test("with nothing surfaced, pinned rows stay inline and the rest go to More", () => {
  assert.deepEqual(placeNavRows(SHIPPED, null), {
    inline: ["hub", "projects", "library", "images", "train"],
    overflow: ["video", "audio", "recipes", "export", "api"],
  });
});

test("a surfaced row takes its own slot and leaves More", () => {
  assert.deepEqual(placeNavRows(SHIPPED, "audio"), {
    inline: ["hub", "projects", "library", "images", "audio", "train"],
    overflow: ["video", "recipes", "export", "api"],
  });
});

test("surfacing a pinned row changes nothing", () => {
  const pinned = SHIPPED.map((row) => ({
    ...row,
    pinned: row.pinned || row.id === "audio",
  }));
  assert.deepEqual(placeNavRows(pinned, "audio"), placeNavRows(pinned, null));
});

test("surfacing never hides the row it shared More with", () => {
  const rows = onlyUnpinned("video", "audio");
  assert.deepEqual(placeNavRows(rows, null).overflow, ["video", "audio"]);
  assert.deepEqual(placeNavRows(rows, "audio").overflow, ["video"]);
});

test("a lone unpinned row is still dropped with More, unless its page surfaces it", () => {
  const rows = onlyUnpinned("audio");
  const away = placeNavRows(rows, null);
  assert.deepEqual(away.overflow, []);
  assert.ok(!away.inline.includes("audio"));
  const here = placeNavRows(rows, "audio");
  assert.deepEqual(here.overflow, []);
  assert.deepEqual(
    here.inline,
    SHIPPED.map((row) => row.id),
  );
});

test("a lone unpinned row of another page stays dropped on Audio", () => {
  const rows = onlyUnpinned("video");
  for (const shown of [null, "audio"]) {
    const placed = placeNavRows(rows, shown);
    assert.deepEqual(placed.overflow, []);
    assert.ok(!placed.inline.includes("video"));
  }
});
