// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Collapsing is a display cap and must not narrow anything else. These pin the
 * relationship the panel relies on: the set "Select all" covers is derived from
 * the matched list, never from the rendered slice. */

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  SOURCES_COLLAPSED_LIMIT,
  filterSources,
  isBulkRemovable,
  sortSources,
  visibleSources,
} = await import("../src/features/rag/lib/source-list.ts");

const docs = (n: number, over: (i: number) => object = () => ({})) =>
  Array.from({ length: n }, (_, i) => ({
    id: `doc-${i}`,
    filename: `report-${String(i).padStart(3, "0")}.pdf`,
    status: "completed" as const,
    managed: false,
    createdAt: `2026-01-01T00:00:${String(i % 60).padStart(2, "0")}`,
    ...over(i),
  }));

type Row = {
  id: string;
  filename: string;
  status: "completed" | "pending" | "running" | "failed";
  managed: boolean;
  createdAt?: string;
};

/** What the panel computes: match -> sort -> (selectable | rendered slice). */
function panel(all: Row[], query = "", expanded = false) {
  const matched = sortSources(filterSources(all, query), "uploaded");
  const searching = query.trim() !== "";
  return {
    selectable: matched.filter(isBulkRemovable),
    shown: visibleSources(matched, { expanded, searching }),
  };
}

test("Select all covers every source, not the collapsed slice", () => {
  // The reported bug: 27 sources, 25 rendered, "Select all" caught only 25.
  const { selectable, shown } = panel(docs(27));
  assert.equal(shown.length, SOURCES_COLLAPSED_LIMIT);
  assert.equal(selectable.length, 27);
});

test("expanding does not change what Select all covers", () => {
  const all = docs(40);
  assert.equal(panel(all, "", false).selectable.length, 40);
  assert.equal(panel(all, "", true).selectable.length, 40);
});

test("Select all is scoped to the search, and spans matches past the limit", () => {
  // 30 matches of "report-0", well past the render cap.
  const all = [
    ...docs(30),
    ...docs(5).map((d, i) => ({
      ...d,
      id: `other-${i}`,
      filename: `memo-${i}.pdf`,
    })),
  ];
  const { selectable } = panel(all, "report-0");
  assert.equal(selectable.length, 30);
  assert.ok(selectable.every((d) => d.filename.startsWith("report-0")));
});

test("ineligible sources are excluded however many are rendered", () => {
  // Linked-folder and indexing rows can never be bulk removed, on screen or not.
  const all = [
    ...docs(26),
    {
      id: "linked",
      filename: "synced.pdf",
      status: "completed" as const,
      managed: true,
    },
    {
      id: "indexing",
      filename: "busy.pdf",
      status: "running" as const,
      managed: false,
    },
  ];
  const { selectable } = panel(all);
  assert.equal(selectable.length, 26);
  assert.ok(!selectable.some((d) => d.id === "linked" || d.id === "indexing"));
});

test("collapsing never changes the matched set itself", () => {
  const all = docs(60);
  const matched = sortSources(filterSources(all, ""), "uploaded");
  for (const expanded of [false, true]) {
    const shown = visibleSources(matched, { expanded, searching: false });
    // The slice is a view of the same ordering, never a different membership.
    assert.deepEqual(
      shown.map((d) => d.id),
      matched.slice(0, shown.length).map((d) => d.id),
      `expanded=${expanded}`,
    );
  }
});
