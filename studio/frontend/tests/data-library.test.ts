// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEFAULT_LIBRARY_FILTERS,
  filterLibraryItems,
  formatLibraryDate,
  groupLibraryItems,
  type LibraryItem,
} from "../src/features/settings/components/data-library.ts";

const labels = {
  noProject: "No project",
  unavailableProject: "Unavailable project",
};
const projects = new Map([
  ["p1", "Research notes"],
  ["p2", "Research notes"],
]);
const items: LibraryItem[] = [
  {
    id: "a",
    title: "Café research",
    createdAt: 10,
    updatedAt: 40,
    projectId: "p1",
    type: "single",
  },
  {
    id: "b",
    title: "Beta",
    createdAt: 30,
    updatedAt: 10,
    projectId: "p2",
    type: "compare",
  },
  { id: "c", title: "日本語", createdAt: 20, type: "single" },
];
const ids = (rows: LibraryItem[]) => rows.map((row) => row.id);
const filter = (overrides: Partial<typeof DEFAULT_LIBRARY_FILTERS>) =>
  filterLibraryItems(
    items,
    { ...DEFAULT_LIBRARY_FILTERS, ...overrides },
    projects,
    labels,
  );

test("search matches normalized title and project terms in any order", () => {
  for (const query of ["  NOTES   CAFE ", "Café Research", "cafe\u0301"])
    assert.deepEqual(ids(filter({ query })), ["a"]);
  assert.deepEqual(ids(filter({ query: "日本" })), ["c"]);
  assert.deepEqual(ids(filter({ query: "No project" })), ["c"]);
  assert.deepEqual(ids(filter({ query: "missing" })), []);
});

test("project identity stays distinct when names are duplicated", () => {
  assert.deepEqual(ids(filter({ project: "project:p1" })), ["a"]);
  assert.deepEqual(ids(filter({ project: "project:p2" })), ["b"]);
  assert.deepEqual(ids(filter({ project: "none" })), ["c"]);
  assert.deepEqual(ids(filter({ project: "project:removed" })), []);
});

test("type, project and text filters compose", () => {
  assert.deepEqual(ids(filter({ type: "compare", query: "research" })), ["b"]);
  assert.deepEqual(ids(filter({ type: "single", project: "project:p2" })), []);
});

test("default ordering preserves the server page order", () => {
  assert.deepEqual(ids(filter({ sort: "default" })), ["a", "b", "c"]);
});

test("sorts use the requested timestamp with a creation fallback", () => {
  assert.deepEqual(ids(filter({ sort: "updated" })), ["a", "c", "b"]);
  assert.deepEqual(ids(filter({ sort: "created" })), ["b", "c", "a"]);
  assert.deepEqual(ids(filter({ sort: "oldest" })), ["a", "c", "b"]);
  assert.deepEqual(ids(filter({ sort: "alphabetical" })).slice(0, 2), [
    "b",
    "a",
  ]);
});

test("equal timestamps and titles have a stable id tie break", () => {
  const same = [
    { id: "z", title: "Same", createdAt: 1 },
    { id: "a", title: "Same", createdAt: 1 },
  ];
  assert.deepEqual(ids(filterLibraryItems(same, DEFAULT_LIBRARY_FILTERS)), [
    "a",
    "z",
  ]);
  assert.deepEqual(ids(same), ["z", "a"]);
});

test("legacy timestamps do not produce invalid date text or unstable ordering", () => {
  assert.equal(formatLibraryDate(Number.NaN), "");
  assert.equal(formatLibraryDate(Number.POSITIVE_INFINITY), "");
  assert.equal(formatLibraryDate(1e30), "");
  const rows = [
    { id: "invalid", title: "Invalid", createdAt: NaN },
    { id: "valid", title: "Valid", createdAt: 1 },
  ];
  assert.deepEqual(ids(filterLibraryItems(rows, DEFAULT_LIBRARY_FILTERS)), [
    "valid",
    "invalid",
  ]);
});

test("groups keep project ids, sorted row order and missing projects", () => {
  const groups = groupLibraryItems(
    [...items, { id: "d", title: "Missing", createdAt: 1, projectId: "gone" }],
    projects,
    labels,
  );
  assert.deepEqual(
    groups.map((group) => group.id),
    ["p1", "p2", "", "gone"],
  );
  assert.equal(groups[2].name, "No project");
  assert.equal(groups[3].name, "Unavailable project");
  assert.deepEqual(
    groups.flatMap((group) => ids(group.items)),
    ["a", "b", "c", "d"],
  );
});

test("filtering a large library reaches records after the rendered page", () => {
  const many = Array.from({ length: 10000 }, (_, i) => ({
    id: String(i),
    title: `Chat ${i}`,
    createdAt: i,
  }));
  assert.deepEqual(
    ids(
      filterLibraryItems(many, {
        ...DEFAULT_LIBRARY_FILTERS,
        query: "Chat 9999",
      }),
    ),
    ["9999"],
  );
  assert.equal(many[0].id, "0");
});

test("localized project labels drive matching and grouping", () => {
  const spanish = {
    noProject: "Sin proyecto",
    unavailableProject: "Proyecto no disponible",
  };
  const rows = [
    ...items,
    { id: "missing", title: "Lost", projectId: "gone", createdAt: 1 },
  ];
  for (const [query, expected] of [
    ["SIN PROYECTO", "c"],
    ["disponible proyecto", "missing"],
  ]) {
    assert.deepEqual(
      ids(
        filterLibraryItems(
          rows,
          { ...DEFAULT_LIBRARY_FILTERS, query },
          projects,
          spanish,
        ),
      ),
      [expected],
    );
  }
  const groups = groupLibraryItems(rows, projects, spanish);
  assert.equal(groups[2].name, spanish.noProject);
  assert.equal(groups[3].name, spanish.unavailableProject);
  assert.deepEqual(
    ids(
      filterLibraryItems(
        rows,
        { ...DEFAULT_LIBRARY_FILTERS, query: "No project" },
        projects,
        spanish,
      ),
    ),
    [],
  );
});

test("media without project labels cannot match an invented project name", () => {
  assert.deepEqual(
    filterLibraryItems(items, {
      ...DEFAULT_LIBRARY_FILTERS,
      query: "No project",
    }),
    [],
  );
});

test("dates use the requested app locale", () => {
  const ms = 1700000000000;
  for (const locale of ["es", "ja", "ar"]) {
    assert.equal(
      formatLibraryDate(ms, locale),
      new Date(ms).toLocaleString(locale, {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      }),
    );
  }
});

test("title ordering and timestamp ties follow the selected app locale", () => {
  const rows = [
    { id: "z", title: "Zebra", createdAt: 1, updatedAt: 1 },
    { id: "a", title: "阿", createdAt: 1, updatedAt: 1 },
    { id: "b", title: "八", createdAt: 1, updatedAt: 1 },
  ];
  const orders = {
    en: ["z", "b", "a"],
    "zh-CN": ["a", "b", "z"],
    ja: ["z", "a", "b"],
  };
  for (const sort of [
    "alphabetical",
    "created",
    "updated",
    "oldest",
  ] as const) {
    for (const [locale, expected] of Object.entries(orders)) {
      assert.deepEqual(
        ids(
          filterLibraryItems(
            rows,
            { ...DEFAULT_LIBRARY_FILTERS, sort },
            undefined,
            undefined,
            locale,
          ),
        ),
        expected,
      );
    }
  }
  assert.deepEqual(
    ids(
      filterLibraryItems(
        rows,
        { ...DEFAULT_LIBRARY_FILTERS, sort: "default" },
        undefined,
        undefined,
        "zh-CN",
      ),
    ),
    ["z", "a", "b"],
  );
  assert.deepEqual(ids(rows), ["z", "a", "b"]);
});
