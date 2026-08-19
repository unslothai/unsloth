// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

// source-list imports its sibling extensionlessly, the way vite resolves.
registerBundlerResolver();

const { sortSources } = await import("../src/features/rag/lib/source-list.ts");

const names = (rows: { filename: string }[]) => rows.map((r) => r.filename);

test("uploaded puts the newest first", () => {
  const rows = [
    { filename: "old.pdf", createdAt: "2026-01-01T00:00:00" },
    { filename: "new.pdf", createdAt: "2026-08-01T00:00:00" },
    { filename: "mid.pdf", createdAt: "2026-04-01T00:00:00" },
  ];
  assert.deepEqual(names(sortSources(rows, "uploaded")), [
    "new.pdf",
    "mid.pdf",
    "old.pdf",
  ]);
});

test("documents with no usable date sort last, not first", () => {
  const rows = [
    { filename: "undated.pdf" },
    { filename: "dated.pdf", createdAt: "2026-01-01T00:00:00" },
    { filename: "null-dated.pdf", createdAt: null },
    { filename: "unparseable.pdf", createdAt: "not a date" },
  ];
  const sorted = names(sortSources(rows, "uploaded"));
  assert.equal(sorted[0], "dated.pdf");
  assert.deepEqual(sorted.slice(1), [
    "null-dated.pdf",
    "undated.pdf",
    "unparseable.pdf",
  ]);
});

test("name sorts alphabetically, not by code unit", () => {
  const rows = [
    { filename: "banana.pdf" },
    { filename: "Apple.pdf" },
    { filename: "cherry.pdf" },
  ];
  // A naive < comparison would put "Apple" and "banana" either side of
  // "cherry" on capitalisation alone.
  assert.deepEqual(names(sortSources(rows, "name")), [
    "Apple.pdf",
    "banana.pdf",
    "cherry.pdf",
  ]);
});

test("size puts the largest first and unknown sizes last", () => {
  const rows = [
    { filename: "small.pdf", sizeBytes: 1024 },
    { filename: "unknown.pdf" },
    { filename: "large.pdf", sizeBytes: 5_000_000 },
    { filename: "null-size.pdf", sizeBytes: null },
    { filename: "empty.pdf", sizeBytes: 0 },
  ];
  assert.deepEqual(names(sortSources(rows, "size")), [
    "large.pdf",
    "small.pdf",
    "empty.pdf",
    "null-size.pdf",
    "unknown.pdf",
  ]);
});

test("a zero-byte file still outranks an unknown size", () => {
  const rows = [
    { filename: "unknown.pdf" },
    { filename: "zero.pdf", sizeBytes: 0 },
  ];
  assert.deepEqual(names(sortSources(rows, "size")), [
    "zero.pdf",
    "unknown.pdf",
  ]);
});

test("ties break on filename so the order does not depend on fetch order", () => {
  const rows = [
    { filename: "b.pdf", createdAt: "2026-01-01T00:00:00" },
    { filename: "a.pdf", createdAt: "2026-01-01T00:00:00" },
    { filename: "c.pdf", createdAt: "2026-01-01T00:00:00" },
  ];
  assert.deepEqual(names(sortSources(rows, "uploaded")), [
    "a.pdf",
    "b.pdf",
    "c.pdf",
  ]);
  const reversed = [...rows].reverse();
  assert.deepEqual(
    names(sortSources(reversed, "uploaded")),
    names(sortSources(rows, "uploaded")),
  );
});

test("returns a new array and leaves the input order alone", () => {
  const rows = [
    { filename: "b.pdf", sizeBytes: 1 },
    { filename: "a.pdf", sizeBytes: 2 },
  ];
  const sorted = sortSources(rows, "size");
  assert.notEqual(sorted, rows);
  assert.deepEqual(names(rows), ["b.pdf", "a.pdf"]);
});
