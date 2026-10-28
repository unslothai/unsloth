// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  columnStats,
  datasetSummary,
  datasetTitle,
  isDatasetValue,
  knownTotal,
  pageCount,
} = await import("../src/features/flows/dataset/dataset-value.ts");

const recipe = {
  kind: "dataset" as const,
  source: "recipe" as const,
  ref: "recipes/recipe_3cf/parquet-files",
  rows: 20,
};
const hub = { kind: "dataset" as const, source: "hf" as const, ref: "org/sql" };
const rowsDataset = {
  kind: "dataset" as const,
  source: "local" as const,
  ref: "ref:abc",
  rows: 7,
  columns: ["a", "b"],
};

test("dataset values are told apart from other outputs", () => {
  assert.equal(isDatasetValue(recipe), true);
  assert.equal(isDatasetValue({ kind: "table", rows: [] }), false);
  assert.equal(isDatasetValue({ kind: "dataset" }), false);
  assert.equal(isDatasetValue(null), false);
});

test("each source reads as a short title and summary, never a raw ref", () => {
  assert.equal(datasetTitle(hub), "org/sql");
  assert.equal(datasetTitle(recipe), "Data Recipe");
  assert.equal(datasetTitle(rowsDataset), "Uploaded");
  const stored = {
    ...rowsDataset,
    ref: "uploads/0123456789abcdef0123456789abcdef_bench-rows.jsonl",
  };
  assert.equal(datasetTitle(stored), "bench-rows.jsonl");
  assert.equal(datasetSummary(recipe), "Data Recipe · 20 rows");
  assert.equal(datasetSummary(rowsDataset), "Uploaded · 7 rows · 2 columns");
  assert.equal(datasetSummary(hub, 1234), "Hugging Face · 1,234 rows");
});

test("paging uses the reader's total, then the value's count, then how far rows came back", () => {
  const page = (total: number | null, offset: number, n: number) => ({
    total,
    offset,
    rows: Array.from({ length: n }, () => ({})),
  });
  assert.equal(knownTotal(hub, page(500, 0, 20), 20), 500);
  assert.equal(knownTotal(rowsDataset, page(null, 0, 7), 20), 7);
  assert.equal(knownTotal(hub, page(null, 20, 5), 20), 25);
  assert.equal(knownTotal(hub, page(null, 0, 20), 20), null);
  assert.equal(pageCount(45, 1000, 20, 0), 3);
  assert.equal(pageCount(5000, 1000, 20, 0), 50);
  assert.equal(pageCount(null, 1000, 20, 40), 4);
  assert.equal(pageCount(0, 1000, 20, 0), 1);
});

test("the Columns tab gets a type, distinct count and nulls per column", () => {
  const stats = columnStats([
    { q: "a", n: 1, tags: ["x"] },
    { q: "a", n: 2.5, extra: null },
    { q: "b", n: null },
  ]);
  const by = Object.fromEntries(stats.map((s) => [s.column_name, s]));
  assert.deepEqual(Object.keys(by), ["q", "n", "tags", "extra"]);
  assert.equal(by.q.simple_dtype, "string");
  assert.equal(by.q.num_unique, 2);
  assert.equal(by.n.simple_dtype, "int | float");
  assert.equal(by.n.num_null, 1);
  assert.equal(by.tags.simple_dtype, "list");
  assert.equal(by.extra.simple_dtype, "null");
});

test("node, inspector, app and run drawer outputs all show datasets with the viewer", () => {
  const read = (path: string) =>
    readFileSync(new URL(`../src/features/flows/${path}`, import.meta.url), "utf8");
  assert.match(read("editor/flow-node.tsx"), /<DatasetOutput value=\{value\} compact \/>/);
  assert.match(read("editor/output-value.tsx"), /<DatasetOutput value=\{value\} \/>/);
  assert.match(read("editor/run-drawer.tsx"), /<DatasetOutput value=\{item\.value\} compact \/>/);
  const viewer = read("dataset/dataset-viewer.tsx");
  assert.match(viewer, /ExecutionDataTab/);
  assert.match(viewer, /ExecutionColumnsTab/);
  assert.match(read("editor/dataset-output.tsx"), /import\("\.\.\/dataset\/dataset-viewer"\)/);
});
