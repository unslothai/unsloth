// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

// source-list imports its sibling extensionlessly, the way vite resolves.
registerBundlerResolver();

const { filterSources } = await import(
  "../src/features/rag/lib/source-list.ts"
);

const docs = [
  { filename: "Practical_statistics_for_data_scientists.pdf" },
  { filename: "BulkRNA_Manuscript_Revised.pdf" },
  { filename: "ETHFellowsGuidelines.pdf" },
  { filename: "notes (draft) [v2].md" },
];

test("an empty or whitespace query keeps every source", () => {
  for (const query of ["", "   ", "\t\n"]) {
    assert.deepEqual(filterSources(docs, query), docs);
  }
});

test("matches a substring anywhere in the filename", () => {
  assert.deepEqual(
    filterSources(docs, "manuscript").map((d) => d.filename),
    ["BulkRNA_Manuscript_Revised.pdf"],
  );
  assert.equal(filterSources(docs, ".pdf").length, 3);
});

test("ignores case on both sides", () => {
  assert.equal(filterSources(docs, "ETHFELLOWS").length, 1);
  assert.equal(filterSources(docs, "ethfellows").length, 1);
});

test("trims the query before matching", () => {
  assert.equal(filterSources(docs, "   BulkRNA   ").length, 1);
});

test("treats the query as literal text, not a pattern", () => {
  // A regex-flavoured read of "(draft)" would match nothing here.
  assert.equal(filterSources(docs, "(draft)").length, 1);
  assert.equal(filterSources(docs, "[v2]").length, 1);
  assert.equal(filterSources(docs, ".*").length, 0);
});

test("no match returns an empty list", () => {
  assert.deepEqual(filterSources(docs, "does-not-exist"), []);
});

test("never mutates or aliases the input array", () => {
  const result = filterSources(docs, "");
  assert.notEqual(result, docs);
  result.pop();
  assert.equal(docs.length, 4);
});
