// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { SOURCES_COLLAPSED_LIMIT, sortSources, visibleSources } = await import(
  "../src/features/rag/lib/source-list.ts"
);

const names = (rows: { filename: string }[]) => rows.map((r) => r.filename);

const settled = (n: number) =>
  Array.from({ length: n }, (_, i) => ({
    id: `doc-${i}`,
    filename: `settled-${String(i).padStart(3, "0")}.pdf`,
    status: "completed" as const,
    createdAt: `2026-01-${String((i % 28) + 1).padStart(2, "0")}T00:00:00`,
    sizeBytes: 1000 + i,
  }));

const optimistic = {
  id: "pending_abc",
  filename: "just-dropped.pdf",
  status: "pending" as const,
};

test("an optimistic upload sorts first in every mode", () => {
  // It has no createdAt and no sizeBytes, so each mode would otherwise rank it
  // last against fully-populated rows.
  for (const mode of ["uploaded", "name", "size"] as const) {
    const sorted = sortSources([...settled(5), optimistic], mode);
    assert.equal(sorted[0].filename, "just-dropped.pdf", mode);
  }
});

test("a row still indexing also stays at the top", () => {
  const running = {
    id: "doc-live",
    filename: "zzz-indexing.pdf",
    status: "running" as const,
    createdAt: "2020-01-01T00:00:00",
  };
  const sorted = sortSources([...settled(5), running], "uploaded");
  assert.equal(sorted[0].filename, "zzz-indexing.pdf");
});

test("a fresh upload survives the collapse limit on a large project", () => {
  // The regression: past 25 sources the new chip was sliced out of the
  // collapsed view, so the upload looked like it never started.
  const docs = sortSources([...settled(40), optimistic], "uploaded");
  const shown = visibleSources(docs, { expanded: false, searching: false });
  assert.equal(shown.length, SOURCES_COLLAPSED_LIMIT);
  assert.ok(
    names(shown).includes("just-dropped.pdf"),
    "the in-flight upload was hidden behind the collapse limit",
  );
});

test("several concurrent uploads all stay visible", () => {
  const uploads = Array.from({ length: 3 }, (_, i) => ({
    id: `pending_${i}`,
    filename: `upload-${i}.pdf`,
    status: "pending" as const,
  }));
  const shown = visibleSources(
    sortSources([...settled(40), ...uploads], "uploaded"),
    { expanded: false, searching: false },
  );
  for (const upload of uploads) {
    assert.ok(names(shown).includes(upload.filename), upload.filename);
  }
});

test("settled rows keep their own ordering behind the in-flight ones", () => {
  const sorted = sortSources([...settled(3), optimistic], "name");
  assert.deepEqual(names(sorted).slice(1), [
    "settled-000.pdf",
    "settled-001.pdf",
    "settled-002.pdf",
  ]);
});

test("a completed upload that never got server metadata still sinks", () => {
  // Documents why handleFiles refreshes after every upload. The in-flight pin
  // releases the moment status flips to "completed", so a row that finished
  // before any refresh -- still carrying no createdAt and no sizeBytes -- sorts
  // to the bottom and falls out of the collapsed slice. Sorting cannot fix
  // this: it has no timestamp to rank by. Only fetching the server row can.
  const stale = {
    id: "doc-done",
    filename: "just-finished.pdf",
    status: "completed" as const,
  };
  const sorted = sortSources([...settled(40), stale], "uploaded");
  assert.equal(sorted.at(-1)?.filename, "just-finished.pdf");
  const shown = visibleSources(sorted, { expanded: false, searching: false });
  assert.ok(!names(shown).includes("just-finished.pdf"));
});

test("ordering is unchanged when nothing is in flight", () => {
  const rows = settled(3);
  assert.deepEqual(
    names(sortSources(rows, "name")),
    names(sortSources(rows, "name")),
  );
  assert.equal(sortSources(rows, "name")[0].filename, "settled-000.pdf");
});
