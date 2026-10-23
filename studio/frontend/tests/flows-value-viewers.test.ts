// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const kinds = await import("../src/features/flows/editor/value-kinds.ts");
const { typesCompatible } = await import(
  "../src/features/flows/editor/connection-rules.ts"
);
const {
  STRUCTURED_KINDS,
  structuredKind,
  structuredSummary,
  sortRows,
  tableToCsv,
  tableToMarkdown,
  tensorAsHeatmap,
  numericStats,
} = kinds;

const table = {
  kind: "table" as const,
  columns: [
    { id: "quant", label: "Quant", type: "text" },
    { id: "tps", label: "Speed", type: "number", unit: "tok/s" },
  ],
  rows: [
    { quant: "Q8_0", tps: 70 },
    { quant: "Q4_K_M", tps: 100.25 },
    { quant: 'odd, "name"|x', tps: null },
  ],
};

test("structured values dispatch by kind; everything else falls through", () => {
  assert.equal(structuredKind(table), "table");
  assert.equal(
    structuredKind({ kind: "chart", type: "bar", x: [], series: [] }),
    "chart",
  );
  assert.equal(
    structuredKind({ kind: "heatmap", rows: [], cols: [], values: [] }),
    "heatmap",
  );
  assert.equal(
    structuredKind({ kind: "heatmap", rows: [], cols: [], ref: "sha256:0" }),
    "heatmap",
  );
  assert.equal(structuredKind({ kind: "tensor", shape: [2, 2] }), "tensor");
  for (const other of [
    null,
    "text",
    3,
    [table],
    { kind: "image", ref: "sha256:0", url: "/x" },
    { kind: "table" },
    { kind: "mystery", rows: [] },
    { a: 1 },
  ]) {
    assert.equal(structuredKind(other), null, JSON.stringify(other));
  }
});

test("every structured kind has a viewer and every renderer checks the registry before its JSON fallback", () => {
  const registry = readSrc("features/flows/editor/value-viewers.tsx");
  for (const kind of STRUCTURED_KINDS)
    assert.match(registry, new RegExp(`\\b${kind}: lazy\\(`));
  const inspector = readSrc("features/flows/editor/inspector.tsx");
  assert.ok(
    inspector.indexOf("isStructuredValue(value)") <
      inspector.indexOf("JSON.stringify(value, null, 2)"),
  );
  const node = readSrc("features/flows/editor/flow-node.tsx");
  assert.ok(
    node.indexOf("<StructuredBadge") <
      node.indexOf("Object.keys(value as object)"),
  );
  assert.match(
    readSrc("features/flows/editor/run-drawer.tsx"),
    /<StructuredOutput value=/,
  );
  // The chart library stays out of the editor chunk: only the lazy full viewer imports it.
  assert.doesNotMatch(registry, /from "recharts"/);
  assert.match(registry, /import\("\.\/value-viewers-full"\)/);
});

test("new value ports connect to json and any, not to unrelated ports", () => {
  for (const type of STRUCTURED_KINDS) {
    assert.ok(
      typesCompatible(
        { id: "o", type, label: "" },
        { id: "i", type: "json", label: "" },
      ),
    );
    assert.ok(
      typesCompatible(
        { id: "o", type, label: "" },
        { id: "i", type: "any", label: "" },
      ),
    );
    assert.ok(
      typesCompatible(
        { id: "o", type, label: "" },
        { id: "i", type, label: "" },
      ),
    );
    assert.ok(
      !typesCompatible(
        { id: "o", type, label: "" },
        { id: "i", type: "image", label: "" },
      ),
    );
  }
});

test("tables sort numerically with empty cells last and export as CSV and markdown", () => {
  const rows = table.rows as Record<string, unknown>[];
  assert.deepEqual(
    sortRows(rows, "tps", true).map((r) => r.tps),
    [100.25, 70, null],
  );
  assert.deepEqual(
    sortRows(rows, "tps", false).map((r) => r.tps),
    [70, 100.25, null],
  );
  assert.equal(
    tableToCsv(table),
    'Quant,Speed (tok/s)\nQ8_0,70\nQ4_K_M,100.25\n"odd, ""name""|x",',
  );
  assert.equal(
    tableToMarkdown(table),
    '| Quant | Speed (tok/s) |\n| --- | ---: |\n| Q8_0 | 70 |\n| Q4_K_M | 100.3 |\n| odd, "name"\\|x |  |',
  );
});

test("summaries count truncated tables by row_count, and 2-D tensors view as heatmaps", () => {
  assert.equal(
    structuredSummary({ ...table, row_count: 5000 }),
    "Table: 5000 rows x 2 columns",
  );
  const t = {
    kind: "tensor" as const,
    shape: [2, 3],
    data: [1, 2, 3, 4, Number.NaN, 6],
  };
  const h = tensorAsHeatmap(t);
  assert.ok(h);
  assert.deepEqual(h.values, [
    [1, 2, 3],
    [4, null, 6],
  ]);
  assert.equal(
    tensorAsHeatmap({ kind: "tensor", shape: [2, 2, 2], data: [] }),
    null,
  );
  assert.deepEqual(numericStats(h.values), {
    min: 1,
    max: 6,
    mean: 16 / 5,
    count: 5,
  });
});

test("lookalike JSON with the wrong element shapes keeps the JSON view", () => {
  assert.equal(structuredKind({ kind: "chart", x: [1], series: [{}] }), null);
  assert.equal(
    structuredKind({ kind: "table", columns: ["a"], rows: [1] }),
    null,
  );
  assert.equal(
    structuredKind({ kind: "heatmap", rows: [], cols: [], values: [1, 2] }),
    null,
  );
  assert.equal(
    structuredKind({ kind: "chart", x: [1], series: [{ name: "a", y: [1] }] }),
    "chart",
  );
});
