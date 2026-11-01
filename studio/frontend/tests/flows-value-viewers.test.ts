// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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
  const inspector = readSrc("features/flows/editor/output-value.tsx");
  assert.ok(
    inspector.indexOf("isStructuredValue(value)") > 0 &&
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

test("line charts over numbers use a numeric x axis; bars and text stay categorical", () => {
  assert.equal(
    kinds.numericChartX({ type: "line", x: [1024, 2048, 32768] }),
    true,
  );
  assert.equal(kinds.numericChartX({ type: "bar", x: [1, 2, 3] }), false);
  assert.equal(kinds.numericChartX({ type: "line", x: ["q4", "q8"] }), false);
  assert.equal(kinds.numericChartX({ type: "line", x: [] }), false);
});

test("numbers stored as text sort by value, not character by character", () => {
  const rows = [{ n: "10" }, { n: "2" }, { n: "1.9" }, { n: "1.10" }];
  assert.deepEqual(
    sortRows(rows, "n", false).map((r) => r.n),
    ["1.10", "1.9", "2", "10"],
  );
});

test("delta cells are not coloured as good or bad: the table does not know which direction is better", () => {
  const src = readFileSync(
    new URL("../src/features/flows/editor/value-viewers-full.tsx", import.meta.url),
    "utf8",
  );
  assert.equal(/role === "delta"/.test(src), false);
});

test("a column mixing numbers and words sorts as text, consistently", () => {
  const rows = [{ v: "15x" }, { v: "2" }, { v: "10" }];
  const once = sortRows(rows, "v", false).map((r) => r.v);
  assert.deepEqual(once, ["10", "15x", "2"]);
  assert.deepEqual(sortRows([...rows].reverse(), "v", false).map((r) => r.v), once);
});

test("a card with no output ports (Preview, Result) draws what the run returned", () => {
  assert.deepEqual(kinds.shownOutputIds([], { value: ["a", "b"] }), ["value"]);
  assert.deepEqual(kinds.shownOutputIds([{ id: "text" }], { text: "x", extra: 1 }), ["text"]);
});
