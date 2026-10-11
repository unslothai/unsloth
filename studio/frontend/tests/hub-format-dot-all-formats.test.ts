// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { matchesFormat } from "../src/features/hub/lib/format-filters.ts";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const HUB_PAGE = read("../src/features/hub/hub-page.tsx");
const CATALOG = read("../src/features/hub/catalog/models-catalog.tsx");
const LISTS = read("../src/features/hub/catalog/models-catalog-lists.tsx");
const TABLE = read("../src/features/hub/catalog/models-table.tsx");
const ROWS = read("../src/features/hub/catalog/models-catalog-rows.tsx");

test("a single format filter admits one dot color, so the dot says nothing", () => {
  const formats = [
    "gguf",
    "safetensors",
    "checkpoint",
    "adapter",
    "mlx",
  ] as const;
  for (const filter of ["gguf", "checkpoint", "mlx"] as const) {
    const admitted = formats.filter((f) => matchesFormat(f, filter));
    // safetensors and checkpoint share one dot.
    const dots = new Set(
      admitted.map((f) => (f === "safetensors" ? "checkpoint" : f)),
    );
    assert.equal(dots.size, 1, filter);
  }
});

test("format dots show only under All formats", () => {
  assert.ok(
    HUB_PAGE.includes(
      'showFormatDots: isDatasetMode || deferredFormatFilter === "all"',
    ),
  );
  assert.ok(CATALOG.includes("showFormatDots={showFormatDots}"));
  assert.equal(LISTS.match(/showFormatDot=\{showFormatDots\}/g)?.length, 4);
});

test("every row drops its format dot when told to", () => {
  assert.equal(
    TABLE.match(/isDataset \|\| !showFormatDot\n\s*\? null/g)?.length,
    3,
  );
  assert.ok(ROWS.includes("{showFormatDot && row.isGguf && ("));
  assert.ok(
    ROWS.includes('{showFormatDot && row.modelFormat === "adapter" && ('),
  );
  // The tooltip legend describes the dots, so it drops them too.
  assert.ok(ROWS.includes("isGguf: showFormatDot && row.isGguf,"));
});

test("status dots stay regardless of the format filter", () => {
  assert.ok(ROWS.includes('<StatusDot tone="success" label="On device" />'));
  assert.ok(TABLE.includes('aria-label="Partial download"'));
});
