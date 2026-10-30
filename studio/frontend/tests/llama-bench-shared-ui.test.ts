// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

const tab = readFileSync(
  new URL("../src/features/benchmarks/llama-bench/llama-bench-tab.tsx", import.meta.url),
  "utf8",
);

test("llama-bench draws with the shared tiles, bar chart and log pane", () => {
  assert.match(tab, /<StatTile/);
  assert.doesNotMatch(tab, /function Tile\(/);
  assert.match(tab, /<BarChart/);
  assert.doesNotMatch(tab, /bg-primary transition-\[width\]/);
  assert.match(tab, /<LogPane\s+text=\{job\.log\.map\(stripAnsi\)\.join\("\\n"\)\}/);
  assert.doesNotMatch(tab, /bg-black\/80/);
});
