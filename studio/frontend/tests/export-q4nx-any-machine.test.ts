// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

// Q4NX conversion runs on the CPU, so the export page must not hide or veto it on a host
// without an NPU: a GPU box trains and converts for a Ryzen AI PC.
const page = readFileSync(
  new URL("../src/features/export/export-page.tsx", import.meta.url),
  "utf8",
);
const card = readFileSync(
  new URL(
    "../src/features/export/components/q4nx-convert-card.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("the export page does not read NPU status", () => {
  assert.doesNotMatch(page, /useNpuStatus|npuStatus|hasNpu/);
});

test("the Q4NX convert card renders unconditionally", () => {
  const cards = page.match(/^.*<Q4nxConvertCard \/>.*$/gm) ?? [];
  assert.equal(cards.length, 2);
  for (const line of cards) {
    assert.equal(line.trim(), "<Q4nxConvertCard />");
  }
});

test("the Q4NX toggle depends only on the quant and the destination", () => {
  const effective = page.match(/const effectiveNpuQ4nx =\s*([^;]+);/);
  assert.ok(effective);
  assert.equal(
    effective[1].replace(/\s+/g, " ").trim(),
    'npuQ4nx && q4nxSourceSelected && destination === "local"',
  );
  const toggle = page.indexOf('aria-label="Also convert to Q4NX (AMD NPU)"');
  assert.ok(toggle > 0);
  const row = page.lastIndexOf('<div className="flex items-start', toggle);
  // Under an NPU gate the row would open right after `{hasNpu && (`.
  assert.doesNotMatch(page.slice(0, row).trimEnd(), /&& \($/);
});

test("the toggle and card copy do not say this machine needs an NPU", () => {
  for (const source of [page, card]) {
    assert.doesNotMatch(source, /Only supported on AMD NPUs/);
    assert.match(source, /converting works on any machine/);
  }
});
