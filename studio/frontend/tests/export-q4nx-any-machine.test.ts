// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

// Q4NX conversion runs on the CPU, so a host without an NPU can convert for a Ryzen AI PC, but
// only once the owner opts in: an NPU host always shows it, any other only with the setting on.
const page = readFileSync(
  new URL("../src/features/export/export-page.tsx", import.meta.url),
  "utf8",
);
const row = readFileSync(
  new URL(
    "../src/features/settings/components/q4nx-export-row.tsx",
    import.meta.url,
  ),
  "utf8",
);
const backend = readFileSync(
  new URL("../../backend/routes/export.py", import.meta.url),
  "utf8",
);
const card = readFileSync(
  new URL(
    "../src/features/export/components/q4nx-convert-card.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("Q4NX shows on an NPU host or with the opt-in, nothing else", () => {
  assert.doesNotMatch(page, /useNpuStatus|hasNpu/);
  const show = page.match(/const showQ4nx =\s*([^;]+);/);
  assert.ok(show);
  assert.equal(
    show[1].replace(/\s+/g, " ").trim(),
    "!!q4nxSettings?.npuSupported || !!q4nxSettings?.enabled",
  );
});

test("the Q4NX convert card renders only behind showQ4nx", () => {
  // Props on the card are fine; the guard is not.
  const cards = page.match(/^.*<Q4nxConvertCard\b[^>]*\/>.*$/gm) ?? [];
  assert.equal(cards.length, 2);
  for (const line of cards) {
    assert.match(line.trim(), /^\{showQ4nx && <Q4nxConvertCard\b[^>]*\/>\}$/);
  }
});

test("the Q4NX toggle is gated on the opt-in, the quant and the destination", () => {
  const effective = page.match(/const effectiveNpuQ4nx =\s*([^;]+);/);
  assert.ok(effective);
  assert.equal(
    effective[1].replace(/\s+/g, " ").trim(),
    'showQ4nx && npuQ4nx && q4nxSourceSelected && destination === "local"',
  );
  const toggle = page.indexOf('aria-label="Also convert to Q4NX (AMD NPU)"');
  assert.ok(toggle > 0);
  const opener = page.lastIndexOf('<div className="flex items-start', toggle);
  assert.match(page.slice(0, opener).trimEnd(), /\{showQ4nx && \($/);
});

test("the setting row is locked on when the host has an NPU", () => {
  assert.match(row, /checked=\{settings\?\.npuSupported \|\| settings\?\.enabled \|\| false\}/);
  assert.match(row, /disabled=\{!settings \|\| saving \|\| settings\.npuSupported\}/);
});

test("both Q4NX export routes refuse without the opt-in on a host with no NPU", () => {
  const gates = backend.match(/if not await run_in_threadpool\(q4nx_export_allowed\):/g) ?? [];
  assert.equal(gates.length, 2);
});

test("the toggle and card copy do not say this machine needs an NPU", () => {
  for (const source of [page, card]) {
    assert.doesNotMatch(source, /Only supported on AMD NPUs/);
    assert.match(source, /converting works on any machine/);
  }
});
