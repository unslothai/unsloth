// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Run preview Hardware row truncates, so a long GPU name used to cut off the
 * VRAM figure appended to it ("AMD Radeon AI PRO R9700 · 31.86 …"). VRAM gets its
 * own row; this pins that it is not folded back into the name.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/studio/wizard/run-preview-card.tsx", import.meta.url),
  "utf8",
);

function metaRow(labelKey: string): string {
  const start = source.indexOf(`label={t("${labelKey}")}`);
  assert.ok(start >= 0, `no MetaRow labelled ${labelKey}`);
  return source.slice(start, source.indexOf("/>", start));
}

test("Hardware row carries the GPU name only, and wraps", () => {
  const row = metaRow("studio.preview.hardware");
  assert.doesNotMatch(row, /memoryTotalGb/);
  assert.match(row, /\bwrap\b/);
});

test("VRAM has its own rounded row with the exact value on hover", () => {
  const row = metaRow("studio.preview.vram");
  assert.match(row, /Math\.round\(gpu\.memoryTotalGb\)\} GiB/);
  assert.match(row, /title=\{`\$\{gpu\.memoryTotalGb\} GiB`\}/);
});
