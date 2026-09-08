// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Estimated Memory Usage header is the only row in the panel carrying a title AND
// two figures, and the panel is w-[min(468px,calc(100vw-1rem))] -- so under a ~460px
// window it shrinks while the figures do not. Measured on the merged build at 320px
// the title had been squeezed from 150px to 11px and rendered as "E...", which is what
// was reported. The layout half is asserted against the source, the idiom
// tensor-parallel-row-gating already uses for markup that cannot be imported; the note
// half is a pure function and is exercised directly.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  glueNoteItems,
  resolveDraftCacheNote,
  resolveKvNote,
} from "../src/features/model-picker/model-config/memory-fit.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PAGE = readFileSync(
  path.join(
    HERE,
    "..",
    "src/features/model-picker/components/model-config-page.tsx",
  ),
  "utf8",
);

const NBSP = " ";

test("the memory header may wrap, so the figures drop rather than eat the title", () => {
  assert.match(CONFIG_PAGE, /\$\{ROW_CLASS\} flex-wrap gap-y-1/);
});

test("the wrapped figures still sit at the right edge", () => {
  // justify-between has nothing to push against on a line holding one item, so
  // without this the figures jump to the left margin when they wrap.
  assert.match(CONFIG_PAGE, /className=\{`ml-auto flex shrink-0 items-center gap-3/);
});

test("the header title is still allowed to truncate as a backstop", () => {
  // Wrapping is what keeps it whole at the widths that matter; truncation stays for a
  // locale whose title does not fit a line of its own. Dropping min-w-0 here would
  // overflow the panel instead.
  assert.match(
    CONFIG_PAGE,
    /className="flex min-w-0 items-center gap-1\.5 rounded-sm text-left/,
  );
});

test("breakdown notes are rendered glued", () => {
  assert.match(CONFIG_PAGE, /\{glueNoteItems\(note\)\}/);
});

test("an item's own spaces do not break", () => {
  const glued = glueNoteItems("f16 · 262,144 tokens · 4 slots");
  assert.equal(glued, `f16 ·${NBSP}262,144${NBSP}tokens ·${NBSP}4${NBSP}slots`);
  // The only ordinary spaces left are the ones a line may break at: one per bullet.
  assert.equal(glued.split(" ").length - 1, 2);
});

test("the bullet leads its item, so a break cannot orphan it", () => {
  for (const item of glueNoteItems("f16 · 4 slots").split(" ").slice(1)) {
    assert.ok(item.startsWith(`·${NBSP}`), `bullet detached from ${item}`);
  }
});

test("a note with no separator keeps every break opportunity it had", () => {
  // Only the KV caption is a list. Weights and Draft cache are ordinary prose, and
  // gluing those bought nothing while costing them the ability to wrap at all, so a
  // long one ran past the caption column into the value instead of wrapping inside it.
  for (const note of [
    "256 of 257 layers on GPU",
    "2.14 GB on GPU",
    "host RAM",
    "f16",
  ]) {
    assert.equal(glueNoteItems(note), note);
    assert.doesNotMatch(glueNoteItems(note), new RegExp(NBSP));
  }
});

test("the notes the row actually builds are left breakable", () => {
  // The two non-list note sources, at their real call sites in model-config-page.
  assert.match(CONFIG_PAGE, /layers on GPU`/);
  const hostNote = resolveDraftCacheNote(0, 1e9);
  assert.equal(hostNote, "host RAM");
  for (const note of ["256 of 257 layers on GPU", hostNote ?? ""]) {
    const spaces = (glueNoteItems(note).match(/ /g) || []).length;
    assert.ok(spaces > 0, `${note} has no break opportunity left`);
  }
});

test("gluing round-trips the note the row actually builds", () => {
  const note = resolveKvNote({
    cacheTypeKv: "q8_0",
    nCtx: 262144,
    nParallel: 4,
    kvOnGpu: false,
  });
  // Same text to a reader, and to anyone grepping the panel: only the spaces differ.
  assert.equal(glueNoteItems(note).replace(new RegExp(NBSP, "g"), " "), note);
});
