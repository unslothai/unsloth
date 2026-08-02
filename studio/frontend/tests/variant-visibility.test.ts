// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  shouldMountVariantExpander,
  toggleAutoExpandedRow,
  visibleGgufVariants,
} from "../src/features/model-picker/components/model-selector/variant-visibility.ts";

/** A repo holding one complete quant, one torn download, and one quant that
 *  only exists on the Hub. */
const COMPLETE = { quant: "Q4_K_M", downloaded: true, partial: false };
const TORN = { quant: "Q8_0", downloaded: false, partial: true };
const REMOTE_ONLY = { quant: "F16", downloaded: false, partial: false };
const VARIANTS = [COMPLETE, TORN, REMOTE_ONLY];

const quants = (rows: readonly { quant: string }[]) =>
  rows.map((row) => row.quant);

test("On Device with the setting off lists what is on disk", () => {
  const shown = visibleGgufVariants(VARIANTS, {
    onDevice: true,
    showAll: false,
  });
  // The torn quant stays: it occupies space and needs a resume.
  assert.deepEqual(quants(shown), ["Q4_K_M", "Q8_0"]);
});

test("On Device with the setting on lists every quant", () => {
  const shown = visibleGgufVariants(VARIANTS, {
    onDevice: true,
    showAll: true,
  });
  assert.deepEqual(quants(shown), ["Q4_K_M", "Q8_0", "F16"]);
});

test("browse lists are never filtered", () => {
  const shown = visibleGgufVariants(VARIANTS, {
    onDevice: false,
    showAll: false,
  });
  assert.deepEqual(quants(shown), ["Q4_K_M", "Q8_0", "F16"]);
});

test("a torn-only repo does not empty its list", () => {
  const shown = visibleGgufVariants([TORN], {
    onDevice: true,
    showAll: false,
  });
  // An empty list renders "No GGUF variants found", stranding the download.
  assert.deepEqual(quants(shown), ["Q8_0"]);
});

test("a repo with nothing on disk lists nothing", () => {
  const shown = visibleGgufVariants([REMOTE_ONLY], {
    onDevice: true,
    showAll: false,
  });
  assert.deepEqual(quants(shown), []);
});

test("variants missing the flags count as not on disk", () => {
  const shown = visibleGgufVariants([{ quant: "Q2_K" }], {
    onDevice: true,
    showAll: false,
  });
  assert.deepEqual(quants(shown), []);
});

test("auto-expansion waits for the sole-quant probe", () => {
  // Every row would otherwise open an expander, and its remote listing,
  // moments before collapsing into a single row.
  assert.equal(
    shouldMountVariantExpander({
      expanded: true,
      autoExpand: true,
      soleQuantsPending: true,
    }),
    false,
  );
});

test("auto-expansion resumes once the probe settles", () => {
  assert.equal(
    shouldMountVariantExpander({
      expanded: true,
      autoExpand: true,
      soleQuantsPending: false,
    }),
    true,
  );
});

test("a row the user opened is not held back", () => {
  assert.equal(
    shouldMountVariantExpander({
      expanded: true,
      autoExpand: false,
      soleQuantsPending: true,
    }),
    true,
  );
});

test("a collapsed row never mounts an expander", () => {
  for (const soleQuantsPending of [true, false]) {
    for (const autoExpand of [true, false]) {
      assert.equal(
        shouldMountVariantExpander({
          expanded: false,
          autoExpand,
          soleQuantsPending,
        }),
        false,
      );
    }
  }
});

test("clicking a row held back by its probe opens it", () => {
  // Auto-expand is on, so the row is not in the collapsed set, but its
  // pending probe means it renders nothing.
  const next = toggleAutoExpandedRow(
    { collapsed: new Set(), reopened: new Set() },
    { repoId: "unsloth/Qwen3-8B-GGUF", showing: false },
  );
  assert.deepEqual([...next.collapsed], []);
  assert.deepEqual([...next.reopened], ["unsloth/Qwen3-8B-GGUF"]);
  // Reopened rows stop following the preference, so the wait no longer applies.
  assert.equal(
    shouldMountVariantExpander({
      expanded: true,
      autoExpand: false,
      soleQuantsPending: true,
    }),
    true,
  );
});

test("clicking a showing row collapses it and clears the reopen mark", () => {
  const next = toggleAutoExpandedRow(
    { collapsed: new Set(), reopened: new Set(["unsloth/Qwen3-8B-GGUF"]) },
    { repoId: "unsloth/Qwen3-8B-GGUF", showing: true },
  );
  assert.deepEqual([...next.collapsed], ["unsloth/Qwen3-8B-GGUF"]);
  assert.deepEqual([...next.reopened], []);
});

test("reopening a collapsed row leaves other rows alone", () => {
  const next = toggleAutoExpandedRow(
    { collapsed: new Set(["a", "b"]), reopened: new Set() },
    { repoId: "a", showing: false },
  );
  assert.deepEqual([...next.collapsed], ["b"]);
  assert.deepEqual([...next.reopened], ["a"]);
});
