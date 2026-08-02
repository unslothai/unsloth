// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { visibleGgufVariants } from "../src/features/model-picker/components/model-selector/variant-visibility.ts";

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
