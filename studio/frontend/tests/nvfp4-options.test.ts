// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { nvfp4SelectionFallback, withNvfp4Option } from "../src/lib/nvfp4-options.ts";

const OPTIONS: [string, string][] = [
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["nvfp4", "NVFP4 (Blackwell)"],
  ["mxfp8", "MXFP8 (Blackwell)"],
];

test("NVFP4 is hidden while the backend switch is off", () => {
  assert.deepEqual(
    withNvfp4Option(OPTIONS, false).map(([value]) => value),
    ["fp8", "int8", "mxfp8"],
  );
});

test("NVFP4 is offered, in place, when the backend enables it", () => {
  assert.deepEqual(withNvfp4Option(OPTIONS, true), OPTIONS);
});

test("the input list is never mutated", () => {
  const copy = OPTIONS.map((o) => [...o]);
  withNvfp4Option(OPTIONS, false);
  assert.deepEqual(OPTIONS, copy);
});

test("a held NVFP4 pick falls back to auto once the backend reports the switch off", () => {
  // The select no longer lists nvfp4, so keeping it would render blank and 400 the next load.
  assert.equal(nvfp4SelectionFallback("nvfp4", true, false), "auto");
  assert.equal(nvfp4SelectionFallback("NVFP4", true, false), "auto");
});

test("other picks, an enabled switch, and a not-yet-known switch keep the selection", () => {
  assert.equal(nvfp4SelectionFallback("fp8", true, false), "fp8");
  assert.equal(nvfp4SelectionFallback("none", true, false), "none");
  assert.equal(nvfp4SelectionFallback("nvfp4", true, true), "nvfp4");
  // Before /api/system answers the switch reads false by default; that must not wipe a real pick.
  assert.equal(nvfp4SelectionFallback("nvfp4", false, false), "nvfp4");
});

test("both pages reset the hidden NVFP4 selects through the fallback", async () => {
  const { readFile } = await import("node:fs/promises");
  const images = await readFile(new URL("../src/features/images/images-page.tsx", import.meta.url), "utf8");
  const video = await readFile(new URL("../src/features/video/video-page.tsx", import.meta.url), "utf8");
  for (const setter of ["setTransformerQuant", "setTextEncoderQuant"]) {
    assert.match(
      images,
      new RegExp(`${setter}\\(\\(v\\) => nvfp4SelectionFallback\\(v, nvfp4DiffusionKnown, nvfp4Diffusion\\)\\)`),
    );
  }
  assert.match(
    video,
    /setTransformerQuant\(\(v\) => nvfp4SelectionFallback\(v, nvfp4DiffusionKnown, nvfp4Diffusion\)\)/,
  );
});
