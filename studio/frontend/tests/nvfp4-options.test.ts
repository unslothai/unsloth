// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { NVFP4_DISABLED_HINT, nvfp4SelectionFallback, withNvfp4Option } from "../src/lib/nvfp4-options.ts";

const OPTIONS: [string, string][] = [
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["nvfp4", "NVFP4 (Blackwell)"],
  ["mxfp8", "MXFP8 (Blackwell)"],
];

test("NVFP4 stays listed while the backend switch is off, disabled with the flag that enables it", () => {
  // Hiding it left users guessing (NVIDIA feedback p.7, p.21): the option names the switch instead.
  const options = withNvfp4Option(OPTIONS, false);
  assert.deepEqual(
    options.map(([value]) => value),
    ["fp8", "int8", "nvfp4", "mxfp8"],
  );
  assert.deepEqual(options[2], ["nvfp4", "NVFP4 (Blackwell)", NVFP4_DISABLED_HINT]);
  assert.match(NVFP4_DISABLED_HINT, /UNSLOTH_NVFP4_DIFFUSION=1/);
  for (const option of [options[0], options[1], options[3]]) assert.equal(option[2], undefined);
});

test("both pages render a disabled option with its reason inline", async () => {
  const { readFile } = await import("node:fs/promises");
  for (const page of ["images/images-page.tsx", "video/video-page.tsx"]) {
    const src = await readFile(new URL(`../src/features/${page}`, import.meta.url), "utf8");
    const select = src.slice(src.indexOf("function AdvancedSelect("));
    const body = select.slice(0, select.indexOf("\n}\n"));
    assert.match(body, /options\.map\(\(\[v, l, disabledReason\]\) =>/);
    assert.match(body, /disabled=\{!!disabledReason\}/);
    assert.match(body, /\{disabledReason\}/);
  }
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
  assert.equal(nvfp4SelectionFallback("nvfp4", true, false), "auto");
  assert.equal(nvfp4SelectionFallback("NVFP4", true, false), "auto");
});

test("other picks, an enabled switch, and a not-yet-known switch keep the selection", () => {
  assert.equal(nvfp4SelectionFallback("fp8", true, false), "fp8");
  assert.equal(nvfp4SelectionFallback("none", true, false), "none");
  assert.equal(nvfp4SelectionFallback("nvfp4", true, true), "nvfp4");
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
