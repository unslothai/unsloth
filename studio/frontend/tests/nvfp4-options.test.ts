// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  NVFP4_DISABLED_REASON,
  nvfp4SelectionFallback,
  withNvfp4Option,
} from "../src/lib/nvfp4-options.ts";

const OPTIONS: [string, string][] = [
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["nvfp4", "NVFP4 (Blackwell)"],
  ["mxfp8", "MXFP8 (Blackwell)"],
];

test("NVFP4 is hidden while the backend switch is off on a host that cannot run it", () => {
  assert.deepEqual(
    withNvfp4Option(OPTIONS, false).map(([value]) => value),
    ["fp8", "int8", "mxfp8"],
  );
});

test("NVFP4 is offered, in place, when the backend enables it", () => {
  assert.deepEqual(withNvfp4Option(OPTIONS, true), OPTIONS);
});

test("a Blackwell host with the switch off lists NVFP4 disabled, in place, with how to enable it", () => {
  const options = withNvfp4Option(OPTIONS, false, true);
  assert.deepEqual(
    options.map(([value]) => value),
    ["fp8", "int8", "nvfp4", "mxfp8"],
  );
  assert.deepEqual(options[2], ["nvfp4", "NVFP4 (Blackwell)", NVFP4_DISABLED_REASON]);
  assert.match(NVFP4_DISABLED_REASON, /UNSLOTH_NVFP4_DIFFUSION=1/);
  // Only NVFP4 is disabled; every other option stays a plain pickable pair.
  for (const option of [options[0], options[1], options[3]]) assert.equal(option.length, 2);
});

test("an enabled switch offers NVFP4 pickable on a Blackwell host too", () => {
  assert.deepEqual(withNvfp4Option(OPTIONS, true, true), OPTIONS);
});

test("a disabled NVFP4 pick still falls back to auto, so it is never sent", () => {
  assert.equal(nvfp4SelectionFallback("nvfp4", true, false), "auto");
});

test("both pages pass the host capability and render the third entry as a disabled item", async () => {
  const { readFile } = await import("node:fs/promises");
  for (const page of ["images/images-page.tsx", "video/video-page.tsx"]) {
    const src = await readFile(new URL(`../src/features/${page}`, import.meta.url), "utf8");
    assert.match(src, /const nvfp4Capable = useNvfp4Capable\(\);/, page);
    assert.match(src, /nvfp4Diffusion,\s*nvfp4Capable,\s*\)/, page);
    assert.match(src, /options\.map\(\(\[v, l, disabledReason\]\)/, page);
    assert.match(src, /disabled=\{disabledReason !== undefined\}/, page);
    assert.match(src, /title=\{disabledReason\}/, page);
  }
  const hooks = await readFile(new URL("../src/hooks/use-host-class.ts", import.meta.url), "utf8");
  assert.match(hooks, /checkpointQuantFormats\.includes\("nvfp4"\)/);
});

test("the input list is never mutated", () => {
  const copy = OPTIONS.map((o) => [...o]);
  withNvfp4Option(OPTIONS, false);
  withNvfp4Option(OPTIONS, false, true);
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
