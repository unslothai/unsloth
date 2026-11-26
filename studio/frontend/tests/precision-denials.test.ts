// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Wan NVFP4 (and FP8 on a card whose rowwise kernel probe failed) was offered, then refused at load (NVIDIA feedback
// p.8, p.39). The backend now reports each scheme's refusal; Precision disables it with the reason on both pages.

import assert from "node:assert/strict";
import test from "node:test";

import { NVFP4_DISABLED_HINT, withNvfp4Option, type SelectOption } from "../src/lib/nvfp4-options.ts";
import {
  activePrecisionDenials,
  deniedSelectionFallback,
  withPrecisionDenials,
} from "../src/lib/precision-denials.ts";
import { readSrc } from "./helpers/kit.ts";

const OPTIONS: SelectOption[] = [
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["nvfp4", "NVFP4 (Blackwell)"],
  ["mxfp8", "MXFP8 (Blackwell)"],
];
const WAN_NVFP4 = "'nvfp4' is ruled out for family 'wan2.2-ti2v': it was measured and failed the NVFP4 accuracy gate";
const WAN = { int8: null, fp8: null, nvfp4: WAN_NVFP4, mxfp8: null };

test("a denied scheme is disabled with the backend's reason; the rest stay pickable", () => {
  const options = withPrecisionDenials(withNvfp4Option(OPTIONS, true), WAN);
  assert.deepEqual(options[2], ["nvfp4", "NVFP4 (Blackwell)", WAN_NVFP4]);
  for (const i of [0, 1, 3]) assert.equal(options[i][2], undefined);
});

test("a failed FP8 kernel probe disables FP8", () => {
  const reason = "'fp8' failed this GPU's kernel probe";
  assert.equal(withPrecisionDenials(OPTIONS, { fp8: reason })[0][2], reason);
});

test("a measured family failure replaces the NVFP4 switch hint, and no denials changes nothing", () => {
  // Turning the switch on would not help Wan, so its reason wins; a family the backend leaves null keeps the hint.
  assert.equal(withPrecisionDenials(withNvfp4Option(OPTIONS, false), WAN)[2][2], WAN_NVFP4);
  assert.equal(withPrecisionDenials(withNvfp4Option(OPTIONS, false), { nvfp4: null })[2][2], NVFP4_DISABLED_HINT);
  assert.deepEqual(withPrecisionDenials(OPTIONS, null), OPTIONS);
  assert.deepEqual(withPrecisionDenials(OPTIONS, undefined), OPTIONS);
});

test("a held denied pick falls back to Auto", () => {
  assert.equal(deniedSelectionFallback("nvfp4", WAN), "auto");
  assert.equal(deniedSelectionFallback("fp8", WAN), "fp8");
  assert.equal(deniedSelectionFallback("auto", WAN), "auto");
  assert.equal(deniedSelectionFallback("nvfp4", null), "nvfp4");
});

test("the loaded build's denials win; with nothing loaded the last planned pick's apply", () => {
  const planned = { fp8: "x" };
  assert.equal(activePrecisionDenials({ loaded: true, precision_options: WAN }, planned), WAN);
  assert.equal(activePrecisionDenials({ loaded: false, precision_options: null }, planned), planned);
  assert.equal(activePrecisionDenials({ loaded: true }, planned), planned);
  assert.equal(activePrecisionDenials(null, undefined), null);
});

for (const path of ["features/images/images-page.tsx", "features/video/video-page.tsx"]) {
  const src = readSrc(path);
  test(`${path}: Precision applies the denials, falls back, and reads the pick's plan`, () => {
    const select = src.slice(src.indexOf('label="Precision"'));
    assert.match(select.slice(0, select.indexOf(") : (")), /withPrecisionDenials\(\s*withNvfp4Option\(/);
    assert.match(src, /const precisionDenials = activePrecisionDenials\(status, plannedPrecisionOptions\);/);
    assert.match(src, /setTransformerQuant\(\(v\) => deniedSelectionFallback\(v, precisionDenials\)\);/);
    assert.match(src, /setPlannedPrecisionOptions\(plan\.precision_options \?\? null\);/);
  });
}
