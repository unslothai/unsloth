// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The video picker header names the resident build's precision: an FP8 load of the BF16 Wan row read "BF16"
// (NVIDIA feedback p.8, p.37). Restoring a clip's settings also restores the precision it ran with.

import assert from "node:assert/strict";
import test from "node:test";

import { withLoadedPrecision } from "../src/features/model-picker/components/model-selector/loaded-precision-trigger.ts";
import {
  VIDEO_CATALOG,
  catalogToModelOptions,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { clipPrecisionSelection } from "../src/lib/clip-precision.ts";
import { loadedTransformerPrecisionLabel } from "../src/lib/resolved-precision.ts";
import { readSrc } from "./helpers/kit.ts";

const WAN = "Wan-AI/Wan2.2-TI2V-5B-Diffusers";
const VIDEO = readSrc("features/video/video-page.tsx");

function header(extra: Record<string, unknown>) {
  const option = catalogToModelOptions(VIDEO_CATALOG, "dense-quant", ["fp8", "int8"]).find((o) => o.id === WAN);
  assert.ok(option, "Wan 2.2 TI2V 5B row");
  const status = { loaded: true, repo_id: WAN, dtype: "bfloat16", model_kind: "pipeline", ...extra };
  return withLoadedPrecision(option, loadedTransformerPrecisionLabel(status as never));
}

test("an FP8 load of the BF16 Wan row reads FP8, a dense load BF16", () => {
  assert.equal(header({ transformer_quant: "fp8" }).descriptionSuffix, "FP8");
  assert.equal(header({ transformer_quant: null }).descriptionSuffix, "BF16");
});

test("the video ModelSelector passes the loaded precision, as the Images page does", () => {
  const selector = VIDEO.slice(VIDEO.indexOf('triggerDataTour="video-model"'));
  assert.match(
    selector.slice(0, selector.indexOf("/>")),
    /loadedPrecisionOverride=\{busy !== "unloading" \? loadedTransformerPrecisionLabel\(status\) : null\}/,
  );
});

test("a clip's engaged precision maps to the select value that reproduces it", () => {
  assert.equal(clipPrecisionSelection("fp8"), "fp8");
  assert.equal(clipPrecisionSelection("INT8"), "int8");
  assert.equal(clipPrecisionSelection(null), "none");
  assert.equal(clipPrecisionSelection("off"), "none");
  // An older sidecar without the field keeps the current pick.
  assert.equal(clipPrecisionSelection(undefined), null);
  assert.equal(clipPrecisionSelection("q4_k_m"), null);
});

test("Restore sets Precision from the clip", () => {
  const restore = VIDEO.slice(VIDEO.indexOf("const restoreSettings = useCallback("));
  const body = restore.slice(0, restore.indexOf('toast.success("Settings restored to inputs")'));
  assert.match(body, /const precision = clipPrecisionSelection\(video\.transformer_quant\);\s*if \(precision\) setTransformerQuant\(precision\);/);
});
