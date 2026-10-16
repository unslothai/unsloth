// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Non-finite activation samples now reach the UI as null; hovering one crashed the
// heatmap on null.toFixed. And the MLX trainer has no activation capture, so a Mac
// run that asked for it would poll for records that never come.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const HEATMAP = readSrc("features/studio/sections/neuron-heatmap-section.tsx");

test("a null heatmap sample is drawn and shown as missing", () => {
  assert.match(HEATMAP, /if \(Number\.isFinite\(v\)\) allVals\.push\(v\)/);
  assert.match(HEATMAP, /!Number\.isFinite\(v\)\s*\? MISSING_COLOR/);
  assert.match(
    HEATMAP,
    /const value\s+= Number\.isFinite\(raw\) \? raw : null;/,
  );
  assert.match(
    HEATMAP,
    /tooltip\.value === null \? "missing \(NaN or inf\)" : tooltip\.value\.toFixed\(4\)/,
  );
  assert.doesNotMatch(HEATMAP, /\{tooltip\.value\.toFixed\(4\)\}/);
});

test("activation capture is off and hidden on the MLX trainer", () => {
  const policy = readSrc(
    "features/studio/sections/use-mlx-training-config-policy.ts",
  );
  assert.match(
    policy,
    /if \(isMac && enableActivationCapture\) \{\s*useTrainingConfigStore\.setState\(\{ enableActivationCapture: false \}\);/,
  );
  const params = readSrc("features/studio/sections/params-section.tsx");
  assert.match(params, /\{activationCaptureApplies\(store\) && !isMac && \(/);
});
