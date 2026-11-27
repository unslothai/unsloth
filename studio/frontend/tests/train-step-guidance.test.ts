// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { scaledRecommendedSteps, stepGuidanceText } from "../src/features/images/train/step-guidance.ts";
import { readSrc } from "./helpers/kit.ts";

// The one family with a step sweep: MiniMax-H3-Image, 150 steps on 18 images at batch 1.
const H3 = { steps: 150, items: 18, batch_size: 1 };

test("the measured count scales with the set size at the same passes per item", () => {
  assert.equal(scaledRecommendedSteps(H3, 18), 150);
  assert.equal(scaledRecommendedSteps(H3, 36), 300);
  assert.equal(scaledRecommendedSteps(H3, 24), 200);
  // a bigger effective batch needs fewer steps for the same passes
  assert.equal(scaledRecommendedSteps(H3, 36, 2), 150);
});

test("far from the measured set the estimate stays within 0.5x to 4x", () => {
  assert.equal(scaledRecommendedSteps(H3, 2), 80);
  assert.equal(scaledRecommendedSteps(H3, 500), 600);
});

test("no guidance or no items gives no scaled count", () => {
  assert.equal(scaledRecommendedSteps(null, 18), null);
  assert.equal(scaledRecommendedSteps(undefined, 18), null);
  assert.equal(scaledRecommendedSteps(H3, 0), null);
  assert.equal(scaledRecommendedSteps({ steps: 0, items: 18 }, 18), null);
});

test("the hint names the family, the measured run and the scaled count", () => {
  assert.equal(stepGuidanceText("MiniMax-H3-Image", null, 18), null);
  assert.equal(
    stepGuidanceText("MiniMax-H3-Image", H3, 0),
    "MiniMax-H3-Image: 150 steps tested best on a set of 18.",
  );
  assert.equal(
    stepGuidanceText("MiniMax-H3-Image", H3, 18),
    "MiniMax-H3-Image: 150 steps tested best on a set of 18, the size of yours.",
  );
  assert.equal(
    stepGuidanceText("MiniMax-H3-Image", H3, 36),
    "MiniMax-H3-Image: 150 steps tested best on a set of 18; about 300 for your 36 items (same passes per item).",
  );
});

test("the panel shows the guidance under Steps and keeps the generic hint", () => {
  const panel = readSrc("features/images/train/diffusion-train-panel.tsx");
  assert.match(panel, /stepGuidanceText\(\s*family\?\.label \?\? familyName,\s*reportedFamily\?\.step_guidance,\s*selectedItemCount,/);
  assert.match(panel, /data-testid="train-step-guidance"/);
  assert.match(panel, /500&ndash;1500 steps suits most small sets\./);
});
