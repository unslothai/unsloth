// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sample images during training: the prompt parser matches the backend's limits, rounds group by
// step in order, and the panel only offers (and only sends) sampling for a family that renders it.

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_SAMPLE_PROMPTS,
  groupSamplesByStep,
  parseSamplePrompts,
} from "../src/features/images/train/diffusion-samples.ts";
import { readSrcAsync } from "./helpers/kit.ts";

test("prompts: one per line, blanks dropped, capped at the backend limit", () => {
  assert.deepEqual(parseSamplePrompts(" a dog \n\n b dog\r\n  "), [
    "a dog",
    "b dog",
  ]);
  assert.equal(
    parseSamplePrompts("1\n2\n3\n4\n5\n6").length,
    MAX_SAMPLE_PROMPTS,
  );
  assert.deepEqual(parseSamplePrompts(""), []);
});

test("rounds group by step, oldest first, prompt order kept", () => {
  const rounds = groupSamplesByStep([
    { step: 100, path: "samples/t/step-100-0.png" },
    { step: 0, path: "samples/t/step-0-0.png" },
    { step: 100, path: "samples/t/step-100-1.png" },
  ]);
  assert.deepEqual(
    rounds.map((r) => [r.step, r.images.map((i) => i.path)]),
    [
      [0, ["samples/t/step-0-0.png"]],
      [100, ["samples/t/step-100-0.png", "samples/t/step-100-1.png"]],
    ],
  );
  assert.deepEqual(groupSamplesByStep(undefined), []);
});

test("the panel gates and zeroes sampling on supports_samples", async () => {
  const source = await readSrcAsync(
    "features/images/train/diffusion-train-panel.tsx",
  );
  assert.match(
    source,
    /const supportsSamples = reportedFamily\?\.supports_samples \?\? false;/,
  );
  assert.match(
    source,
    /sample_every: supportsSamples \? Math\.max\(0, Math\.floor\(sampleEvery\)\) : 0/,
  );
  assert.match(
    source,
    /<DiffusionSamples jobId=\{viewRun\.job_id\} samples=\{viewRun\.samples\} \/>/,
  );
  assert.match(
    source,
    /<DiffusionSamples jobId=\{status\.job_id\} samples=\{status\.samples\} \/>/,
  );
});
