// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { generatePhaseLabel, sameGenerateProgress } = await import(
  "../src/lib/media-generate-phase.ts"
);

test("each phase gets its own label next to the spinner", () => {
  assert.equal(generatePhaseLabel({ phase: "encode", step: 0, total: 8 }), "Encoding prompt…");
  assert.equal(generatePhaseLabel({ phase: "denoise", step: 0, total: 8 }), "Denoising please wait…");
  assert.equal(
    generatePhaseLabel({ phase: "denoise", step: 3, total: 8 }),
    "Denoising please wait… \u2022 Step 3/8",
  );
  assert.equal(generatePhaseLabel({ phase: "decode", step: 8, total: 8 }), "Decoding…");
  assert.equal(
    generatePhaseLabel({ phase: "decode", step: 8, total: 8 }, { hasAudio: true }),
    "Decoding video and audio…",
  );
  assert.equal(generatePhaseLabel({ phase: "export", step: 8, total: 8 }), "Encoding video…");
});

test("a missing phase (sd.cpp) reads as denoising, with the ETA when known", () => {
  const label = generatePhaseLabel({ phase: null, step: 2, total: 4, eta_seconds: 5 });
  assert.match(label, /^Denoising please wait… \u2022 Step 2\/4 · ~/);
});

test("a new preview re-renders even when nothing else moved", () => {
  const a = { step: 2, eta_seconds: 1, phase: "denoise", preview_seq: 1 };
  assert.equal(sameGenerateProgress(a, { ...a }), true);
  assert.equal(sameGenerateProgress(a, { ...a, preview_seq: 2 }), false);
  assert.equal(sameGenerateProgress({ step: 1, phase: "encode" }, { step: 1, phase: "denoise" }), false);
});
