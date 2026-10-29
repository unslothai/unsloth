// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { assetLabel } = await import("../src/features/hub/download-manager/required-assets.ts");
const { hubPlanStepLabel } = loadWithStubs<{
  hubPlanStepLabel: (plan: unknown, running: boolean) => string;
}>(new URL("../src/features/hub/download-manager/hub-plan-progress.tsx", import.meta.url), {
  "react/jsx-runtime": {},
  "./download-manager-controller": {},
  "./download-presentation": {},
  "./download-progress-bar": {},
  "./required-assets": { assetLabel },
});

const model = { repoId: "unsloth/Z-Image-Turbo-GGUF", files: ["z-image-turbo-Q8_0.gguf"], bytes: 7, checkpoint: true };
const companions = {
  repoId: "unsloth/Z-Image-Turbo",
  files: ["text_encoder/model.safetensors", "vae/diffusion_pytorch_model.safetensors"],
  bytes: 8,
  checkpoint: false,
};

test("the card names the running step and where it is in the plan", () => {
  const plan = { id: 1, entries: [model, companions], remaining: [model, companions] };
  assert.equal(hubPlanStepLabel(plan, true), "Model file · 1 of 2");
  assert.equal(hubPlanStepLabel({ ...plan, remaining: [companions] }, true), "Encoder & decoder · 2 of 2");
  assert.equal(hubPlanStepLabel({ ...plan, entries: [companions], remaining: [companions] }, true), "Encoder & decoder");
  assert.equal(hubPlanStepLabel(plan, false), "Queued");
});

test("the card's bar is the running job's, as its downloads panel row shows it (#14061)", () => {
  const source = readSrc("features/hub/download-manager/hub-plan-progress.tsx");
  assert.match(source, /presentedProgress\(job\)/);
  assert.doesNotMatch(source, /plan\.entries\.reduce/, "a plan-wide total is back on the card");
});
