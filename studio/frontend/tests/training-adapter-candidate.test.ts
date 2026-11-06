// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { validateTrainingModelCandidate } = await import(
  "../src/features/training/lib/freeform-model-validation.ts"
);

test("a scanned adapter folder is not refused as untrainable", () => {
  // The inventory marks adapters canTrain=false because they are not bases.
  const capabilities = {
    canChat: true,
    canTrain: false,
    canExport: false,
  } as never;
  assert.deepEqual(
    validateTrainingModelCandidate({
      id: "/outputs/run/checkpoint-30",
      modelFormat: "adapter",
      capabilities,
    }),
    { ok: true },
  );
  assert.deepEqual(
    validateTrainingModelCandidate({
      id: "/models/merged",
      modelFormat: "safetensors",
      capabilities,
    }),
    { ok: false, reasonKey: "studio.modelPicker.reasonNotTrainable" },
  );
});

test("a single adapter file is refused; its folder is the thing to pick", () => {
  for (const id of [
    "/outputs/run/checkpoint-30/adapter_config.json",
    "/outputs/run/adapter_model.safetensors",
    "C:\\outputs\\run\\adapter_model.bin",
  ]) {
    assert.deepEqual(validateTrainingModelCandidate({ id }), {
      ok: false,
      reasonKey: "studio.modelPicker.reasonAdapter",
    });
  }
  assert.deepEqual(
    validateTrainingModelCandidate({ id: "/outputs/run/checkpoint-30" }),
    { ok: true },
  );
});
