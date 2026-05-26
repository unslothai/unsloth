// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { validateTrainingModelCandidate } from "../src/features/training/lib/freeform-model-validation.ts";

test("freeform training validation rejects GGUF ids and files", () => {
  assert.equal(
    validateTrainingModelCandidate({ id: "unsloth/foo-GGUF" }).ok,
    false,
  );
  assert.equal(
    validateTrainingModelCandidate({ id: "/models/foo/model.gguf" }).ok,
    false,
  );
});

test("freeform training validation rejects adapters and unsupported quantized names", () => {
  assert.equal(
    validateTrainingModelCandidate({
      id: "/outputs/checkpoint/adapter_model.safetensors",
    }).ok,
    false,
  );
  assert.equal(
    validateTrainingModelCandidate({ id: "Org/Model-AWQ" }).ok,
    false,
  );
});

test("freeform training validation rejects non-trainable known inventory rows", () => {
  assert.equal(
    validateTrainingModelCandidate({
      id: "Org/ChatOnly",
      capabilities: {
        canTrain: false,
        canChat: true,
        canDelete: false,
        canDownload: false,
        requiresVariant: false,
        supportsLora: false,
        supportsVision: false,
      },
    }).ok,
    false,
  );
});

test("freeform training validation allows ordinary HF ids and local folders", () => {
  assert.equal(
    validateTrainingModelCandidate({ id: "unsloth/Llama-3.2-3B" }).ok,
    true,
  );
  assert.equal(
    validateTrainingModelCandidate({ id: "/models/llama-3.2-3b" }).ok,
    true,
  );
});
