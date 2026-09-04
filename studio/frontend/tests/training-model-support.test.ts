// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  isTrainableModelFormat,
  isUntrainableModelFormat,
} from "../src/features/training/lib/model-support.ts";

test("only runnable weight formats qualify as cached training models", () => {
  assert.equal(isTrainableModelFormat("safetensors"), true);
  assert.equal(isTrainableModelFormat("checkpoint"), true);
  assert.equal(isTrainableModelFormat("unknown"), false);
  assert.equal(isTrainableModelFormat("adapter"), false);
  assert.equal(isTrainableModelFormat("gguf"), false);
  assert.equal(isTrainableModelFormat(null), false);
});

test("remote models with unknown format are not classified as untrainable", () => {
  assert.equal(isUntrainableModelFormat("unknown"), false);
  assert.equal(isUntrainableModelFormat(null), false);
  assert.equal(isUntrainableModelFormat("adapter"), true);
  assert.equal(isUntrainableModelFormat("gguf"), true);
});
