// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { downloadInventoryHintKind } from "../src/features/hub/download-manager/download-manager-types.ts";

test("classifies quant variants as GGUF", () => {
  assert.equal(downloadInventoryHintKind("model", "Q4_K_M"), "gguf");
});

test("classifies legacy scoped variants as model downloads", () => {
  assert.equal(downloadInventoryHintKind("model", "@diffusion"), "model");
});

test("uses the caller's format for scoped downloads", () => {
  assert.equal(
    downloadInventoryHintKind("model", "@diffusion", "gguf"),
    "gguf",
  );
  assert.equal(
    downloadInventoryHintKind("model", "@diffusion", "model"),
    "model",
  );
});

test("keeps dataset jobs out of model inventory", () => {
  assert.equal(downloadInventoryHintKind("dataset", null), "dataset");
});
