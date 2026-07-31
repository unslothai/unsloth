// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isHuggingFaceDatasetSelected } from "../src/features/training/lib/dataset-selection.ts";

test("Hugging Face source remains authoritative for filename-like repo ids", () => {
  assert.equal(
    isHuggingFaceDatasetSelected("huggingface", "owner/data.arrow"),
    true,
  );
  assert.equal(
    isHuggingFaceDatasetSelected("huggingface", "owner/data.jsonl"),
    true,
  );
  assert.equal(isHuggingFaceDatasetSelected("upload", "data.arrow"), false);
  assert.equal(isHuggingFaceDatasetSelected("huggingface", "  "), false);
});
