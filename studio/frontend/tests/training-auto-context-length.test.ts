// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const {
  recommendContextLengthForTokenCount,
  resolveRecommendedContextLength,
} = await import("../src/features/training/api/token-preview-api.ts");

test("recommendContextLengthForTokenCount selects the smallest standard bucket covering the row (#18236)", () => {
  assert.equal(recommendContextLengthForTokenCount(0), null);
  assert.equal(recommendContextLengthForTokenCount(null), null);
  assert.equal(recommendContextLengthForTokenCount(120), 512);
  assert.equal(recommendContextLengthForTokenCount(512), 512);
  assert.equal(recommendContextLengthForTokenCount(513), 1024);
  assert.equal(recommendContextLengthForTokenCount(1420), 2048);
  assert.equal(recommendContextLengthForTokenCount(3100), 4096);
  assert.equal(recommendContextLengthForTokenCount(3100, 2048), 2048);
});

test("resolveRecommendedContextLength uses backend pool recommendation or falls back to samples", () => {
  assert.deepEqual(
    resolveRecommendedContextLength({
      recommended_max_seq_length: 4096,
      max_row_tokens: 2890,
      samples: [
        {
          tokens: [],
          input_ids: [],
          num_tokens: 2048,
          num_trained: 1000,
          num_hidden: 0,
          truncated_from: 2400,
          text: "sample",
        },
      ],
    }),
    { recommended: 4096, maxTokens: 2890 },
  );

  assert.deepEqual(
    resolveRecommendedContextLength({
      samples: [
        {
          tokens: [],
          input_ids: [],
          num_tokens: 512,
          num_trained: 300,
          num_hidden: 0,
          truncated_from: 1350,
          text: "sample",
        },
      ],
    }),
    { recommended: 2048, maxTokens: 1350 },
  );
});
