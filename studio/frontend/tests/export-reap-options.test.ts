// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_REAP_SETTINGS,
  reapAllowedSparsities,
  reapKeptExperts,
  reapRequest,
  reapSummaryLabel,
} from "../src/features/export/lib/reap.ts";

const QWEN3_30B = { num_experts: 128, top_k: 8 };

test("REAP is off by default and sends nothing", () => {
  assert.equal(DEFAULT_REAP_SETTINGS.enabled, false);
  assert.equal(reapRequest(DEFAULT_REAP_SETTINGS, QWEN3_30B), null);
  assert.equal(reapSummaryLabel(DEFAULT_REAP_SETTINGS, QWEN3_30B), null);
});

test("an enabled setting sends sparsity, samples and a trimmed dataset", () => {
  const on = { ...DEFAULT_REAP_SETTINGS, enabled: true, dataset: "  " };
  assert.deepEqual(reapRequest(on, QWEN3_30B), {
    sparsity: 0.25,
    num_calibration_samples: 512,
    dataset: null,
  });
  assert.equal(
    reapRequest({ ...on, dataset: " org/data " }, QWEN3_30B)?.dataset,
    "org/data",
  );
  assert.equal(
    reapSummaryLabel({ ...on, sparsity: 0.5 }, QWEN3_30B),
    "50% of experts removed (64 of 128 kept)",
  );
});

test("kept experts match Core's int(E * sparsity) drop", () => {
  assert.equal(reapKeptExperts({ num_experts: 60, top_k: 4 }, 0.25), 45);
  assert.equal(reapKeptExperts({ num_experts: 7, top_k: 2 }, 0.5), 4);
});

test("a sparsity Core would refuse is not offered or sent", () => {
  // 4 experts, top-3: 50% would keep 2 < 3; 25% keeps 3.
  const small = { num_experts: 4, top_k: 3 };
  assert.deepEqual(reapAllowedSparsities(small), [0.25]);
  const chose50 = { ...DEFAULT_REAP_SETTINGS, enabled: true, sparsity: 0.5 };
  assert.equal(reapRequest(chose50, small)?.sparsity, 0.25);
  // 3 experts, top-2: 25% drops none, 50% keeps 2.
  assert.deepEqual(reapAllowedSparsities({ num_experts: 3, top_k: 2 }), [0.5]);
  // Nothing valid: the card is hidden and no request goes out.
  const none = { num_experts: 2, top_k: 2 };
  assert.deepEqual(reapAllowedSparsities(none), []);
  assert.equal(reapRequest(chose50, none), null);
});
