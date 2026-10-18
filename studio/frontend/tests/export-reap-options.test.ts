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

test("group-limited routers count like Core: per group, at least 2 left in each", () => {
  // DeepSeek-V3: 256 experts, 8 groups, top-8 over 4 groups.
  const dsv3 = { num_experts: 256, top_k: 8, n_group: 8, topk_group: 4 };
  assert.deepEqual(reapAllowedSparsities(dsv3), [0.25, 0.5]);
  assert.equal(reapKeptExperts(dsv3, 0.25), 192);
  // 8 experts in 4 groups: 50% would leave 1 per group, which Core refuses.
  const tiny = { num_experts: 8, top_k: 2, n_group: 4, topk_group: 2 };
  assert.deepEqual(reapAllowedSparsities(tiny), []);
  // 12 experts in 4 groups at 25%: int(3) dropped, round(0.75) = 1 per group, 4 in total.
  const odd = { num_experts: 12, top_k: 2, n_group: 4, topk_group: 2 };
  assert.equal(reapKeptExperts(odd, 0.25), 8);
  // 20 experts in 4 groups at 50%: 10 dropped, round(2.5) = 2 per group (half to even), 8 in total.
  const half = { num_experts: 20, top_k: 2, n_group: 4, topk_group: 2 };
  assert.equal(reapKeptExperts(half, 0.5), 12);
});
