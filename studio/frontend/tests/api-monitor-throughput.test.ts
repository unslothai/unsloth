// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { computeStats } from "../src/features/api-monitor/stats.ts";
import type { ApiMonitorEntry } from "../src/features/chat/types/api.ts";

// A request that waited 9s behind a busy slot, then generated 50 tokens in 1s.
// Rated against the whole request that is 5 tok/s; against the decode window, 50.
function queuedEntry(overrides: Partial<ApiMonitorEntry> = {}): ApiMonitorEntry {
  return {
    id: "apireq_1",
    endpoint: "/v1/chat/completions",
    method: "POST",
    model: "local-model",
    prompt_preview: "hello",
    reply_preview: "hi",
    prompt_truncated: false,
    reply_truncated: false,
    status: "completed",
    started_at: 1_000_000,
    updated_at: 1_000_010,
    finished_at: 1_000_010,
    duration_ms: 10_000,
    ttft_ms: 9_000,
    decode_ms: 1_000,
    decode_ms_authoritative: true,
    completion_tokens: 50,
    ...overrides,
  } as ApiMonitorEntry;
}

test("throughput rates the decode window, not the queue wait", () => {
  const stats = computeStats([queuedEntry()]);
  assert.equal(stats.tokensPerSecond, 50);
  // The wait is still visible as duration, just not charged to the model.
  assert.equal(stats.avgDurationMs, 10_000);
});

test("a streamed window without engine timings is not rated", () => {
  // How many tokens rode in the first chunk is unknowable here, and reasoning tokens
  // are counted in the usage but generated before the window opens. Both inflate.
  const stats = computeStats([
    queuedEntry({ decode_ms_authoritative: false }),
  ]);
  assert.equal(stats.tokensPerSecond, null);
});

test("a reply with no decode window is left out of the rate", () => {
  const stats = computeStats([
    queuedEntry({ ttft_ms: null, decode_ms: null, decode_ms_authoritative: false }),
  ]);
  // Not folded back in at the whole-request rate, which would report 5 tok/s.
  assert.equal(stats.tokensPerSecond, null);
  assert.equal(stats.totalTokens, 50);
});

test("rating is total tokens over total decode time, not a mean of rates", () => {
  // One tiny request must not outweigh a long one.
  const stats = computeStats([
    queuedEntry({ id: "a", completion_tokens: 1, decode_ms: 1_000 }),
    queuedEntry({ id: "b", completion_tokens: 99, decode_ms: 1_000 }),
  ]);
  assert.equal(stats.tokensPerSecond, 50);
});
