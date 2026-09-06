// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { computeStats } from "../src/features/api-monitor/stats.ts";
import type { ApiMonitorEntry } from "../src/features/chat/types/api.ts";

// Waited 9s behind a busy slot, then generated 50 tokens in 1s: 5 tok/s rated against
// the whole request, 50 against the decode window.
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

test("a reply with no decode window is left out of the rate", () => {
  const stats = computeStats([
    queuedEntry({ ttft_ms: null, decode_ms: null }),
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

test("context usage follows the busiest running request", () => {
  const stats = computeStats([
    queuedEntry({ id: "done", context_usage: 0.94, updated_at: 1_000_030 }),
    queuedEntry({
      id: "live-low",
      status: "running",
      finished_at: null,
      duration_ms: null,
      context_usage: 0.41,
      updated_at: 1_000_040,
    }),
    queuedEntry({
      id: "live-high",
      status: "running",
      finished_at: null,
      duration_ms: null,
      context_usage: 0.78,
      updated_at: 1_000_050,
    }),
  ]);

  assert.equal(stats.contextUsage, 0.78);
  assert.equal(stats.active, 2);
});

test("context usage falls back to the latest known request when idle", () => {
  const stats = computeStats([
    queuedEntry({ id: "newer", context_usage: 0.58, updated_at: 1_000_050 }),
    queuedEntry({ id: "older", context_usage: 0.91, updated_at: 1_000_040 }),
  ]);

  assert.equal(stats.contextUsage, 0.58);
});

test("context usage is clamped before it reaches the live monitor", () => {
  const stats = computeStats([
    queuedEntry({ context_usage: 1.4 }),
  ]);
  assert.equal(stats.contextUsage, 1);
});

test("floating monitor surfaces context usage and the live-call counter", () => {
  const source = readFileSync(
    new URL("../src/features/api-monitor/api-monitor-overlay.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /label="Context"/);
  assert.match(source, /stats\.contextUsage/);
  assert.match(source, /entry\.context_usage/);
  assert.match(source, /label="Live calls"/);
});
