// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  promptProgressMetrics,
  recordPromptProgress,
} from "../src/features/chat/utils/generation-progress.ts";

test("real prompt progress produces percentage, uncached throughput, and ETA", () => {
  const metrics = promptProgressMetrics({
    total: 1_000,
    processed: 600,
    cache: 100,
    timeMs: 250,
  });

  assert.equal(metrics.percentage, 60);
  assert.equal(metrics.tokensPerSecond, 2_000);
  assert.equal(metrics.etaMs, 200);
});

test("observed batch slowdown is projected across the remaining batches", () => {
  const metrics = promptProgressMetrics(
    { total: 900, processed: 600, cache: 0, timeMs: 992.992 },
    [
      { processed: 0, cache: 0, timeMs: 0 },
      { processed: 100, cache: 0, timeMs: 100 },
      { processed: 200, cache: 0, timeMs: 220 },
      { processed: 300, cache: 0, timeMs: 364 },
      { processed: 400, cache: 0, timeMs: 536.8 },
      { processed: 500, cache: 0, timeMs: 744.16 },
    ],
  );

  assert.ok(Math.abs((metrics.tokensPerSecond ?? 0) - 401.878) < 0.001);
  assert.ok(Math.abs((metrics.etaMs ?? 0) - 1_086.898) < 0.001);
});

test("progress history remains owned by one run", () => {
  const first = recordPromptProgress(
    "run-a",
    { total: 1_000, processed: 100, cache: 0, timeMs: 50 },
  );
  const second = recordPromptProgress(
    "run-a",
    { total: 1_000, processed: 200, cache: 0, timeMs: 110 },
    first,
  );
  const retry = recordPromptProgress(
    "run-b",
    { total: 1_000, processed: 100, cache: 0, timeMs: 60 },
    second,
  );

  assert.equal(second.history.length, 1);
  assert.equal(retry.history.length, 0);
});

test("different batch sizes are compared by per-token cost", () => {
  const metrics = promptProgressMetrics(
    { total: 500, processed: 300, cache: 0, timeMs: 300 },
    [
      { processed: 0, cache: 0, timeMs: 0 },
      { processed: 100, cache: 0, timeMs: 100 },
    ],
  );

  assert.equal(metrics.tokensPerSecond, 1_000);
  assert.equal(metrics.etaMs, 200);
});

test("prompt progress history stays bounded", () => {
  let progress = recordPromptProgress(
    "run",
    { total: 10_000, processed: 0, cache: 0, timeMs: 0 },
  );
  for (let index = 1; index <= 100; index += 1) {
    progress = recordPromptProgress(
      "run",
      {
        total: 10_000,
        processed: index * 100,
        cache: 0,
        timeMs: index * 10,
      },
      progress,
    );
  }

  assert.equal(progress.history.length, 32);
});

test("one scheduling spike cannot make the forecast explode", () => {
  const metrics = promptProgressMetrics(
    { total: 700, processed: 400, cache: 0, timeMs: 1_300 },
    [
      { processed: 0, cache: 0, timeMs: 0 },
      { processed: 100, cache: 0, timeMs: 100 },
      { processed: 200, cache: 0, timeMs: 200 },
      { processed: 300, cache: 0, timeMs: 300 },
    ],
  );

  assert.ok(Number.isFinite(metrics.etaMs));
  assert.ok((metrics.etaMs ?? 0) < 10_000);
});
