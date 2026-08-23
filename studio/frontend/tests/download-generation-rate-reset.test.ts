// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A generation change means another backend owns the transfer, so the previous
// one's samples describe a different run. Nothing else catches it: a restart
// resumes from the same cache so the counter never goes backwards, and the
// runtime holding the buffer is not recreated. The poll gap across the restart
// then lands inside the measured span: 100 MB/s published 13 MB/s.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type TransferSample,
  appendSample,
  computeTransferStats,
} from "../src/lib/transfer-stats.ts";

const MB = 1e6;

test("a generation change clears the rate samples before the next one is taken", () => {
  const source = readFileSync(
    new URL(
      "../src/features/hub/download-manager/poll-loop.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const clear = source.indexOf(
    "rt.speedSamples.length = 0",
    source.indexOf("generationChanged) {"),
  );
  const sample = source.indexOf("applySpeedSample(rt,");
  assert.ok(clear > 0, "a generation change should clear the samples");
  assert.ok(sample > 0, "the speed sample should still be taken");
  assert.ok(
    clear < sample,
    "clearing after sampling would price the new run on the old buffer",
  );
});

test("a restart does not make the resumed transfer look slow", () => {
  const rates = (clearOnGeneration: boolean) => {
    const samples: TransferSample[] = [];
    const out: number[] = [];
    let bytes = 0;
    const poll = (t: number, b: number, generationChanged: boolean) => {
      if (generationChanged && clearOnGeneration) samples.length = 0;
      appendSample(samples, t, b);
      const stats = computeTransferStats(samples, 100_000 * MB);
      out.push(stats.stable ? stats.rateBytesPerSecond : 0);
    };
    for (let t = 0; t <= 60; t += 1) {
      bytes = t * 100 * MB;
      poll(t, bytes, false);
    }
    // The backend restarts. 12s without a successful poll, which is inside the
    // 30s degraded-poll reset, so nothing else clears the buffer. It resumes at
    // the same byte count and the same speed.
    const after: number[] = [];
    for (let t = 73; t <= 84; t += 1) {
      poll(t, bytes + (t - 73) * 100 * MB, t === 73);
      after.push(out[out.length - 1]);
    }
    return after;
  };

  const stale = rates(false).filter((r) => r > 0);
  const cleared = rates(true).filter((r) => r > 0);
  assert.ok(
    Math.min(...stale) < 20 * MB,
    "without the reset the restart gap should still drag the rate down",
  );
  assert.ok(
    Math.min(...cleared) > 95 * MB,
    `after the reset the resumed transfer should read ~100 MB/s, got ${(Math.min(...cleared) / MB).toFixed(1)}`,
  );
});
