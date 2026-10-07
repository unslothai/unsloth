// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Switching dictation downloads must restart the estimator; the shared manager does it
// structurally since another model is another job.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type TransferSample,
  appendSample,
  computeTransferStats,
} from "../src/lib/transfer-stats.ts";

import { readSrc } from "./helpers/kit.ts";

const MB = 1e6;

const voiceTabSource = readSrc("features/settings/tabs/voice-tab.tsx");

test("each download's samples belong to its own job, not to the tab", () => {
  const pollLoopSource = readSrc("features/hub/download-manager/poll-loop.ts");
  assert.ok(
    /speedSamples:\s*\[\]/.test(pollLoopSource),
    "each job runtime should start with its own empty sample buffer",
  );
  const voiceTabSource = readSrc("features/settings/tabs/voice-tab.tsx");
  for (const gone of ["computeTransferStats", "appendSample", "downloadSamplesRef"]) {
    assert.ok(
      !voiceTabSource.includes(gone),
      `voice-tab should not re-implement the estimator (${gone})`,
    );
  }
});

test("a new model's rate is not priced over the previous model's samples", () => {
  const published = (reset: boolean) => {
    const samples: TransferSample[] = [];
    let watched: string | null = null;
    let rate = 0;
    const poll = (model: string, bytes: number, total: number, t: number) => {
      if (reset && model !== watched) samples.length = 0;
      watched = model;
      appendSample(samples, t, bytes);
      const stats = computeTransferStats(samples, total);
      rate = stats.stable ? stats.rateBytesPerSecond : 0;
    };
    for (let t = 0; t <= 20; t += 1) poll("A", t * 200 * MB, 4_000 * MB, t);
    // The resumed counter starts above the last one's end, so nothing regresses to trigger a reset.
    let worst = 0;
    for (let t = 21; t <= 30; t += 1) {
      poll("B", 4_000 * MB + (t - 21) * 5 * MB, 8_000 * MB, t);
      worst = Math.max(worst, rate);
    }
    return worst;
  };

  assert.ok(
    published(true) <= 6 * MB,
    `with the reset, published ${(published(true) / MB).toFixed(1)} MB/s for a 5 MB/s transfer`,
  );
  assert.ok(
    published(false) > 50 * MB,
    "without the reset the old model's samples should still poison the rate",
  );
});
