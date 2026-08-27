// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Switching straight from one dictation download to another must restart the
// estimator. The guard once sat AFTER the ref it compares against was assigned,
// so it never fired and the new run was priced over the old one's samples: a
// 5 MB/s download read as 200 MB/s with 20s left. appendSample cannot save it
// either, since a resumed model can start above where the last one stopped.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type TransferSample,
  appendSample,
  computeTransferStats,
} from "../src/lib/transfer-stats.ts";

const MB = 1e6;

const voiceTabSource = readFileSync(
  new URL("../src/features/settings/tabs/voice-tab.tsx", import.meta.url),
  "utf8",
);

test("the watched model is compared before it is adopted", () => {
  const compare = voiceTabSource.indexOf(
    "download.model !== watchedDownloadRef.current",
  );
  const assign = voiceTabSource.indexOf(
    "watchedDownloadRef.current = download.model",
  );
  assert.ok(compare > 0, "the model-change guard should still exist");
  assert.ok(assign > 0, "the watched model should still be recorded");
  assert.ok(
    compare < assign,
    "comparing after assigning makes the guard unreachable",
  );
});

// What that guard is worth: the same two downloads, with and without the reset.
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
    // A fast model finishes 4 GB at 200 MB/s.
    for (let t = 0; t <= 20; t += 1) poll("A", t * 200 * MB, 4_000 * MB, t);
    // Then a slow one resumes from its own 4 GB partial at 5 MB/s. Its counter
    // starts at or above where the last one stopped, so nothing regresses.
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
