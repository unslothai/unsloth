// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The model-load toast polls on a 2s interval, clamped to about once a minute
// while hidden. Those gaps time the poller, not the transfer, and the estimator
// reads gaps as the burst cadence, so progress stopping near the moment the user
// returns left a stale rate on screen for a minute. The hub and voice pollers
// already drop hidden samples; this pins that the chat one does too.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  type TransferSample,
  appendSample,
  computeTransferStats,
} from "../src/lib/transfer-stats.ts";

const MB = 1e6;

test("the chat estimator drops samples taken while the tab is hidden", () => {
  const source = readFileSync(
    new URL(
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const guard = source.indexOf("document.hidden");
  assert.ok(guard > 0, "the chat poller should skip a hidden tab");
  const clear = source.indexOf("samples.length = 0", guard);
  const sample = source.indexOf("appendSample(samples,", guard);
  assert.ok(
    clear > 0 && clear < sample,
    "the hidden branch must clear, not sample",
  );
});

test("throttled hidden samples do not hold a stale rate once progress stops", () => {
  const heldSeconds = (dropWhileHidden: boolean) => {
    const samples: TransferSample[] = [];
    let rate = 0;
    const poll = (t: number, b: number, hidden: boolean) => {
      if (dropWhileHidden && hidden) {
        samples.length = 0;
        rate = 0;
        return;
      }
      appendSample(samples, t, b);
      const stats = computeTransferStats(samples, 100_000 * MB);
      rate = stats.stable ? stats.rateBytesPerSecond : 0;
    };

    let bytes = 0;
    let t = 0;
    for (; t <= 60; t += 2) {
      bytes = t * 50 * MB;
      poll(t, bytes, false);
    }
    // Hidden: the 2s interval is clamped to a minute, still downloading.
    for (; t <= 660; t += 60) {
      bytes = t * 50 * MB;
      poll(t, bytes, true);
    }
    // Progress stops just as the tab is shown again.
    let held = 0;
    for (const end = t + 300; t <= end; t += 2) {
      poll(t, bytes, false);
      if (rate > 0) held += 2;
    }
    return held;
  };

  assert.ok(
    heldSeconds(false) > 30,
    "without the guard the throttled cadence should still hold a stale rate",
  );
  assert.equal(
    heldSeconds(true),
    0,
    "with the guard nothing survives the hidden stretch",
  );
});
