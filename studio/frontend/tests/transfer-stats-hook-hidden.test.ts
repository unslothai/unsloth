// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Hidden tabs clamp the 1.5s poll to about a minute, and the effect is keyed on bytes, so a
// stalled counter never recomputes and a stale estimate would stay on screen.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type TransferSample,
  appendSample,
  computeTransferStats,
} from "../src/lib/transfer-stats.ts";

import { readSrc } from "./helpers/kit.ts";

const MB = 1e6;

const hookSource = readSrc("features/chat/hooks/use-transfer-stats.ts");

test("the shared hook skips sampling while the tab is hidden", () => {
  const guard = hookSource.indexOf("document.hidden");
  assert.ok(guard > 0, "the hook should skip a hidden tab");
  const clear = hookSource.indexOf("samplesRef.current.length = 0", guard);
  const sample = hookSource.indexOf("appendSample(samplesRef.current", guard);
  assert.ok(
    clear > 0 && clear < sample,
    "the hidden branch must clear, not sample",
  );
});

// The effect only runs when bytes change, so model that rather than a timer.
test("a stale estimate does not outlive the transfer that produced it", () => {
  const displayed = (guardHidden: boolean) => {
    const samples: TransferSample[] = [];
    let shown = { rate: 0, stable: false };
    let lastBytes = -1;
    const render = (t: number, bytes: number, hidden: boolean) => {
      if (bytes === lastBytes) return;
      lastBytes = bytes;
      if (guardHidden && hidden) {
        samples.length = 0;
        shown = { rate: 0, stable: false };
        return;
      }
      appendSample(samples, t, bytes);
      const s = computeTransferStats(samples, 100_000 * MB);
      shown = { rate: s.stable ? s.rateBytesPerSecond : 0, stable: s.stable };
    };

    let bytes = 0;
    let t = 0;
    for (; t <= 60; t += 1.5) {
      bytes += 75 * MB;
      render(t, bytes, false);
    }
    for (; t <= 660; t += 60) {
      bytes += 3_000 * MB;
      render(t, bytes, true);
    }
    for (; t <= 660 + 1_800; t += 1.5) render(t, bytes, false);
    return shown;
  };

  assert.ok(
    displayed(false).stable,
    "without the guard the dead transfer should still be showing a rate",
  );
  assert.equal(
    displayed(true).rate,
    0,
    "with the guard nothing survives the hidden stretch",
  );
});
