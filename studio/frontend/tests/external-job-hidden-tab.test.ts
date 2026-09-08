// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// External trackers own their own timers: the STT mirror runs a 750ms interval,
// and a hidden tab's are clamped to about once a minute. The estimator reads
// gaps between increases as the burst cadence, so those gaps stretch the stall
// window and leave a stale rate on the Downloads row. The guard lives in
// updateExternalJob so every external tracker inherits it.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
Object.assign(globalThis.window, { addEventListener: () => {} });

const visibility = { hidden: false };
Object.defineProperty(globalThis, "document", {
  configurable: true,
  value: {
    get hidden() {
      return visibility.hidden;
    },
  },
});

const { startExternalJob, updateExternalJob } = await import(
  "../src/features/hub/download-manager/external-jobs.ts"
);
const { getState } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);

const KEY = "model:org/stt-mirror";
const MB = 1e6;

test("a hidden tab's throttled samples never become a published rate", () => {
  // updateExternalJob reads the wall clock itself, so drive it: without real
  // spacing the estimator would be unstable anyway and the test would pass for
  // the wrong reason.
  const realNow = Date.now;
  let now = realNow();
  Date.now = () => now;
  try {
    startExternalJob({
      key: KEY,
      repoId: "org/stt-mirror",
      variant: null,
      expectedBytes: 8_000 * MB,
      cancel: () => {},
    });

    // Visible and healthy first, at the tracker's own 750ms cadence.
    visibility.hidden = false;
    let bytes = 0;
    for (let i = 0; i <= 60; i += 1) {
      now += 750;
      bytes += 37.5 * MB; // 50 MB/s at 750ms
      updateExternalJob(KEY, {
        downloadedBytes: bytes,
        expectedBytes: 8_000 * MB,
      });
    }
    assert.ok(
      getState().jobs[KEY].bytesPerSec > 0,
      "the visible phase should publish a rate, or this test proves nothing",
    );

    // Backgrounded: browsers clamp the interval to about once a minute.
    visibility.hidden = true;
    for (let i = 0; i < 10; i += 1) {
      now += 60_000;
      bytes += 3_000 * MB;
      updateExternalJob(KEY, {
        downloadedBytes: bytes,
        expectedBytes: 8_000 * MB,
      });
    }

    const job = getState().jobs[KEY];
    assert.ok(job, "the external job should still exist");
    assert.equal(job.bytesPerSec, 0, "a hidden tab must publish no rate");
    assert.equal(job.etaSeconds, 0, "a hidden tab must publish no ETA");
    // Progress itself is still worth recording; only the timing is not.
    assert.equal(job.downloadedBytes, bytes);
  } finally {
    Date.now = realNow;
  }
});
