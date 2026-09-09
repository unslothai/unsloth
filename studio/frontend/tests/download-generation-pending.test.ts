// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// syncServerGeneration persists the new generation before the tick decides
// whether to poll progress, and status is polled twice as often as progress
// (500ms against 1000ms). A generation observed on a status-only tick was
// therefore already stored by the next one, so generationChanged read false and
// the samples from the previous server were never dropped. Holding the change
// until a progress poll consumes it is what makes the reset reliable.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  POLL_INTERVAL_MS,
  PROGRESS_POLL_INTERVAL_MS,
} from "../src/features/hub/download-manager/download-manager-config.ts";

const source = readFileSync(
  new URL("../src/features/hub/download-manager/poll-loop.ts", import.meta.url),
  "utf8",
);

test("status ticks outnumber progress polls, so the race is the normal case", () => {
  assert.ok(
    PROGRESS_POLL_INTERVAL_MS > POLL_INTERVAL_MS,
    "the race only exists because progress is polled less often than status",
  );
});

test("a generation change is held until a progress poll consumes it", () => {
  const set = source.indexOf("rt.pendingGenerationChange = true");
  const read = source.indexOf("rt.pendingGenerationChange === true");
  const clear = source.indexOf("rt.pendingGenerationChange = false");
  assert.ok(set > 0, "a generation change should be recorded on the runtime");
  assert.ok(read > 0, "the progress path should read the held change");
  assert.ok(clear > read, "it should be cleared only after it is read");
  // It must not be read straight out of syncServerGeneration's return value
  // again, or the early return swallows it exactly as before.
  assert.doesNotMatch(
    source,
    /const generationChanged = syncServerGeneration\(/,
    "reading the return value directly reintroduces the swallow",
  );
});

// The behaviour the flag buys, modelled on the real cadence: a change seen on a
// status-only tick still reaches the progress path.
test("a change seen between progress polls still reaches reconcile", () => {
  const consumed: boolean[] = [];
  const rt: { pendingGenerationChange?: boolean } = {};
  let storedGeneration = 1;
  let lastProgressAt = 0;

  const tick = (now: number, serverGeneration: number, sticky: boolean) => {
    // syncServerGeneration: persists immediately, reports the change once.
    const changed = serverGeneration !== storedGeneration;
    storedGeneration = serverGeneration;
    if (sticky && changed) rt.pendingGenerationChange = true;
    // shouldPollProgress: the early return that drops the signal.
    if (now - lastProgressAt < PROGRESS_POLL_INTERVAL_MS) return;
    lastProgressAt = now;
    if (sticky) {
      consumed.push(rt.pendingGenerationChange === true);
      rt.pendingGenerationChange = false;
    } else {
      consumed.push(changed);
    }
  };

  const run = (sticky: boolean) => {
    consumed.length = 0;
    rt.pendingGenerationChange = false;
    storedGeneration = 1;
    lastProgressAt = 0;
    // The backend restarts at t=500ms, which is a status-only tick.
    for (let now = 0; now <= 3_000; now += POLL_INTERVAL_MS) {
      tick(now, now >= 500 ? 2 : 1, sticky);
    }
    return consumed.some(Boolean);
  };

  assert.equal(run(false), false, "without the flag the change is swallowed");
  assert.equal(run(true), true, "with it the progress path still sees it");
});
