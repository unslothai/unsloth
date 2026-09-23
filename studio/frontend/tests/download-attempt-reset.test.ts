// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  ManagedDownload,
  ProgressLike,
} from "../src/features/hub/download-manager/download-manager-types.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { floorHoldEnded, resolveProgressUpdate } =
  await import("../src/features/hub/download-manager/progress-reconcile.ts");

const GB = 1024 ** 3;
const EXPECTED = 5 * GB;

function job(overrides: Partial<ManagedDownload> = {}): ManagedDownload {
  return {
    key: "model:org/model-gguf#q4_k_m",
    kind: "model",
    repoId: "org/model-GGUF",
    variant: "Q4_K_M",
    state: "running",
    downloadedBytes: 2 * GB,
    completedBytes: 0,
    expectedBytes: EXPECTED,
    completeOnDisk: false,
    fraction: 0.4,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 1,
    serverGeneration: 7,
    serverAttempt: 1,
    ...overrides,
  };
}

function restartedReading(): ProgressLike {
  return {
    downloaded_bytes: 6 * 1024 * 1024,
    completed_bytes: 0,
    expected_bytes: EXPECTED,
    progress: (6 * 1024 * 1024) / EXPECTED,
    cache_measured: true,
  } as ProgressLike;
}

test("the status sync treats an attempt change like a generation change", () => {
  const source = readSrc("features/hub/download-manager/poll-loop.ts");
  const sync = source.indexOf("function syncServerGeneration(");
  const body = source.slice(
    sync,
    source.indexOf("function runCounterChanged("),
  );
  assert.match(body, /status\.attempt/);
  assert.match(body, /job\.serverAttempt/);
  assert.match(body, /if \(generationChanged\) return "generation";/);
  assert.match(body, /return attemptChanged \? "attempt" : null;/);
});

test("an attempt change holds the GGUF floor off until the bytes left grow", () => {
  const source = readSrc("features/hub/download-manager/poll-loop.ts");
  assert.match(source, /if \(runChange === "attempt"\) \{\s*rt\.floorHold = \{/);
  assert.match(source, /skipFloor: rt\.floorHold != null,/);
  assert.match(
    source,
    /floorHoldEnded\(rt\.floorHold, expected, downloadedBytes, Date\.now\(\)\)\s*\)\s*\{\s*rt\.floorHold = null;/,
  );
});

const MB = 1024 * 1024;

function reading(
  bytes: number,
  expected = EXPECTED,
  progress = bytes / expected,
): ProgressLike {
  return {
    downloaded_bytes: bytes,
    completed_bytes: 0,
    expected_bytes: expected,
    progress,
    cache_measured: true,
  } as ProgressLike;
}

// Replays polls after an attempt change: the first is the reset poll, and `hold` mirrors poll-loop.
function replayRetry(polls: ProgressLike[], hold: boolean): number[] {
  let current = job();
  let floorHold = hold
    ? {
        remainingBytes: current.expectedBytes - current.downloadedBytes,
        until: 60_000,
      }
    : null;
  const fractions: number[] = [];
  polls.forEach((poll, i) => {
    const next = resolveProgressUpdate(current, poll, {
      resetMonotonic: i === 0,
      skipFloor: floorHold !== null,
    });
    if (
      floorHold &&
      floorHoldEnded(floorHold, next.expected, next.downloadedBytes, i * 2_000)
    ) {
      floorHold = null;
    }
    current = { ...current, ...next, expectedBytes: next.expected };
    fractions.push(next.fraction);
  });
  return fractions;
}

// Shape measured in a real session (Qwen3-14B-GGUF BF16, Xet worker SIGKILLed at 34%): the reclaim
// bumps the attempt before the retry worker purges the killed partial, so the reset poll reads that
// partial -- a few more bytes than the last poll but a fraction a hair lower -- and the restart only
// shows up on later polls.
test("a reset spent on the killed run's partial still lets the bar follow the restart", () => {
  const polls = [
    reading(2 * GB + 100 * MB, EXPECTED, 0.3995),
    reading(2 * GB + 100 * MB),
    reading(6 * MB),
    reading(200 * MB),
    reading(400 * MB),
  ];
  const pinned = replayRetry(polls, false);
  assert.ok(
    pinned.slice(1).every((f) => f >= 0.39),
    `without the hold the bar stays at the old run's mark: ${pinned}`,
  );

  const followed = replayRetry(polls, true);
  assert.ok(followed[2] < 0.01, `the restart should show: ${followed}`);
  assert.ok(followed[4] > followed[3] && followed[4] < 0.1, `${followed}`);
});

// A split GGUF whose first 1 GB shard finished during the killed attempt: the reclaim re-measures the
// completed baseline, so the stale reading loses 1 GB from both counters before the partial is purged.
test("a re-measured completed baseline does not end the hold before the purge", () => {
  const polls = [
    reading(1 * GB, 4 * GB),
    reading(1 * GB, 4 * GB),
    reading(6 * MB, 4 * GB),
    reading(200 * MB, 4 * GB),
  ];
  const followed = replayRetry(polls, true);
  assert.ok(followed[2] < 0.01, `the restart should show: ${followed}`);
  assert.ok(followed[3] > followed[2] && followed[3] < 0.1, `${followed}`);
});

test("the floor hold ends when the bytes left grow or at its deadline", () => {
  const hold = { remainingBytes: 3 * GB, until: 10_000 };
  assert.equal(floorHoldEnded(hold, 5 * GB, 2 * GB, 0), false);
  assert.equal(floorHoldEnded(hold, 4 * GB, 1 * GB, 0), false);
  assert.equal(floorHoldEnded(hold, 5 * GB, 6 * MB, 0), true);
  assert.equal(floorHoldEnded(hold, 5 * GB, 2 * GB, 10_000), true);
});

test("a same-generation retry drops the bar to the new attempt's bytes", () => {
  const reset = resolveProgressUpdate(job(), restartedReading(), {
    resetMonotonic: true,
  });
  assert.equal(reset.downloadedBytes, 6 * 1024 * 1024);
  assert.ok(
    reset.fraction < 0.01,
    `fraction ${reset.fraction} should follow the bytes`,
  );

  const held = resolveProgressUpdate(job(), restartedReading());
  assert.equal(held.downloadedBytes, 6 * 1024 * 1024);
  assert.equal(held.fraction, 0.4);
});

test("an adopted job carries the persisted attempt so a retry while closed is detected", () => {
  const source = readSrc("features/hub/download-manager/poll-loop.ts");
  assert.match(
    source,
    /const seedAttempt = carryOverSeed \? existing\?\.serverAttempt : undefined;/,
  );
  assert.match(source, /\? \{ serverAttempt: seedAttempt \}/);
});

test("the attempt survives persistence next to the generation", () => {
  const source = readSrc(
    "features/hub/download-manager/download-manager-state.ts",
  );
  assert.match(source, /value\.serverAttempt/);
  assert.match(source, /job\.serverAttempt/);
});
