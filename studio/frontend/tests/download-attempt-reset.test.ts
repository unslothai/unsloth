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

test("an attempt change holds the GGUF floor off until the reading drops", () => {
  const source = readSrc("features/hub/download-manager/poll-loop.ts");
  assert.match(source, /if \(runChange === "attempt"\) \{\s*rt\.floorHold = \{/);
  assert.match(source, /skipFloor: rt\.floorHold != null,/);
  assert.match(
    source,
    /floorHoldEnded\(rt\.floorHold, downloadedBytes, Date\.now\(\)\)\s*\)\s*\{\s*rt\.floorHold = null;/,
  );
});

function reading(bytes: number, progress = bytes / EXPECTED): ProgressLike {
  return {
    downloaded_bytes: bytes,
    completed_bytes: 0,
    expected_bytes: EXPECTED,
    progress,
    cache_measured: true,
  } as ProgressLike;
}

// Shape measured in a real session (Qwen3-14B-GGUF BF16, Xet worker SIGKILLed at 34%): the reclaim
// bumps the attempt before the retry worker purges the killed partial, so the reset poll reads that
// partial -- a few more bytes than the last poll but a fraction a hair lower -- and the restart only
// shows up on later polls.
function replayRetry(hold: boolean): number[] {
  const MB = 1024 * 1024;
  const polls: ProgressLike[] = [
    reading(2 * GB + 100 * MB, 0.3995),
    reading(2 * GB + 100 * MB),
    reading(6 * MB),
    reading(200 * MB),
    reading(400 * MB),
  ];
  let current = job();
  let floorHold = hold
    ? { bytes: current.downloadedBytes, until: 60_000 }
    : null;
  const fractions: number[] = [];
  polls.forEach((poll, i) => {
    const next = resolveProgressUpdate(current, poll, {
      resetMonotonic: i === 0,
      skipFloor: floorHold !== null,
    });
    if (floorHold && floorHoldEnded(floorHold, next.downloadedBytes, i * 2_000)) {
      floorHold = null;
    }
    current = { ...current, ...next, fraction: next.fraction };
    fractions.push(next.fraction);
  });
  return fractions;
}

test("a reset spent on the killed run's partial still lets the bar follow the restart", () => {
  const pinned = replayRetry(false);
  assert.ok(
    pinned.slice(1).every((f) => f >= 0.39),
    `without the hold the bar stays at the old run's mark: ${pinned}`,
  );

  const followed = replayRetry(true);
  assert.ok(followed[2] < 0.01, `the restart should show: ${followed}`);
  assert.ok(followed[4] > followed[3] && followed[4] < 0.1, `${followed}`);
});

test("the floor hold ends on a byte drop or at its deadline", () => {
  const hold = { bytes: 2 * GB, until: 10_000 };
  assert.equal(floorHoldEnded(hold, 2 * GB + 1, 0), false);
  assert.equal(floorHoldEnded(hold, 6 * 1024 * 1024, 0), true);
  assert.equal(floorHoldEnded(hold, 2 * GB, 10_000), true);
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
