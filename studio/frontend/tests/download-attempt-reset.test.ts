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

const { resolveProgressUpdate } =
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
  assert.match(body, /return generationChanged \|\| attemptChanged;/);
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
