// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { carriesOverSeed, idleProbeVerdict, seededMeasuredTransfer } = await import(
  "../src/features/hub/download-manager/adopt-rules.ts"
);

const POLL_LOOP = readSrc("features/hub/download-manager/poll-loop.ts");

test("adopting a new generation drops the previous run's byte seed", () => {
  assert.equal(carriesOverSeed(true, 4, 5), false);
  assert.equal(carriesOverSeed(true, 4, 4), true, "the same run keeps its counters");
});

test("an unknown generation is not evidence of a new run", () => {
  assert.equal(carriesOverSeed(true, 4, undefined), true);
  assert.equal(carriesOverSeed(true, undefined, 5), true);
  assert.equal(carriesOverSeed(true, 4, Number.NaN), true);
  assert.equal(carriesOverSeed(false, 4, 5), false, "a fresh start never seeds");
});

test("only an absent cache retires a hydrated job, not a zero reading", () => {
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x"), "active");
  assert.equal(idleProbeVerdict(0, null), "gone");
  assert.equal(idleProbeVerdict(1024, null), "active", "bytes outrank a missing path");
});

test("an older backend that omits cache_path leaves the job adoptable", () => {
  assert.equal(idleProbeVerdict(0, undefined), "active");
});

test("a variant whose own files are gone does not survive on a sibling's cache dir", () => {
  assert.equal(
    idleProbeVerdict(0, "/hub/models--unsloth--x", false),
    "gone",
    "the repo dir is the wrong granularity for a variant",
  );
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x", null), "active");
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x", undefined), "active");
  assert.equal(idleProbeVerdict(4096, "/hub/models--unsloth--x", false), "gone");
  assert.equal(idleProbeVerdict(4096, "/hub/models--unsloth--x", null), "active");
  assert.equal(idleProbeVerdict(0, null, true), "gone", "no cache at all is still gone");
});

test("a measured scan with no cache path retires the job however it was serialized", () => {
  // The endpoint uses response_model_exclude_none, so a measured-empty answer omits cache_path.
  assert.equal(idleProbeVerdict(0, undefined, null, true), "gone");
  assert.equal(idleProbeVerdict(0, null, null, true), "gone");
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x", null, true), "active");
  assert.equal(idleProbeVerdict(0, undefined, null, false), "active");
  assert.equal(idleProbeVerdict(0, undefined, null, undefined), "active");
});


test("a scan that never happened does not retire a job", () => {
  assert.equal(idleProbeVerdict(0, null, null, false), "active");
  assert.equal(idleProbeVerdict(0, null, false, false), "active");
  assert.equal(idleProbeVerdict(0, null, null, true), "gone");
  assert.equal(idleProbeVerdict(0, null, null, undefined), "gone");
});

test("live idle polls retire an explicitly missing target before the grace period", () => {
  const start = POLL_LOOP.indexOf("function handleIdleAfterProgress");
  const end = POLL_LOOP.indexOf("\nfunction handleTickError", start);
  const handler = POLL_LOOP.slice(start, end);

  assert.match(
    handler,
    /idleProbeVerdict\([\s\S]*progressResp\.target_present[\s\S]*\) === "gone"/,
    "an authoritative missing-target response must bypass the idle grace period",
  );
  assert.match(
    POLL_LOOP,
    /handleIdleAfterProgress\(rt, key, madeProgress, progressResp\)/,
    "the live poll must pass its progress response to the idle verdict",
  );
});

test("the held-transfer marker travels with the counters it describes", () => {
  assert.equal(seededMeasuredTransfer(true, false), false);
  assert.equal(seededMeasuredTransfer(true, true), true);
  assert.equal(seededMeasuredTransfer(false, false), undefined);
  assert.equal(seededMeasuredTransfer(true, undefined), undefined);
});

test("the adoption path actually seeds the marker onto the job", () => {
  assert.match(
    POLL_LOOP,
    /measuredTransfer:\s*seedMeasuredTransfer/,
    "startJob computes the seeded held-transfer marker but no longer puts it on "
      + "the job, so an adopted run restores it as undefined (measured) and "
      + "prices the retry against the dead run's bytes",
  );
});
